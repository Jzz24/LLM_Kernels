#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Equivalence Validation for Mixture-of-Experts Communication Strategies

This module validates the numerical equivalence between different MoE implementations:
- All-to-All communication pattern
- All-Gather + All-Reduce communication pattern

The validation includes detailed numerical analysis of outputs and intermediate activations.
"""

import os
import time
import argparse
import logging
from typing import Dict, Tuple, Optional, Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from all2all_moe import Config
from all2all_moe import DeepseekV3MoE as AllToAllMoE
from sglang_moe_all_gather_all_reduce import EPMoE
from utils import ResultCollector, ResultAnalyzer, setup_logging


# Configuration constants
DEFAULT_SEED = 42


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Initialize the distributed environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def synchronize_weights(source_model: AllToAllMoE, target_model: EPMoE) -> None:
    """
    Synchronize weights between two different MoE implementations,
    handling the structural differences between models.
    
    Args:
        source_model: AllToAllMoE model (with individual experts)
        target_model: EPMoE model (with fused expert weights)
    """
    with torch.no_grad():
        # Copy gate weights
        target_model.gate.weight.data.copy_(source_model.gate.weight.data)
        if hasattr(source_model.gate, "bias") and source_model.gate.bias is not None:
            target_model.gate.bias.data.copy_(source_model.gate.bias.data)

        is_quantized = (hasattr(target_model, 'config') and 
                target_model.config.quantize_method is not None and 
                target_model.config.use_block_quant)
        if is_quantized:
            from quantization.int8_kernel import weight_quant
            block_size = target_model.config.w_quant_block_size[0]
        
        # Copy expert weights - respecting the distributed expert partitioning
        for i, expert in enumerate(source_model.experts):
            # Only copy experts assigned to this GPU's partition
            if (expert is not None and 
                i >= target_model.start_expert_id and 
                i < target_model.start_expert_id + target_model.num_experts_per_partition):
                
                # Calculate local expert index within this partition
                local_i = i - target_model.start_expert_id

                if is_quantized:
                    w13_fp16 = torch.empty(
                        2 * target_model.intermediate_size,
                        target_model.hidden_size,
                        device=target_model.w13_weight.device,
                        dtype=torch.float16
                    )
                    
                    w13_fp16[:target_model.intermediate_size, :] = expert.gate_proj.weight.data
                    w13_fp16[target_model.intermediate_size:, :] = expert.up_proj.weight.data
                    
                    w13_quant, w13_scale = weight_quant(w13_fp16, block_size=block_size)
                    target_model.w13_weight[local_i] = w13_quant
                    target_model.w13_weight_scale[local_i] = w13_scale
                    
                    w2_quant, w2_scale = weight_quant(expert.down_proj.weight.data, block_size=block_size)
                    target_model.w2_weight[local_i] = w2_quant
                    target_model.w2_weight_scale[local_i] = w2_scale
                else:
                    # Copy gate_proj (W1) to the first half of w13_weight
                    target_model.w13_weight.data[local_i, :target_model.intermediate_size, :] = (
                        expert.gate_proj.weight.data
                    )
                    
                    # Copy up_proj (W3) to the second half of w13_weight
                    target_model.w13_weight.data[local_i, target_model.intermediate_size:, :] = (
                        expert.up_proj.weight.data
                    )
                    
                    # Copy down_proj (W2)
                    target_model.w2_weight.data[local_i] = expert.down_proj.weight.data
                
        logger = logging.getLogger("moe_validation")
        logger.info(f"Model weights synchronized successfully")


def create_models(config: Config, world_size: int) -> Tuple[AllToAllMoE, EPMoE]:
    """
    Create both MoE model implementations with identical configurations.
    
    Args:
        config: Model configuration
        world_size: Number of processes/GPUs
        
    Returns:
        Tuple of (all_to_all_model, all_gather_model)
    """
    model_all_to_all = AllToAllMoE(config).eval().cuda()
    
    model_all_gather = EPMoE(config).eval().cuda()
    
    return model_all_to_all, model_all_gather


def validate_models(
    rank: int, 
    world_size: int,
    batch_size: int = 2,
    seq_len: int = 256,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Run validation comparing two MoE implementations.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of validation results
    """
    logger = setup_logging()
    logger.info(f"Rank {rank}: Starting validation with {world_size} GPUs")
    
    # Initialize distributed environment
    setup_distributed(rank, world_size)
    
    # Set tensor type and random seed
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(seed)
    
    # Create models with identical configuration
    config = Config()
    config.ep_size = world_size
    model_all_to_all, model_all_gather = create_models(config, world_size)
    
    # Ensure identical weights
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.nn.init.normal_(model_all_to_all.gate.weight, mean=0.0, std=0.1)
        
        # Copy weights to all-gather model
        synchronize_weights(model_all_to_all, model_all_gather)
    

    input_data = torch.randn(
        batch_size, seq_len, config.hidden_size, 
        device=f"cuda:{rank}", 
        dtype=torch.float16
    )
    logger.info(f"Rank {rank}: Input data sum: {input_data.sum().item():.4f}")
    
    # Synchronize before timing
    dist.barrier()
    
    # Run all-to-all model
    logger.info(f"Rank {rank}: Running all-to-all model")
    all_to_all_start = time.time()
    with torch.no_grad():
        output_all_to_all = model_all_to_all(input_data)
    all_to_all_end = time.time()
    all_to_all_time = all_to_all_end - all_to_all_start
    
    # Run all-gather model
    logger.info(f"Rank {rank}: Running all-gather model")
    all_gather_start = time.time()
    with torch.no_grad():
        output_all_gather = model_all_gather(input_data)
    all_gather_end = time.time()
    all_gather_time = all_gather_end - all_gather_start
    
    # Compare outputs
    logger.info(f"Rank {rank}: Comparing outputs")
    results = ResultAnalyzer.compare_tensors(
        output_all_gather, 
        output_all_to_all, 
        var_name="output",
        rank=rank
    )
    
    # Add timing information
    logger.info(f"Rank {rank}: All-gather time: {all_gather_time:.4f}s, "
                f"All-to-all time: {all_to_all_time:.4f}s, "
                f"Speedup: {all_gather_time/all_to_all_time:.2f}x")
    
    # Cleanup
    dist.destroy_process_group()
    
    return results


def main():
    """Parse arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Validate numerical equivalence between MoE implementations"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--world-size", type=int, default=2, 
                        help="Number of GPUs (defaults to available GPUs)")

    args = parser.parse_args()
    
    # Determine world size
    world_size = args.world_size or torch.cuda.device_count()
    world_size = min(world_size, torch.cuda.device_count())
    
    if world_size > 1:
        # Run with multiple GPUs
        mp.spawn(
            validate_models,
            args=(world_size, args.batch_size, args.seq_len, args.seed),
            nprocs=world_size,
            join=True
        )
    else:
        # Run with single GPU
        validate_models(0, 1, args.batch_size, args.seq_len, args.seed,)


if __name__ == "__main__":
    main()