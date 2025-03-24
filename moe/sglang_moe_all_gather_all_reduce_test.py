#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPMoE (All-Gather + All-Reduce) Implementation Test

Tests the functionality of the EPMoE implementation with the All-Gather + All-Reduce pattern.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang_moe_all_gather_all_reduce import EPMoE
from utils import setup_logging
from config import Config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def initialize_quantized_weights(model, seed=42, std=0.01, block_size=128, quantize=False):
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.nn.init.normal_(model.gate.weight, mean=0.0, std=0.1)

        w13_fp16 = torch.empty(
            model.num_experts_per_partition,
            2 * model.intermediate_size,
            model.hidden_size,
            device=model.w13_weight.device,
            dtype=torch.float16
        )

        torch.nn.init.normal_(w13_fp16, mean=0.0, std=std)
        
        w2_fp16 = torch.empty(
            model.num_experts_per_partition,
            model.hidden_size,
            model.intermediate_size,
            device=model.w2_weight.device,
            dtype=torch.float16
        )
        torch.nn.init.normal_(w2_fp16, mean=0.0, std=std)
        

        if not quantize:
            model.w13_weight.data.copy_(w13_fp16)
            model.w2_weight.data.copy_(w2_fp16)
        else:
            from quantization.int8_kernel import weight_quant
            for expert_idx in range(model.num_experts_per_partition):
                # 量化 w13_weight
                w13_quant, w13_scale = weight_quant(w13_fp16[expert_idx], block_size=block_size)
                model.w13_weight[expert_idx] = w13_quant
                model.w13_weight_scale[expert_idx] = w13_scale
                
                # 量化 w2_weight
                w2_quant, w2_scale = weight_quant(w2_fp16[expert_idx], block_size=block_size)
                model.w2_weight[expert_idx] = w2_quant
                model.w2_weight_scale[expert_idx] = w2_scale
    return model

def run_on_gpu(rank, world_size):
    """Run EPMoE model on a single GPU."""
    setup_distributed(rank, world_size)
    torch.set_default_dtype(torch.float16)
    logger = setup_logging(f"epmoe_test_rank{rank}")

    batch_size = 2
    seq_len = 512
    seed = 42

    config = Config()
    config.ep_size = world_size
    model = EPMoE(config).eval().cuda()
    logger.info(f"Model creation completed")

    quantize_flag = config.quantize_method is not None and config.use_block_quant
    model = initialize_quantized_weights(model, seed=seed, std=0.01, block_size=128, quantize=quantize_flag)

    input_data = torch.randn(
        batch_size, seq_len, config.hidden_size, 
        device=f"cuda:{rank}", 
        dtype=torch.float16
    )
    
    input_sum = input_data.sum().item()
    logger.info(f"Input data sum: {input_sum:.4f}")
    
    # 同步进程
    dist.barrier()
    
    # 前向传播
    logger.info(f"Starting forward pass")
    with torch.no_grad():
        output = model(input_data)
    
    output_sum = output.float().sum().item()
    logger.info(f"GPU {rank} Output sum: {output_sum:.4f}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        world_size = 2
        mp.spawn(run_on_gpu, args=(world_size,), nprocs=world_size, join=True)
    else:
        run_on_gpu(0, 1)