#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPMoE (All-Gather + All-Reduce) Implementation Test

Tests the functionality of the EPMoE implementation with the All-Gather + All-Reduce pattern.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang_moe_all_gather_all_reduce import EPMoE
from utils import setup_logging


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_on_gpu(rank, world_size):
    """Run EPMoE model on a single GPU."""
    setup_distributed(rank, world_size)
    logger = setup_logging(f"epmoe_test_rank{rank}")
    
    torch.set_default_dtype(torch.float16)
    num_experts = 256  # 必须能被world_size整除
    top_k = 8
    hidden_size = 7168
    intermediate_size = 2048
    batch_size = 2
    seq_len = 256
    seed = 42

    # 创建模型
    model = EPMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tp_size=world_size,
        activation="silu",
        renormalize=True
    ).eval().cuda()
    
    logger.info(f"Model creation completed")
    
    # 初始化权重
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.nn.init.normal_(model.gate.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(model.w13_weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(model.w2_weight, mean=0.0, std=0.01)
    
    input_data = torch.randn(
        batch_size, seq_len, hidden_size, 
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
    
    output_sum = output.sum().item()
    logger.info(f"GPU {rank} Output sum: {output_sum:.4f}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        world_size = 2
        mp.spawn(run_on_gpu, args=(world_size,), nprocs=world_size, join=True)
    else:
        run_on_gpu(0, 1)