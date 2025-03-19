#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-to-All MoE Implementation Test

Tests the functionality of the All-to-All MoE implementation.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from all2all_moe import DeepseekV3MoE, Config
from utils import setup_logging


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_on_gpu(rank, world_size):
    """Run AllToAllMoE model on a single GPU."""
    setup_distributed(rank, world_size)
    logger = setup_logging(f"all2all_test_rank{rank}")
    
    torch.set_default_dtype(torch.float16)
    seed = 42
    batch_size = 2
    seq_len = 256
    
    # 创建模型配置
    config = Config()
    config.ep_size = world_size
    
    # 创建模型
    model = DeepseekV3MoE(config).eval().cuda()
    logger.info(f"Model creation completed")
    
    # 初始化权重
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.nn.init.normal_(model.gate.weight, mean=0.0, std=0.1)
    
    # 创建输入数据
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
    
    output_sum = output.sum().item()
    logger.info(f"GPU {rank} Output sum: {output_sum:.4f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        world_size = 2
        mp.spawn(run_on_gpu, args=(world_size,), nprocs=world_size, join=True)
    else:
        run_on_gpu(0, 1)