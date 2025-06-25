# This script implements All-Reduce using two All-to-All operations.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 可以切换 DTYPE 来对比不同精度的表现
DTYPE = torch.float16

def run(rank, world_size, tensor_size):

    # --- 1. 初始化分布式环境 ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6585'
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size)

    print(f"Process {rank}/{world_size} initialized on {torch.cuda.current_device()}.")

    # --- 2. 准备数据 ---
    device = f'cuda:{rank}'
    t_full = torch.rand(tensor_size, device=device, dtype=DTYPE)
    t_full_pytorch = t_full.clone()

    # 将每个GPU上的完整张量切分成 world_size 个块
    t_chunks = list(t_full.split(tensor_size // world_size))
    
    dist.barrier()
    

    # 1. All-to-All: 将数据块分散到对应的GPU上
    scatter_buffer = [torch.zeros_like(chunk) for chunk in t_chunks]
    dist.all_to_all(scatter_buffer, t_chunks)

    # 2. Local Sum: 在每个GPU上独立地对接收到的数据块求和
    local_sum_chunk = torch.sum(torch.stack(scatter_buffer).float(), dim=0).to(DTYPE)
    t_chunks[rank].copy_(local_sum_chunk)

    dist.barrier()

    # --- 3. All-Gather ---
    dist.all_gather(t_chunks, t_chunks[rank])
    dist.barrier()

    # --- 4. 结果验证 ---
    dist.all_reduce(t_full_pytorch, op=dist.ReduceOp.SUM)

    if torch.allclose(t_full, t_full_pytorch, rtol=1e-3, atol=1e-2):
        print(f"Rank {rank}: ✅ Correctness check PASSED!")
    else:
        print(f"Rank {rank}: ❌ Correctness check FAILED!")
        diff = torch.abs(t_full - t_full_pytorch).float()
        ref_abs = torch.abs(t_full_pytorch).float()
        percent = diff / (ref_abs + 1e-8)
        
        cos_sim = torch.nn.functional.cosine_similarity(t_full.flatten().float(), 
                                                        t_full_pytorch.flatten().float(), dim=0)

        print(f"   - Rank {rank} average absolute difference: {diff.mean().item()}")
        print(f"   - Rank {rank} max absolute difference: {diff.max().item()}")
        print(f"   - Rank {rank} average percent error: {percent.mean().item() * 100:.6f}%")
        print(f"   - Rank {rank} max percent error: {percent.max().item() * 100:.6f}%")
        print(f"   - Rank {rank} cosine similarity: {cos_sim.item():.6f}")

    # --- 5. 清理 ---
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("This script requires at least 2 GPUs to run.")
    else:
        print(f"Found {world_size} GPUs. Starting single-node multi-GPU All-Reduce test.")
        
        TENSOR_SIZE = 1024 * 1024 * 128

        if TENSOR_SIZE % world_size != 0:
            TENSOR_SIZE = (TENSOR_SIZE // world_size) * world_size
            print(f"Tensor size adjusted to {TENSOR_SIZE} to be divisible by {world_size} GPUs.")

        mp.spawn(run,
                 args=(world_size, TENSOR_SIZE),
                 nprocs=world_size,
                 join=True)