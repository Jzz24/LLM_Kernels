# The Ring All-Reduce logic in this script is based on the implementation from:
# https://github.com/rajesh-s/mlsys-allreduce/tree/main

import os
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp

DTYPE = torch.float16

def run(rank, world_size, tensor_size):

    # --- 1. 初始化分布式环境 ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6585'
    
    # 为当前进程设置对应的GPU设备并初始化进程组
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size)

    print(f"Process {rank}/{world_size} initialized on {torch.cuda.current_device()}.")

    # --- 2. 准备数据 ---
    device = f'cuda:{rank}'
    t_full = torch.rand(tensor_size, device=device, dtype=DTYPE)
    t_full_pytorch = t_full.clone()

    # 将完整张量切分成块
    t_chunks = list(t_full.split(tensor_size // world_size))
    
    # 在GPU上创建接收缓冲区
    zero_buffer = torch.zeros_like(t_chunks[0])
    dist.barrier()
    
    # --- 3. 自定义 Ring All-Reduce ---

    # --- 阶段 A: Reduce-scatter 循环 ---
    for i in range(world_size - 1):
        send_chunk_idx = (rank - i + world_size) % world_size
        recv_chunk_idx = (rank - 1 - i + world_size) % world_size
        
        # 奇偶排序通信，阻塞通信（send/recv）避免死锁，无需手动wait()同步
        if (rank % 2) == 0:
            dist.send(t_chunks[send_chunk_idx], dst=(rank + 1) % world_size)
            dist.recv(zero_buffer, src=(rank - 1 + world_size) % world_size)
        else:
            dist.recv(zero_buffer, src=(rank - 1 + world_size) % world_size)
            dist.send(t_chunks[send_chunk_idx], dst=(rank + 1) % world_size)
        
        # 累加值
        # Note: t_chunks仍指向t_full的切片视图，并未重新分配内存，需要inplace修改t_chunks，才能影响t_full
        t_chunks[recv_chunk_idx].copy_((t_chunks[recv_chunk_idx].float() + zero_buffer.float()).to(DTYPE))
    
    # --- 阶段 B: All-gather 循环 ---
    for i in range(world_size - 1):
        send_chunk_idx = (rank + 1 - i + world_size) % world_size
        recv_chunk_idx = (rank - i + world_size) % world_size
        
        if (rank % 2) == 0:
            dist.send(t_chunks[send_chunk_idx], dst=(rank + 1) % world_size)
            dist.recv(t_chunks[recv_chunk_idx], src=(rank - 1 + world_size) % world_size)
        else:
            dist.recv(t_chunks[recv_chunk_idx], src=(rank - 1 + world_size) % world_size)
            dist.send(t_chunks[send_chunk_idx], dst=(rank + 1) % world_size)
    
    dist.barrier()


    # --- 4. 结果验证 ---
    # 使用 PyTorch 内置的 all_reduce 作为正确性参考
    dist.all_reduce(t_full_pytorch, op=dist.ReduceOp.SUM)

    if torch.allclose(t_full, t_full_pytorch, rtol=1e-3, atol=1e-2):
        print(f"Rank {rank}: ✅ Correctness check PASSED!")
    else:
        print(f"Rank {rank}: ❌ Correctness check FAILED!")
        diff = torch.abs(t_full - t_full_pytorch).float()
        ref_abs = torch.abs(t_full_pytorch).float()
        percent = diff / (ref_abs + 1e-8)  # 防止除零
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(t_full.flatten().float(), 
                                                        t_full_pytorch.flatten().float(), dim=0)

        print(f"   - Rank {rank} average absolute difference: {diff.mean().item()}")
        print(f"   - Rank {rank} max absolute difference: {diff.max().item()}")
        print(f"   - Rank {rank} average percent error: {percent.mean().item():.6f}")
        print(f"   - Rank {rank} max percent error: {percent.max().item():.6f}")
        print(f"   - Rank {rank} cosine similarity: {cos_sim.item():.6f}")



    # --- 5. 清理 ---
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # 获取可用的GPU数量
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("This script requires at least 2 GPUs to run.")
    else:
        print(f"Found {world_size} GPUs. Starting single-node multi-GPU All-Reduce test.")
        
        # 定义要进行All-Reduce的张量大小 (例如 128M 个浮点数)
        TENSOR_SIZE = 1024 * 1024 * 128

        # 确保张量大小可以被GPU数量整除
        if TENSOR_SIZE % world_size != 0:
            TENSOR_SIZE = (TENSOR_SIZE // world_size) * world_size
            print(f"Tensor size adjusted to {TENSOR_SIZE} to be divisible by {world_size} GPUs.")

        mp.spawn(run,
                 args=(world_size, TENSOR_SIZE),
                 nprocs=world_size,
                 join=True)