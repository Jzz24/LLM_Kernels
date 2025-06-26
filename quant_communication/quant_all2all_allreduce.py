# This script implements a quantized All-Reduce using All-to-All and All-Gather operations.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from quant_utils import int8_quant, int8_dequant, fp8_quant, fp8_dequant

DTYPE = torch.float16
QUANT_TYPE = 'int8' # 'int8' or 'fp8'
QUANT_BLOCK_SIZE = 32

def run(rank, world_size, tensor_size):

    # --- 1. Initialize distributed environment ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6585'
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size)

    # --- Select quantization functions and parameters based on QUANT_TYPE ---
    if QUANT_TYPE == 'int8':
        quant_func = int8_quant
        dequant_func = int8_dequant
        quant_block_size = QUANT_BLOCK_SIZE
        quant_dtype = torch.int8
        scale_dtype = torch.float16
        transmission_dtype = torch.int8
    elif QUANT_TYPE == 'fp8':
        quant_func = fp8_quant
        dequant_func = fp8_dequant
        quant_block_size = QUANT_BLOCK_SIZE
        quant_dtype = torch.float8_e4m3fn
        scale_dtype = torch.float16
        transmission_dtype = torch.int8 # NCCL does not support FP8 directly, must use INT8 view
    else:
        raise ValueError(f"Unsupported QUANT_TYPE: {QUANT_TYPE}")

    # --- 2. Prepare data ---
    device = f'cuda:{rank}'
    t_full = torch.randn(tensor_size, device=device, dtype=DTYPE)
    t_full_pytorch = t_full.clone()

    # Split the full tensor on each GPU into world_size chunks
    t_chunks = list(t_full.split(tensor_size // world_size))
    
    dist.barrier()
    
    # --- 3. Custom Quantized All-to-All All-Reduce ---

    # --- Stage A: Quantized Reduce-Scatter (implemented with All-to-All) ---
    
    # 1. Quantize each chunk to be sent
    quant_data_to_send_list = []
    scale_data_to_send_list = []
    for chunk in t_chunks:
        q_data, q_scale = quant_func(chunk, block_size=quant_block_size)
        quant_data_to_send_list.append(q_data.view(transmission_dtype))
        scale_data_to_send_list.append(q_scale)

    # 2. Prepare receive buffers
    recv_quant_list = [torch.zeros_like(q) for q in quant_data_to_send_list]
    recv_scale_list = [torch.zeros_like(s) for s in scale_data_to_send_list]

    # 3. Perform two All-to-All operations
    dist.all_to_all(recv_quant_list, quant_data_to_send_list)
    dist.all_to_all(recv_scale_list, scale_data_to_send_list)

    # 4. Dequantize the received chunks
    dequantized_chunks = []
    for q_data, q_scale in zip(recv_quant_list, recv_scale_list):
        dequant_chunk = dequant_func(q_data.view(quant_dtype), q_scale, output_dtype=DTYPE)
        dequantized_chunks.append(dequant_chunk)

    # 5. Sum the received chunks locally
    local_sum_chunk = torch.sum(torch.stack(dequantized_chunks).float(), dim=0).to(DTYPE)

    dist.barrier()

    # --- Stage B: Quantized All-Gather ---

    # 1. Quantize the locally summed chunk
    q_sum, s_sum = quant_func(local_sum_chunk, block_size=quant_block_size)

    # 2. Prepare receive buffers for All-Gather
    gathered_quant_list = [torch.zeros_like(q_sum.view(transmission_dtype)) for _ in range(world_size)]
    gathered_scale_list = [torch.zeros_like(s_sum) for _ in range(world_size)]

    # 3. Perform two All-Gather operations
    dist.all_gather(gathered_quant_list, q_sum.view(transmission_dtype))
    dist.all_gather(gathered_scale_list, s_sum)

    # 4. Dequantize all gathered chunks and populate t_chunks
    for i in range(world_size):
        dequant_chunk = dequant_func(gathered_quant_list[i].view(quant_dtype), gathered_scale_list[i], output_dtype=DTYPE)
        t_chunks[i].copy_(dequant_chunk)

    dist.barrier()

    # --- 4. Result validation ---
    dist.all_reduce(t_full_pytorch, op=dist.ReduceOp.SUM)

    if torch.allclose(t_full, t_full_pytorch, rtol=1e-2, atol=1e-1):
        print(f"Rank {rank}: ✅ Correctness check PASSED!")
    else:
        if rank == 0:
            print(f"Rank {rank}: ❌ Correctness check FAILED!")
            print (t_full[::32], t_full_pytorch[::32])
            
            # --- Error Calculation ---
            original_flat = t_full_pytorch.flatten().float()
            quantized_flat = t_full.flatten().float()
            
            # Absolute Error
            abs_diff = torch.abs(original_flat - quantized_flat)
            
            # Relative Error
            ref_abs = torch.abs(original_flat)
            rel_diff = abs_diff / (ref_abs + 1e-8) # Add epsilon to avoid division by zero
            
            # Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(original_flat, quantized_flat, dim=0)

            print(f"   - Rank {rank} Cosine Similarity: {cos_sim.item():.6f}")
            print(f"   - Rank {rank} Max Absolute Error: {abs_diff.max().item():.6f}")
            print(f"   - Rank {rank} Mean Absolute Error: {abs_diff.mean().item():.6f}")
            print(f"   - Rank {rank} Max Relative Error: {rel_diff.max().item() * 100:.4f}%")
            print(f"   - Rank {rank} Mean Relative Error: {rel_diff.mean().item() * 100:.4f}%")

            # --- Find and print the top 5 elements with the largest relative error ---
            print("\n--- Top 5 largest relative errors ---")
            topk_rel_err, topk_indices = torch.topk(rel_diff, 5)
            
            for i in range(5):
                idx = topk_indices[i].item()
                print(f"  - Index: {idx}")
                print(f"    Original value:  {original_flat[idx].item():.8f}")
                print(f"    Quantized value: {quantized_flat[idx].item():.8f}")
                print(f"    Absolute error:  {abs_diff[idx].item():.8f}")
                print(f"    Relative error:  {rel_diff[idx].item() * 100:.2f}%")

    # --- 5. Cleanup ---
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("This script requires at least 2 GPUs to run.")
    else:
        print(f"Found {world_size} GPUs. Starting single-node multi-GPU All-Reduce test.")
        
        TENSOR_SIZE = 1024 * 1024 * 128
        
        # Ensure each chunk per GPU is divisible by the quantization block size
        chunk_size_per_gpu = TENSOR_SIZE // world_size
        if chunk_size_per_gpu % QUANT_BLOCK_SIZE != 0:
            chunk_size_per_gpu = (chunk_size_per_gpu // QUANT_BLOCK_SIZE) * QUANT_BLOCK_SIZE
        
        TENSOR_SIZE = chunk_size_per_gpu * world_size
        
        if TENSOR_SIZE == 0:
            raise ValueError("Tensor size is too small for the given world_size and block_size.")
            
        print(f"Tensor size adjusted to {TENSOR_SIZE} to be divisible by world_size and quant_block_size.")

        mp.spawn(run,
                 args=(world_size, TENSOR_SIZE),
                 nprocs=world_size,
                 join=True)