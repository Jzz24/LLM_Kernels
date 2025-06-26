# The Ring All-Reduce logic in this script is based on the implementation from:
# https://github.com/rajesh-s/mlsys-allreduce/tree/main

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from quant_utils import int8_quant, int8_dequant, fp8_quant, fp8_dequant

DTYPE = torch.bfloat16
QUANT_TYPE = 'int8' # 'int8' or 'fp8'
Quant_BLOCK_SIZE = 32


def run(rank, world_size, tensor_size):

    # --- 1. Initialize distributed environment ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6585'
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size)


    if QUANT_TYPE == 'int8':
        quant_func = int8_quant
        dequant_func = int8_dequant
        quant_block_size = Quant_BLOCK_SIZE
        quant_dtype = torch.int8
        scale_dtype = torch.float16
        transmission_dtype = torch.int8 # INT8 can be transmitted directly
    elif QUANT_TYPE == 'fp8':
        quant_func = fp8_quant
        dequant_func = fp8_dequant
        quant_block_size = Quant_BLOCK_SIZE
        quant_dtype = torch.float8_e4m3fn # Logical quantization type
        scale_dtype = torch.float16
        transmission_dtype = torch.int8 # NCCL does not support FP8 directly, must use INT8 view
    else:
        raise ValueError(f"Unsupported QUANT_TYPE: {QUANT_TYPE}")


    # --- 2. Prepare data ---
    device = f'cuda:{rank}'
    t_full = torch.rand(tensor_size, device=device, dtype=DTYPE)
    t_full_pytorch = t_full.clone()

    # Split the full tensor into chunks
    t_chunks = list(t_full.split(tensor_size // world_size))
    
    # --- Create receive buffers ---
    recv_quant_buffer = torch.zeros_like(t_chunks[0], dtype=transmission_dtype)
    scale_shape = (*t_chunks[0].size()[:-1], t_chunks[0].size(-1) // quant_block_size)
    recv_scale_buffer = torch.zeros(scale_shape, device=device, dtype=scale_dtype)
    
    dist.barrier()
    
    # --- 3. Custom Quantized Ring All-Reduce ---

    # --- Stage A: Quantized Reduce-scatter loop ---
    for i in range(world_size - 1):
        send_chunk_idx = (rank - i + world_size) % world_size
        recv_chunk_idx = (rank - 1 - i + world_size) % world_size
        
        chunk_to_send = t_chunks[send_chunk_idx]
        quant_data_to_send, scale_data_to_send = quant_func(chunk_to_send, block_size=quant_block_size)

        # Before sending, view the quantized data as the transmission type
        quant_data_to_send_view = quant_data_to_send.view(transmission_dtype)

        if (rank % 2) == 0:
            dist.send(quant_data_to_send_view, dst=(rank + 1) % world_size)
            dist.send(scale_data_to_send, dst=(rank + 1) % world_size)
            dist.recv(recv_quant_buffer, src=(rank - 1 + world_size) % world_size)
            dist.recv(recv_scale_buffer, src=(rank - 1 + world_size) % world_size)
        else:
            dist.recv(recv_quant_buffer, src=(rank - 1 + world_size) % world_size)
            dist.recv(recv_scale_buffer, src=(rank - 1 + world_size) % world_size)
            dist.send(quant_data_to_send_view, dst=(rank + 1) % world_size)
            dist.send(scale_data_to_send, dst=(rank + 1) % world_size)
        
        # After receiving, view the buffer back to the logical quantized type and dequantize
        dequant_input = recv_quant_buffer.view(quant_dtype)
        dequant_data = dequant_func(dequant_input, recv_scale_buffer, output_dtype=DTYPE)
        t_chunks[recv_chunk_idx].copy_((t_chunks[recv_chunk_idx].float() + dequant_data.float()).to(DTYPE))
    
    # --- Stage B: Quantized All-gather loop ---
    for i in range(world_size - 1):
        send_chunk_idx = (rank + 1 - i + world_size) % world_size
        recv_chunk_idx = (rank - i + world_size) % world_size
        
        chunk_to_send = t_chunks[send_chunk_idx]
        quant_data_to_send, scale_data_to_send = quant_func(chunk_to_send, block_size=quant_block_size)

        # Before sending, view the quantized data as the transmission type
        quant_data_to_send_view = quant_data_to_send.view(transmission_dtype)

        if (rank % 2) == 0:
            dist.send(quant_data_to_send_view, dst=(rank + 1) % world_size)
            dist.send(scale_data_to_send, dst=(rank + 1) % world_size)
            dist.recv(recv_quant_buffer, src=(rank - 1 + world_size) % world_size)
            dist.recv(recv_scale_buffer, src=(rank - 1 + world_size) % world_size)
        else:
            dist.recv(recv_quant_buffer, src=(rank - 1 + world_size) % world_size)
            dist.recv(recv_scale_buffer, src=(rank - 1 + world_size) % world_size)
            dist.send(quant_data_to_send_view, dst=(rank + 1) % world_size)
            dist.send(scale_data_to_send, dst=(rank + 1) % world_size)

        # After receiving, view the buffer back to the logical quantized type and dequantize
        dequant_input = recv_quant_buffer.view(quant_dtype)
        dequant_data = dequant_func(dequant_input, recv_scale_buffer, output_dtype=DTYPE)
        t_chunks[recv_chunk_idx].copy_(dequant_data)

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

        chunk_size = TENSOR_SIZE // world_size
        if chunk_size % Quant_BLOCK_SIZE != 0:
            chunk_size = (chunk_size // Quant_BLOCK_SIZE) * Quant_BLOCK_SIZE
        TENSOR_SIZE = chunk_size * world_size
        
        if TENSOR_SIZE == 0:
            raise ValueError("Tensor size is too small for the given world_size and block_size.")
            
        print(f"Tensor size adjusted to {TENSOR_SIZE} to be divisible by world_size and quant_block_size.")

        mp.spawn(run,
                 args=(world_size, TENSOR_SIZE),
                 nprocs=world_size,
                 join=True)