import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import time

from utils import QuantUtils


os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # H100是Hopper架构对应SM 9.0
_current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile CUDA extension
marlin_cuda = load(
    name='marlin_cuda',
    sources=[
        os.path.join(_current_dir, 'marlin_w4a16_gemm.cpp'),
        os.path.join(_current_dir, 'marlin_w4a16_gemm.cu')
    ],
    verbose=False,
    extra_cuda_cflags=['-O3']
)

def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply"""
    return marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)

def check_results(result, reference, threshold=0.01):
    """Check computation accuracy"""
    abs_diff = torch.abs(result - reference)
    max_abs_error = torch.max(abs_diff).item()
    mean_abs_error = torch.mean(abs_diff).item()
    
    error_count = torch.sum(abs_diff > threshold).item()
    total_elements = result.numel()
    accuracy = (total_elements - error_count) / total_elements * 100
    
    print(f"Max error: {max_abs_error:.6f}, Mean error: {mean_abs_error:.6f}")
    print(f"Accuracy: {accuracy:.2f}% ({total_elements - error_count}/{total_elements} errors)")
    
    return error_count == 0

def test_marlin_w4a16_gemm():

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Test parameters
    # M, N, K = 256, 512, 1024
    M, N, K = 64*2, 64*8, 128*8
    groupsize = 128
    
    print(f"Matrix size: M={M}, N={N}, K={K}, groupsize={groupsize}")
    
    # 1. Initialize random matrices
    A = torch.randn(M, K, dtype=torch.half, device=device) * 0.1
    W_fp16 = torch.randn(K, N, dtype=torch.half, device=device) * 0.1
    
    # 2. Quantize weights
    ref_dequant, W_int4, scales = QuantUtils.fake_quantize(W_fp16, groupsize=groupsize)
    
    # 3. Pack weights to Marlin format
    packed_weight, packed_scales = QuantUtils.pack(W_int4, scales, groupsize=groupsize)
    
    # 4. Reference computation with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # torch.cuda.synchronize()
    start_event.record()
    C_ref = torch.matmul(A, ref_dequant)
    end_event.record()
    torch.cuda.synchronize()
    ref_time = start_event.elapsed_time(end_event)
    
    # 5. Marlin computation
    C_marlin = torch.zeros(M, N, dtype=torch.half, device=device)
    workspace = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device) # unused
    

    start_event.record()
    mul(A, packed_weight, C_marlin, packed_scales, workspace)
    end_event.record()
    torch.cuda.synchronize()
    marlin_time = start_event.elapsed_time(end_event)
    
    print(f"Reference time: {ref_time:.2f}ms")
    print(f"Marlin time: {marlin_time:.2f}ms")
    print(f"Speedup: {ref_time/marlin_time:.2f}x")
    
    # 6. Verify results
    is_correct = check_results(C_marlin, C_ref, threshold=0.001)
    
    if is_correct:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")

if __name__ == "__main__":
    test_marlin_w4a16_gemm()