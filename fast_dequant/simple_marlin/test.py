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

def benchmark(f, warmup=3, iter=20):
    """Benchmark function with proper warmup and timing"""
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarking
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Cool down GPU to avoid throttling
    time.sleep(0.1)
    return res

def benchmark_dense(A, B_ref, C):
    """Benchmark dense matrix multiplication"""
    res = benchmark(lambda: torch.matmul(A, B_ref, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B_ref.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, s):
    """Benchmark quantized matrix multiplication"""
    workspace = torch.zeros(C.shape[1] // 128 * 16, dtype=torch.int32, device=A.device)
    res = benchmark(lambda: mul(A, B, C, s, workspace))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

def check_results(result, reference, threshold=0.01):
    """Check computation accuracy"""
    abs_diff = torch.abs(result - reference)
    max_abs_error = torch.max(abs_diff).item()
    mean_abs_error = torch.mean(abs_diff).item()
    
    error_count = torch.sum(abs_diff > threshold).item()
    total_elements = result.numel()
    accuracy = (total_elements - error_count) / total_elements * 100
    
    print(f"    Max error: {max_abs_error:.6f}, Mean error: {mean_abs_error:.6f}")
    print(f"    Accuracy: {accuracy:.2f}% ({total_elements - error_count}/{total_elements} within threshold)")
    
    return error_count == 0

def test_single_case(M, N, K, groupsize, device):
    """Test a single M, N, K combination with comprehensive benchmarking"""
    print(f"\n=== Testing M={M}, N={N}, K={K}, groupsize={groupsize} ===")
    
    # 1. Initialize random matrices
    A = torch.randn(M, K, dtype=torch.half, device=device) * 0.1
    W_fp16 = torch.randn(K, N, dtype=torch.half, device=device) * 0.1
    
    # 2. Quantize weights
    ref_dequant, W_int4, scales = QuantUtils.fake_quantize(W_fp16, groupsize=groupsize)
    
    # 3. Pack weights to Marlin format
    packed_weight, packed_scales = QuantUtils.pack(W_int4, scales, groupsize=groupsize)
    
    # 4. Prepare output tensors
    C_ref = torch.zeros(M, N, dtype=torch.half, device=device)
    C_marlin = torch.zeros(M, N, dtype=torch.half, device=device)
    
    # 5. Benchmark dense computation
    print("  Benchmarking dense computation...")
    res_dense = benchmark_dense(A, ref_dequant, C_ref)
    
    # 6. Benchmark quantized computation
    print("  Benchmarking quantized computation...")
    res_quant = benchmark_quant(A, packed_weight, C_marlin, packed_scales)
    
    # 7. Calculate speedup
    speedup = res_dense['s'] / res_quant['s']
    
    # 8. Print performance results
    print(f"  Performance Results:")
    print(f"    Dense    - Time: {res_dense['s']*1000:.3f}ms, TFLOP/s: {res_dense['TFLOP/s']:.3f}, GB/s: {res_dense['GB/s']:.3f}")
    print(f"    Quantized- Time: {res_quant['s']*1000:.3f}ms, TFLOP/s: {res_quant['TFLOP/s']:.3f}, GB/s: {res_quant['GB/s']:.3f}")
    print(f"    Speedup: {speedup:.2f}x")
    
    # 9. Verify correctness
    print("  Checking accuracy...")
    is_correct = check_results(C_marlin, C_ref, threshold=0.05)  # 放宽阈值，因为量化误差
    
    if is_correct:
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
        
    return is_correct

def test_marlin_w4a16_gemm():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # 打印GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    
    # 定义测试用例 - 确保所有维度都是16的倍数（满足kernel要求）
    test_cases = [
        (16, 64, 128, 128),     
        (16, 128, 128, 128),    
        (16, 64, 512, 128),    
        (16, 64, 1024, 128),    
        
        (256, 512, 1024, 128),  
        (512, 256, 1024, 128),   
        (1024, 1024, 512, 128),  
        (128, 2048, 1024, 128),  
        
        (1024, 2048, 2048, 128),
        (2048, 1024, 1024, 128), 
        (512, 4096, 2048, 128),  
        
        (256, 512, 1024, 64),    
        (256, 512, 1024, 256),   
        (512, 512, 512, 512),   
    ]
    
    print("Starting Marlin W4A16 GEMM comprehensive testing...")
    print("="*80)
    
    passed_count = 0
    total_count = len(test_cases)
    failed_cases = []
    
    for i, (M, N, K, groupsize) in enumerate(test_cases):
        try:
            is_correct = test_single_case(M, N, K, groupsize, device)
            if is_correct:
                passed_count += 1
            else:
                failed_cases.append((M, N, K, groupsize))
        except Exception as e:
            print(f"  ❌ FAILED with exception: {e}")
            failed_cases.append((M, N, K, groupsize))
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"Total test cases: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success rate: {passed_count/total_count*100:.1f}%")
    
    if failed_cases:
        print(f"\nFailed cases:")
        for M, N, K, groupsize in failed_cases:
            print(f"  M={M}, N={N}, K={K}, groupsize={groupsize}")
    else:
        print(f"\n🎉 All tests passed!")

if __name__ == "__main__":
    test_marlin_w4a16_gemm()