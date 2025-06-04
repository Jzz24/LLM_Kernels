import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import time

from utils import QuantUtils


os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # H100是Hopper架构对应SM 9.0
# Get current directory
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
    M, N, K = 64, 64*2, 128
    groupsize = 128
    
    print(f"Matrix size: M={M}, N={N}, K={K}, groupsize={groupsize}")
    
    # 1. Initialize random matrices
    A = torch.randn(M, K, dtype=torch.half, device=device) * 0.1
    W_fp16 = torch.randn(K, N, dtype=torch.half, device=device) * 0.1
    
    # 2. Quantize weights
    ref_dequant, W_int4, scales = QuantUtils.fake_quantize(W_fp16, groupsize=groupsize)
    
    # 3. Pack weights to Marlin format
    packed_weight, packed_scales = QuantUtils.pack(W_int4, scales, groupsize=groupsize)
    print ("k_offset=0, 16*64 packed weight, thread 0", packed_weight[0])
    print ("k_offset=0, 16*16 ref_dequant,\n", ref_dequant[:16,1])
    print ("k_offset=0, 16*16 scales,\n", scales[:,:16])
    # print ("k_offset=0, 16*16 packed scales,\n", packed_scales[:,:16])

    import ipdb; ipdb.set_trace()
    
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
    print (C_marlin, "\n", C_ref)
    print (C_marlin.max(), C_marlin.min())

    is_correct = check_results(C_marlin, C_ref, threshold=0.001)
    
    if is_correct:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")

def test_w_matrix_layout():
    """测试W矩阵在CUDA kernel中的排布"""
    device = torch.device("cuda")
    
    # 使用最小的测试矩阵便于验证
    M, N, K = 64, 64, 16
    groupsize = -1
    
    print(f"Testing W matrix layout: K={K}, N={N}")
    
    # 创建旋转模式的W矩阵
    W_fp16 = torch.zeros(K, N, dtype=torch.float32)
    
    for j in range(N):  # 列
        for i in range(K):  # 行
            W_fp16[i, j] = (j + i) % 16
    
    W_fp16 = W_fp16.to(torch.half).to(device)
    
    for i in range(min(K, 16)):
        row_str = f"Row {i:2d}: "
        for j in range(16):
            row_str += f"{W_fp16[i, j].item():2.0f} "
        print(row_str)
    
    # 不使用量化缩放，直接转换为int4
    W_int4 = W_fp16.clone().to(torch.int32)
    
    # 修正 scales 的形状
    if groupsize == -1:
        # 无分组：每列一个scale，但需要按照pack格式
        scales = torch.ones(1, N, dtype=torch.half, device=device)
    else:
        # 分组量化
        num_groups = K // groupsize
        scales = torch.ones(num_groups, N, dtype=torch.half, device=device)
    
    print(f"Scales shape: {scales.shape}")
    print(f"W_int4 shape: {W_int4.shape}")
    
    # 打包权重
    packed_weight, packed_scales = QuantUtils.pack(W_int4, scales, groupsize=groupsize)
    
    print(f"\nPacked weight shape: {packed_weight.shape}")
    print(f"Packed scales shape: {packed_scales.shape}")
    
    # 验证打包结果
    print("\n=== Manual packing verification ===")
    # [0,2,4,6,1,3,5,7]
    subtile_0_thread_0 = [0, 8, 8, 0, 1, 9, 9, 1]
    
    # 手动计算第一个int32
    manual_int32 = 0
    for i in range(8):
        val = subtile_0_thread_0[i] & 0xF
        manual_int32 |= (val << (i * 4))
    
    actual_int32 = packed_weight[0][0]
    print(f"Manual calculated int32: 0x{manual_int32:08x}")
    print(f"Actual packed int32: 0x{actual_int32:08x}")
    import ipdb; ipdb.set_trace()
    
    # 调用CUDA kernel
    A = torch.ones(M, K, dtype=torch.half, device=device)
    C_marlin = torch.zeros(M, N, dtype=torch.half, device=device)
    workspace = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device)
    
    print("\n=== Calling CUDA kernel for verification ===")
    mul(A, packed_weight, C_marlin, packed_scales, workspace)
    
    return W_fp16, W_int4, packed_weight
            

if __name__ == "__main__":
    # test_w_matrix_layout()
    test_marlin_w4a16_gemm()