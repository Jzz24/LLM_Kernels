# -*- coding: utf-8 -*-
"""
TorchAO MXFP8 Quantization Core
Derived from: torchao/prototype/mx_formats/
"""

import torch
import math
from enum import Enum
from typing import Optional, Union

# ==================== 1. Constants ====================
# FP8 E4M3 max value: 448.0
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# E8M0 exponent bias
E8M0_EXPONENT_BIAS = 127
# E8M0 NaN encoding
E8M0_EXPONENT_NAN_VAL = 255
# Default block size
BLOCK_SIZE_DEFAULT = 32
SUPPORTED_ELEM_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


# ==================== 2. Config ====================
class ScaleCalculationMode(Enum):
    """Methods for MX block scale calculation"""
    FLOOR = "floor"  # OCP MX spec: X = 2^floor(log2(max_abs(v))-max_exp)
    RCEIL = "rceil"  # cuBLAS: ceil(max_abs(v) / max_pos)
    CEIL = "ceil"    # 2^ceil(log2(max_abs(v))-max_exp)
    EVEN = "even"    # Round to even


# ==================== 3. Utils ====================
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ==================== 4. MXTensor Core ====================
class MXTensor:
    
    def __init__(
        self,
        qdata: torch.Tensor,           # Quantized elements
        scale: torch.Tensor,           # E8M0 scales
        elem_dtype: torch.dtype,       # Target dtype (e.g., float8_e4m3fn)
        block_size: int,               # Block size (default 32)
        orig_dtype: torch.dtype,       # Original dtype
    ):
        self.qdata = qdata
        self.scale = scale
        self.elem_dtype = elem_dtype
        self.block_size = block_size
        self.orig_dtype = orig_dtype
    
    @staticmethod
    def to_mx(
        x: torch.Tensor,
        elem_dtype: torch.dtype,
        block_size: int,
        scale_calculation_mode: ScaleCalculationMode,
    ) -> "MXTensor":
        """
        Convert a high-precision tensor into MX format.
        
        Args:
            x: Input high-precision tensor. Shape: (*, K)
            elem_dtype: Reduced precision target dtype (e.g., torch.float8_e4m3fn).
            block_size: Number of elements per block along the last dimension (K). OCP MX specifies 32.
            scale_calculation_mode: The rule used to determine the scale value for the block.
            
        Returns:
            MXTensor: A wrapper containing:
                - qdata: The quantized values. Shape: (*, K), dtype is `elem_dtype`.
                - scale: The block scales in E8M0 format (uint8). Shape: (*, ceil(K / block_size)).
        """
        assert elem_dtype in SUPPORTED_ELEM_DTYPES, f"Unsupported elem_dtype: {elem_dtype}"
        assert block_size == 32, f"block_size must be 32, got {block_size}"
        assert x.dim() >= 2, "x must have at least 2 dimensions"
        
        orig_shape = x.shape
        orig_dtype = x.dtype
        device = x.device
        
        # Determine blocking layout
        leading_dims = orig_shape[:-1]
        K = orig_shape[-1]
        M = math.prod(leading_dims)
        
        # Reshape to 2D
        x_2d = x.reshape(M, K).to(torch.float32)
        
        # Compute scales
        num_blocks_k = ceil_div(K, block_size)
        scales = torch.zeros(M, num_blocks_k, device=device, dtype=torch.float32)
        
        for block_idx in range(num_blocks_k):
            start = block_idx * block_size
            end = min(start + block_size, K)
            block_data = x_2d[:, start:end]
            
            amax = torch.max(torch.abs(block_data), dim=1)[0]
            
            if scale_calculation_mode == ScaleCalculationMode.FLOOR:
                max_pos = F8E4M3_MAX if elem_dtype == torch.float8_e4m3fn else 57344.0
                amax_clamped = torch.clamp(amax, min=1e-12)
                exponent = torch.floor(torch.log2(amax_clamped / max_pos))
                scales[:, block_idx] = torch.exp2(exponent)
            elif scale_calculation_mode == ScaleCalculationMode.RCEIL:
                max_pos = F8E4M3_MAX if elem_dtype == torch.float8_e4m3fn else 57344.0
                ratio = amax / max_pos
                exponent = torch.ceil(torch.log2(ratio))
                scales[:, block_idx] = torch.exp2(exponent)
            elif scale_calculation_mode == ScaleCalculationMode.CEIL:
                max_exp = 8 if elem_dtype == torch.float8_e4m3fn else 15
                exponent = torch.ceil(torch.log2(amax) - max_exp)
                scales[:, block_idx] = torch.exp2(exponent)
            elif scale_calculation_mode == ScaleCalculationMode.EVEN:
                pass
        
        # Quantize elements
        scales_reshaped = scales.unsqueeze(2)
        x_reshaped = x_2d.reshape(M, num_blocks_k, block_size)
        x_scaled = x_reshaped / scales_reshaped
        
        max_val = F8E4M3_MAX if elem_dtype == torch.float8_e4m3fn else 57344.0
        x_scaled = torch.clamp(x_scaled, -max_val, max_val)
        x_q = x_scaled.to(elem_dtype)
        
        # Convert scales to E8M0
        exponent_values = torch.log2(scales)
        exponent_biased = torch.round(exponent_values).to(torch.int32) + E8M0_EXPONENT_BIAS
        exponent_biased = torch.clamp(exponent_biased, 0, 255)
        
        nan_mask = torch.isnan(scales)
        exponent_biased[nan_mask] = E8M0_EXPONENT_NAN_VAL
        scales_e8m0 = exponent_biased.to(torch.uint8)
        
        # Restore shapes
        x_q = x_q.reshape(orig_shape)
        scales_e8m0 = scales_e8m0.reshape(*leading_dims, num_blocks_k)
        
        return MXTensor(
            qdata=x_q,
            scale=scales_e8m0,
            elem_dtype=elem_dtype,
            block_size=block_size,
            orig_dtype=orig_dtype,
        )

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize MX tensor back to original precision (orig_dtype).
        """
        orig_shape = self.qdata.shape
        K = orig_shape[-1]
        leading_dims = orig_shape[:-1]
        M = math.prod(leading_dims) if len(leading_dims) > 0 else 1
        num_blocks_k = ceil_div(K, self.block_size)
        
        x_f32 = self.qdata.to(torch.float32).reshape(M, num_blocks_k, self.block_size)
        
        # Restore scale (E8M0 -> float32)
        exponent = self.scale.to(torch.int32) - E8M0_EXPONENT_BIAS
        scales_f32 = torch.exp2(exponent.to(torch.float32))
        
        nan_mask = (self.scale == E8M0_EXPONENT_NAN_VAL)
        scales_f32[nan_mask] = float('nan')
        
        scales_reshaped = scales_f32.reshape(M, num_blocks_k, 1)
        x_dequant = (x_f32 * scales_reshaped).reshape(orig_shape)
        
        return x_dequant.to(self.orig_dtype)
    

def simulate_mx_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR
) -> torch.Tensor:
    """
    Simulate an MX FP8 matrix multiplication block by block: C = A @ B.T.
    Inside each block, it performs FP8 dot products, then applies the specific
    block scales, and accumulates into FP32, exactly mirroring hardware behavior.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Weight tensor of shape (N, K)
        scale_mode: Scale calculation mode for quantization.
        
    Returns:
        C: The output tensor of shape (M, N)
    """
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, "Inner dimension mismatch"
    block_size = 32
    num_blocks = ceil_div(K, block_size)

    # 1. Quantize A along the K dimension (last dim of A)
    A_mx = MXTensor.to_mx(
        A,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scale_calculation_mode=scale_mode
    )
    
    # 2. Quantize B along the K dimension (last dim of B, since it's N x K)
    B_mx = MXTensor.to_mx(
        B,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scale_calculation_mode=scale_mode
    )

    ## TODO: dump mx_fp8 input and weights here
    
    # 3. Extract the quantized FP8 values directly
    # Shape: (M, num_blocks, block_size)
    A_q = A_mx.qdata.reshape(M, num_blocks, block_size)
    # Shape: (N, num_blocks, block_size)
    B_q = B_mx.qdata.reshape(N, num_blocks, block_size)
    
    # 4. Decode block scales back to floating point for the scaling step
    A_scale_f32 = torch.exp2(A_mx.scale.to(torch.int32) - E8M0_EXPONENT_BIAS).to(torch.float32)
    B_scale_f32 = torch.exp2(B_mx.scale.to(torch.int32) - E8M0_EXPONENT_BIAS).to(torch.float32)
    
    # Handle NaNs from scales
    A_scale_f32[A_mx.scale == E8M0_EXPONENT_NAN_VAL] = float('nan')
    B_scale_f32[B_mx.scale == E8M0_EXPONENT_NAN_VAL] = float('nan')

    # 5. Initialize the output accumulator C
    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    
    # 6. Perform the blocked matrix multiplication
    for i in range(num_blocks):
        # A_block is (M, block_size), B_block is (N, block_size)
        A_block = A_q[:, i, :]
        B_block = B_q[:, i, :]
        
        # Simulated Hardware MAC FP8 @ FP8
        # 虽然 NVIDIA Hopper 架构的 Tensor Core 支持原生的 FP8 乘法并累加到 FP32，
        # 但在目前多数 PyTorch 版本中，常规的 `torch.matmul` 并未直接实现原生 `Float8_e4m3fn` 
        # 张量的 "addmm"（矩阵乘法）后端分发。如果直接传入两个 fp8 张量，会报 NotImplementedError。
        # 在实际硬件底层或自定义 Triton / CUDA Kernel 中，它们是以 fp8 输入，以 fp32 累加。
        # 我们这里通过将它们转为 fp32 后再相乘，可以在数值精度上完全等价地模拟这一过程。
        partial_C = torch.matmul(A_block.to(torch.float32), B_block.to(torch.float32).t())
        
        # Fetch scales for this specific block: scale_A is (M, 1), scale_B is (1, N)
        scale_A_i = A_scale_f32[:, i].unsqueeze(1)
        scale_B_i = B_scale_f32[:, i].unsqueeze(0)
        
        # Apply the scaled partial block out to the high-precision fp32 accumulator
        C += partial_C * scale_A_i * scale_B_i
        
    return C.to(A.dtype)

def test_mx_tensor_quantization():
    print("--- Testing MXTensor Quantization ---")
    x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    
    mx_tensor = MXTensor.to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    
    print(f"Quantized shape: {mx_tensor.qdata.shape}, dtype: {mx_tensor.qdata.dtype}")
    print(f"Scale (E8M0) shape: {mx_tensor.scale.shape}, dtype: {mx_tensor.scale.dtype}")
    
    x_dequant = mx_tensor.dequantize()
    cos_sim = torch.nn.functional.cosine_similarity(
        x.flatten().to(torch.float32), 
        x_dequant.flatten().to(torch.float32), 
        dim=0
    )
    print(f"Cosine Similarity: {cos_sim.item():.6f}")

def test_simulated_mx_fp8_matmul():
    print("\n--- Testing Simulated MXFP8 Matmul ---")
    M, K, N = 128, 4096, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    # For models, B (weight) is typically natively stored as [N, K]
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Use RCEIL to avoid the aggressive outlier clipping of FLOOR mode
    C_mx = simulate_mx_fp8_matmul(A, B, ScaleCalculationMode.RCEIL)
    ## TODO: dump mx_fp8 gemm outputs here

    C_ref = torch.matmul(A, B.t())
    
    cos_sim_gemm = torch.nn.functional.cosine_similarity(
        C_ref.flatten().to(torch.float32), 
        C_mx.flatten().to(torch.float32), 
        dim=0
    )
    print(f"Original shape: A({A.shape}) @ B.T({B.t().shape}) -> C({C_ref.shape})")
    print(f"GEMM Cosine Similarity (vs BF16): {cos_sim_gemm.item():.6f}")

if __name__ == "__main__":
    test_mx_tensor_quantization()
    test_simulated_mx_fp8_matmul()