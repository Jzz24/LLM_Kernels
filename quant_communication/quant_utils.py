from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def int8_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 127.
    # y = tl.extra.cuda.libdevice.round(x / s)
    y = tl.floor(x / s + 0.5) #round
    y = y.to(y_ptr.dtype.element_ty)
    s = s.to(s_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def int8_quant(x: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 32.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with dtype `torch.float16`.
    """
    # x.shape -> (bs, seq_len, hidden_dim) or (bs*seq_len, hidden_dim)
    # x.shape -> (*, hidden_dim)
    # quantize block -> (1, 32)
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    int8_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def int8_dequant_kernel(y_ptr, s_ptr, x_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the input tensor `y_ptr` using scaling factors from `s_ptr` and stores the result in `x_ptr`.

    Args:
        y_ptr (triton.Pointer): Pointer to the input quantized tensor (int8).
        s_ptr (triton.Pointer): Pointer to the input scaling factors tensor.
        x_ptr (triton.Pointer): Pointer to the output dequantized tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """

    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    y = tl.load(y_ptr + offs)
    s = tl.load(s_ptr + pid)  # Each block has one scale
    
    # Dequantize: x = y * s
    # Cast y to the dtype of s before multiplication
    x = y.to(s.dtype) * s
    
    # Cast to the final output type and store
    x = x.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, x)


def int8_dequant(y: torch.Tensor, s: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantizes the input tensor `y` using the scaling factors `s`.

    Args:
        y (torch.Tensor): The quantized input tensor with dtype `torch.int8`.
        s (torch.Tensor): The tensor of scaling factors.
        output_dtype (torch.dtype): The desired dtype for the output tensor.

    Returns:
        torch.Tensor: The dequantized tensor with the same dtype as `s`.
    """
    assert y.dtype == torch.int8, "Input y must be torch.int8"
    
    x = torch.empty_like(y, dtype=output_dtype)

    assert y.size(-1) % s.size(-1) == 0, "Last dimension of y must be a multiple of last dimension of s"
    block_size = y.size(-1) // s.size(-1)
    grid = (s.numel(), )
    
    int8_dequant_kernel[grid](y, s, x, BLOCK_SIZE=block_size)
    
    return x


@triton.jit
def fp8_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    s = s.to(s_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def fp8_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    fp8_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def fp8_dequant_kernel(y_ptr, s_ptr, x_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the input tensor `y_ptr` using scaling factors from `s_ptr` and stores the result in `x_ptr`.

    Args:
        y_ptr (triton.Pointer): Pointer to the input quantized tensor (float8).
        s_ptr (triton.Pointer): Pointer to the input scaling factors tensor.
        x_ptr (triton.Pointer): Pointer to the output dequantized tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    y = tl.load(y_ptr + offs)
    s = tl.load(s_ptr + pid)
    
    # Dequantize: x = y * s. Cast both to a common higher precision type for multiplication.
    x = y.to(s.dtype) * s
    
    x = x.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, x)


def fp8_dequant(y: torch.Tensor, s: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantizes the input tensor `y` using the scaling factors `s`.

    Args:
        y (torch.Tensor): The quantized input tensor with dtype `torch.float8_e4m3fn`.
        s (torch.Tensor): The tensor of scaling factors.
        output_dtype (torch.dtype): The desired dtype for the output tensor.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    assert y.dtype == torch.float8_e4m3fn, "Input y must be torch.float8_e4m3fn"
    
    x = torch.empty_like(y, dtype=output_dtype)

    assert y.size(-1) % s.size(-1) == 0, "Last dimension of y must be a multiple of last dimension of s"
    block_size = y.size(-1) // s.size(-1)
    grid = (s.numel(), )
    
    fp8_dequant_kernel[grid](y, s, x, BLOCK_SIZE=block_size)
    
    return x


if __name__ == '__main__':
    # --- 测试配置 ---
    shape = (1024, 1024, 1024)
    dtype = torch.float16 # input and output dtype
    device = 'cuda'

    x_original = torch.randn(*shape, dtype=dtype, device=device)

    # --- 测试 INT8 ---
    print("--- Testing INT8 Quantization ---")
    int8_block_size = 32
    y_quant_int8, s_quant_int8 = int8_quant(x_original, block_size=int8_block_size)
    x_dequant_int8 = int8_dequant(y_quant_int8, s_quant_int8, output_dtype=dtype)
    cos_sim_int8 = torch.nn.functional.cosine_similarity(x_original.flatten().float(), x_dequant_int8.flatten().float(), dim=0)
    print(f"INT8 Quant/Dequant Cosine Similarity: {cos_sim_int8.item():.6f}")

    # --- 测试 FP8 ---
    print("\n--- Testing FP8 Quantization ---")
    fp8_block_size = 128
    y_quant_fp8, s_quant_fp8 = fp8_quant(x_original, block_size=fp8_block_size)
    x_dequant_fp8 = fp8_dequant(y_quant_fp8, s_quant_fp8, output_dtype=dtype)
    cos_sim_fp8 = torch.nn.functional.cosine_similarity(x_original.flatten().float(), x_dequant_fp8.flatten().float(), dim=0)
    print(f"FP8 Quant/Dequant Cosine Similarity: {cos_sim_fp8.item():.6f}")