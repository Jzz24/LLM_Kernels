# import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple


@triton.jit
def weight_dequant_int8_kernel(y_ptr, x_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the int8 tensor `y` to float32 using block-wise dequantization.

    Args:
        y_ptr: Pointer to the input quantized tensor.
        x_ptr: Pointer to the output dequantized tensor.
        s_ptr: Pointer to the scaling factors tensor for int8.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n).to(tl.float32)
    x = y * s
    tl.store(x_ptr + offs, x, mask=mask)

@triton.jit
def weight_dequant_int4_kernel(y_ptr, x_ptr, s4_ptr, z4_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the int4 tensor `y` to float32 using block-wise dequantization.

    Args:
        y_ptr: Pointer to the input int4 tensor.
        x_ptr: Pointer to the output dequantized tensor.
        s4_ptr: Pointer to the scaling factors tensor for int4.
        z4_ptr: Pointer to the zero points tensor for int4.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.int8)
    s4 = tl.load(s4_ptr + pid_m * n + pid_n).to(tl.int8)
    z4 = tl.load(z4_ptr + pid_m * n + pid_n).to(tl.int8)
    x = (y - z4) * s4
    tl.store(x_ptr + offs, x, mask=mask)

def weight_dequant(y: torch.Tensor, s: torch.Tensor, s4: torch.Tensor, z4: torch.Tensor, block_size: int = 128, block_size_int4: int = 16) -> torch.Tensor:
    """
    Dequantizes the input tensor `y` from int4 to float32 using block-wise dequantization.

    Args:
        y (torch.Tensor): The input quantized tensor.
        s (torch.Tensor): The scaling factors for int8.
        s4 (torch.Tensor): The scaling factors for int4.
        z4 (torch.Tensor): The zero points for int4.
        block_size (int, optional): The size of the blocks to be used for dequantization. Default is 128.
        block_size_int4 (int, optional): The size of the blocks for int4 dequantization. Default is 16.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    assert y.is_contiguous(), 'Input tensor must be contiguous'
    assert y.dim() == 2, 'Input tensors must have 2 dimensions'
    assert y.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    assert block_size % block_size_int4 == 0, 'block_size must be divisible by block_size_int4'

    M, N = y.size()
    dequant_int4_x = torch.empty_like(y, dtype=torch.int8)
    dequant_int8_x = torch.empty_like(y, dtype=torch.float16)

    grid = (triton.cdiv(M, block_size_int4), triton.cdiv(N, block_size_int4))
    weight_dequant_int4_kernel[grid](y, dequant_int4_x, s4, z4, M, N, block_size_int4)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    weight_dequant_int8_kernel[grid](dequant_int4_x, dequant_int8_x, s, M, N, block_size)

    return dequant_int8_x


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
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
    y = tl.extra.cuda.libdevice.round(x / s)
    y = y.to(y_ptr.dtype.element_ty)
    s = s.to(s_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with dtype `torch.float16`.
    """
    # x.shape -> (bs, seq_len, hidden_dim) or (bs*seq_len, hidden_dim)
    # x.shape -> (*, hidden_dim)
    # quantize block -> (1, 128)
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_quant_int8_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x` to int8 using block-wise quantization.

    Args:
        x_ptr: Pointer to the input tensor.
        y_ptr: Pointer to the output quantized tensor.
        s_ptr: Pointer to the scaling factors tensor for int8.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 119.0
    y = tl.clamp(tl.extra.cuda.libdevice.round(x / s), -119.0, 119.0).to(tl.int8)
    s = s.to(tl.float16)
    tl.store(s_ptr + pid_m * n + pid_n, s)
    tl.store(y_ptr + offs, y, mask=mask)

@triton.jit
def weight_quant_int4_kernel(y_ptr, s4_ptr, z4_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Further quantizes the int8 tensor `y` to int4 using block-wise quantization.

    Args:
        y_ptr: Pointer to the input int8 tensor.
        s4_ptr: Pointer to the scaling factors tensor for int4.
        z4_ptr: Pointer to the zero points tensor for int4.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
        BLOCK_SIZE_INT4 (tl.constexpr): The size of the block for int4 quantization.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)
    s4 = tl.extra.cuda.libdevice.round((tl.max(y) - tl.min(y)) / 15.0)  # Scale for int4
    z4 = tl.extra.cuda.libdevice.round(-tl.min(y) / s4)  # Zero point for int4
    y_uint4 = tl.clamp((
        tl.extra.cuda.libdevice.round(y / s4) + z4), 0, 15).to(tl.int8)
    s4 = s4.to(tl.int8)
    z4 = z4.to(tl.int8)
    tl.store(y_ptr + offs, y_uint4, mask=mask)
    tl.store(s4_ptr + pid_m * n + pid_n, s4)
    tl.store(z4_ptr + pid_m * n + pid_n, z4)


def weight_quant(x: torch.Tensor, block_size: int = 128, block_size_int4: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization and further quantizes to int4.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        block_size_int4 (int, optional): The size of the blocks for int4 quantization. Default is 16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The quantized tensor, the scaling factors for int8, the scaling factors for int4, and the zero points for int4.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.dim() == 2, 'Input tensors must have 2 dimensions'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    assert block_size % block_size_int4 == 0, 'block_size must be divisible by block_size_int4'

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8) # dtype int8, actually uint4, needs to pack
    s = x.new_empty(((M + block_size - 1) // block_size, (N + block_size - 1) // block_size), dtype=torch.float16)
    s4 = x.new_empty(((M + block_size_int4 - 1) // block_size_int4, 
                      (N + block_size_int4 - 1) // block_size_int4), dtype=torch.int8) # dtype int8
    z4 = x.new_empty(((M + block_size_int4 - 1) // block_size_int4, 
                      (N + block_size_int4 - 1) // block_size_int4), dtype=torch.int8) # dtype int8, actually uint4, needs to pack

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_quant_int8_kernel[grid](x, y, s, M, N, block_size)

    grid = (triton.cdiv(M, block_size_int4), triton.cdiv(N, block_size_int4))
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_quant_int4_kernel[grid](y, s4, z4, M, N, block_size_int4)

    return y, s, s4, z4


w4a8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=w4a8_gemm_configs, key=['N', 'K'])
@triton.jit
def w4a8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    b_s4_ptr, b_z4_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_INT4: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        b_s4_ptr (tl.tensor): Pointer to the scaling factors for int4 quantization of matrix B.
        b_z4_ptr (tl.tensor): Pointer to the zero points for int4 quantization of matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension. Aligns with the int8 quantization block size.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    k_int4 = tl.cdiv(K, BLOCK_SIZE_INT4)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :] # (BLOCK_M, BLOCK_K)
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None] # (BLOCK_K, BLOCK_N)
    a_s_ptrs = a_s_ptr + offs_m * k # (BLOCK_M,) ,activation quant block (1, 128)
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k # (BLOCK_N,) ,weight quant block (128, 128)

    int4_offs_n = (offs_n // BLOCK_SIZE_INT4) #(BLOCK_N,)
    int4_offs_k = (offs_k // BLOCK_SIZE_INT4) #(BLOCK_K,)
    b_s4_ptrs = b_s4_ptr + int4_offs_n[None, :] * k_int4 + int4_offs_k[:, None] # (BLOCK_K, BLOCK_N)
    b_z4_ptrs = b_z4_ptr + int4_offs_n[None, :] * k_int4 + int4_offs_k[:, None] # (BLOCK_K, BLOCK_N)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)

        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        int4_s = tl.load(b_s4_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=1.0)
        int4_z = tl.load(b_z4_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)

        a_s = tl.load(a_s_ptrs).to(tl.float32)
        b_s = tl.load(b_s_ptrs).to(tl.float32)
        # dequantize_scale = a_s[:, None] * b_s[None, :]
        # shape -> (BLOCK_M, BLOCK_N)
        # tl.static_print("dequant int4:", ((b - int4_z) * int4_s).dtype)
        b = ((b - int4_z) * int4_s)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K # row-major, ptrs move along K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
        b_s4_ptrs += (BLOCK_SIZE_K // BLOCK_SIZE_INT4)
        b_z4_ptrs += (BLOCK_SIZE_K // BLOCK_SIZE_INT4)
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def w4a8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, 
              b_s: torch.Tensor, b_s4: torch.Tensor, b_z4: torch.Tensor, BLOCK_SIZE_INT4: int=16) -> torch.Tensor:
    """
    Perform a matrix multiplication using int8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The int8 scaling factor for the second input matrix, must be contiguous.
        b_s4 (torch.Tensor): The int4 scaling factor for the second input matrix for int4 quantization, must be contiguous.
        b_z4 (torch.Tensor): The int4 zero point for the second input matrix for int4 quantization, must be contiguous.
        block_size_int4 (int, optional): The block size for int4 quantization. Default is 16.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    # a for int8 activation, shape (M, K), (bsz*seq_len, hidden_size)
    # b for int4 weight,     shape (N, K), (out_features, in_features) (out_features, hidden_size)
    # gemm -> a @ b_t
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    assert b_s4.is_contiguous() and b_z4.is_contiguous(), 'Scaling factor and zero point tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    w4a8_gemm_kernel[grid](a, b, c, a_s, b_s, b_s4, b_z4, M, N, K, BLOCK_SIZE_INT4)
    return c


def test_w4a8_gemm():
    # Create random tensors for a and b
    a = torch.randn(256, 7168, dtype=torch.float16, device='cuda').contiguous()
    b = torch.randn(512, 7168, dtype=torch.float16, device='cuda').contiguous()
    
    # Quantize tensors
    a_quant, a_s = act_quant(a)
    b_quant, b_s, b_s4, b_z4 = weight_quant(b)

    # Perform int8 GEMM
    c = w4a8_gemm(a_quant, a_s, b_quant, b_s, b_s4, b_z4)
    c_float = a.matmul(b.t())
    
    # Print the result
    assert c.isnan().sum() == 0, 'Result of int8 GEMM contains NaNs'
    assert c.isinf().sum() == 0, 'Result of int8 GEMM contains Infs'

    cos_sim = torch.nn.functional.cosine_similarity(c.view(-1), c_float.view(-1), dim=0)
    print (f'Cosine similarity with float32 GEMM: {cos_sim.item()}')

def test_weight_quant_dequant():
    # Create a random FP16 tensor
    x = torch.randn(256, 7168, dtype=torch.float16, device='cuda').contiguous()
    # x = torch.load('test.pt').contiguous()
    
    # Quantize the tensor
    y_quant, s, s4, z4 = weight_quant(x)
    
    # Dequantize the tensor
    x_dequant = weight_dequant(y_quant, s, s4, z4)
    
    # Calculate cosine similarity between the original and dequantized tensors
    cos_sim = torch.nn.functional.cosine_similarity(x.view(-1), x_dequant.view(-1), dim=0)
    
    print(f'Cosine similarity between original and dequantized tensor: {cos_sim.item()}')

if __name__ == "__main__":
    test_weight_quant_dequant()
    test_w4a8_gemm()