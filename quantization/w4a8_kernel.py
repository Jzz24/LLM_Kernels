# import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple

from quant_utils import Int4QuantUtils
from bitpack import pack_weights_over_rows

PACKING_BITS = 32
W_BITS = 4

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
def weight_dequant_int4_kernel(y_ptr, x_ptr, s4_ptr, z4_ptr, M, N, 
                               BLOCK_SIZE: tl.constexpr,
                               GROUP_SIZE: tl.constexpr,
                               unpack_mask: tl.constexpr = 0x0F,
                               elements_per_sample: tl.constexpr = 8,
                               w_bits: tl.constexpr = 4):
    """
    Dequantizes the packed uint4 tensor `y` to int8 using per-row quantization.
    
    Args:
        y_ptr: Pointer to the packed input uint4 tensor.
        x_ptr: Pointer to the output dequantized tensor.
        s4_ptr: Pointer to the scaling factors tensor.
        z4_ptr: Pointer to the zero points tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_SIZE: The processing block size (can be different from GROUP_SIZE).
        GROUP_SIZE: The quantization group size for scales/zeros (typically 128).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate offsets for this block
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Linear offsets
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load packed values
    packed_offs = (offs_m[:, None] // elements_per_sample * N + offs_n[None, :])
    packed_y = tl.load(y_ptr + packed_offs, mask=mask, other=0)

    # Unpack
    shift = offs_m[:, None] % elements_per_sample * w_bits
    unpacked_y = ((packed_y >> shift) & unpack_mask).to(tl.int8)
    
    # Calculate scale/zero indices based on GROUP_SIZE
    # Each row has scales/zeros for every GROUP_SIZE columns
    blocks_per_row = tl.cdiv(N, GROUP_SIZE)
    group_idx = offs_n[None, :] // GROUP_SIZE
    
    # For each element, find its proper scale/zero
    # Each row's scales are stored consecutively in groups of blocks_per_row
    scale_offs = offs_m[:, None] * blocks_per_row + group_idx
    scale_mask = (offs_m[:, None] < M) & (group_idx < blocks_per_row)
    
    # Load scales and zeros
    s4_vals = tl.load(s4_ptr + scale_offs, mask=scale_mask, other=1).to(tl.int8)
    z4_vals = tl.load(z4_ptr + scale_offs, mask=scale_mask, other=0).to(tl.int8)
    
    # Dequantize
    x = (unpacked_y - z4_vals) * s4_vals
    
    # Store results
    tl.store(x_ptr + offs, x, mask=mask)


def weight_dequant(y: torch.Tensor, s: torch.Tensor, s4: torch.Tensor, z4: torch.Tensor, 
                  block_size: int = 128, group_size_int4: int = 128, pack_direction: str = "row") -> torch.Tensor:
    """
    Dequantizes the input tensor `y` from int4 to float16 using block-wise dequantization.
    The int4 quantization has a granularity of 1×128 (one scale+zero point per row per 128 columns).

    Args:
        y (torch.Tensor): The input quantized tensor.
        s (torch.Tensor): The scaling factors for int8.
        s4 (torch.Tensor): The scaling factors for int4, shape [M, N/group_size_int4].
        z4 (torch.Tensor): The zero points for int4, shape [M, N/group_size_int4].
        block_size (int, optional): The size of the blocks for int8 dequantization. Default is 128.
        group_size_int4 (int, optional): The size of the blocks for int4 dequantization. Default is 128.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    assert y.is_contiguous(), 'Input tensor must be contiguous'
    assert y.dim() == 2, 'Input tensors must have 2 dimensions'
    assert y.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    assert block_size % group_size_int4 == 0, 'block_size must be divisible by group_size_int4'
    assert pack_direction == "row", "Only row packing is supported for dequantization"

    M, N = y.size()
    elements_per_sample = PACKING_BITS // W_BITS
    UNPACK_M = M * elements_per_sample
    dequant_int4_x = torch.empty((UNPACK_M, N), dtype=torch.int8, device=y.device)
    dequant_int8_x = torch.empty((UNPACK_M, N), dtype=torch.float16, device=y.device)

    # First dequantize from int4 to int8
    grid = (triton.cdiv(UNPACK_M, group_size_int4), triton.cdiv(N, group_size_int4))
    weight_dequant_int4_kernel[grid](y, dequant_int4_x, s4, z4, UNPACK_M, N, block_size, group_size_int4, 
                                     2**W_BITS - 1, elements_per_sample, W_BITS)
    
    # Then dequantize from int8 to float16
    grid = (triton.cdiv(UNPACK_M, block_size), triton.cdiv(N, block_size))
    weight_dequant_int8_kernel[grid](dequant_int4_x, dequant_int8_x, s, UNPACK_M, N, block_size)

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
    # y = tl.extra.cuda.libdevice.round(x / s)
    y = tl.floor(x / s + 0.5)
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
    # y = tl.clamp(tl.extra.cuda.libdevice.round(x / s), -119.0, 119.0).to(tl.int8)
    y = tl.clamp(tl.floor(x / s + 0.5), -119.0, 119.0).to(tl.int8)
    s = s.to(tl.float16)
    tl.store(s_ptr + pid_m * n + pid_n, s)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def weight_quant_int4_kernel(y_ptr, s4_ptr, z4_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Further quantizes the int8 tensor `y` to int4 using per-row quantization.
    Each row of BLOCK_SIZE elements gets its own scale and zero point.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    blocks_per_row = tl.cdiv(N, BLOCK_SIZE)
    
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    offs = (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)
    
    max_vals = tl.max(y, axis=1)
    min_vals = tl.min(y, axis=1)
    s4 = tl.floor((max_vals - min_vals) / 15.0 + 0.5)
    z4 = tl.floor(-min_vals / s4 + 0.5)
    
    y_uint4 = tl.clamp(
        tl.floor(y / s4[:, None] + 0.5) + z4[:, None], 
        0, 15
    ).to(tl.int8)
    
    tl.store(y_ptr + offs, y_uint4, mask=mask)

    row_mask = offs_m < M
    scale_offs = offs_m * blocks_per_row + pid_n
    
    tl.store(s4_ptr + scale_offs, s4.to(tl.int8), mask=row_mask)
    tl.store(z4_ptr + scale_offs, z4.to(tl.int8), mask=row_mask)


def weight_quant(x: torch.Tensor, block_size: int = 128, group_size_int4: int = 128, 
                 pack_direction: str = "row") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization and further quantizes to int4.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        group_size_int4 (int, optional): The size of the blocks for int4 quantization. Default is 128.
        pack_direction (str, optional): The direction for packing the quantized weights. Default is "row".

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The quantized tensor, the scaling factors for int8, the scaling factors for int4, and the zero points for int4.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.dim() == 2, 'Input tensors must have 2 dimensions'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    assert block_size % group_size_int4 == 0, 'block_size must be divisible by group_size_int4'

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8) # todo: dtype int8, actually uint4, needs to pack to int32
    s = x.new_empty(((M + block_size - 1) // block_size, (N + block_size - 1) // block_size), dtype=torch.float16)
    s4 = x.new_empty(M, (N + group_size_int4 - 1) // group_size_int4, dtype=torch.int8) # dtype int8
    z4 = x.new_empty(M, (N + group_size_int4 - 1) // group_size_int4, dtype=torch.int8) # dtype int8, actually uint4, needs to pack

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_quant_int8_kernel[grid](x, y, s, M, N, block_size)

    grid = (triton.cdiv(M, group_size_int4), triton.cdiv(N, group_size_int4))
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_quant_int4_kernel[grid](y, s4, z4, M, N, group_size_int4)

    # Pack the uint4 weight into int32
    # Note: the int4 weight should shift to uint4, due to triton shift bugs with negative values !
    assert y.min() >= 0 and y.max() <= 15, 'Quantized values must be in the range [0, 15]'

    # y_packed = Int4QuantUtils.pack(y, storage_bits=32, q_bits=4, direction="row")
    if pack_direction == "row":
        assert M % (PACKING_BITS // W_BITS) == 0, f'row nums must be divisible by {PACKING_BITS // W_BITS}'
        y_packed, elements_per_sample = pack_weights_over_rows(y, W_BITS, PACKING_BITS, transpose=False)
    elif pack_direction == "col":
        pass

    return y_packed, s, s4, z4


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
                    GROUP_SIZE_INT4: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation with int8 activations and int4 weights.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A (int8 activations).
        b_ptr (tl.tensor): Pointer to the second input matrix B (int4 weights).
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the int8 scaling factors for matrix B.
        b_s4_ptr (tl.tensor): Pointer to the int4 scaling factors for matrix B.
        b_z4_ptr (tl.tensor): Pointer to the int4 zero points for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_INT4 (tl.constexpr): group size for int4 quantization (128).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    k_int4 = tl.cdiv(K, GROUP_SIZE_INT4)
    
    # Calculate offsets
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to input matrices
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]  # (BLOCK_M, BLOCK_K)
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]  # (BLOCK_K, BLOCK_N)
    
    # Pointers to int8 scaling factors
    a_s_ptrs = a_s_ptr + offs_m * k  # (BLOCK_M,), activation quant block (1, 128)
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k  # (BLOCK_N,), weight quant block (128, 128)

    # Pointers to int4 scaling factors and zero points
    b_s4_ptrs = b_s4_ptr + offs_n[None, :] * k_int4 + offs_k[:, None] // GROUP_SIZE_INT4 # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    b_z4_ptrs = b_z4_ptr + offs_n[None, :] * k_int4 + offs_k[:, None] // GROUP_SIZE_INT4 # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for i in range(k):

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)

        # int4_s and int4_z is int8 dtype
        # the int4 dequantization output is int8 dtype
        int4_s = tl.load(b_s4_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=1.0)
        int4_z = tl.load(b_z4_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        b_dequant = (b - int4_z) * int4_s
        # tl.static_print("dequant int4:", b_dequant.dtype)
        
        # Load int8 scales
        a_s = tl.load(a_s_ptrs).to(tl.float32)
        b_s = tl.load(b_s_ptrs).to(tl.float32)
        
        # Compute matrix multiplication with scales
        accumulator += tl.dot(a, b_dequant) * a_s[:, None] * b_s[None, :]
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
        b_s4_ptrs += (BLOCK_SIZE_K // GROUP_SIZE_INT4)
        b_z4_ptrs += (BLOCK_SIZE_K // GROUP_SIZE_INT4)
    
    # Convert accumulator to output dtype
    c = accumulator.to(c_ptr.dtype.element_ty)
    
    # Store results
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def w4a8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, 
              b_s: torch.Tensor, b_s4: torch.Tensor, b_z4: torch.Tensor, GROUP_SIZE_INT4: int=128) -> torch.Tensor:
    """
    Perform a matrix multiplication using int8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The int8 scaling factor for the second input matrix, must be contiguous.
        b_s4 (torch.Tensor): The int4 scaling factor for the second input matrix for int4 quantization, must be contiguous.
        b_z4 (torch.Tensor): The int4 zero point for the second input matrix for int4 quantization, must be contiguous.
        group_size_int4 (int, optional): The block size for int4 quantization. Default is 16.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    # a for int8 activation, shape (M, K), (bsz*seq_len, hidden_size)
    # b for int4 weight,     shape (N, K), (out_features, in_features) (out_features, hidden_size)
    # gemm -> a @ b_t

    # TODO: move unpack function to the w4a8_gemm_kernel inside
    b = Int4QuantUtils.unpack(b, storage_bits=32, q_bits=4, direction="row")
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    assert b_s4.is_contiguous() and b_z4.is_contiguous(), 'Scaling factor and zero point tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    w4a8_gemm_kernel[grid](a, b, c, a_s, b_s, b_s4, b_z4, M, N, K, GROUP_SIZE_INT4)
    return c


def test_w4a8_gemm():
    # Create random tensors for a and b
    a = torch.randn(1024, 7168, dtype=torch.float16, device='cuda').contiguous()
    b = torch.randn(1024, 7168, dtype=torch.float16, device='cuda').contiguous()
    
    # Quantize tensors
    a_quant, a_s = act_quant(a)
    b_quant, b_s, b_s4, b_z4 = weight_quant(b)

    # Perform int8 GEMM
    c = w4a8_gemm(a_quant, a_s, b_quant, b_s, b_s4, b_z4)
    c_float = a.matmul(b.t())
    
    # Print the result
    assert c.isnan().sum() == 0, 'Result of int8 GEMM contains NaNs'
    assert c.isinf().sum() == 0, 'Result of int8 GEMM contains Infs'

    cos_sim = torch.nn.functional.cosine_similarity(c.view(-1).float(), c_float.view(-1).float(), dim=0)
    print (f'Cosine similarity with float32 GEMM: {cos_sim.item()}')

def test_weight_quant_dequant():
    # Create a random FP16 tensor
    x = torch.randn(1024, 7168, dtype=torch.float16, device='cuda').contiguous()
    
    # Quantize the tensor
    y_quant, s, s4, z4 = weight_quant(x)
    
    # Dequantize the tensor
    x_dequant = weight_dequant(y_quant, s, s4, z4)
    
    # Calculate cosine similarity between the original and dequantized tensors
    cos_sim = torch.nn.functional.cosine_similarity(x.view(-1).float(), x_dequant.view(-1).float(), dim=0)
    
    print(f'Cosine similarity between original and dequantized tensor: {cos_sim.item()}')

if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    test_weight_quant_dequant()
    # test_w4a8_gemm()