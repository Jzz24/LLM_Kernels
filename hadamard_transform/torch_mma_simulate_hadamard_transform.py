import torch
import numpy as np
from scipy.linalg import hadamard

def _create_block_diagonal_h(size, dtype, device):
    """辅助函数：创建块对角的哈达玛矩阵。"""
    if size == 0: # 处理 rem_size 可能为0的情况
        return torch.eye(16, dtype=dtype, device=device)
    num_blocks = 16 // size
    H_small = torch.from_numpy(hadamard(size)).to(dtype).to(device)
    H_block_diagonal = torch.zeros(16, 16, dtype=dtype, device=device)
    for i in range(num_blocks):
        start, end = i * size, (i + 1) * size
        H_block_diagonal[start:end, start:end] = H_small
    return H_block_diagonal


def reshape_column_major(tensor: torch.Tensor, new_shape_last_two: tuple):
    """
    将输入的三维张量 (B, H, W)，对其最后两个维度 (H, W)
    按照“列优先”的规则重塑为新的形状 (B, new_H, new_W)
    """
    B, H, W = tensor.shape
    new_H, new_W = new_shape_last_two

    if H * W != new_H * new_W:
        raise ValueError("The total number of elements in the last two dimensions must remain unchanged")

    # (B, H, W) -> (B, W, H)
    tensor_t = tensor.transpose(-2, -1).contiguous()
    # (B, W, H) -> (B, new_W, new_H)
    reshaped_tensor_t = tensor_t.view(B, new_W, new_H)
    # (B, new_W, new_H) -> (B, new_H, new_W)
    reshaped_tensor = reshaped_tensor_t.transpose(-2, -1).contiguous()
    
    return reshaped_tensor

def reshape_from_column_major(reshaped_tensor: torch.Tensor, original_shape_last_two: tuple):
    """
    将一个经过列优先重排的张量，还原回其最初的行优先形状。
    将输入的三维张量 (B, new_H, new_W)，对其最后两个维度 (new_H, new_W)
    还原为原始形状 (B, H, W), 为上述 reshape_column_major 的逆操作
    """
    B, new_H, new_W = reshaped_tensor.shape
    H, W = original_shape_last_two

    if H * W != new_H * new_W:
        raise ValueError("The total number of elements in the last two dimensions must remain unchanged")

    # 1. 逆转最后的转置: (B, new_H, new_W) -> (B, new_W, new_H)
    reshaped_tensor_t = reshaped_tensor.transpose(-2, -1).contiguous()

    # 2. 逆转 view 操作: (B, new_W, new_H) -> (B, W, H)
    tensor_t = reshaped_tensor_t.view(B, W, H)

    # 3. 逆转最初的转置: (B, W, H) -> (B, H, W)
    original_tensor = tensor_t.transpose(-2, -1).contiguous()

    return original_tensor

def hadamard_kernel_simulation_left_multiply(a: torch.Tensor, had_size: int):

    assert had_size > 0 and (had_size & (had_size - 1)) == 0, "had_size must be a power of 2"
    num_elements = a.numel()
    assert num_elements % 256 == 0, "The number of elements in the input tensor must be a multiple of 256"
    num_chunks = num_elements // 256
    device, dtype = a.device, a.dtype

    X = a.view(num_chunks, 16, 16)
    log_had_size = int(np.log2(had_size))
    
    X_T = X.transpose(-2, -1)
    H_16 = torch.from_numpy(hadamard(16)).to(dtype).to(device)

    if had_size <= 16:
        # 阶段一: had_size <= 16
        H_16_diag = _create_block_diagonal_h(had_size, dtype, device)
        Y_temp = torch.matmul(H_16_diag, X_T)
        Y = Y_temp.transpose(-2, -1)
    elif 16 < had_size <= 256:
        # 阶段二: 16 < had_size <= 256
        k = had_size // 16
        H_16_diag = _create_block_diagonal_h(k, dtype, device)

        Y_temp_T = torch.matmul(H_16, X_T)
        Y_temp1 = Y_temp_T.transpose(-2, -1)
        Y = torch.matmul(H_16_diag, Y_temp1)
    else: # had_size > 256
        num_vectors = num_elements // had_size
        k = had_size // 256

        # 模拟 Z = H_16 @ (H_16 @ X.T).T
        Temp_T = torch.matmul(H_16, X_T) #第一次左乘
        Temp = Temp_T.transpose(-2, -1)
        Z_16x16 = torch.matmul(H_16, Temp)  # 第二次左乘 (num_vectors*k, 16, 16)
        
        # shape (num_vectors, k, 256)
        Z = Z_16x16.view(num_vectors, k, 256)

        # # 步骤 3: 模拟 Y = H_k @ Z
        # 如何将这一步转换成为h_16的左乘呢
        # H_k = torch.from_numpy(hadamard(k)).to(dtype).to(device)
        # Y = torch.matmul(H_k, Z)

        # 步骤 3: 模拟 Y = H_k @ Z，并用 H_16 的 MMA 指令完成
        # 这一步模拟 CUDA l=1 阶段的行为
        if k <= 16:
            # reshape -> (num_vectors, 16, 16*k)
            Z_re = reshape_column_major(Z, (16, k * 16))

            # 3b. 准备 H_16 的块对角矩阵，作用于 Z_re 的行
            H_block = _create_block_diagonal_h(k, dtype, device)
            
            # 3c. 执行左乘
            Y_re = torch.matmul(H_block, Z_re)

            # 3d. 将结果逆向重塑回 (num_vectors, k, 256)
            Y = reshape_from_column_major(Y_re, (k, 256))
        else:
            # 如果 k > 16 (例如 H_8192)，则需要更复杂的递归或循环，
            # 但对于 H_4096 (k=16) 及以下，此逻辑已足够。
            # 为保持模拟与理论一致，我们直接使用 matmul
            H_k = torch.from_numpy(hadamard(k)).to(dtype).to(device)
            Y = torch.matmul(H_k, Z)

    return Y.reshape_as(a)

def hadamard_ground_truth(a: torch.Tensor, had_size: int):
    num_vectors = a.numel() // had_size
    h_matrix = torch.from_numpy(hadamard(had_size)).to(a.dtype).to(a.device)
    expected = torch.matmul(a.view(num_vectors, had_size), h_matrix)
    return expected.view_as(a)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    had_sizes_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # 定义每个had_size要测试的chunk数量 (每个chunk是256个元素)
    chunks_to_test = [1, 2, 4, 8, 16]

    failed_cases = []

    for h_size in had_sizes_to_test:
        for n_chunks in chunks_to_test:
            num_elements = n_chunks * 256
            
            if num_elements % h_size != 0:
                continue

            case_name = f"Test H_{h_size} with {n_chunks} chunk(s) ({num_elements} elements)"
            
            input_tensor = torch.randn(num_elements, device=device)

            expected = hadamard_ground_truth(input_tensor.clone(), h_size)
            
            output_left_multiply = hadamard_kernel_simulation_left_multiply(input_tensor.clone(), h_size)

            is_left_correct = torch.allclose(output_left_multiply, expected, atol=1e-3, rtol=1e-3)

            if not is_left_correct:
                failed_cases.append(case_name)

    print("\n" + "="*60)
    if not failed_cases:
        print("All tests passed!")
    else:
        print("The following tests failed:")
        for case in failed_cases:
            print(f"  - {case}")
    print("="*60)