import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# 获取当前文件所在目录
_current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT编译CUDA扩展
marlin_cuda = load(
    name='marlin_cuda',
    sources=[
        # 调整路径，因为不再在simple_marlin包中
        os.path.join(_current_dir, 'simple_marlin/cuda/marlin_cuda.cpp'),
        os.path.join(_current_dir, 'simple_marlin/cuda/marlin_cuda_kernel.cu')
    ],
    verbose=True,
    # 可选编译优化标志
    extra_cuda_cflags=['-O3']
)

def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    return marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)

# 添加测试代码
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试参数
    m, k, n = 16, 64, 256  # 示例维度
    
    # to do