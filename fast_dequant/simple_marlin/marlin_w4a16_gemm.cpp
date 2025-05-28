#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA kernel function
int simple_marlin_cuda(
    const void* A,
    const void* B,
    void* C,
    const void* s,
    int prob_m,
    int prob_n,
    int prob_k,
    int groupsize = -1,
    cudaStream_t stream = 0
);

// Error codes
const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

// Main PyTorch binding function for Marlin W4A16 GEMM
void mul(
    const torch::Tensor& A,        // FP16 input matrix [M, K]
    const torch::Tensor& B,        // INT4 packed weight matrix 
    torch::Tensor& C,              // FP16 output matrix [M, N]
    const torch::Tensor& s,        // FP16 scale factors
    torch::Tensor& workspace,      // Workspace (unused but kept for compatibility)
    int thread_k = -1,
    int thread_n = -1,
    int sms = -1,
    int max_par = 16
) {
    // Get problem dimensions
    int prob_m = A.size(0);
    int prob_k = A.size(1);
    int prob_n = C.size(1);
    
    // Validate alignment requirements (must be divisible by 16)
    if (prob_m % 16 != 0 || prob_n % 16 != 0 || prob_k % 16 != 0) {
        AT_ERROR("Matrix dimensions must be divisible by 16. Got M=", prob_m, 
                 ", N=", prob_n, ", K=", prob_k);
    }
    
    // Ensure all tensors are on the same CUDA device
    int dev = A.get_device();
    if (B.get_device() != dev || C.get_device() != dev || s.get_device() != dev) {
        AT_ERROR("All tensors must be on the same CUDA device");
    }
    
    // Validate data types
    if (A.scalar_type() != torch::kHalf || C.scalar_type() != torch::kHalf) {
        AT_ERROR("A and C matrices must be of type torch.float16");
    }
    
    if (s.scalar_type() != torch::kHalf) {
        AT_ERROR("Scale tensor must be of type torch.float16");
    }
    
    if (B.scalar_type() != torch::kInt32) {
        AT_ERROR("B matrix must be packed INT4 data (torch.int32)");
    }
    
    // Ensure tensors are contiguous
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous(); 
    auto C_contig = C.contiguous();
    auto s_contig = s.contiguous();
    
    // Determine groupsize from scale tensor shape
    int groupsize = -1;
    if (s.dim() == 2) {
        int scale_rows = s.size(0);
        int scale_cols = s.size(1);
        
        if (scale_cols != prob_n) {
            AT_ERROR("Scale tensor N dimension mismatch: expected ", prob_n, 
                     " but got ", scale_cols);
        }
        
        groupsize = prob_k / scale_rows;
        
        // Validate groupsize
        if (prob_k % groupsize != 0) {
            AT_ERROR("K dimension not divisible by groupsize: K=", prob_k, 
                     ", groupsize=", groupsize);
        }
    }
    
    // Call the CUDA kernel
    int err = simple_marlin_cuda(
        A_contig.data_ptr(),
        B_contig.data_ptr(),
        C_contig.data_ptr(),
        s_contig.data_ptr(),
        prob_m,
        prob_n,
        prob_k,
        groupsize,
        at::cuda::getCurrentCUDAStream(dev)
    );
    
    // Handle errors
    if (err == ERR_PROB_SHAPE) {
        AT_ERROR("Matrix dimensions not compatible with Marlin kernel requirements");
    } else if (err == ERR_KERN_SHAPE) {
        AT_ERROR("Kernel configuration error or insufficient GPU resources");
    } else if (err != 0) {
        AT_ERROR("Unknown error in Marlin kernel: error code ", err);
    }
}

// Register Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Marlin W4A16 GEMM: Optimized INT4 quantized matrix multiplication";
    
    // 只保留核心的 mul 函数
    m.def("mul", &mul, 
          "Marlin W4A16 GEMM multiplication",
          py::arg("A"),
          py::arg("B"), 
          py::arg("C"),
          py::arg("s"),
          py::arg("workspace"),
          py::arg("thread_k") = -1,
          py::arg("thread_n") = -1,
          py::arg("sms") = -1,
          py::arg("max_par") = 16);
}