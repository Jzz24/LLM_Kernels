 #include <torch/all.h>
 #include <torch/python.h>
 #include <ATen/cuda/CUDAContext.h>
 #include <cuda_runtime.h>
 
 // Forward declaration of the CUDA kernel function
 // This should match the signature in your CUDA file
 int marlin_cuda(
     const void* A,
     const void* B,
     void* C,
     int m,
     int n,
     int k,
     int batch_size = 1,
     int dev = 0,
     cudaStream_t stream = 0
 );
 
 // Error codes
 const int ERR_SHAPE_MISMATCH = 1;
 const int ERR_UNSUPPORTED_TYPE = 2;
 
 // PyTorch binding function
 void gemm(
     const torch::Tensor& A,
     const torch::Tensor& B,
     torch::Tensor& C,
     int batch_size = 1
 ) {
     // Get problem dimensions
     int m = A.size(0);
     int k = A.size(1);
     int n = B.size(1);
     
     // Validate input dimensions
     if (B.size(0) != k) {
         AT_ERROR("Matrix dimensions don't match: A is ", m, "x", k, 
                  " but B is ", B.size(0), "x", n);
     }
     
     if (C.size(0) != m || C.size(1) != n) {
         AT_ERROR("Output matrix C has wrong dimensions: expected ", m, "x", n,
                  " but got ", C.size(0), "x", C.size(1));
     }
     
     // Ensure all tensors are on the same CUDA device
     int dev = A.get_device();
     if (B.get_device() != dev || C.get_device() != dev) {
         AT_ERROR("All tensors must be on the same CUDA device");
     }
     
     // Validate data types (assuming half/fp16)
     if (A.scalar_type() != torch::kHalf || 
         B.scalar_type() != torch::kHalf || 
         C.scalar_type() != torch::kHalf) {
         AT_ERROR("All matrices must be of type torch.float16");
     }
     
     // Call the CUDA kernel
     int err = simple_gemm_cuda(
         A.data_ptr(),
         B.data_ptr(),
         C.data_ptr(),
         m, n, k,
         batch_size,
         dev,
         at::cuda::getCurrentCUDAStream(dev)
     );
     
     // Handle errors
     if (err == ERR_SHAPE_MISMATCH) {
         AT_ERROR("Matrix dimensions not compatible with kernel implementation");
     } else if (err == ERR_UNSUPPORTED_TYPE) {
         AT_ERROR("Unsupported data type for GEMM kernel");
     } else if (err != 0) {
         AT_ERROR("Unknown error in GEMM kernel: ", err);
     }
 }
 
 // Register Python bindings
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("gemm", &gemm, 
           "Simple GEMM implementation for FP16 matrices",
           py::arg("A"),
           py::arg("B"),
           py::arg("C"),
           py::arg("batch_size") = 1);
 }