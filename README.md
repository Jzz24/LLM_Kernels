# LLM_Kernels
A high-performance implementation and verification toolkit for LLM kernels.
## MoE
- [x] Multiple communication strategies (All-to-All, AllGather)
- [x] Group GEMM acceleration
- [x] Quantized Group GEMM
## Quantization
- [x] fp8 blockwise gemm
- [x] int8 gemm
- [x] w4a8 gemm
- [ ] int4 weight pack/unpack (cuda)
## MHA
- [ ] sage attention
