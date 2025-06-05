# LLM_Kernels
A simple implementation and verification toolkit for LLM kernels.

## Quantization
- [x] fp8 blockwise gemm
- [x] int8 gemm
- [x] w4a8 gemm（triton）
- [x] int4 weight pack/unpack
- [x] w4a16 gemm (cuda simple Marlin)
- [ ] w4a8 gemm (cuda, simple Qserve)
- [ ] fp4/6/8 fake quantize function
## MoE
- [x] Multiple communication strategies (All-to-All, AllGather)
- [x] Group GEMM acceleration
- [x] Quantized Group GEMM
## Attention
- [ ] sage attention
