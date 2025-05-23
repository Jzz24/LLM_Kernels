#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// 保持和 Marlin 相同的 Vec 结构定义
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales

// 使用 Marlin 中相同的 ldsm4 实现
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// 使用 Marlin 中相同的 lop3 实现
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// 使用 Marlin 中相同的 dequant 实现
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

// 使用 Marlin 中相同的 scale 实现
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// 使用 Marlin 中相同的 mma 实现
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// 简化版 INT4 矩阵乘法 kernel
template <
  const int threads=256,
  const int thread_m_blocks=4, // 每个线程块处理的输出行数 (16*thread_m_blocks)
  const int thread_n_blocks=8  // 每个线程块处理的输出列数 (16*thread_n_blocks)
>
__global__ void simple_int4_gemm(
  const int4* __restrict__ A, // fp16 输入矩阵 [M, K]
  const int4* __restrict__ B, // 量化权重 [K/8, N]
        int4* __restrict__ C, // fp16 输出矩阵 [M, N]
  const int4* __restrict__ s, // 量化系数 [K/16, N]
  int prob_m,  // 批次维度 m
  int prob_n,  // 输出维度 n
  int prob_k   // 输入维度 k
) {
  // 计算当前线程块处理的区域
  constexpr int thread_k_blocks = 4;  // 每个线程块处理的k维度块数 (16*thread_k_blocks)
  
  int block_m = blockIdx.y * 16 * thread_m_blocks;
  int block_n = blockIdx.x * 16 * thread_n_blocks;
  
  // 线程在块内的位置
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  
  // 每个warp处理的16×64的输出区域
  int warp_m = (warp_id / (thread_n_blocks/4)) * 16;
  int warp_n = (warp_id % (thread_n_blocks/4)) * 64;
  
  // 共享内存
  extern __shared__ int4 sh[];
  int4* sh_a = sh;                  // 输入矩阵A共享内存
  int4* sh_b = sh_a + (16 * thread_m_blocks * thread_k_blocks / 8); // 权重矩阵B共享内存
  int4* sh_s = sh_b + (thread_k_blocks * 16 * thread_n_blocks / 8); // 量化系数共享内存
  
  // 寄存器存储
  FragA frag_a[thread_m_blocks];    // A矩阵片段
  I4 frag_b_quant;                  // 量化的B矩阵
  FragC frag_c[thread_m_blocks][4][2] = {0}; // 累加器
  FragS frag_s[4];                  // 量化系数
  
  // 共享内存索引计算
  int a_sh_stride = 16 * thread_k_blocks / 8;
  int a_sh_rd = a_sh_stride * ((lane_id) % 16) + (lane_id) / 16;
  a_sh_rd += 2 * ((warp_id * 32) / (thread_n_blocks / 4));
  
  int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_sh_rd = lane_id;
  
  int s_sh_stride = 16 * thread_n_blocks / 8;
  int s_sh_rd = 8 * ((lane_id / 32) % (thread_n_blocks / 4)) + (lane_id % 32) / 4;
  
  // 主循环：沿K维度迭代
  for (int k_offset = 0; k_offset < prob_k; k_offset += 16 * thread_k_blocks) {
    // 加载输入矩阵A到共享内存
    for (int i = threadIdx.x; i < 16 * thread_m_blocks * thread_k_blocks / 8; i += threads) {
      int row_block = i / (thread_k_blocks * 2);
      int col_block = (i % (thread_k_blocks * 2)) / 2;
      int row = 16 * row_block + 8 * ((i % 2) / 1);
      int col = 16 * col_block + 8 * (i % 1);
      
      if (block_m + row < prob_m && k_offset + col < prob_k) {
        sh_a[i] = A[(block_m + row) * prob_k / 8 + (k_offset / 8) + col / 8];
      } else {
        sh_a[i] = {0};
      }
    }
    
    // 加载权重矩阵B到共享内存
    for (int i = threadIdx.x; i < thread_k_blocks * thread_n_blocks * 2; i += threads) {
      int row = i / (thread_n_blocks * 2);
      int col = i % (thread_n_blocks * 2);
      
      if (k_offset / 8 + row < prob_k / 8 && block_n / 8 + col < prob_n / 8) {
        sh_b[i] = B[(k_offset / 8 + row) * prob_n / 8 + (block_n / 8) + col];
      } else {
        sh_b[i] = {0};
      }
    }
    
    // 加载量化系数到共享内存
    for (int i = threadIdx.x; i < thread_n_blocks * 2; i += threads) {
      if (k_offset / 16 < prob_k / 16 && block_n / 8 + i < prob_n / 8) {
        sh_s[i] = s[(k_offset / 16) * prob_n / 8 + (block_n / 8) + i];
      } else {
        sh_s[i] = {0};
      }
    }
    
    __syncthreads();
    
    // 内循环：计算当前K块与输出块的乘积
    for (int k_iter = 0; k_iter < thread_k_blocks; k_iter++) {
      // 加载A矩阵片段
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        int a_sh_rd_idx = a_sh_rd + a_sh_stride * 16 * i + 2 * k_iter;
        ldsm4(frag_a[i], &sh_a[a_sh_rd_idx]);
      }
      
      // 处理连续的8列 (4个n_iter)
      #pragma unroll
      for (int n_iter = 0; n_iter < 4; n_iter++) {
        // 加载量化的B矩阵
        int b_idx = (k_iter * 2) * b_sh_stride + warp_n / 8 + n_iter * 8 + b_sh_rd % (b_sh_stride / 4);
        frag_b_quant = *reinterpret_cast<I4*>(&sh_b[b_idx]);
        
        // 加载量化系数
        frag_s[n_iter] = *reinterpret_cast<FragS*>(&sh_s[warp_n / 8 + n_iter]);
        
        // 计算两列 (n_offset 0-1)
        #pragma unroll
        for (int n_offset = 0; n_offset < 2; n_offset++) {
          // 解量化权重
          int b_quant = frag_b_quant[n_offset * 2];
          FragB frag_b0 = dequant(b_quant);
          scale(frag_b0, frag_s[n_iter], 0);
          
          // 第二组权重
          int b_quant_shift = frag_b_quant[n_offset * 2 + 1];
          FragB frag_b1 = dequant(b_quant_shift);
          scale(frag_b1, frag_s[n_iter], 1);
          
          // 矩阵乘法
          #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
            mma(frag_a[i], frag_b0, frag_c[i][n_iter][n_offset]);
            mma(frag_a[i], frag_b1, frag_c[i][n_iter][n_offset]);
          }
        }
      }
    }
    __syncthreads();
  }
  
  // warp内结果归约（如果每个warp有多个线程处理相同输出）
  // 简化版本省略了这部分
  
  // 将结果写回全局内存
  const int active_threads = 32 * thread_n_blocks / 4;
  if (threadIdx.x < active_threads) {
    int c_row_offset = block_m + warp_m;
    int c_col_offset = block_n + warp_n + (threadIdx.x % 32) / 4 * 8;
    
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      int row = c_row_offset + i * 16 + (threadIdx.x / 32) * 4 + (threadIdx.x % 4);
      if (row < prob_m) {
        for (int j = 0; j < 4; j++) {
          int col = c_col_offset + j * 2;
          if (col < prob_n) {
            int4 c_out;
            // Pack results to float16
            for (int k = 0; k < 2; k++) {
              #pragma unroll
              for (int l = 0; l < 4; l++) {
                reinterpret_cast<half*>(&c_out)[k*4+l] = __float2half(frag_c[i][j][k][l]);
              }
            }
            C[row * prob_n / 8 + col / 8] = c_out;
          }
        }
      }
    }
  }
}

// 动态共享内存大小计算
int get_smem_size(int thread_m_blocks, int thread_n_blocks, int thread_k_blocks) {
  // A共享内存 + B共享内存 + 缩放因子共享内存
  return (16 * thread_m_blocks * thread_k_blocks / 8 + 
          thread_k_blocks * 16 * thread_n_blocks / 8 +
          thread_n_blocks * 2) * sizeof(int4);
}

// 启动kernel的主机函数
// to do