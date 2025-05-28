#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// Vec类型定义 - 保持不变
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;  // 16x16 A矩阵片段
using FragB = Vec<half2, 2>;  // 8x16 B矩阵片段  
using FragC = Vec<float, 4>;  // 16x8 累加器片段
using FragS = Vec<half2, 1>;  // 量化缩放因子

// 核心硬件指令 - 保持不变
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

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
    *reinterpret_cast<const half2*>(&MUL), 
    *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

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

// 简化的 Marlin kernel - warp tile 16x64
template <int THREADS, int THREAD_M_BLOCKS, int THREAD_N_BLOCKS>
__global__ void SimpleMarlin(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ s,
  int prob_m,
  int prob_n,
  int prob_k,
  int groupsize = -1
) {
  // 每个warp处理16x64的tile
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  
  // 计算当前线程块的起始位置
  const int block_m = blockIdx.y * (16 * THREAD_M_BLOCKS);
  const int block_n = blockIdx.x * (16 * THREAD_N_BLOCKS);
  
  // 共享内存布局 - 简化版本
  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (16 * THREAD_M_BLOCKS * 16 / 8); // A tile
  int4* sh_s = sh_b + (16 * 16 * THREAD_N_BLOCKS / 32); // B tile
  
  // 寄存器片段存储 - 注意frag_c的维度
  FragA frag_a[THREAD_M_BLOCKS]; // 4个Vec<half2, 4>
  I4 frag_b_quant; // Vec<int, 4>;
  FragC frag_c[THREAD_M_BLOCKS][4][2];  // 最后一维是2，对应两次MMA, Vec<float, 4>
  FragS frag_s[4]; // 4个Vec<half2, 1>;

  // 初始化累加器
  #pragma unroll
  for (int i = 0; i < THREAD_M_BLOCKS; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          #pragma unroll
          for (int k = 0; k < 2; k++) {  // 两个MMA结果
              #pragma unroll
              for (int l = 0; l < 4; l++) {
                  frag_c[i][j][k][l] = 0.0f;
              }
          }
      }
  }
  
  // K维度循环
  for (int k_offset = 0; k_offset < prob_k; k_offset += 16) {
    
    // 加载A矩阵到共享内存 - 简化版本
    // if (threadIdx.x < 128) {
    //   int a_row = block_m + (threadIdx.x / 8) * 2;
    //   int a_col = k_offset + (threadIdx.x % 8) * 2;
    //   if (a_row < prob_m && a_col < prob_k) {
    //     int a_idx = a_row * (prob_k / 8) + a_col / 8;
    //     sh_a[threadIdx.x] = A[a_idx];
    //   }
    // }
    if (threadIdx.x < 128) { // 128个线程处理64*16的fp16矩阵分块
      int row = threadIdx.x / 2;     // 0-63行
      int col = threadIdx.x % 2;     // 0-1列
  
      if ((block_m + row) < prob_m && (k_offset + col * 8) < prob_k) {
        int a_idx = (block_m + row) * (prob_k / 8) + (k_offset / 8) + col;
        sh_a[threadIdx.x] = A[a_idx];
      }
    }
    
    // // 加载B矩阵到共享内存 - 简化版本
    // if (threadIdx.x < 64) {
    //   int b_row = k_offset + threadIdx.x / 4;
    //   int b_col = block_n + (threadIdx.x % 4) * 16;
    //   if (b_row < prob_k && b_col < prob_n) {
    //     int b_idx = b_row * (prob_n / 32) + b_col / 32;
    //     sh_b[threadIdx.x] = B[b_idx];
    //   }
    // }
    
    // // 加载缩放因子
    // if (groupsize != -1 && threadIdx.x < 32) {
    //   int s_row = k_offset / groupsize;
    //   int s_col = block_n + threadIdx.x * 2;
    //   if (s_col < prob_n) {
    //     int s_idx = s_row * (prob_n / 8) + s_col / 8;
    //     sh_s[threadIdx.x] = s[s_idx];
    //   }
    // }
    
    // __syncthreads();
    
    // // 从共享内存加载到寄存器
    // if (warp_id < (THREAD_M_BLOCKS * THREAD_N_BLOCKS / 4)) {
    //   int warp_m = (warp_id / (THREAD_N_BLOCKS / 4)) * 16;
    //   int warp_n = (warp_id % (THREAD_N_BLOCKS / 4)) * 64;
      
    //   // 加载A片段
    //   #pragma unroll
    //   for (int i = 0; i < THREAD_M_BLOCKS; i++) {
    //     int a_sh_offset = (warp_m + i * 16 + (lane_id / 4) * 8) * 2 + (lane_id % 4) * 2;
    //     ldsm4(frag_a[i], &sh_a[a_sh_offset / 8]);
    //   }
      
    //   // 加载缩放因子
    //   if (groupsize != -1) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //       int s_sh_offset = (warp_n / 16 + j) * 2 + lane_id / 16;
    //       frag_s[j] = *reinterpret_cast<FragS*>(&sh_s[s_sh_offset / 8]);
    //     }
    //   }
      
    //   // 执行4次sub-tile计算
    //   #pragma unroll
    //   for (int sub_tile = 0; sub_tile < 4; sub_tile++) {
    //       // 加载B片段
    //       int b_sh_offset = sub_tile * 16 + (lane_id / 4) * 2 + (lane_id % 4);
    //       frag_b_quant = *reinterpret_cast<I4*>(&sh_b[b_sh_offset / 32]);
          
    //       // 对4个int4值进行dequant和MMA
    //       #pragma unroll
    //       for (int b_frag = 0; b_frag < 4; b_frag++) {
    //           // 获取32位打包的INT4数据
    //           int b_quant = frag_b_quant[b_frag];
    //           int b_quant_shift = b_quant >> 8;  // 关键：右移8位获取高位数据
              
    //           // 第一次解量化和MMA（低8位）
    //           FragB frag_b0 = dequant(b_quant);
    //           if (groupsize != -1) {
    //               scale(frag_b0, frag_s[sub_tile], 0);
    //           }
              
    //           // 第二次解量化和MMA（高8位）
    //           FragB frag_b1 = dequant(b_quant_shift);
    //           if (groupsize != -1) {
    //               scale(frag_b1, frag_s[sub_tile], 1);
    //           }
              
    //           // 执行两次MMA - 这是关键！
    //           #pragma unroll
    //           for (int i = 0; i < THREAD_M_BLOCKS; i++) {
    //               mma(frag_a[i], frag_b0, frag_c[i][sub_tile][0]);  // 第一次MMA
    //               mma(frag_a[i], frag_b1, frag_c[i][sub_tile][1]);  // 第二次MMA
    //           }
    //       }
    //   }
    // }
  }
  
  // 写回结果 - 简化版本
  if (warp_id < (THREAD_M_BLOCKS * THREAD_N_BLOCKS / 4)) {
    int warp_m = (warp_id / (THREAD_N_BLOCKS / 4)) * 16;
    int warp_n = (warp_id % (THREAD_N_BLOCKS / 4)) * 64;
    
    #pragma unroll
    for (int i = 0; i < THREAD_M_BLOCKS; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        int c_row = block_m + warp_m + i * 16 + (lane_id / 4) * 2;
        int c_col = block_n + warp_n + j * 16 + (lane_id % 4) * 2;
        
        if (c_row < prob_m && c_col < prob_n) {
          // 转换为half并写回
          int4 c_out;
          #pragma unroll
          for (int k = 0; k < 4; k++) {
            // reinterpret_cast<half*>(&c_out)[k] = __float2half(frag_c[i][j][k]);
            float result = frag_c[i][j][0][k] + frag_c[i][j][1][k]; // 合并两次MMA结果
            reinterpret_cast<half*>(&c_out)[k] = __float2half(result);
          }
          
          int c_idx = c_row * (prob_n / 8) + c_col / 8;
          C[c_idx] = c_out;
        }
      }
    }
  }
}

// 启动函数
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
) {
  // 固定配置：每个warp处理16x64
  const int THREADS = 256;
  const int THREAD_M_BLOCKS = 4;  // 64行
  const int THREAD_N_BLOCKS = 4;  // 64列
  
  // 检查维度对齐
  if (prob_m % 16 != 0 || prob_n % 16 != 0 || prob_k % 16 != 0) {
    return -1;
  }
  
  // 计算网格维度
  dim3 blocks(
    ceildiv(prob_n, 16 * THREAD_N_BLOCKS),
    ceildiv(prob_m, 16 * THREAD_M_BLOCKS)
  );
  
  // 计算共享内存大小
  int smem_size = (16 * THREAD_M_BLOCKS * 16 / 8 +  // A, dtype=fp16, int4(128bit) // fp16(16bit) = 8
                   16 * 16 * THREAD_N_BLOCKS / 32 +  // B, dtype=uint4, int4(128bit) // 4bit = 32
                   THREAD_N_BLOCKS * 16 / 8) * sizeof(int4);  // scales, dtype=fp16
  
  
  
  // // 启动kernel
  // SimpleMarlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS>
  //   <<<blocks, THREADS, smem_size, stream>>>(
  //     reinterpret_cast<const int4*>(A),
  //     reinterpret_cast<const int4*>(B),
  //     reinterpret_cast<int4*>(C),
  //     reinterpret_cast<const int4*>(s),
  //     prob_m, prob_n, prob_k, groupsize
  //   );
    
  // return cudaGetLastError() == cudaSuccess ? 0 : -1;
  // 清除之前的错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error before kernel launch: %s\n", cudaGetErrorString(err));
    // 继续执行，因为这是之前的错误
  }
  
  // 启动kernel
  SimpleMarlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS>
    <<<blocks, THREADS, smem_size, stream>>>(
      reinterpret_cast<const int4*>(A),
      reinterpret_cast<const int4*>(B),
      reinterpret_cast<int4*>(C),
      reinterpret_cast<const int4*>(s),
      prob_m, prob_n, prob_k, groupsize
    );
  
  // 检查kernel启动错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Launch Error: %s (code %d)\n", 
           cudaGetErrorString(err), (int)err);
    return (int)err;
  }
  
  // 等待kernel执行完成并检查执行错误
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    printf("CUDA Execution Error: %s (code %d)\n", 
           cudaGetErrorString(err), (int)err);
    return (int)err;
  }
  
  printf("CUDA kernel completed successfully!\n");
  return 0;
}