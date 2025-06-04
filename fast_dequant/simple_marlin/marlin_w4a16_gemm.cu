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
using FragB = Vec<half2, 2>;  // 16x8 B矩阵片段  
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
        half* __restrict__ C,
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
  FragS frag_s[4]; // 4个Vec<half2, 1>; 4对应 sub_tile数量, half2对应每个sub_tile的2个mma

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
  
    // 128个线程处理64*16的fp16矩阵分块
    if (threadIdx.x < 128) {
      int row = threadIdx.x / 2;     // 0-63行
      int col = threadIdx.x % 2;     // 0-1列

      if ((block_m + row) < prob_m && (k_offset + col * 8) < prob_k) {
        int a_idx = (block_m + row) * (prob_k / 8) + (k_offset / 8) + col;
        sh_a[threadIdx.x] = A[a_idx];
      }
    }

    // 32个线程处理16*64的4bits矩阵分块
    if (threadIdx.x < 32) {
      // int b_row = threadIdx.x / 2;      // 0-15行
      // int b_col = threadIdx.x % 2;      // 0-1列（int4列）
      
      // int b_idx = (k_offset + b_row) * (prob_n / 32) + (block_n / 32) + b_col;
      // sh_b[threadIdx.x] = B[b_idx];

      // // debug
      // // 对于blockIdx.x == 1的情况, 16*64的4bit分块，包装成int4 128bit, 也就是1*32个int4 128bits
      // if (k_offset == 0) {
      //   printf("B[%d] = %d, %d, %d, %d\n", b_idx, B[b_idx].x, B[b_idx].y, B[b_idx].z, B[b_idx].w);
      // }


      // 新的基于数据块（chunk-based）的B矩阵索引计算
      
      int k_strip_idx = k_offset / 16;                     // 当前是第几个K条带 (每个条带16行)
      int n_tile_idx = blockIdx.x;                         // 当前块处理这个K条带内的第几个N瓦片 (每个瓦片64列)
      int num_n_tiles_per_k_strip = prob_n / 64;           // 每个K条带总共有多少个N瓦片

      // 计算当前16x64瓦片（即32个int4）在全局B数组中的起始偏移
      int chunk_base_offset_in_B = (k_strip_idx * num_n_tiles_per_k_strip + n_tile_idx) * 32;

      // 每个线程 (threadIdx.x 从 0 到 31) 加载这个数据块中的一个int4
      int b_idx = chunk_base_offset_in_B + threadIdx.x;
      
      sh_b[threadIdx.x] = B[b_idx]; // 从全局内存B加载到共享内存sh_b
    }

    if (groupsize != -1 && threadIdx.x < 8) {  // 只需要8个线程！
      int s_row = k_offset / groupsize;         // 当前group的行索引
      int s_col_start = block_n + threadIdx.x * 8;  // 每个线程负责8个连续的FP16
      
      if (s_col_start < prob_n) {
        int s_idx = s_row * (prob_n / 8) + (block_n / 8) + threadIdx.x;
        sh_s[threadIdx.x] = s[s_idx];
      }
    }

    __syncthreads();
    
    // 从共享内存加载到寄存器并执行MMA计算
    const int total_warps = THREAD_M_BLOCKS * THREAD_N_BLOCKS / 4;  // 计算参与MMA的warp数量
    if (warp_id < total_warps) {  // 动态计算warp数量
      // 计算warp在M和N维度的分配
      const int warps_n = THREAD_N_BLOCKS / 4;  // N维度的warp数量，1
      const int warp_m_idx = warp_id / warps_n;  // 当前warp在M维度的索引，（0，1，2，3）
      const int warp_n_idx = warp_id % warps_n;  // 当前warp在N维度的索引，（0）
      
      const int warp_m = warp_m_idx * 16;        // warp在M维度的起始位置
      const int warp_n = warp_n_idx * 64;        // warp在N维度的起始位置

      // 加载A片段
      int load_row = warp_m + lane_id % 16;      // 行地址：warp_m + (0-15)
      int load_col = lane_id / 16;               // 列地址：0 或 1
      int a_sh_offset = load_row * 2 + load_col; // int4偏移

      if (a_sh_offset < 128) {
        ldsm4(frag_a[warp_m_idx], &sh_a[a_sh_offset]);
      }
      
      // 加载缩放因子 - 基于warp的N位置
      // if (groupsize != -1) {
      //   #pragma unroll
      //   for (int j = 0; j < 4; j++) { 
      //     int s_sh_offset = (warp_n / 16 + j) * 2 + (lane_id / 16);
      //     if (s_sh_offset < 8) {
      //       frag_s[j] = *reinterpret_cast<FragS*>(&sh_s[s_sh_offset]);
      //     }
      //   }
      // }
      if (groupsize != -1) {
        // 每4个线程读取同一个int4
        int s_sh_offset = lane_id / 4;  
        
        if (s_sh_offset < 8) {
          // 直接使用half2指针读取
          half2* scales_ptr = reinterpret_cast<half2*>(&sh_s[s_sh_offset]);
          
          // 为每个sub_tile分配对应的缩放因子
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            frag_s[j][0] = scales_ptr[j];  // 直接赋值half2
          }
        }
      }
      
      // 简化的B矩阵处理
      // 四个warp全部加载？
      frag_b_quant = *reinterpret_cast<I4*>(&sh_b[lane_id]);

      #pragma unroll
      for (int sub_tile = 0; sub_tile < 4; sub_tile++) {
        int b_quant = frag_b_quant[sub_tile];
        int b_quant_shift = b_quant >> 8;
        
        // 解量化
        FragB frag_b0 = dequant(b_quant);
        FragB frag_b1 = dequant(b_quant_shift);
        
        // 应用缩放因子
        if (groupsize != -1) {
          scale(frag_b0, frag_s[sub_tile], 0);
          scale(frag_b1, frag_s[sub_tile], 1);
        }

        if (blockIdx.x == 1 && warp_id == 0 && 4<=lane_id <= 7 && sub_tile == 0 && k_offset == 0) {
          // printf("warp_id: %d, b_quant: %d, b_quant_shift: %d\n, blockidx.x: %d", warp_id, b_quant, b_quant_shift, blockIdx.x);
          // printf("=== SUB_TILE 0, THREAD 0 DEBUG ===\n");
          // printf("b_quant = 0x%08x, b_quant_shift = 0x%08x\n", b_quant, b_quant_shift);
          printf("blockidx.x %d, thread %d, FRAG_B0: [%.3f, %.3f, %.3f, %.3f], frag_s: [%.3f, %.3f]\n",
            blockIdx.x, lane_id,
           __half2float(frag_b0[0].x), __half2float(frag_b0[0].y),
           __half2float(frag_b0[1].x), __half2float(frag_b0[1].y),
           __half2float(frag_s[sub_tile][0].x), __half2float(frag_s[sub_tile][0].y)); // 2次mma
    
          // printf("FRAG_B1: [%.3f, %.3f, %.3f, %.3f]\n",
          //  __half2float(frag_b1[0].x), __half2float(frag_b1[0].y),
          //  __half2float(frag_b1[1].x), __half2float(frag_b1[1].y));
        }
        
        // MMA计算 - 使用正确的索引
        mma(frag_a[warp_m_idx], frag_b0, frag_c[warp_m_idx][sub_tile][0]);
        mma(frag_a[warp_m_idx], frag_b1, frag_c[warp_m_idx][sub_tile][1]);
      }
    }

    __syncthreads();  // 确保所有计算完成
  }

  // 写回结果 - 也使用动态计算
  const int total_warps = THREAD_M_BLOCKS * THREAD_N_BLOCKS / 4;
  if (warp_id < total_warps) {  // 与计算逻辑保持一致
    const int warps_n = THREAD_N_BLOCKS / 4; // 4/4，一个warp处理N维度
    const int warp_m_idx = warp_id / warps_n; // （0, 1, 2, 3）4个warp并行
    const int warp_n_idx = warp_id % warps_n; // 0
    
    const int warp_m = warp_m_idx * 16; //（0， 16， 32， 48）
    const int warp_n = warp_n_idx * 64;

    #pragma unroll
    for (int j = 0; j < 4; j++) {  // 4个N维度的sub_tile
      #pragma unroll
      for (int col_block = 0; col_block < 2; col_block++) {  // 每个sub_tile的2个8列分块
        
        // 计算基础位置
        int base_row = block_m + warp_m + (lane_id / 4);
        int base_col = block_n + warp_n + j * 16 + col_block * 8 + (lane_id % 4) * 2;
        
        // 写回上半部分 (c[0], c[1])
        if (base_row < prob_m && base_col + 1 < prob_n) {
          int addr_0 = base_row * prob_n + base_col; 
          C[addr_0] = __float2half(frag_c[warp_m_idx][j][col_block][0]);
          C[addr_0 + 1] = __float2half(frag_c[warp_m_idx][j][col_block][1]);
        }
        
        // 写回下半部分 (c[2], c[3])
        int high_row = base_row + 8;
        if (high_row < prob_m && base_col + 1 < prob_n) {
          int addr_1 = high_row * prob_n + base_col;
          C[addr_1] = __float2half(frag_c[warp_m_idx][j][col_block][2]);
          C[addr_1 + 1] = __float2half(frag_c[warp_m_idx][j][col_block][3]);
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
      reinterpret_cast<half*>(C),
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



              // * 缩放因子索引计算详解：
              // * 
              // * 1. 共享内存布局 sh_s：
              // *    sh_s[0]: [scale_0  - scale_7 ]    // sub_tile 0 的前8个缩放因子
              // *    sh_s[1]: [scale_8  - scale_15]    // sub_tile 0 的后8个缩放因子  
              // *    sh_s[2]: [scale_16 - scale_23]    // sub_tile 1 的前8个缩放因子
              // *    sh_s[3]: [scale_24 - scale_31]    // sub_tile 1 的后8个缩放因子
              // *    sh_s[4]: [scale_32 - scale_39]    // sub_tile 2 的前8个缩放因子
              // *    sh_s[5]: [scale_40 - scale_47]    // sub_tile 2 的后8个缩放因子
              // *    sh_s[6]: [scale_48 - scale_55]    // sub_tile 3 的前8个缩放因子
              // *    sh_s[7]: [scale_56 - scale_63]    // sub_tile 3 的后8个缩放因子
              // * 
              // * 2. 索引计算分解：
              // *    - (warp_n / 16 + j) = (0/16 + j) = j  // 当前只有1个warp处理N维度
              // *    - j * 2: 每个sub_tile占用2个连续的int4位置
              // *    - (lane_id / 16): 0(lane 0-15) 或 1(lane 16-31)，选择前半部分或后半部分
              // * 
              // * 3. 具体映射关系：
              // *    sub_tile 0: lane 0-15 → sh_s[0], lane 16-31 → sh_s[1]
              // *    sub_tile 1: lane 0-15 → sh_s[2], lane 16-31 → sh_s[3]  
              // *    sub_tile 2: lane 0-15 → sh_s[4], lane 16-31 → sh_s[5]
              // *    sub_tile 3: lane 0-15 → sh_s[6], lane 16-31 → sh_s[7]
              // * 
              // * 4. FragS存储策略：
              // *    - FragS = Vec<half2, 1> 只存储2个half
              // *    - 从8个half的int4中选择前2个half存储到frag_s[j]
              // *    - 这2个half在后续的scale函数中分别用于frag_b0和frag_b1的缩放
              // * 
              // * 5. 设计理念：
              // *    - 虽然32个线程都存储相同的缩放因子(存在冗余)
              // *    - 但避免了复杂的线程间通信和数据分发逻辑
              // *    - 简化了MMA计算时的缩放因子访问：直接使用frag_s[sub_tile]
              // */