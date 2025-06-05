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
using FragA = Vec<half2, 4>;  // 16x16 A矩阵片段，对应一次 mma 4*2个元素，行优先
using FragB = Vec<half2, 2>;  // 16x8 B矩阵片段，对应一次 mma 2*2个元素，列优先
using FragC = Vec<float, 4>;  // 16x8 累加器片段，对应一次 mma 2*2个元素，行优先
using FragS = Vec<half2, 1>;  // 量化缩放因子, 对应每个sub_tile的两次mma, 一次解量化一列

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
    int4* sh_b = sh_a + (16 * THREAD_M_BLOCKS * 16 / 8);
    int4* sh_s = sh_b + (16 * 16 * THREAD_N_BLOCKS / 32);
    
    // 寄存器片段存储
    // 矩阵A，thread_m_blocks=4个warp, 处理64*16的fp16数据片段
    // 矩阵B，一个warp处理16*64的4bits数据片段，4bits数据由python pack代码已离线进行重排，
    //       并将其内存重定义为1*32的int4, warp内每个线程负责一个int4，一个int4包含一个thread
    //       总计8个mma指令所需所有的32个4bits数据。8个mma = 4个sub_tile * 2次MMA。
    //       重排的目的，是为了最大化访存。
    // 矩阵C，thread_m_blocks=4个warp, 处理64*64的fp16数据片段，thread_n_blocks=4, 2对应两次MMA,
    // 矩阵S，一个warp处理16*64的scale, 对应4个sub_tile的缩放因子，每个sub_tile有2次MMA
    FragA frag_a[THREAD_M_BLOCKS]; // 4个Vec<half2, 4>
    I4 frag_b_quant; // 1个Vec<int, 4>;
    FragC frag_c[THREAD_M_BLOCKS][4][2];  // 4*4*2=16个 Vec<float, 4>
    FragS frag_s[4]; // 4个Vec<half2, 1>

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < THREAD_M_BLOCKS; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < 2; k++) {  // 两个MMA结果
                frag_c[i][j][k] = {0.0f};
            }
        }
    }
    // K维度循环
    for (int k_offset = 0; k_offset < prob_k; k_offset += 16) {
    
        // A矩阵128个线程处理64*16的fp16矩阵分块, 每个thread处理8个fp16=128bits, 对齐cache line
        if (threadIdx.x < THREAD_M_BLOCKS * 32) {
            int row = threadIdx.x / 2;     // 0-63行
            int col = threadIdx.x % 2;     // 0-1列

            if ((block_m + row) < prob_m && (k_offset + col * 8) < prob_k) {
                int a_idx = (block_m + row) * (prob_k / 8) + (k_offset / 8) + col;
                sh_a[threadIdx.x] = A[a_idx];
            }
        }

        // B矩阵32个线程处理16*64的4bits矩阵分块
        // B.shape = (k, n), dtype= 4 bits
        // reordered_and_packed_B.shape = (K//16, N*16//8), dtype = int32
        // reinterpret_cast<int4*> shape = (K//16, N*16//8//4), dtype = int4
        if (threadIdx.x < 32) {
            // 基于数据块（chunk-based）的B矩阵索引计算
            int k_strip_idx = k_offset / 16;                    
            int n_tile_idx = blockIdx.x;                         // 当前块处理这个K条带内的第几个N片段 (每个片段64列)
            int num_n_tiles_per_k_strip = prob_n / 64;           // 每个K条带总共有多少个N片段

            // 计算当前16x64分块（即32个int4）在全局B数组中的起始偏移
            int chunk_base_offset_in_B = (k_strip_idx * num_n_tiles_per_k_strip + n_tile_idx) * 32;
            // 每个线程 (threadIdx.x 从 0 到 31) 加载这个数据块中的一个int4
            int b_idx = chunk_base_offset_in_B + threadIdx.x;
            sh_b[threadIdx.x] = B[b_idx];
        }

        // scale矩阵，32个线程处理1*64的fp16矩阵分块
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
            const int warps_n = THREAD_N_BLOCKS / 4;  // N维度的warp数量，1
            const int warp_m_idx = warp_id / warps_n;  // 当前warp在M维度的索引，（0，1，2，3）
            const int warp_n_idx = warp_id % warps_n;  // 当前warp在N维度的索引，（0）
            
            const int warp_m = warp_m_idx * 16;        // warp在M维度的起始位置
            const int warp_n = warp_n_idx * 64;        // warp在N维度的起始位置

            // 加载A片段, 使用ldmatrix指令自动加载到寄存器
            int load_row = warp_m + lane_id % 16;      // 行地址：warp_m + (0-15)
            int load_col = lane_id / 16;               // 列地址：0 或 1
            int a_sh_offset = load_row * 2 + load_col; // int4偏移

            if (a_sh_offset < THREAD_M_BLOCKS * 32) {
                ldsm4(frag_a[warp_m_idx], &sh_a[a_sh_offset]);
            }
            
            // scale同样经过离线reorder,
            // 将4个sub_tile的总计8个mma指令，所需的8个fp16 scale，组合在一起
            // 根据mma指令的B矩阵寄存器排布特性，每4个thread共享一个scale
            if (groupsize != -1) {
                // 根据mma指令的排布，列优先，每4个线程读取同一个int4
                int s_sh_offset = lane_id / 4;  
                
                if (s_sh_offset < 8) {
                    // 直接使用half2指针读取
                    half2* scales_ptr = reinterpret_cast<half2*>(&sh_s[s_sh_offset]);
                    
                    // 为每个sub_tile分配对应2个mma的缩放因子
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        frag_s[j][0] = scales_ptr[j];  // 直接赋值half2
                    }
                }
            }
            
            // 简化的B矩阵处理
            // 四个warp全部加载？似乎很浪费
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
                
                mma(frag_a[warp_m_idx], frag_b0, frag_c[warp_m_idx][sub_tile][0]);
                mma(frag_a[warp_m_idx], frag_b1, frag_c[warp_m_idx][sub_tile][1]);
            }
        }

        __syncthreads();
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
    
    
    SimpleMarlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS>
        <<<blocks, THREADS, smem_size, stream>>>(
            reinterpret_cast<const int4*>(A),
            reinterpret_cast<const int4*>(B),
            reinterpret_cast<half*>(C),
            reinterpret_cast<const int4*>(s),
            prob_m, prob_n, prob_k, groupsize
        );
        
    return 0;
}