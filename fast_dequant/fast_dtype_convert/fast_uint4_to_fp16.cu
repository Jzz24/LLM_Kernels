#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>


// uint4的解量化确实需要预先对uint4权重进行交织处理，输入的每32个bit中，
// 高16bit分别是e7 e5 e3 e1，低16bit分别是e6 e4 e2 e0，
// 因为只有这样才用方便的结合lop3指令，然后最终输出的8个fp16 result数组里面，
// 才能从高位到低位，依次排布 e7 e6 e5 ... e1 e0
// 通过离线的权重交织处理，然后在kernel内部去还原顺序，方便后续的tensorcore矩阵乘法去使用；
// 如果不预先进行交织的话，在kernel内部解量化后，还要手动调整顺序，因此影响了计算的性能

// 定义4位无符号整数数组
struct uint4_t {
    uint32_t vals;  // 每个字节包含两个4位值
};

// 用于处理8个uint4值的转换
__global__ void uint4_to_fp16_kernel(const uint4_t* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 8 < size) {  // 每个线程处理8个元素
        // 加载8个4位整数(打包在一个32位整数中)
        uint32_t i4s;
        if (idx * 8 + 8 <= size) {
            i4s = input[idx].vals;
        } else {
            // 处理边界情况
            i4s = 0;
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(&i4s);
            for (int i = 0; i < 4 && idx * 8 + i * 2 < size; ++i) {
                uint8_t val1 = (idx * 8 + i * 2 < size) ? (input[idx].vals >> (i * 8)) & 0xF : 0;
                uint8_t val2 = (idx * 8 + i * 2 + 1 < size) ? (input[idx].vals >> (i * 8 + 4)) & 0xF : 0;
                byte_ptr[i] = (val2 << 4) | val1;
            }
        }
        
        // 输出变量
        __half result[8];
        uint32_t* h = reinterpret_cast<uint32_t*>(result);
        
        // CUTLASS 转换逻辑
        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOTTOM_MASK = 0x000f000f; // select 0/4 uint4
        static constexpr uint32_t TOP_MASK = 0x00f000f0; // select 1/5 uint4
        // static constexpr uint32_t BOTTOM_MASK = 0x0000 00ff; // select 0/1 uint4
        // static constexpr uint32_t TOP_MASK = 0x00f000f0; // select 2/3 uint4
        static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
        
        // 分割输入数据
        const uint32_t top_i4s = i4s >> 8;
        
        // 提取并转换各个4位值
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        
        // 应用转换公式
        static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
        static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
        static constexpr uint32_t NEG_72 = 0xd480d480;

        // 个人理解: -72 = -64 - 8, massita expr 1024/16 + ((x+8)*16)/16 - 64 - 8 = x
        // Y_FP16 = 1024 + (x+8)*16, x = Y_FP16/16 - 64 - 8
        // (1024 + (x+8)*16)/16 = 64 + x + 8
        
        // 完成转换
        asm volatile("sub.f16x2 %0, %1, %2;\n" 
                    : "=r"(h[0]) 
                    : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" 
                    : "=r"(h[1]) 
                    : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        
        asm volatile("sub.f16x2 %0, %1, %2;\n" 
                    : "=r"(h[2]) 
                    : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" 
                    : "=r"(h[3]) 
                    : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        
        // 存储结果
        if (idx * 8 + 8 <= size) {
            // 批量存储所有8个值
            *reinterpret_cast<uint4*>(output + idx * 8) = *reinterpret_cast<uint4*>(result);
        } else {
            // 逐个存储边界情况
            for (int i = 0; i < 8 && idx * 8 + i < size; ++i) {
                output[idx * 8 + i] = result[i];
            }
        }
    }
}

// 简化版的转换逻辑 - 用于验证
__global__ void uint4_to_fp16_simple_kernel(const uint4_t* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 计算包含此元素的uint4_t索引和位置
        int input_idx = idx / 8;
        int bit_pos = (idx % 8) * 4;
        
        // 提取4位值
        uint32_t packed = input[input_idx].vals;
        uint32_t shift = bit_pos;
        uint8_t value = (packed >> shift) & 0xF;
        
        float float_val = static_cast<float>(value) - 8.0f;
        output[idx] = __float2half(float_val);
    }
}

// 将整数打包到uint4_t中
void pack_uint4(uint8_t* values, uint4_t* packed, int size) {
    for (int i = 0; i < size; i += 8) {
        uint32_t packed_val = 0;
        for (int j = 0; j < 8 && i + j < size; ++j) {
            uint32_t val = values[i + j] & 0xF;
            packed_val |= (val << (j * 4));
        }
        packed[i/8].vals = packed_val;
    }
}

// 主函数
int main() {
    const int size = 1024 * 1024;
    
    // 主机内存分配
    uint8_t* h_input_values = new uint8_t[size];  // 原始4位值
    uint4_t* h_input = new uint4_t[(size + 7) / 8]; // 打包的4位值
    __half* h_output = new __half[size];
    float* h_expected = new float[size];
    
    // 初始化测试数据 (0-15范围的值)
    for (int i = 0; i < size; ++i) {
        h_input_values[i] = i % 16;  // 4位能表示0-15
        h_expected[i] = static_cast<float>(h_input_values[i]) - 8.0f;  // 转换公式
    }
    
    // 打包数据到uint4_t格式
    pack_uint4(h_input_values, h_input, size);
    
    // 设备内存分配
    uint4_t* d_input;
    __half* d_output;
    cudaMalloc(&d_input, ((size + 7) / 8) * sizeof(uint4_t));
    cudaMalloc(&d_output, size * sizeof(__half));
    
    // 数据传输到设备
    cudaMemcpy(d_input, h_input, ((size + 7) / 8) * sizeof(uint4_t), cudaMemcpyHostToDevice);
    
    // 创建计时器
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 启动优化kernel并计时
    cudaEventRecord(start);
    
    int threads = 256;
    int blocks = (size + 8 * threads - 1) / (8 * threads);
    uint4_to_fp16_kernel<<<blocks, threads>>>(d_input, d_output, size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("优化版kernel执行时间: %.3f ms\n", milliseconds);
    
    // 获取结果
    cudaMemcpy(h_output, d_output, size * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // 验证结果 
    bool all_correct = true;
    for (int i = 0; i < size; ++i) {
        float converted = __half2float(h_output[i]);
        if (fabs(converted - h_expected[i]) > 0.01f) {
            printf("错误: [%d] 期望值 %f, 实际值 %f, 输入值 %d\n", 
                   i, h_expected[i], converted, h_input_values[i]);
            all_correct = false;
            if (i > 20) break;  // 只显示前20个错误
        }
    }
    
    if (all_correct) {
        printf("优化版转换测试通过! 所有%d个元素正确转换。\n", size);
    }
    
    // 使用简化版kernel进行对比测试
    cudaEventRecord(start);
    
    uint4_to_fp16_simple_kernel<<<(size+255)/256, 256>>>(d_input, d_output, size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("简化版kernel执行时间: %.3f ms\n", milliseconds);
    
    // 性能提升比较
    cudaMemcpy(h_output, d_output, size * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // 验证简化版结果
    all_correct = true;
    for (int i = 0; i < size; ++i) {
        float converted = __half2float(h_output[i]);
        if (fabs(converted - h_expected[i]) > 0.01f) {
            printf("简化版错误: [%d] 期望值 %f, 实际值 %f\n", 
                   i, h_expected[i], converted);
            all_correct = false;
        }
    }
    
    if (all_correct) {
        printf("简化版转换测试通过!\n");
    }
    
    // 资源清理
    delete[] h_input_values;
    delete[] h_input;
    delete[] h_output;
    delete[] h_expected;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}