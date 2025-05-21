#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>

// 直接的uint8到fp16转换kernel
__global__ void uint8_to_fp16_kernel(const uint8_t* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 4 < size) {  // 每个线程处理4个元素
        // 创建简单数组存储输入和输出
        uint8_t source[4];
        __half result[4];
        
        // 加载4个输入值
        if (idx * 4 + 4 <= size) {
            *reinterpret_cast<uint32_t*>(source) = *reinterpret_cast<const uint32_t*>(input + idx * 4);
        } 
        else {
            for (int i = 0; i < 4 && idx * 4 + i < size; ++i) {
                source[i] = input[idx * 4 + i];
            }
        }
        
        // 转换逻辑 - 直接使用CUTLASS的转换PTX指令
        uint32_t* h = reinterpret_cast<uint32_t*>(result);
        uint32_t i8s = *reinterpret_cast<uint32_t*>(source);
        
        // PTX指令执行字节重排和转换
        static constexpr uint32_t mask_for_elt_01 = 0x5150;
        static constexpr uint32_t mask_for_elt_23 = 0x5352; 
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464; //b0110 0100 0110 0100 0110 0100 0110 0100
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480; 
        
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" 
                     : "=r"(h[0]) 
                     : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
        
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" 
                     : "=r"(h[1]) 
                     : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
        
        // 减去魔法数(1152)转换回有符号值
        asm volatile("sub.f16x2 %0, %1, %2;\n" 
                     : "=r"(h[0]) 
                     : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
        
        asm volatile("sub.f16x2 %0, %1, %2;\n" 
                     : "=r"(h[1]) 
                     : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
        
        // 存储结果
        if (idx * 4 + 4 <= size) {
            *reinterpret_cast<uint2*>(output + idx * 4) = *reinterpret_cast<uint2*>(result);
        } else {
            for (int i = 0; i < 4 && idx * 4 + i < size; ++i) {
                output[idx * 4 + i] = result[i];
            }
        }
    }
}

// 简化版的转换逻辑
__global__ void uint8_to_fp16_simple_kernel(const uint8_t* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 直接将uint8转换为fp16，并应用-128偏移
        uint8_t value = input[idx];
        output[idx] = __float2half(static_cast<float>(value) - 128.0f);
    }
}

// 主函数 - 直接调用kernel
int main() {
    const int size = 1024;
    
    // 主机内存分配和初始化
    uint8_t* h_input = new uint8_t[size];
    __half* h_output = new __half[size];
    float* h_expected = new float[size];
    
    // 填充测试数据
    for (int i = 0; i < size; ++i) {
        h_input[i] = i % 256;
        h_expected[i] = static_cast<float>(h_input[i]) - 128.0f;
    }
    
    // 设备内存分配
    uint8_t* d_input;
    __half* d_output;
    cudaMalloc(&d_input, size * sizeof(uint8_t));
    cudaMalloc(&d_output, size * sizeof(__half));
    
    // 数据传输到设备
    cudaMemcpy(d_input, h_input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    int threads = 256;
    int blocks = (size + 4 * threads - 1) / (4 * threads);
    uint8_to_fp16_kernel<<<blocks, threads>>>(d_input, d_output, size);
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    // 结果传回主机
    cudaMemcpy(h_output, d_output, size * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < size; ++i) {
        float converted = __half2float(h_output[i]);
        if (fabs(converted - h_expected[i]) > 0.01f) {
            printf("错误: [%d] 期望值 %f, 实际值 %f\n", 
                   i, h_expected[i], converted);
        }
    }
    
    // 资源清理
    delete[] h_input;
    delete[] h_output;
    delete[] h_expected;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}

// nvcc -O3 fast_uint8_to_fp16.cu -o test_dequant -arch sm_90