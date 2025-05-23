#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>

// 优化版 uint8 到 bf16 转换内核
__global__ void uint8_to_bf16_kernel(const uint8_t* input, __nv_bfloat16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 4 < size) {  // 每个线程处理4个元素
        // 创建简单数组存储输入和输出
        uint8_t source[4];
        __nv_bfloat16 result[4];
        
        // 加载4个输入值
        if (idx * 4 + 4 <= size) {
            *reinterpret_cast<uint32_t*>(source) = *reinterpret_cast<const uint32_t*>(input + idx * 4);
        } else {
            for (int i = 0; i < 4 && idx * 4 + i < size; ++i) {
                source[i] = input[idx * 4 + i];
            }
        }
        
        // 获取uint8数据作为一个32位值
        uint32_t i8s = *reinterpret_cast<uint32_t*>(source);
        
        // BF16转换需要通过FP32中间表示
        // 0x4B000000对应FP32的2^23=8388608
        static constexpr uint32_t fp32_base = 0x4B000000;
        float fp32_intermediates[4];
        uint32_t* fp32_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
        
        // 构建FP32值，BF16没有足够的尾数进行IADD技巧
        fp32_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);  // 0x{4B0000}{e0}
        fp32_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);  // 0x{4B0000}{e1}
        fp32_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);  // 0x{4B0000}{e2}
        fp32_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);  // 0x{4B0000}{e3}
        
        // 减去基数+128，使无符号整数变为有符号
        for (int i = 0; i < 4; ++i) {
            fp32_intermediates[i] -= 8388736.0f;  // 8388608 + 128
        }
        
        // 截断FP32表示并打包为BF16
        uint32_t* bf16_result = reinterpret_cast<uint32_t*>(&result);
        bf16_result[0] = __byte_perm(fp32_casted[0], fp32_casted[1], 0x7632); // 分别取fp32e0, fp31e1出高16位
        bf16_result[1] = __byte_perm(fp32_casted[2], fp32_casted[3], 0x7632);
        
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
__global__ void uint8_to_bf16_simple_kernel(const uint8_t* input, __nv_bfloat16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 直接将uint8转换为bf16，并应用-128偏移
        uint8_t value = input[idx];
        output[idx] = __float2bfloat16(static_cast<float>(value) - 128.0f);
    }
}

// 主函数
int main() {
    const int size = 1024;
    
    // 主机内存分配和初始化
    uint8_t* h_input = new uint8_t[size];
    __nv_bfloat16* h_output = new __nv_bfloat16[size];
    float* h_expected = new float[size];
    
    // 填充测试数据
    for (int i = 0; i < size; ++i) {
        h_input[i] = i % 256;
        h_expected[i] = static_cast<float>(h_input[i]) - 128.0f;
    }
    
    // 设备内存分配
    uint8_t* d_input;
    __nv_bfloat16* d_output;
    cudaMalloc(&d_input, size * sizeof(uint8_t));
    cudaMalloc(&d_output, size * sizeof(__nv_bfloat16));
    
    // 数据传输到设备
    cudaMemcpy(d_input, h_input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    int threads = 256;
    int blocks = (size + 4 * threads - 1) / (4 * threads);
    uint8_to_bf16_kernel<<<blocks, threads>>>(d_input, d_output, size);
    
    // 同步等待完成
    cudaDeviceSynchronize();
    
    // 结果传回主机
    cudaMemcpy(h_output, d_output, size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // 验证结果 - BF16排列不同于FP16
    bool all_correct = true;
    for (int i = 0; i < size; ++i) {
        float converted = __bfloat162float(h_output[i]);
        if (fabs(converted - h_expected[i]) > 0.1f) {  // BF16精度较低，使用较大容差
            printf("错误: [%d] 期望值 %f, 实际值 %f\n", 
                   i, h_expected[i], converted);
            all_correct = false;
        }
    }
    
    if (all_correct) {
        printf("转换测试通过! 所有元素正确转换。\n");
    }
    
    // 运行简单版本进行比较
    uint8_to_bf16_simple_kernel<<<(size+255)/256, 256>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    
    // 资源清理
    delete[] h_input;
    delete[] h_output;
    delete[] h_expected;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
// nvcc -O3 fast_uint8_to_bf16.cu -o test_dequant -arch sm_90