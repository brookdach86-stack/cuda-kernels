#include <cuda_runtime.h>
#include <iostream>

// Grid-stride loop 版本的 SAXPY 核函数
__global__ void saxpyKernelStride(int n, float a, const float* x, float* y) {
    // 1. 计算当前线程的初始索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 计算整个网格的总线程数作为跨步大小
    int stride = blockDim.x * gridDim.x;

    // 3. 使用循环处理数据：每个线程处理 index, index + stride, index + 2*stride ...
    for (int i = index; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    // 即使 N 非常大（例如超过硬件线程限制），这种写法也能处理
    int N = 1 << 20; 
    float a = 2.5f;
    size_t size = N * sizeof(float);

    // 主机内存分配与初始化
    float *h_X = (float*)malloc(size);
    float *h_Y = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_X[i] = 1.0f;
        h_Y[i] = 10.0f;
    }

    // 设备内存分配
    float *d_X, *d_Y;
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);

    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

    // 【关键点】：我们可以自由定义网格大小
    // 这里我们不根据 N 来计算，而是根据硬件习惯设定一个固定值
    int threadsPerBlock = 256;
    int numBlocks = 128; // 即使 numBlocks * 256 < N，循环也会确保所有元素被处理

    std::cout << "启动核函数:Grid 大小 = " << numBlocks << ", Block 大小 = " << threadsPerBlock << std::endl;
    std::cout << "总线程数 = " << numBlocks * threadsPerBlock << "，处理数据量 = " << N << std::endl;

    saxpyKernelStride<<<numBlocks, threadsPerBlock>>>(N, a, d_X, d_Y);

    // 拷贝结果回主机
    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

    // 验证部分结果
    // 预期：2.5 * 1.0 + 10.0 = 12.5
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_Y[i] != 12.5f) {
            success = false;
            break;
        }
    }
    
    std::cout << "结果验证: " << (success ? "通过！" : "失败！") << std::endl;
    std::cout << "h_Y[0] = " << h_Y[0] << std::endl;

    // 清理
    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_X);
    free(h_Y);

    return 0;
}