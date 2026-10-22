#include <cuda_runtime.h>
#include <iostream>

// SAXPY 核函数：Y[i] = a * X[i] + Y[i]
__global__ void saxpyKernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i]; // 每个线程独立操作，互不干扰，不需要 atomicAdd
    }
}

int main() {
    int N = 1 << 20;
    float a = 2.0f; // 标量系数
    size_t size = N * sizeof(float);

    // 1. 分配并初始化主机内存
    float* h_X = (float*)malloc(size);
    float* h_Y = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_X[i] = 1.0f;
        h_Y[i] = 2.0f;
    }

    // 2. 分配设备内存
    float* d_X, * d_Y;
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);

    // 3. 拷贝数据到设备
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

    // 4. 定义启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 运行核函数
    saxpyKernel << <blocksPerGrid, threadsPerBlock >> > (N, a, d_X, d_Y);

    // 5. 将结果（整个数组 Y）拷贝回主机
    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

    // 验证第一个和最后一个元素
    // 预期：2.0 * 1.0 + 2.0 = 4.0
    std::cout << "Y[0]: " << h_Y[0] << " (预期 4.0)" << std::endl;
    std::cout << "Y[N-1]: " << h_Y[N - 1] << " (预期 4.0)" << std::endl;

    // 6. 释放内存
    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_X);
    free(h_Y);

    return 0;
}