#include <iostream>
#include <cuda_runtime.h>
#include "common.h"
#include <cstdlib>

// CPU 向量加法
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// GPU 向量加法核函数
__global__ void vectorAddGPU(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20; // 约100万个元素
    size_t size = N * sizeof(float);

    // 分配内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size);
    float* h_C_GPU = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // --- 计时开始 ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); // 移动到了 main 内部
    CUDA_CHECK(cudaEventCreate(&stop));

    // GPU 计时
    CUDA_CHECK(cudaEventRecord(start));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAddGPU << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU 时间: %f ms\n", milliseconds);

    // CPU 计时并对比（可以使用简单的 clock() 或更精确的计时器）
    // ... 这里可以添加 CPU 计时 ...

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);

    return 0;
}