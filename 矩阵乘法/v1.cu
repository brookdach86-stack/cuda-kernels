/*
核函数设计思路:

线程网络与块结构：使用二维网格和二维线程块，自然地映射到二维结果矩阵C

索引计算：每个线程通过blockIdx、blockDim和threadIdx计算出自己负责的C矩阵元素坐标(i, j)

循环计算：线程通过一个循环，累加A[i][k] * B[k][j]的结果

边界检查：确保线程索引不会超出矩阵维度
*/

#include <stdio.h>
#include <cuda_runtime.h>

// 核函数: 每个线程计算C矩阵的一个元素
// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel_v1(float *C, float *A, float *B, int M, int K, int N)
{
    // 计算当前线程负责的C矩阵全局行索引和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查: 确保线程索引不超出矩阵维度
    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU版本，用于验证GPU结果
void matmul_cpu(float *C, float *A, float *B, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 比较两个矩阵是否近似相等
bool check_result(float *C_gpu, float *C_cpu, int M, int N, float eps = 1e-4f)
{
    for (int i = 0; i < M * N; ++i)
    {
        if (fabsf(C_gpu[i] - C_cpu[i]) > eps)
        {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    // 矩阵维度: C = A[M x K] * B[K x N]
    int M = 1024;
    int K = 1024;
    int N = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 在主机端分配内存
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_gpu = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);

    // 用随机数初始化矩阵
    srand(42);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)rand() / RAND_MAX;

    // 在设备端分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // 将输入数据从主机拷贝到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 配置线程块和网格
    // dim3 blockDim(32, 32);
    // dim3 gridDim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // 启动核函数并计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_kernel_v1<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU kernel time: %.3f ms\n", gpu_time);

    // 将结果从设备拷贝回主机
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // CPU验证
    matmul_cpu(h_C_cpu, h_A, h_B, M, K, N);
    if (check_result(h_C_gpu, h_C_cpu, M, N))
    {
        printf("Result verification: PASSED\n");
    }
    else
    {
        printf("Result verification: FAILED\n");
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}