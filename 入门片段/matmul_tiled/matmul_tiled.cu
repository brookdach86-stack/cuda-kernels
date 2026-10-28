#include <iostream>
#include <cuda_runtime.h>
#include <cuda/cmath>

const int TILE = 16;

/*
Tilling 矩阵乘法核函数(C =A*B)
K轴:消去轴，是A的列数也是B的行数，A和B相乘的必要条件
*/
__global__ void matmulTiled(float *A, float *B, float *C, int M, int N, int K)
{
    // 1.建立共享内存
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // 2.锁定当前线程负责的最终结果矩阵C的绝对坐标
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f; // 线程个人的寄存器口袋，用来累加结果

    // 3.tiling 核心:沿着K轴往后推
    int blocks = cuda::ceil_div(K, TILE);

    // 4.协同搬运A的方砖进共享内存，触发合并全局访问
    for (int t = 0; t < blocks; t++)
    {
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

#pragma unroll // 强制摊平循环，将读取和计算直接逼进寄存器
        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < M)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    // 矩阵维度：M=32, N=32, K=32 (刚好分配为 2x2 个 Tile)
    int M = 32, N = 32, K = 32;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 主机端分配内存并初始化
    float *h_A = new float[M * K], *h_B = new float[K * N], *h_C = new float[M * N];
    for (int i = 0; i < M * K; ++i)
        h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = 2.0f;

    // 设备端分配显存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 数据拷贝至 GPU
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 线程块配置 (二维配置)
    dim3 dimBlock(TILE, TILE); // 一个 Block 里有 16x16=256 个工人
    dim3 dimGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // 启动完全体 Tiling 核函数
    matmulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // 同步并拷贝回结果
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 验证结果：每个点预期都是 1 * 2 * 32 = 64
    std::cout << "Tiling 矩阵乘法运行成功! C[0]: " << h_C[0] << " (预期值: 64)" << std::endl;

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}