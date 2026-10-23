/*
核函数设计思路（v2 - 共享内存优化版）:

v1 的问题：
  每个线程计算 C 的一个元素时，需要遍历 K 次，每次从全局内存读取 A 和 B。
  全局内存带宽 ~900 GB/s，共享内存带宽 ~10 TB/s，差距巨大。
  而且相邻线程读取的 A、B 数据有大量重叠，但 v1 没有复用，白白浪费了带宽。

v2 的优化：
  将矩阵分块（tile），每次把一个 tile 的 A 和 B 子矩阵载入共享内存，
  线程块内的所有线程共享这些数据，在共享内存中完成部分乘累加，
  然后加载下一个 tile 继续累加。

  计算量 / 全局访存量：
    v1: O(1)          — 每次计算都需要一次全局内存读取
    v2: O(TILE_SIZE)  — 一个 tile 的数据被 TILE_SIZE 个线程复用

  以 TILE_SIZE=32 为例，全局内存访问减少为原来的 1/32。

共享内存布局：
  As[TILE_SIZE][TILE_SIZE] — 缓存 A 的一个 tile
  Bs[TILE_SIZE][TILE_SIZE] — 缓存 B 的一个 tile

执行流程：
  for (tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE):
    1. 所有线程协作，将 A 和 B 的当前 tile 从全局内存加载到共享内存
    2. __syncthreads() 确保加载完成
    3. 在共享内存中做部分乘累加
    4. __syncthreads() 确保计算完成后再加载下一个 tile
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// TILE_SIZE = 共享内存里能装下的矩阵分块大小 = 线程块一边的线程数
#define TILE_SIZE 32

__global__ void matmul_kernel_v2(float *C, float *A, float *B, int M, int K, int N)
{
    // 共享内存：缓存 A 和 B 的 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 当前线程负责的 C 矩阵全局位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 按 tile 步进 K 维度
    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE)
    {
        // 步骤1: 所有线程协作加载 A 和 B 的 tile 到共享内存
        // 每个线程加载一个元素，TILE_SIZE x TILE_SIZE 个线程刚好覆盖整个 tile
        int tA_row = row;                    // A 的全局行号
        int tA_col = tile_idx + threadIdx.x; // A 的全局列号
        int tB_row = tile_idx + threadIdx.y; // B 的全局行号
        int tB_col = col;                    // B 的全局列号

        // 加载 A 的 tile（带边界检查）
        if (tA_row < M && tA_col < K)
            As[threadIdx.y][threadIdx.x] = A[tA_row * K + tA_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // 加载 B 的 tile（带边界检查）
        if (tB_row < K && tB_col < N)
            Bs[threadIdx.y][threadIdx.x] = B[tB_row * N + tB_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // 步骤2: 同步，确保 tile 全部加载完成
        __syncthreads();

        // 步骤3: 在共享内存中做部分乘累加，一行乘一列的内积
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 步骤4: 同步，确保所有线程计算完毕后再加载下一个 tile
        __syncthreads();
    }

    // 写回全局内存
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

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
    int M = 8192;
    int K = 8192;
    int N = 8192;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_gpu = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);

    srand(42);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 线程块配置：TILE_SIZE x TILE_SIZE，与共享内存 tile 大小一致
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_kernel_v2<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU kernel time (v2 shared memory): %.3f ms\n", gpu_time);

    return 0;
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    matmul_cpu(h_C_cpu, h_A, h_B, M, K, N);
    if (check_result(h_C_gpu, h_C_cpu, M, N))
        printf("Result verification: PASSED\n");
    else
        printf("Result verification: FAILED\n");

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