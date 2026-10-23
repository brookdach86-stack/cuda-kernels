
/*
 * v4: 双缓冲 GEMM
 *
 * 核心思想：两套共享内存轮流用
 *   - 加载下一块数据的同时，计算当前块
 *   - 加载和计算重叠，GPU 不空等
 *
 * 单缓冲： [加载] [计算] [加载] [计算] ...  ← 加载时计算单元闲着
 * 双缓冲： [加载块1]
 *          [加载块2] [计算块1]                ← 重叠！
 *          [加载块3] [计算块2]
 *                    [计算块3]                ← 收尾
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

// ===================================================================
// 第 1 段：核函数签名 + 双份共享内存声明
// ===================================================================
__global__ void matmul_double_buf(float *C, float *A, float *B, int M, int K, int N)
{
    // 关键：As[2][...] 两个 buffer，不是 As[1][...]
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // ===================================================================
    // 第 2 段：预加载第 0 块
    // ===================================================================
    As[0][threadIdx.y][threadIdx.x] = (row < M && threadIdx.x < K)
                                          ? A[row * K + threadIdx.x]
                                          : 0.0f;
    Bs[0][threadIdx.y][threadIdx.x] = (threadIdx.y < K && col < N)
                                          ? B[threadIdx.y * N + col]
                                          : 0.0f;
    __syncthreads();

    // ===================================================================
    // 第 3 段：循环加载 buf[read] 同时计算 buf[comp]
    // ===================================================================
    int read = 1; // 下一块加载到 buf[1]
    int comp = 0; // 当前计算 buf[0]

    for (int bk = TILE_SIZE; bk < K; bk += TILE_SIZE)
    {
        // 3a. 加载下一块
        int col_a = bk + threadIdx.x;
        int row_b = bk + threadIdx.y;

        As[read][threadIdx.y][threadIdx.x] = (row < M && col_a < K)
                                                 ? A[row * K + col_a]
                                                 : 0.0f;
        Bs[read][threadIdx.y][threadIdx.x] = (row_b < K && col < N)
                                                 ? B[row_b * N + col]
                                                 : 0.0f;

        // 3b. 同时计算上一块
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[comp][threadIdx.y][k] * Bs[comp][k][threadIdx.x];

        __syncthreads();

        // 3c. ping-pong 交换
        read ^= 1; // 0变1, 1变0
        comp ^= 1;
    }

    // ===================================================================
    // 第 4 段：收尾，计算最后一块
    // ===================================================================
    for (int k = 0; k < TILE_SIZE; k++)
        sum += As[comp][threadIdx.y][k] * Bs[comp][k][threadIdx.x];

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ===================================================================
// CPU 验证 + main
// ===================================================================
void cpu_matmul(float *C, float *A, float *B, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main()
{
    int M = 1024, K = 1024, N = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_C_cpu = (float *)malloc(bytes_C);

    for (int i = 0; i < M * K; i++)
        h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++)
        h_B[i] = (float)(rand() % 100) / 10.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_double_buf<<<grid, block>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU kernel time (v4 double buffer): %.3f ms\n", gpu_time);

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cpu_matmul(h_C_cpu, h_A, h_B, M, K, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float err = fabsf(h_C[i] - h_C_cpu[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("max error: %f\n", max_err);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}