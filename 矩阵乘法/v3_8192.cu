/*
v3: 寄存器分块（干净版）
每个线程算 TM×TN = 8×8 = 64 个 C 元素
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matmul_v3(float *C, float *A, float *B, int M, int K, int N)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row = threadIdx.y * TM;
    int thread_col = threadIdx.x * TN;

    float Csub[TM][TN] = {{0.0f}};

    // ========== 分块 ==========
    for (int bk = 0; bk < K; bk += BK)
    {
        // ========== 加载 As[128][8] ==========
        for (int r = threadIdx.y; r < BM; r += blockDim.y)
        {
            int c = threadIdx.x;
            if (c < BK)
            {
                int gr = block_row + r, gc = bk + c;
                As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
        }

        // ========== 加载 Bs[8][128] ==========
        int br = threadIdx.y;
        if (br < BK)
        {
            for (int c = threadIdx.x; c < BN; c += blockDim.x)
            {
                int gr = bk + br, gc = block_col + c;
                Bs[br][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        __syncthreads();

        // ========== 计算 ==========
        for (int k = 0; k < BK; k++)
        {
            float Areg[TM];
            float Breg[TN];
            for (int ti = 0; ti < TM; ti++)
                Areg[ti] = As[thread_row + ti][k];
            for (int tj = 0; tj < TN; tj++)
                Breg[tj] = Bs[k][thread_col + tj];
            for (int ti = 0; ti < TM; ti++)
                for (int tj = 0; tj < TN; tj++)
                    Csub[ti][tj] += Areg[ti] * Breg[tj];
        }

        __syncthreads();
    }

    // ========== 写回 ==========
    int gr = block_row + thread_row;
    int gc = block_col + thread_col;
    for (int ti = 0; ti < TM; ti++)
        for (int tj = 0; tj < TN; tj++)
        {
            int r = gr + ti, c = gc + tj;
            if (r < M && c < N)
                C[r * N + c] = Csub[ti][tj];
        }
}

// ========== CPU 验证 ==========
float *cpu_mmul(float *A, float *B, int M, int K, int N)
{
    float *C = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    return C;
}

int check(float *g, float *c, int n, float eps = 1e-3f)
{
    for (int i = 0; i < n; i++)
        if (fabsf(g[i] - c[i]) > eps)
        {
            printf("FAIL at %d: %f vs %f\n", i, g[i], c[i]);
            return 0;
        }
    return 1;
}

int main()
{
    int M = 8192, K = 8192, N = 8192;
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
        A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        B[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    matmul_v3<<<grid, block>>>(dC, dA, dB, M, K, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float t;
    cudaEventElapsedTime(&t, s, e);
    printf("GPU: %.3f ms\n", t);

    return 0;
    float *gpuC = (float *)malloc(M * N * sizeof(float));
    cudaMemcpy(gpuC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float *cpuC = cpu_mmul(A, B, M, K, N);
    printf(check(gpuC, cpuC, M * N) ? "PASS\n" : "FAIL\n");

    free(A);
    free(B);
    free(gpuC);
    free(cpuC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}