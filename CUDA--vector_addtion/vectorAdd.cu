#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // 1.分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // 2.分配设备内存
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3.将输入数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4.配置并启动 Kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    vectorAdd << <gridSize, blockSize >> > (d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f)
        {
            printf("Verification FAILED at index %d!\n", i);
            break;
        }
    }
    printf("Verification PASSED!\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

