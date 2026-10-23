/* 实现了4级粗粒度并行的reduce操作，每个block处理COARSEN个元素，每个线程处理1个元素，每个线程内进行COARSEN次循环，每次循环处理1个元素，每个循环内进行COARSEN次原子操作 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define COARSEN 4 // 4级粗粒度并行，循环粗化

__global__ void reduce_kernel(float *d_out, float *d_data, int N)
{
    int tid = threadIdx.x;
    int block_idx = blockIdx.x * blockDim.x * COARSEN;

    __shared__ float smem[BLOCK_SIZE];
    float local_sum = 0.0f;
    for (int i = 0; i < COARSEN; ++i)
    {
        // 三级定位
        int idx = block_idx + tid * COARSEN + i;
        if (idx < N)
        {
            local_sum += d_data[idx];
        }
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(&d_out[0], smem[0]);
    }
}

float sum_cpu(float *h_data, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sum += h_data[i];
    }
    return sum;
}

// 造数据，随机数在[low, high]之间防溢出
void init_float_rand(float *arr, int n, int low, int high)
{
    for (int i = 0; i < n; i++)
    {
        float r = (float)rand() / RAND_MAX;
        arr[i] = low + r * (high - low);
    }
}

int main()
{
    int N = 1 << 24;
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(sizeof(float));
    init_float_rand(h_data, N, -1.0, 1.0);

    float sum = sum_cpu(h_data, N);
    float *d_data, *d_out;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    dim3 grid = (N + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);
    cudaEvent_t st, sp;
    cudaEventCreate(&st);
    cudaEventCreate(&sp);
    cudaEventRecord(st);
    reduce_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_data, N);
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float ms;
    cudaEventElapsedTime(&ms, st, sp);
    printf("time: %.3f ms\n", ms);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %f, h_out = %f\n", sum, h_out[0]);

    free(h_data);
    free(h_out);
    cudaFree(d_data);
    cudaFree(d_out);
    return 0;
}