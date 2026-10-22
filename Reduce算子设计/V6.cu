/*
核心改动：两步归约，彻底消除 atomicAdd 竞争。
V5：很多 block 抢一个桶；V6：每个 block 发一个桶，最后再汇总。
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define COARSEN 4

__global__ void reduce_kernel(float *d_partial, float *d_data, int N)
{
    int tid = threadIdx.x;
    int block_idx = blockIdx.x * blockDim.x * COARSEN;

    __shared__ float smem[BLOCK_SIZE];
    float local_sum = 0.0f;
    for (int i = 0; i < COARSEN; ++i)
    {
        int idx = block_idx + tid * COARSEN + i;
        if (idx < N)
        {
            local_sum += d_data[idx];
        }
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride /= 2)
    {
        if (tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float val = smem[tid];
    for (int stride = 16; stride >= 1; stride /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }

    if (tid == 0)
    {
        d_partial[blockIdx.x] = val;
    }
}

__global__ void reduce_final(float *d_out, float *d_partial, int num_blocks)
{
    int tid = threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    //d_partial.size() = num_blocks
    smem[tid] = (tid < num_blocks) ? d_partial[tid] : 0.0f;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride /= 2)
    {
        if (tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float val = smem[tid];
    for (int stride = 16; stride >= 1; stride /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }

    if (tid == 0)
    {
        d_out[0] = val;
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
    int N = 123341;
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(sizeof(float));
    init_float_rand(h_data, N, -1.0, 1.0);

    float sum = sum_cpu(h_data, N);

    float *d_data, *d_out, *d_partial;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);
    cudaMalloc(&d_partial, grid * sizeof(float));

    reduce_kernel<<<grid, BLOCK_SIZE>>>(d_partial, d_data, N);
    reduce_final<<<1, BLOCK_SIZE>>>(d_out, d_partial, grid);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %f, h_out = %f\n", sum, h_out[0]);

    free(h_data);
    free(h_out);
    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_partial);
    return 0;
}