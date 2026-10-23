/*
 * 两阶段归约流程:
 *
 *   CPU 造数据 (123344 个 float)
 *          │
 *          ▼
 *   reduce_kernel (61 个 block)
 *          │
 *          ▼
 *   d_partial[0..60] (61 个局部和)
 *          │
 *          ▼
 *   reduce_final (1 个 block)
 *          │
 *          ▼
 *   d_out[0] (最终总和)
 *          │
 *          ▼
 *   CPU 验证
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define VEC_SIZE 4 // float4 向量化, 每个线程一次处理 4 个 float

// 每个 block 处理 BLOCK_SIZE * VEC_SIZE = 2048 个元素

__global__ void reduce_kernel(float *d_partial, float *d_data, int N)
{
    int tid = threadIdx.x;
    __shared__ float smem[BLOCK_SIZE];

    float local_sum = 0.0f;

    float4 *d_data4 = (float4 *)d_data;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx * VEC_SIZE < N)
    {
        float4 v = d_data4[idx];
        local_sum += v.x + v.y + v.z + v.w;
    }
    smem[tid] = local_sum;
    __syncthreads();

    if (tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32)
        smem[tid] += smem[tid + 32];

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

    float local_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE)
    {
        local_sum += d_partial[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    if (tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32)
        smem[tid] += smem[tid + 32];

    if (tid < 32)
    {
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

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main()
{
    int N = 123341;
    int N_aligned = (N + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
    size_t bytes = N_aligned * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(sizeof(float));

    init_float_rand(h_data, N, -1.0f, 1.0f);
    for (int i = N; i < N_aligned; i++)
        h_data[i] = 0.0f;

    float sum = sum_cpu(h_data, N_aligned);

    float *d_data, *d_out, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int grid = (N_aligned + BLOCK_SIZE * VEC_SIZE - 1) / (BLOCK_SIZE * VEC_SIZE);
    CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

    reduce_kernel<<<grid, BLOCK_SIZE>>>(d_partial, d_data, N_aligned);
    reduce_final<<<1, BLOCK_SIZE>>>(d_out, d_partial, grid);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("sum = %f, h_out = %f\n", sum, h_out[0]);

    free(h_data);
    free(h_out);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}