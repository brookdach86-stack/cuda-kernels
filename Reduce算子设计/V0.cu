#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel(float *d_out, float *d_data, int N)
{
    int tid = threadIdx.x;
    int i = 2 * tid;

    for (int stride = 1; stride < N; stride *= 2)
    {
        if (i % (2 * stride) == 0 && i + stride < N)
        {
            d_data[i] += d_data[i + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_out[0] = d_data[0];
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

int main()
{
    int N = 2048;
    int bytes = N * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_data[i] = 1.0f;
    }

    float *d_data;
    float *d_out;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int threads = N / 2;
    kernel<<<1, threads>>>(d_out, d_data, N);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_sum = sum_cpu(h_data, N);
    printf("GPU sum: %f\n", h_out[0]);
    printf("CPU sum: %f\n", cpu_sum);
    printf("Match: %s\n", (h_out[0] == cpu_sum) ? "YES" : "NO"); //验证

    free(h_data);
    free(h_out);
    cudaFree(d_data);
    cudaFree(d_out);

    return 0;
}