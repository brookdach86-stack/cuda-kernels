#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

__global__ void reduce_kernel(float *d_out, float *d_data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];
    if (tid < N) {
        smem[i] = d_data[tid];
    } else {
        smem[i] = 0.0f;
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        if (i < stride && tid < N) {
            smem[i] += smem[i + stride];
        }
        __syncthreads();
    }

    if (i == 0) {
        atomicAdd(&d_out[0], smem[0]); // 原子加法，确保线程安全, 只有线程0会写入d_out[0]
    }
}

float sum_cpu(float *h_data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += h_data[i];
    }
    return sum;
}

int main(){
    int n = 2048;
    size_t size = n * sizeof(float);
    float *h_data = (float *)malloc(size);
    float *h_out = (float *)malloc(sizeof(float));
    for(int i = 0; i < n; i++){
        h_data[i] = (float)rand() / RAND_MAX;
    }

    float sum = sum_cpu(h_data, n);
    printf("CPU sum: %f\n", sum);

    float *d_data, *d_out;

    cudaMalloc(&d_data, size);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    reduce_kernel<<<1, BLOCK_SIZE>>>(d_out, d_data, n);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU sum: %f\n", h_out[0]);
    printf("CPU sum: %f\n", sum);

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    free(h_out);
    return 0;
}