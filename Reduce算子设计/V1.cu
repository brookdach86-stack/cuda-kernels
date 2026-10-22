#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define BLOCK_SIZE 2048

__global__ void reduce_kernel(float *d_out, float *d_data, int N){
    int tid = threadIdx.x;
    for(int stride = blockDim.x; stride >= 1; stride /= 2){
        if(tid < N){
            d_data[tid] += d_data[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_out[0] = d_data[0];
    }
}

float sum_cpu(float *h_data, int N){
    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += h_data[i];
    }
    return sum;
}

int main(){
    int N = 2048;
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(sizeof(float));
    for(int i = 0; i < N; i++){
        h_data[i] = (float)rand() / RAND_MAX;
    }

    float sum = sum_cpu(h_data, N);

    float *d_data, *d_out;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    reduce_kernel<<<1, BLOCK_SIZE>>>(d_out, d_data, N);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU sum: %f\n", h_out[0]);
    printf("CPU sum: %f\n", sum);

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    free(h_out);

    return 0;
}
