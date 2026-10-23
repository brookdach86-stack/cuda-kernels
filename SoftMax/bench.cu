#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 8192;
const int C = 4096;

#include "v0.cu"
#include "v1.cu"
#include "v2.cu"

int main() {
    size_t bytes = (size_t)N * C * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    for (int i = 0; i < N * C; i++) h_data[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    float *d_data, *d_out;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    softmax_gpu<<<(N + 255)/256, 256>>>(d_out, d_data, N, C);
    cudaDeviceSynchronize();

    online_softmax<<<(N + 255)/256, 256>>>(d_out, d_data, N, C);
    cudaDeviceSynchronize();

    warp_online_softmax<<<(N*32 + 255)/256, 256>>>(d_out, d_data, N, C);
    cudaDeviceSynchronize();

    cudaFree(d_data); cudaFree(d_out);
    free(h_data); free(h_out);
    return 0;
}