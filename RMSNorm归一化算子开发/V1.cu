/*
1. Warp间数据规约
2. 计算RMS(x)
3. 乘上缩放因子，计算最终结果
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void kernel(float *d_out, float *d_data, float *d_weights, int N, int C){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id =  index / blockDim.x;
    int lane_id = index % blockDim.x;

    if (warp_id >= N) return;

    // 基于 Warp 的数据分片寻址
    float *inp = d_data + warp_id * C ;
    float *out = d_out + warp_id * C ;

    float local_sum = 0.0f;
    for(int stride = 0; stride < C; stride += BLOCK_SIZE){
        local_sum += inp[stride] * inp[stride];
    }

    unsigned mask = 0xffffffff;
    for(int offset = BLOCK_SIZE; offset >= 1; offset /= 2){
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    local_sum = __shfl_down(mask, local_sum, 0);
    local_sum = 1 / sqrt(local_sum / C + 1e-9);
    for(int i = 0; i < C; i++){
        out[i] = inp[i] * local_sum * d_weights[i];
    }
}

void init(float *data, int N, int low, int high){
    for (int i = 0; i < N; ++i) {
        float r = (float)rand() / RAND_MAX;
        data[i] = low + (high - low) * r;
    }
}

void rmsnorm_cpu(float *h_out, float *h_data, float *weights, int N, int C){
    for(int i = 0; i < N; ++i){
        float *inp  = h_data + i * C;
        float *out = h_out + i * C;

        float local_sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            local_sum += inp[j] * inp[j];
        }
        local_sum = 1 / sqrt(local_sum / C + 1e-9);
        for (int j = 0; j < C; ++j) {
            out[j] = inp[j] * local_sum * weights[j];
        }
    }
}

int check_res(const float *actual, const float *expected, int n, float tol){
    int errors = 0;
    float max_diff = 0.0f;
    for(int i = 0; i < n; ++i){
        float diff = fabs(actual[i] - expected[i]);
        if(diff > max_diff) max_diff = diff;
        if(diff > tol) errors++;
    }
    printf("  max_diff    = %.2e\n", max_diff);
    printf("  GPU vs CPU  : %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0;
}

int main(){
   int B = 64, T = 1024, C = 4096;
    int N = B * T;
    size_t bytes = N * C * sizeof(float);
    size_t weights_bytes = C * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);
    float *h_weights = (float *)malloc(weights_bytes);

    init(h_data, N * C, -1.0, 1.0);
    init(h_weights, C, 0, 1.0);

    rmsnorm_cpu(h_out, h_data, h_weights, N, C);

    float *d_data, *d_out, *d_weights;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_bytes, cudaMemcpyHostToDevice);

    kernel<<<N, BLOCK_SIZE>>>(d_out, d_data, d_weights, N, C);
    
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    check_res(h_out, out, N * C, 1e-5);

    free(h_data);
    free(h_weights);
    free(h_out);
    free(out);
    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_weights);

    return 0;
}