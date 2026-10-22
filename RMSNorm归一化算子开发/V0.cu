#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define BLOCK_SIZE 512

// GPU版本
__global__ void kernel(float *d_out, float *d_data, float *weights, int B, int T, int C){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int N = B * T;

    if(tid >= N) return;
    else{
        float *inp = d_data + tid * C;
        float *out = d_out + tid * C;
        float local_sum = 0.0f;
        for(int i = 0; i < C; i++){
            local_sum += inp[i] * inp[i];
        }
        local_sum = sqrt(local_sum / C + 1e-9);
        for(int i = 0; i < C; i++){
            out[i] = inp[i] / local_sum * weights[i];
        }
    }
}
// CPU版本
void rmsnorm_cpu(float *h_out, float *h_data, float *weights, int B, int T,int C){
    for(int row = 0; row < B * T; row++){
        float *inp = h_data + row * C;
        float *out = h_out + row * C;
        float local_sum = 0.0f;
        for(int i = 0; i < C; i++){
            local_sum += inp[i] * inp[i];
        }
        local_sum = 1 / sqrt(local_sum / C + 1e-6);
        for(int i = 0; i < C; i++){
            out[i] = inp[i] * local_sum * weights[i];
        }
    }
}

int check_results(const float *actual, const float *expected, int N, float tol){
    int errors = 0;
    float max_diff = 0.0f;
    for(int i = 0; i < N; i++){
        float diff = fabs(actual[i] - expected[i]);
        if(diff > max_diff) max_diff = diff;
        if(diff > tol) errors++;
    }
    printf("  max_diff    = %.2e\n", max_diff);
    printf("  GPU vs CPU  : %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0;
}

void init(float *h_data, int N, int low, int high){
    for(int i = 0; i < N; i++){
        float r = (float)rand() / RAND_MAX;
        h_data[i] = low + r * (high - low);
    }
}

int main(){
    int B = 64, T = 1024, C = 2048;
    int N = B * T * C;
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);
    float *weights = (float *)malloc(C * sizeof(float));

    init(h_data, N, -1.0, 1.0);
    init(weights, C, 0.0, 1.0);

    float *d_data, *d_out, *d_weights;



    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_weights, C * sizeof(float));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, bytes, cudaMemcpyHostToDevice);

    kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_data, d_weights, B, T, C);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    rmsnorm_cpu(h_out, h_data, weights, B, T, C);

    int errors = check_results(h_out, out, N, 1e-6);
    printf("  errors      = %d\n", errors);

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_weights);

    free(h_data);
    free(h_out);
    free(out);
    free(weights);

    return 0;
}