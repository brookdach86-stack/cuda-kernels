#include <cuda_runtime.h>
#include <stdio.h>

void rmsnorm_cpu(float *h_out,float *h_data,float *h_weights,int B,int T,int C){
    for (int i  = 0; i < B * T; ++i){
        float *inp = h_data + i * C;
        float *out = h_out + i * C;

        float local_sum = 0.0f;
        for(int j = 0; j < C; ++j){
            local_sum += inp[j] * inp[j];
        }
        local_sum = 1 / sqrt(local_sum / C + 1e-6);
        for(int j = 0; j < C; ++j){
            out[j] = inp[j] * local_sum * h_weights[j];
        }
    }
}