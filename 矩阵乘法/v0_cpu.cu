#include<stdio.h>
#include<cuda_runtime.h>

void multi_cpu(float *C,float *A,float *B,int M,int K,int N){
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0.0f;
            for(int k = 0; k < K; ++k){
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

//当M=N=K时，时间复杂度为𝑂⁡(𝑀⁢𝑁⁢𝐾) =𝑂⁡(𝑁3)，空间复杂度为𝑂⁡(𝑀⁢𝐾 +𝐾⁢𝑁 +𝑀⁢𝑁) =𝑂⁡(𝑁2)。