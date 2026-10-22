#include<cuda_runtime.h>
#include<iostream>
#include<math.h>

__global__ void dotProductKernel(const float*A,const float*B,float* result,int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N){
        float temp = A[i]*B[i];
        atomicAdd(result,temp);
    }
}

int main(){
    int N = 1<<20;
    size_t size = N*sizeof(float);

    float*h_a = (float*)malloc(size);
    float*h_b = (float*)malloc(size);
    float h_c = 0.0f;

    for(int i = 0;i<N;i++){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, sizeof(float));

    int blocks = 256;
    int grids = (N+blocks-1)/blocks;

    dotProductKernel<<<grids,blocks>>>(d_a,d_b,d_c,N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_c,d_c,sizeof(float),cudaMemcpyDeviceToHost);

    std::cout << "向量点乘结果: " << h_c << std::endl;
    std::cout << "预期结果: " << (float)N * 2.0f << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    return 0;
}

