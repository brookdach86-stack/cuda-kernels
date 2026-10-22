#include<iostream>
#include<cuda_runtime.h>
#include<cuda/cmath>
using namespace std;

__global__ void myKernel(float *data,int N){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < N){
        data[idx] = idx;
    }
}

//分配统一内存
int main(){
    int N = 1024;
    float*data; 
    int block = 256;
    int grid = cuda::ceil_div(N,block);

    cudaMallocManaged(&data,N*sizeof(float));
    
    for(int i=0;i<N;i++){
        data[i] = (float)i;
    }

    myKernel<<<grid,block>>>(data,N);
    cudaDeviceSynchronize();

    printf("Result:%f\n",data[3]);
    cudaFree(data);
    return 0;
}