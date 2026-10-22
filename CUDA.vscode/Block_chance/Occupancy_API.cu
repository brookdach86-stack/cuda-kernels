#include<cstdio>
#include<cuda_runtime.h>

__global__ void myKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

int main(){
    int N = 1000000;
    float *d_input,*d_output;
    
    cudaMalloc(&d_input,N*sizeof(float));
    cudaMalloc(&d_output,N*sizeof(float));
    
    int blocksize;
    int minGridsize;

    //cudaOccupancyMaxPotentialBlockSize 是一个硬件适配器,找到最适配的线程块数量
    cudaOccupancyMaxPotentialBlockSize(&minGridsize,&blocksize,myKernel,0,0);

    int gridsize = (N+blocksize-1)/blocksize;

    myKernel<<<gridsize,blocksize>>>(d_input,d_output,N);

    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess){
        printf("Kernel excution error:%s\n",cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err!=cudaSuccess){
        printf("Kernel execution error:%s\n",cudaGetErrorString(err));
    }else{
        printf("Success!");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}