// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
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
    int blocksize = 256;
    int gridsize = (N+blocksize-1)/blocksize;

    float *d_input,*d_output;
    cudaMalloc(&d_input,N*sizeof(float));
    cudaMalloc(&d_output,N*sizeof(float));
    
    myKernel<<<gridsize,blocksize>>>(d_input,d_output,N);

    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess){
        printf("Kernel excution error:%s\n",cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    if (err!=cudaSuccess){
        printf("Kernel execution error:%s\n",cudaGetErrorString(err));
    }else {
        printf("Kernel executed successfully!\n");
    }    

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}