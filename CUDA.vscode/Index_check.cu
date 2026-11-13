// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <iostream>
#include <cuda/cmath> //cuda.ceil_div()
#include <cuda_runtime.h>

__global__ void vecAdd(float* A,float* B,float* C,int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

int main(){
    int threads = 256;
    float *A, *B, *C;
    // int blocks = (vectorLength + threads - 1) / threads;
    int blocks = cuda::ceil_div(1000, threads); // 使用cuda::ceil_div()函数进行向上取整
    vecAdd<<<blocks, threads>>>(A, B, C, 1000);
}