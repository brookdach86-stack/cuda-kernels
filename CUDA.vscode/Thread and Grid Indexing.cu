// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <iostream>
#include<cuda_runtime.h>
__global__ void vecAdd(float* A,float* B,float* C){
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    C[workIndex] = A[workIndex] + B[workIndex];
}
int main(){
    float *A,*B,*C;
    //Kernel invovation
    vecAdd<<<4,256>>>(A,B,C);
}