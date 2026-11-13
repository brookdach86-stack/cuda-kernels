// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <cuda_runtime.h>
#include <iostream>
__global__ void vecAdd(float*A,float*B,float*C){
}
int main(){
    float *A = nullptr; 
    float *B = nullptr; 
    float *C = nullptr;
    //Kernel invovation
    vecAdd<<<1,256>>>(A,B,C);
}