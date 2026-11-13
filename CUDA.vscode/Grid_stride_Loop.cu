// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include<cuda_runtime.h>
#include<iostream>
#include<cstdio>

__global__ void processLargeArray(float* data,int N){
    int idx = blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;//总线程数

    //每个线程以 stride 为步长遍历
    for(int i = idx;i<N; i+= stride){
        data[i] = data[i]*2.0f;
    } 
}