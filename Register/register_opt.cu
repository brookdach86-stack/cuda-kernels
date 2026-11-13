// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <cuda_runtime.h>
#include <iostream>

__global__ void computeRegister(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // 依然是这个占 8KB 的大数组
    float arr[2048]; 

    for (int i = 0; i < 2048; i++) {
        arr[i] = data[idx] + (float)i;
    }
    
    float sum = 0.0f;
    
    // 【代码微调】：适度展开这个固定次数的求和循环，免去每次循环判断 i < 2048 的耗时
    #pragma unroll
    for (int i = 0; i < 2048; i++) {
        sum += arr[i];
    }
    
    data[idx] = sum;
}

int main() {
    return 0;
}