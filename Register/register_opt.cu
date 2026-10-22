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