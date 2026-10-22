#include <cuda_runtime.h>
#include <iostream>

__global__ void computeRegister(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // 1. 把局部数组开到 2048 个 float（占 8KB，一个线程无论如何都不可能拥有这么多寄存器）
    float arr[2048]; 

    // 2. 强行把这个大数组的每一个位置都写满、读一遍，断了编译器的优化后路
    for (int i = 0; i < 2048; i++) {
        arr[i] = data[idx] + (float)i;
    }
    
    // 3. 把它们累加起来输出
    float sum = 0.0f;
    for (int i = 0; i < 2048; i++) {
        sum += arr[i];
    }
    
    data[idx] = sum;
}

int main() {
    return 0;
}