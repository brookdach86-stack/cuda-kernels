#include <iostream>
#include <cuda_runtime.h>

// 1. GPU 算子核心：告诉 GPU 要执行的任务
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // 2. 启动 GPU 任务
    // <<<1, 5>>> 的意思是：启动 1 个线程块，里面包含 5 个并行线程
    helloFromGPU<<<1, 5>>>();

    // 3. 让 CPU 等待 GPU 所有的线程执行完毕
    cudaDeviceSynchronize();

    return 0;
}
