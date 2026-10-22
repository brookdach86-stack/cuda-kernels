#include <iostream>
#include <cuda_runtime.h>

//__device__ 工具函数:Warp 内广播,只能在GPU内调用
__device__ float warp_broadcast(float val, int srcLane)
{
    // 所有线程获取 srcLane 线程的 val 值
    return __shfl_sync(0xFFFFFFFF, val, srcLane);
}

__global__ void demo_kernel()
{
    float val = threadIdx.x * 1.0f;
    float result = warp_broadcast(val, 5);
    printf("thread %d: result = %f\n", threadIdx.x, result);
}

int main()
{
    demo_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}