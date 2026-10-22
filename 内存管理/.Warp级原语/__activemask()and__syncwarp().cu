#include <stdio.h>
#include <cuda_runtime.h>

__global__ void demoKernel(int *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid % 2 == 0)
    {
        unsigned mask = __activemask(); // 显示当前线程的mask值
        data[tid] = tid * 10;
        __syncwarp(mask); // mask中标记1的线程等待所有线程到达
    }
    else{
        unsigned mask = __activemask();
        data[tid] = tid*100;
        __syncwarp(mask);
    }
}