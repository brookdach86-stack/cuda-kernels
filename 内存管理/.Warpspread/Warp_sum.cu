#include <iostream>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void sum_kernel()
{
    float val = threadIdx.x + 1.0f;
    float sum = warp_reduce_sum(val);

    if (threadIdx.x == 0)
    {
        printf("warp sum = %f\n", sum);
    }
}

int main()
{
    sum_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}