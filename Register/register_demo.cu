#include <iostream>
#include <cuda_runtime.h>

__global__ void computerRegister(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    float acc = 0.0f;

    for (int i = 0; i < 10; i++)
    {
        temp = data[idx + i * N];
        acc += temp * temp; // 寄存器高频累加
    }
    data[idx] = acc;
}

int main()
{
    return 0;
}