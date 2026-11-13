// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
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