#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorized_sum(float *input, float *output, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 全局线程ID
    int stride = blockDim.x * gridDim.x;             // grid总线程数

    float sum = 0.0f;

    int vec_N = N / 4;                                     // 向量化组数
    float4 *vec_input = reinterpret_cast<float4 *>(input); // 转为float4指针

    // 向量化加载：每次读4个float
    for (int i = tid; i < vec_N; i += stride)
    {
        float4 val = vec_input[i];
        sum += val.x + val.y + val.z + val.w;
    }

    // 尾部处理：剩余不足4个的元素
    int tail_start = vec_N * 4;
    for (int i = tail_start + tid; i < N; i += stride)
    {
        sum += input[i];
    }

    output[blockIdx.x] = sum; // 简化，实际需要block内归约
}