// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void coalesced_read(float* data, float* out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        out[tid] = data[tid];  // data[0], data[1], ..., data[31] 连续
    }
}
