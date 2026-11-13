// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 1 << 20;
    int threadsPerBlock = 256;
    float* d_A, * d_B, * d_C;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    return 0;
}
