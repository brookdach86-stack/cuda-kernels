// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <stdio.h>
#include <cuda_runtime.h>

//atomicExch(found,tid)是一个原子交换操作，如果found为0，将tid赋值给found，否则不进行任何操作
__global__ void early_exit_kernel(int*data,int target,int*found){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pred =(data[tid] == target);
    if(__any_sync(0xFFFFFFFF,pred)){
        if(pred){
            atomicExch(found,tid);
        }
    }
}
