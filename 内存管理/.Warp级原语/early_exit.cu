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
