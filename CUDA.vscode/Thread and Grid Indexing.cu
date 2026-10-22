#include <iostream>
#include<cuda_runtime.h>
__global__ void vecAdd(float* A,float* B,float* C){
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    C[workIndex] = A[workIndex] + B[workIndex];
}
int main(){
    float *A,*B,*C;
    //Kernel invovation
    vecAdd<<<4,256>>>(A,B,C);
}