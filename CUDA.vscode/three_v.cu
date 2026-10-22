#include <cuda_runtime.h>
#include <iostream>
__global__ void vecAdd(float*A,float*B,float*C){
}
int main(){
    float *A = nullptr; 
    float *B = nullptr; 
    float *C = nullptr;
    //Kernel invovation
    vecAdd<<<1,256>>>(A,B,C);
}