#include <iostream>
#include <cstdio>
#include <cuda/cmath>
#include <cuda_runtime.h>
using namespace std;

//分配统一内存 -> CPU初始化 -> GPU核函数计算 -> 同步 -> CPU验证结果 -> 释放内存

//GPU 核函数：向量加法
__global__ void vecAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    const int N = 1 << 20; // 向量大小
    const size_t size = N * sizeof(float);//一个float占4字节，size == 要分配的字节数

    // 1.在主机上分配内存(Unified Memory)
    float* A;
    float* B;
    float* C;

    // 使用cudaMallocManaged分配统一内存，CPU和GPU都可以访问,二级指针指向指针内存地址
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    /*
    分配CPU传统内存，用于后期的串行验证
    1.malloc CPU内存 申请size字节的内存
    2.comparisonResult用于存储GPU计算结果与CPU计算结果的比较结果
    */
    float *comparisonResult = (float*)malloc(size);
    
    // 2.CPU初始化 
    for (int i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i*2);
    }

    // 3.核函数计算
    int treads = 256;
    int blocks = cuda::ceil_div(N,treads); //向上取整，保证每个元素都被处理
    vecAdd<<<blocks, treads>>>(A, B, C, N); //调用核函数，<<<blocks, threads>>>指定网格和块的维度

    // 4.同步
    cudaDeviceSynchronize(); //等待GPU完成所有任务
 
    //5.CPU串行验证
    for(int i = 0; i < N; ++i)
    {
        comparisonResult[i] = (A[i] + B[i]);
    }
    
    //6.验证GPU计算结果与CPU计算结果是否一致
    bool isCorrect = true;
    for(int i = 0; i < N; ++i){
        if(fabs(C[i] - comparisonResult[i]) > 1e-5)
        {
            isCorrect = false;
            break;
        }
    }
    if(isCorrect)
    {
        cout << "GPU计算结果正确!" << endl;
    }
    else
    {
        cout << "GPU计算结果错误!" << endl;
    }
    cudaFree(A); //释放GPU内存
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}