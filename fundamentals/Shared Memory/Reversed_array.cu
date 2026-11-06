/*
1.主机准备 cudaMalloc分配全局内存
2.协作加载 threadIdx.x -> 共享内存tile
3.栅栏同步 __syncthreads
4.逆序写回 线程根据对称索引partner反向读取tile
*/

#include<cuda_runtime.h>
#include<iostream>

//声明与使用核函数
__global__ void sharedMenReverse(float* d_out,const float*d_in){ 
    __shared__ float tile[256]; // 1.静态声明共享内存
    int t_idx = threadIdx.x;
    int idx =blockIdx.x*blockDim.x + t_idx;

    tile[t_idx] = d_in[idx]; // 2.加载到共享内存
    __syncthreads(); // 3.同步等待
    
    int partner = blockDim.x-1-t_idx;
    d_out[idx] = tile[partner];
}   

int main(){
    const int N = 256;
    size_t size = N*sizeof(float);

    float h_in[N],h_out[N];
    for(int i = 0;i < N;i++){
        h_in[i] = i;
    }

    float*d_in,*d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out,size);

    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);

    //1个Block,256个线程
    sharedMenReverse<<<1,N>>>(d_out,d_in);

    for(int i = 0;i<3;i++){
        std::cout<<h_out[i]<<"";
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}