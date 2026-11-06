/*
1. 物理分块:把256个元素的tile数组切分成32个交错的Bank
2. 制造两路冲突:把核函数的读取步长改为2,线程0和线程16挤在同一个Bank0窗口,造成2-way冲突,
3. 制造全员冲突:修改读取步长为32,让Warp内32个线程全部死磕在Bank0,造成32-Way极端冲突，高并发变成单线程
*/

#include <cuda_runtime.h>
#include <iostream>

// 1. 无冲突核函数：线程 i 读 tile[i] 
__global__ void noConflictKernel(float* d_out, const float* d_in) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    
    tile[tid] = d_in[tid];
    __syncthreads();

    // 步长为 1：线程 0->Bank 0, 线程 1->Bank 1... 完美错开
    d_out[tid] = tile[tid]; 
}

// 2. 32路冲突核函数：线程 i 读 tile[i * 32] 
__global__ void maxConflictKernel(float* d_out, const float* d_in) {
    // 放大数组到 8192 防止 tid * 32 越界死机
    __shared__ float tile[8192]; 
    int tid = threadIdx.x;
    
    tile[tid] = d_in[tid];
    __syncthreads();

    // 步长为 32
    d_out[tid] = tile[tid * 32]; 
}

int main() {
    const int N = 256;
    size_t size = N * sizeof(float);
    float h_in[N]; for(int i = 0; i < N; i++) h_in[i] = i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size); 
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 1. CUDA 计时事件准备
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    float ms = 0;

    // 步骤 2：跑无冲突版 (只跑 1 个 Warp 32线程来看纯硬件行为)
    cudaEventRecord(start); 
    noConflictKernel<<<1, 32>>>(d_out, d_in);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "无冲突耗时: " << ms << " ms" << std::endl;

    // 步骤 3：跑 32 路全员冲突版
    cudaEventRecord(start);
    maxConflictKernel<<<1, 32>>>(d_out, d_in);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "全员撞车耗时: " << ms << " ms" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}