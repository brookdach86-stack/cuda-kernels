#include <iostream>
#include <cuda_runtime.h>
#include <cuda/cmath>
using namespace std;

__global__ void myKernel(float *data, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] = idx;
    }
}

int main()
{
    int device;
    int N = 1024;
    float *data = nullptr;
    int block = 256;
    int grid = cuda::ceil_div(N, block);

    // 获取当前使用显卡ID
    cudaGetDevice(&device);

    // 统一分配内存
    cudaMallocManaged(&data, N * sizeof(float));

    // 性能优化：预取
    cudaMemLocation cpuLoc{};
    cpuLoc.type = cudaMemLocationTypeHost; // 指定目标是主机(CPU)
    cudaMemPrefetchAsync(data, N * sizeof(float), cpuLoc,0); // 预取


    for (int i = 0; i < N; ++i)
    {
        data[i] = (float)i;
    }

    // 主动把数据运输到GPU
    cudaMemLocation gpuLoc{};
    gpuLoc.type = cudaMemLocationTypeDevice;
    gpuLoc.id = device;
    cudaMemPrefetchAsync(data, N * sizeof(float), gpuLoc, 0);

    myKernel<<<grid, block>>>(data, N);

    cudaDeviceSynchronize();
    cout << "GPU计算完成,最后一个元素值" << data[N - 1] << endl;

    cudaFree(data);
    return 0;
}