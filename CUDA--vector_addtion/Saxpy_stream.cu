#include<cuda_runtime.h>
#include<iostream>

__global__ void saxpyKernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 20;
    const int nStreams = 4;
    const int streamSize = N / nStreams;
    const size_t streamBytes = streamSize * sizeof(float);
    float a = 2.0f;

    float* h_x, * h_y;
    cudaMallocHost(&h_x, N * sizeof(float));
    cudaMallocHost(&h_y, N * sizeof(float));

    for (int i = 0;i < N;i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float* d_x, * d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    //创建流
    cudaStream_t streams[nStreams];
    for (int i = 0;i < nStreams;i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0;i < nStreams;i++) {
        int offset = i * streamSize;

        // 异步从主机拷贝到设备 (H2D)
        cudaMemcpyAsync(&d_x[offset], &h_x[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_y[offset], &h_y[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);

        // 启动核函数 (注意第4个参数是 streams[i])
        int threadsPerBlock = 256;
        int blocksPerGrid = (streamSize + threadsPerBlock - 1) / threadsPerBlock;
        saxpyKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(streamSize, a, &d_x[offset], &d_y[offset]);

        // 异步从设备拷贝回主机 (D2H)
        cudaMemcpyAsync(&h_y[offset], &d_y[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // 5. 同步所有流，等待任务完成
    cudaDeviceSynchronize();

    // 验证结果
    std::cout << "验证结果[0]: " << h_y[0] << " (预期 12.0)" << std::endl;
    std::cout << "验证结果[末尾]: " << h_y[N-1] << " (预期 12.0)" << std::endl;

    // 6. 清理
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFreeHost(h_x); // 释放锁页内存
    cudaFreeHost(h_y);

    return 0;
}