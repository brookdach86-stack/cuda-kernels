#include <cuda_runtime.h>

__global__ void batchProcess(float* data, int batchSize, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (col < width && row < height && batch < batchSize) {
        int idx = batch * height * width + row * width + col;
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    int w = 16, h = 16, batchSize = 4;
    int total_elements = w * h * batchSize;
    float *data;

    // 分配统一内存并初始化
    cudaMallocManaged(&data, total_elements * sizeof(float));
    for (int i = 0; i < total_elements; i++) data[i] = 1.0f;

    // 3D 启动配置
    dim3 block(16, 16, 1);
    dim3 grid((w + 15) / 16, (h + 15) / 16, batchSize);
    batchProcess<<<grid, block>>>(data, batchSize, h, w);

    // 同步并释放
    cudaDeviceSynchronize();
    cudaFree(data);
    
    return 0;
}