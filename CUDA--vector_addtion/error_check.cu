#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 补全：定义向量加法核函数 ---
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
int main() {
    // --- 补全：定义变量 ---
    int N = 10000;
    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float* d_A, * d_B, * d_C;

    // --- 使用宏进行错误检查 ---
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size)); // 补全：分配 B
    CUDA_CHECK(cudaMalloc(&d_C, size)); // 补全：分配 C

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice)); // 补全：拷贝 B

    // --- 补全：配置执行配置 ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Kernel 启动的错误检查需要两步
    vectorAdd << <gridSize, blockSize >> > (d_A, d_B, d_C, N);

    // 检查是否启动成功（比如参数错误、显存溢出等）
    CUDA_CHECK(cudaGetLastError());          // 检查启动参数是否正确
    // 检查核函数内部执行是否有错（这是同步操作）
    CUDA_CHECK(cudaDeviceSynchronize());     // 检查执行过程是否有错误

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 验证结果
    if (h_C[0] == 3.0f) printf("运行成功！\n");

    // --- 补全：释放内存 ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}