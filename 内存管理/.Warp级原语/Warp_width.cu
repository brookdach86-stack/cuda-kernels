#include <stdio.h>
#include <cuda_runtime.h>

// 核心函数：利用 width=8 进行子 Warp 归约求和
__device__ float sub_warp_sum(float val) {
    // offset 从 4 开始，因为 width=8，最大跨度是 8/2 = 4
    for (int offset = 4; offset > 0; offset >>= 1) {
        // 关键：最后一个参数 8 限制了数据交换只在每 8 个线程内部进行
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, 8);
    }
    return val;
}

__global__ void test_kernel() {
    int tid = threadIdx.x;
    
    // 1. 初始化数据：模拟 4 个不同的小组
    float val = 0.0f;
    if (tid < 8)       val = 1.0f; // 第 1 组 (Lane 0-7)
    else if (tid < 16) val = 2.0f; // 第 2 组 (Lane 8-15)
    else if (tid < 24) val = 3.0f; // 第 3 组 (Lane 16-23)
    else               val = 4.0f; // 第 4 组 (Lane 24-31)

    // 2. 执行子 Warp 求和
    float sum = sub_warp_sum(val);

    // 3. 打印结果：只让每个小组的“老大”（Lane 0, 8, 16, 24）汇报
    // tid % 8 == 0 正好选中这四个组长
    if (tid % 8 == 0) {
        printf("Group Leader [Lane %d]: My group sum is %.0f\n", tid, sum);
    }
}

int main() {
    // 启动 1 个 Block，包含 32 个线程（正好一个 Warp）
    test_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}