#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(float* d_out, const float* d_in) {
    // 业务代码...
}

int main() {
    // ==================== 4.4 核心配置命令 ====================

    // 【命令 1】：解锁限制，允许动态共享内存申请到硬件极限（比如 A100 解锁到 164KB）
    cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024);

    // 【命令 2】：设置偏好。告诉 GPU 把这块 SRAM 蛋糕多切点给共享内存（PreferShared）
    cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);

    // =========================================================

    // 正常启动你的核函数
    // myKernel<<<grid, block, sharedSize>>>(d_out, d_in);
    
    std::cout << "4.4 硬件配置完成！" << std::endl;
    return 0;
}