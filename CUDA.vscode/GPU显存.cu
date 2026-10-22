#include <iostream>
#include <cuda_runtime.h> // 必须引入 CUDA 运行时头文件

using namespace std;

int main() {
    // 1. 在 CPU 上声明一个一维浮点数指针，此时它是个空指针
    float* d_tensor = nullptr; 
    
    // 计算 1024 个浮点数需要多少字节（Byte）的显存
    size_t bytes = 1024 * sizeof(float);

    // 2. 【核心语法】：调用 cudaMalloc
    // 第一个参数必须写 (void**)&d_tensor —— 意思是把指针的地址传给 GPU 驱动，让它直接往里写显存的真实地址,二重指针
    // 第二个参数是申请的字节数
    cudaError_t err = cudaMalloc((void**)&d_tensor, bytes);

    // 3. 安全检查（算子开发的标准习惯）：检查显存是否开辟成功（比如防范显存溢出 OOM）
    if (err != cudaSuccess) {
        cout << "显存开辟失败！错误原因: " << cudaGetErrorString(err) << endl;
        return -1;
    }
    
    cout << "🚀 GPU 显存成功开辟!d_tensor 已经指向了 RTX 5060 芯片内部的硬件地址。\n";

    // 4. 【唯一正确配对】：必须用 cudaFree 释放
    cudaFree(d_tensor);
    cout << "💥 显存安全回收。\n";

    return 0;
}