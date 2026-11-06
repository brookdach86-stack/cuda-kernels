//古老写法
__global__ void optimizedKernel(float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 【核心用法】：传入数据的内存地址 &input[idx]
    // 强制编译器去走只读缓存，由于它不占 L1 的坑位，可以大大缓解 L1 缓存的压力
    float val = __ldg(&input[idx]); 
    output[idx] = val * 2.0f;
}

// ✨ 现代标准写法（推荐，NVCC 会自动在底层开启类似 __ldg 的只读缓存优化）
__global__ void myKernel(const float* __restrict__ input) {
    int idx = threadIdx.x;
    float val = input[idx]; // 编译器看懂了 const 和 restrict，自动优化
}

