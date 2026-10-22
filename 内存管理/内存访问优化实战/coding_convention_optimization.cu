/*
归约（Reduction）内核：用于对数组求和
核心思想：使用共享内存逐步"折叠"数据，将多个元素归约为一个结果
*/

__global__ void reduce_sequential(float *input, float *output, int N)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    __shared__ float sdata[256];

    // 加载数据到共享内存（越界补0）
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 逐步归约：每轮参与计算的线程数减半
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // 每轮必须同步
    }

    // 只有线程0写最终结果
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// 调用方式：
// int blockSize = 256;
// int numBlocks = (N + blockSize - 1) / blockSize;
// reduce_sequential<<<numBlocks, blockSize>>>(d_input, d_output, N);

/*
归约过程示例（假设 8 个线程，数据 [1,2,3,4,5,6,7,8]）：

轮次    s值    操作                          结果
初始    -      -                             [1, 2, 3, 4, 5, 6, 7, 8]
第1轮   4      sdata[0]+=sdata[4] 等         [6, 8, 10, 12, 5, 6, 7, 8]
第2轮   2      sdata[0]+=sdata[2] 等         [16, 20, 10, 12, 5, 6, 7, 8]
第3轮   1      sdata[0]+=sdata[1]            [36, 20, 10, 12, 5, 6, 7, 8]

最终 sdata[0] = 36 就是总和

关键点：
- 共享内存：避免反复访问全局内存
- __syncthreads()：每轮归约后同步，确保所有线程完成
- s >>= 1：每轮参与计算的线程数减半
- tid == 0：只有线程0写最终结果
- "Sequential"：归约是串行的，每轮必须等上一轮完成
*/