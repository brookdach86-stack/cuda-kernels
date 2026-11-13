// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
// 3. 原子操作：跨 Block 安全累加到全局显存
__global__ void demo_atomic(float* global_sum, const float* d_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(global_sum, d_in[idx]); // 硬件级防冲突累加
    }
}