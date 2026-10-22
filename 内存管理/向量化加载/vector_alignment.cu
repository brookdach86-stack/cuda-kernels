// 向量化对齐
#include <stdio.h>
#include <cuda_runtime.h>

// 对齐要求：
// float4 要求 16B 对齐
// cudaMalloc 分配的地址是 256B 对齐的 → 满足
// 但如果有偏移,需要特殊处理:先标量处理头部未对齐部分,再向量化处理主体

__global__ void safe_vectorized(float *dst, float *src, int offset, int N)
{
    // offset: 起始偏移(float 元素计);
    // (offset % 4) 是 offset 对 4 取余,得到 offset 对齐到 4 的倍数的偏移量
    // (4 - (offset % 4)) 是 4 减去 offset 对齐到 4 的倍数的偏移量
    int align_offset = (4 - (offset % 4)) % 4;

    // 主体向量化区间 [vec_start, vec_start + n4*4)
    int vec_start = offset + align_offset;
    int n4 = (N - vec_start) / 4;        // 可向量化的 float4 个数，主体线程数
    int tail_start = vec_start + n4 * 4; // 尾部标量起点

    float4 *src4 = reinterpret_cast<float4 *>(src + vec_start);
    float4 *dst4 = reinterpret_cast<float4 *>(dst + vec_start);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 头部一个线程负责处理 1 个元素
    if (tid < align_offset)
    {
        // 头部:标量处理 [offset, vec_start)
        dst[offset + tid] = src[offset + tid];
    }
    else if (tid < align_offset + n4)
    {
        // 主体:向量化处理,每个线程 1 个 float4(16B)
        int vid = tid - align_offset;
        dst4[vid] = src4[vid];
    }
    else if (tid < N - offset)
    {
        // 尾部:标量处理 [tail_start, N)，一个线程负责处理 1 个元素
        int tail_idx = tid - align_offset - n4;
        dst[tail_start + tail_idx] = src[tail_start + tail_idx];
    }
}