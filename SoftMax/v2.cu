#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define WARP_SIZE 32

__global__ void warp_online_softmax(float *d_out, float *d_data, int N, int C)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int row_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (row_id >= N)
    {
        return;
    }

    float *inp = d_data + row_id * C;
    float *out = d_out + row_id * C;

    float maxval = -INFINITY;
    float sum = 0.0f;

    // 步进WARP_SIZE，每个线程处理WARP_SIZE个元素
    for (int i = lane_id; i < C; i += WARP_SIZE)
    {
        float val = inp[i];
        if (val > maxval)
        {
            sum *= expf(maxval - val);
            maxval = val;
        }
        sum += expf(val - maxval);
    }

    // 蝴蝶规约，汇总WARP_SIZE个线程的结果
    unsigned mask = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float sum_other = __shfl_down_sync(mask, sum, offset);
        float maxval_other = __shfl_down_sync(mask, maxval, offset);
        float maxval_new = fmaxf(maxval, maxval_other);
        sum = sum * expf(maxval - maxval_new) + sum_other * expf(maxval_other - maxval_new); // 合并两个线程
        maxval = maxval_new;
    }

    sum = __shfl_sync(mask, sum, 0);
    maxval = __shfl_sync(mask, maxval, 0);

    float r = 1.0f / sum;
    for (int i = lane_id; i < C; i += WARP_SIZE)
    {
        out[i] = expf(inp[i] - maxval) * r;
    }
}