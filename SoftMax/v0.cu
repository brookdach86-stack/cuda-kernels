#include <math.h>
#include <float.h>

void softmax_cpu(float *input, float *output, int n)
{
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (input[i] > max_val)
        {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }
}


__global__ void softmax_gpu(float *d_out, float *d_data, int N, int C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid;
    if (row >= N)
    {
        return;
    }

    float *inp = d_data + row * C;
    float *out = d_out + row * C;

    float max_val = -INFINITY;
    for (int i = 0; i < C; i++)
    {
        if (inp[i] > max_val)
        {
            max_val = inp[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < C; i++)
    {
        sum += expf(inp[i] - max_val);
    }
    for (int i = 0; i < C; i++)
    {
        out[i] = expf(inp[i] - max_val) / sum;
    }
}
