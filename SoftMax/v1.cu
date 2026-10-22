#include<cuda_runtime.h>
#include<math.h>
#include<float.h>
#include<stdio.h>

__global__ void online_softmax(float *d_out,float *d_data,int N,int C){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = tid;
    if (row >= N){
        return;
    }
    float *inp = d_data + row * C;
    float *out = d_out + row * C;

    float sum = 0.0f;
    float maxval = -INFINITY;
    for (int i = 0; i < C; i++)
    {
        float val = inp[i];
        if (inp[i] > maxval){
            sum *= expf(maxval - val);
            maxval = val;
        }
        sum += expf(val - maxval);
    }
    float r = 1 / sum;
    for(int i = 0; i < C; ++i){
        out[i] = expf(inp[i] - maxval) * r;
    }
}