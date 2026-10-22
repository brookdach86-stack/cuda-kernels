#include<stdio.h>
#include<cuda_runtime.h>

__device__ float warp_reduce_sum(float val){
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 块级规约核函数
__global__ void block_reduce_kernel(float* input,float*output,int n){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float val = (tid < n) ? input[tid] : 0.0f;
    val = warp_reduce_sum(val);

    __shared__ float warp_sums[32];
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (laneId == 0){
        warp_sums[warpId] = val;
    }
    __syncthreads();

    int numWarp = blockDim.x / 32;
    val = (threadIdx.x < numWarp) ? warp_sums[threadIdx.x] : 0.0f;
    if (warpId == 0){
        val = warp_reduce_sum(val);
    }
    if (threadIdx.x == 0){
        output[blockIdx.x] = val;
    }
}

int main(){
    int n = 1024;
    int numBlocks = n/32;

    float* input = (float*)malloc(n*sizeof(float));
    float* output = (float*)malloc(numBlocks*sizeof(float));

    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }

    float* input_d;
    float* output_d;

    cudaMalloc(&input_d,n*sizeof(float));
    cudaMalloc(&output_d,n*sizeof(float));
    cudaMemcpy(input_d,input,n*sizeof(float),cudaMemcpyHostToDevice);

    block_reduce_kernel<<<numBlocks,32>>>(input_d,output_d,n);

    cudaMemcpy(output,output_d,numBlocks*sizeof(float),cudaMemcpyDeviceToHost);

    //output计算的是每个block的规约结果
    printf("output[0] = %f\n",output[0]);// 32.000000

    free(input);
    free(output);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}