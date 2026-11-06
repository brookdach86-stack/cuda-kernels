#include<cstdio>
#include<cuda_runtime.h>

__global__ void grayscaleKernel(unsigned char*rgb,unsigned char*gray,int width,int height){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col < width && row < height){
        int rgbIdx = (row*width + col)*3;
        int grayIdx = row*width + col;
        //转灰，依次由红绿蓝系数计算灰度
        gray[grayIdx] = (unsigned char)(
            0.299f * rgb[rgbIdx] +
            0.587f * rgb[rgbIdx + 1] +
            0.114f * rgb[rgbIdx + 2]
        );
    }
}

//启动配置
int main(){
    int w = 1024,h = 1024,pixels = w*h;
    unsigned char *rgb,*gray;

    //分配统一内存
    cudaMallocManaged(&rgb,pixels*3);
    cudaMallocManaged(&gray,pixels);

    dim3 block(16,16);
    dim3 grid((w + 15)/16,(h+15)/16);
    grayscaleKernel<<<grid,block>>>(rgb,gray,w,h);
    cudaDeviceSynchronize();

    printf("GPU Grayscale conversion successful!\n");

    cudaFree(rgb);
    cudaFree(gray);
    return 0;
}