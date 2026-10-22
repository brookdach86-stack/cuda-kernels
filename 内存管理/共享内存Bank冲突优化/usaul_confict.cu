#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// 1. Shared Memory 分块转置（读取时 32-way Bank Conflict）
__global__ void transposeSharedMem(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (gx < width && gy < height) {
        tile[threadIdx.y][threadIdx.x] = idata[gy * width + gx];
    }
    __syncthreads();

    int ox = blockIdx.y * TILE_SIZE + threadIdx.x;
    int oy = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (ox < height && oy < width) {
        odata[oy * height + ox] = tile[threadIdx.x][threadIdx.y];//height原本是行数转置后变成列数
    }
}

// 2. Padding 优化（消除 Bank Conflict）
// ★ 相对于第1部分的唯一改动：TILE_SIZE → TILE_SIZE+1
__global__ void transposePadding(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // ★ 32×33，多一列 padding，用于处理边界情况，避免越界访问，确保每个线程都有数据可读

    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (gx < width && gy < height) {
        tile[threadIdx.y][threadIdx.x] = idata[gy * width + gx];
    }
    __syncthreads();

    int ox = blockIdx.y * TILE_SIZE + threadIdx.x;
    int oy = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (ox < height && oy < width) {
        odata[oy * height + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

// 3. Swizzle 优化（XOR 变换消除 Bank Conflict）
// ★ 相对于第1部分的改动：列索引做 XOR 变换，不浪费 shared memory
__global__ void transposeSwizzle(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE]; // ★ 仍然是 32×32，不加 padding

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_SIZE + tx;
    int gy = blockIdx.y * TILE_SIZE + ty;

    if (gx < width && gy < height) {
        tile[ty][tx ^ ty] = idata[gy * width + gx]; // ★ 写入：列索引 tx → tx^ty
    }
    __syncthreads();

    int ox = blockIdx.y * TILE_SIZE + tx;
    int oy = blockIdx.x * TILE_SIZE + ty;
    if (ox < height && oy < width) {
        odata[oy * height + ox] = tile[tx][ty ^ tx]; // ★ 读取：列索引 ty → ty^tx
    }
}