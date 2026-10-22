/*
核心思路：用共享内存做"中转站"，将全局内存的非合并写入转化为共享内存的列读取(通过 Padding 消除 Bank Conflict),再以合并方式写出
*/

// 每个线程处理 TILE_DIM / BLOCK_ROWS 个元素
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float *out, float *in, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 Padding 消除 Bank Conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取：每个 warp 读取连续元素
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < width && (y + j) < height)
        {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    // 合并写入：转置后的索引，连续线程写连续地址
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < height && (y + j) < width)
        {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 调用方式：
// dim3 block(TILE_DIM, BLOCK_ROWS);
// dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);