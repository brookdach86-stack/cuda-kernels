#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void kernel(float *d_out, float *d_data, float *weights, int N, int C)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int row = index / blockDim.x;
    int tid = index % blockDim.x;

    if (row < N)
    {

        // ====== Step 1: 定位当前行数据 ======
        float *inp = d_data + row * C;
        float *out = d_out + row * C;

        int factor = C / BLOCK_SIZE;
        __shared__ float smem[BLOCK_SIZE];

        // ====== Step 2: 分片算部分平方和 ======
        float local_sum = 0.0f;
        for (int i = 0; i < factor; ++i)
        {
            local_sum += inp[tid + i * BLOCK_SIZE] * inp[tid + i * BLOCK_SIZE];
        }

        // ====== Step 3: 写入 shared memory + 同步 ======
        smem[tid] = local_sum;
        __syncthreads();

        // ====== Step 4: 树形规约，smem[0] = Σx² ======
        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
        {
            if (tid < stride)
            {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }

        // ====== Step 5: 算缩放因子 norm = 1 / RMS(x) ======
        float norm = 1 / sqrt(smem[0] / C + 1e-6);

        // ====== Step 6: 分片输出 out[i] = x[i] * norm * weight[i] ======
        for (int i = tid; i < C; i += BLOCK_SIZE)
        {
            out[i] = inp[i] * norm * weights[i];
        }
    }
}

void init(float *data, int N, int low, int high)
{
    for (int i = 0; i < N; ++i)
    {
        float r = (float)rand() / RAND_MAX;
        data[i] = low + (high - low) * r;
    }
}

void rmsnorm_cpu(float *h_out, float *h_data, float *weights, int N, int C)
{
    for (int i = 0; i < N; ++i)
    {
        float *inp = h_data + i * C;
        float *out = h_out + i * C;

        float local_sum = 0.0f;

        for (int j = 0; j < C; ++j)
        {
            local_sum += inp[j] * inp[j];
        }

        local_sum = 1 / sqrt(local_sum / C + 1e-6);

        for (int j = 0; j < C; ++j)
        {
            out[j] = inp[j] * local_sum * weights[j];
        }
    }
}

int check_results(const float *actual, const float *expected,
                  int n, float tol)
{
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = fabsf(actual[i] - expected[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > tol)
            errors++;
    }
    printf("  max_diff    = %.2e\n", max_diff);
    printf("  GPU vs CPU  : %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0;
}

int main()
{
    int B = 64, T = 1024, C = 2048;
    int N = B * T;
    size_t bytes = B * T * C * sizeof(float);
    size_t weights_bytes = C * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    float *h_weights = (float *)malloc(weights_bytes);
    float *h_out = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);

    init(h_data, B * T * C, -1.0, 1.0);
    init(h_weights, C, 0, 1.0);
    rmsnorm_cpu(h_out, h_data, h_weights, B * T, C);

    float *d_data, *d_out, *d_weights;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_bytes, cudaMemcpyHostToDevice);

    kernel<<<N, BLOCK_SIZE>>>(d_out, d_data, d_weights, N, C);
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    check_results(h_out, out, N * C, 1e-5);

    free(h_data);
    free(h_weights);
    free(h_out);
    free(out);
    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_weights);

    return 0;
}