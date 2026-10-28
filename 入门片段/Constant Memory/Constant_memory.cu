#include <iostream>
#include <cuda_runtime.h>

__constant__ float d_factors[4];

__global__ void multiplyKernel(float *data)
{
    int idx = threadIdx.x;
    data[idx] = data[idx] * d_factors[idx % 4];
}

int main()
{
    float *d_data;
    float h_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    cudaMalloc(&d_data, 4 * sizeof(float));
    cudaMemcpy(d_data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice);

    float h_factors[4] = {10.0f, 20.0f, 30.0f, 40.0f};

    cudaMemcpyToSymbol(d_factors, h_factors, 4 * sizeof(float));

    multiplyKernel<<<1, 4>>>(d_data);

    cudaMemcpy(h_data, d_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++)
    {
        std::cout << h_data[i] << "";
    }
    cudaFree(d_data);
    return 0;
}