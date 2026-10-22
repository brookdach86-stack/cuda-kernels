/* 显存管理 */
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

//定义核函数：GPU并行计算向量加法
__global__ void vecAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) 
    {
        C[idx] = A[idx] + B[idx]; 
    }
}

//CPU串行验证函数
void serialVecAdd(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i) 
    {
        C[i] = A[i] + B[i]; 
    }
}

//六步式流水线实现显存管理
void explicitMemExample(int vectorLength){
    //声明两套指针
    size_t size = vectorLength * sizeof(float); 
    float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr,*h_comparisionResult = nullptr; //h_表示host(CPU)
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr; //d_表示device(GPU)

    //步骤2：分配两边的内存
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    h_comparisionResult = (float*)malloc(size); //CPU验证结果

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for(int i = 0;i<vectorLength;i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i*2);
    }
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); //从CPU复制数据到GPU
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size); //初始化GPU内存，把空数据缓冲掉

    //计算网格与线程块，启动GPU核函数
    int threads = 256;
    int blocks = (vectorLength + threads - 1) /threads;
    //传给核函数的参数必须全是GPU的显存指针d_
    vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, vectorLength);
    cudaDeviceSynchronize(); //等待GPU完成计算
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); //从GPU复制结果回CPU
    serialVecAdd(h_A, h_B, h_comparisionResult, vectorLength); //CPU串行计算验证结果
    
    bool match = true;
    for(int i = 0;i<vectorLength;i++){
        if(abs(h_C[i]-h_comparisionResult[i]) > 1e-5){
            match = false;
            std::cout << "Mismatch at index " << i << ": GPU result " << h_C[i] << ", CPU result " << h_comparisionResult[i] << std::endl;
            break;
        }
}

if (match){
    std::cout << "Results match!" << std::endl;
}else{
    std::cout << "Results do not match!" << std::endl;
}

    //清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    free(h_comparisionResult);
}
int main(){
    explicitMemExample(1<<20);
    return 0;
}
