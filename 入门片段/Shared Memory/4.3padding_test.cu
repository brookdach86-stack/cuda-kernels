// ==================== 4.2 冲突版（原代码） ====================
__global__ void maxConflictKernel(float* d_out, const float* d_in) {
    __shared__ float tile[8192]; // 原本的共享内存
    int tid = threadIdx.x;
    
    tile[tid] = d_in[tid];
    __syncthreads();

    // 💥 冲突点：步长为 32，所有人取模 32 全挤在 Bank 0
    d_out[tid] = tile[tid * 32]; 
}

// ==================== 4.3 缓冲修正版（只加/改 2 行） ====================
__global__ void fixConflictKernel(float* d_out, const float* d_in) {
    // 【修改 1】：多开一些缓冲格子（原本只够存，现在每 32 个数后面多塞 1 个空格）
    __shared__ float tile[8192 + 256]; 
    int tid = threadIdx.x;
    
    // 【修改 2】：写入时加入缓冲偏移（每到 32 的倍数，就往后多挪 1 个格子）
    int offset_idx = tid + (tid / 32); 
    tile[offset_idx] = d_in[tid];
    __syncthreads();

    // ✨ 奇迹时刻：现在读 tile[offset_idx * 32] 时，因为中间有空格缓冲
    // 每一个线程算出来的物理物理 Bank 刚好全错开了！
    d_out[tid] = tile[offset_idx * 32];