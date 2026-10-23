# CUDA 学习笔记

从零开始手写 CUDA kernel。顺序是这样的：先搞懂怎么起一个 kernel（向量加），再搞懂 GPU 内存怎么用（寄存器→共享内存→全局内存），然后学 warp 怎么通信，最后把所有东西揉进几个大 kernel——Reduce、SoftMax、RMSNorm、矩阵乘。

---

## 学了什么

- 向量加 → Grid/Block 的索引到底怎么算
- 寄存器满了会溢出，溢出掉到 local memory，性能直接寄
- 共享内存 32 个 bank，线程抢同个 bank 就串行等（bank conflict），可以用 swizzle 或者 padding 解决
- 全局内存要合并访问，一次读 128bit（float4）最快
- Warp 是 32 个线程捆一起跑的最小单位，`__shfl_down_sync` 在寄存器层面交换数据
- Reduce 从 V0 写到 V7：shared memory → 多 block → warp shuffle → 循环展开
- SoftMax 不用跑三趟，online 算法一边找 max 一边累 exp，一趟搞定，这就是 FlashAttention 的底层
- RMSNorm：每个 token 独立归一化，关键是 warp 级归约
- 矩阵乘 GEMM V0→V4，从 naive 一路优化到双缓冲，跑了 Nsight Compute 看数

---

## 1. 向量加、Grid Stride

| 文件 | 干嘛的 |
| ------ | -------- |
| `CUDA--vector_addtion/saxpy.cu` | 第一个 kernel，SAXPY |
| `CUDA--vector_addtion/vectorAdd.cu` | 向量加法 |
| `CUDA.vscode/Grid_stride_Loop.cu` | Grid-Stride Loop，数组比线程多也能跑 |
| `CUDA.vscode/Index_check.cu` | blockIdx、threadIdx 怎么映射到数据 |
| `CUDA.vscode/Thread and Grid Indexing.cu` | 1D/2D/3D 线程组织 |
| `CUDA.vscode/GPU显存.cu` | cudaMalloc、cudaMemcpy、cudaFree |

`blockIdx × blockDim + threadIdx`，就记住这一条。

---

## 2. 寄存器

| 文件 | 干嘛的 |
| ------ | -------- |
| `Register/register_demo.cu` | 寄存器和 local memory 的区别 |
| `Register/register_spill.cu` | 变量用太多 → 寄存器放不下 → 溢出 → 慢 |
| `Register/register_opt.cu` | 少用几个变量，别让它溢出 |

一个 SM 总共 65536 个寄存器，线程数 × 每线程寄存器数超过这个数就溢出到 local memory（在显存里），性能断崖。

---

## 3. 共享内存 & 访存

| 文件 | 干嘛的 |
| ------ | -------- |
| `内存管理/合并访问/coalesced_access.cu` | 同一个 warp 的线程读连续地址最快 |
| `内存管理/向量化加载/main.cu` | float4 一次读 128bit |
| `内存管理/向量化加载/vector_alignment.cu` | 不对齐会多读一次 |
| `内存管理/共享内存Bank冲突优化/usaul_confict.cu` | bank conflict：多线程撞同一 bank → 排队 |
| `内存管理/内存访问优化实战/matrix_transpose_optimization.cu` | 矩阵转置，用 shared memory 把列写变行写 |
| `内存管理/内存访问优化实战/coding_convention_optimization.cu` | 写法不同，编译器优化效果差很多 |

bank conflict 就是 32 个线程里有 ≥2 个线程落到同一个 bank 地址，一次访问变成多次。解决就两种：padding（多占一列）或 swizzle（XOR 打散）。

---

## 4. Warp 原语

| 文件 | 干嘛的 |
| ------ | -------- |
| `CUDA.vscode/Warp/warp_shuffle.cu` | `__shfl_down_sync`，warp 内传数据 |
| `内存管理/.Warp级原语/Warp_match.cu` | `__match_any_sync`，找相同值的线程 |
| `内存管理/.Warp级原语/Warp_vote.cu` | `__any_sync / __all_sync`，warp 投票 |
| `内存管理/.Warp级原语/Warp_width.cu` | `__shfl_xor_sync`，蝴蝶交换 |
| `内存管理/.Warp级原语/__activemask()and__syncwarp().cu` | activemask + syncwarp |

warp 是 CUDA 最小的执行单元，32 个线程锁死。shuffle 在寄存器层面搬数据，比 shared memory 快一个档。

---

## 5. Reduce 规约 V0→V7

| 版本 | 干了什么 |
| ------ | --------- |
| `V0.cu` | 最简单的，全局内存 stride，非合并访问 |
| `V1.cu` | 调了读取顺序，合并访问 |
| `V2_share.cu` | 数据搬进 shared memory 再归约 |
| `V3_Blocks.cu` | 每个 block 算一部分，host 加起来 |
| `V4.cu` | warp 内不用 `__syncthreads`，省同步开销 |
| `V5.cu` | 模板参数展开，减少分支 |
| `V6.cu` | `#pragma unroll` 全展开 |
| `V7.cu` | block 归约 + warp shuffle，最终版 |

还没跑 Nsight，暂时预估：V0→V2 ~8x，V2→V3 ~3x，V3→V7 ~2x。

---

## 6. SoftMax

| 版本 | 干了什么 |
| ------ | --------- |
| `cpu.cu` | CPU 版，max→exp→sum→normalize，三趟 |
| `v0.cu` | GPU naive，每行一个线程 |
| `v1.cu` | 一行搬进 shared memory |
| `v2.cu` | online softmax，一个 warp 一趟扫完 |

online 就是一边找 max 一边累 sum，旧的自动打折缩水。FlashAttention 用的就是这个。

---

## 7. RMSNorm

| 版本 | 干了什么 |
| ------ | --------- |
| `CPU.cu` | CPU baseline |
| `V0.cu` | 每行一个线程 |
| `V1.cu` | 每行一个 block，shared memory 归约 |
| `V2.cu` | 多行共享 shared memory，分段归约 |

RMSNorm 每个 token 独立算 `rms = sqrt(mean(x²))`。瓶颈在归约。

---

## 8. 矩阵乘法 GEMM V0→V4（已跑 Nsight）

实测环境：RTX 5060 Laptop，1024×1024×1024，CUDA 12.x。

| 版本 | 干了什么 | 耗时 | vs V1 | DRAM 利用率 | SM 利用率 |
| ------ | ------ | ------ | ------- | ------------ | ---------- |
| `v0_cpu.cu` | CPU 三重循环 | — | — | — | — |
| `v1.cu` | naive，全局内存反复读 | 2.133 ms | 1.00× | 0.90% | 91.82% |
| `v2.cu` | shared memory 分块 32×32 | 1.858 ms | 1.15× | 0.89% | 84.50% |
| `v3.cu` | 寄存器分块 8×8 | 1.220 ms | 1.75× | 1.73% | 20.23% |
| `v4_double_buffer.cu` | 双缓冲，加载和计算重叠 | 1.722 ms | 1.24× | 1.22% | 82.31% |

几个有意思的事：

1. DRAM 全部不到 2% —— 1024×1024 三个矩阵加起来 12 MB，5060 的 L2 有 32 MB，全在缓存里。所以 V2 共享内存没怎么快（只 1.15x）。要真的触发带宽瓶颈得开到 4096 以上。

2. V3 的 SM 只有 20% 但是最快。因为寄存器分块把 block 线程数压到 256，大部分 SM 闲着，但每个线程一次算 64 个结果，效率很高。用 1/5 的算力跑赢了 V2。

3. V4 双缓冲在小矩阵上效果一般，加载延迟本来就短。矩阵大到 8192 以上才能看到 ~1.3x 的提升。

优化思路就是一步步减少访存：全局 → 共享 → 寄存器 → 加载和计算叠起来跑。

---

## 还差的事

- [x] GEMM V1→V4 实测（RTX 5060 + Nsight Compute）
- [ ] 4096 / 8192 大矩阵测一次，看看什么时候触发带宽瓶颈
- [ ] Reduce V0→V7 也跑一遍 Nsight
- [ ] SoftMax / RMSNorm 补实测数字
- [ ] Tensor Core GEMM
