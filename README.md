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

| 版本 | 干了什么 | 耗时 (N=16M) | 备注 |
| ------ | --------- | ------ | ---- |
| `V0.cu` | 最简单的，全局内存 stride，非合并访问 | — | 单 block，N>2048 跑不动 |
| `V1.cu` | 调了读取顺序，合并访问 | — | 单 block |
| `V2_share.cu` | 数据搬进 shared memory 再归约 | — | 单 block |
| `V3_Blocks.cu` | 每个 block 算一部分，host 加起来 | — | 单 block |
| `V4.cu` | warp 内不用 `__syncthreads`，省同步开销 | 0.385 ms | 结果正确 |
| `V5.cu` | 模板参数展开，减少分支 | 0.550 ms | 结果有偏差，待查 |
| `V6.cu` | `#pragma unroll` 全展开 | 0.455 ms | 结果有偏差，待查 |
| `V7.cu` | block 归约 + warp shuffle，最终版 | 0.487 ms | 结果有偏差，待查 |

实测环境：RTX 5060 Laptop，N = 1<<24 = 16M，cudaEvent 计时 + Nsight Compute。

| 版本 | 耗时 | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% | 备注 |
| ------ | ------ | ------- | ----- | ------- | ------- | ---------- | ---- |
| `V0.cu` | — | — | — | — | — | — | 单 block，N>2048 跑不动 |
| `V1.cu` | — | — | — | — | — | — | 单 block |
| `V2_share.cu` | — | — | — | — | — | — | 单 block |
| `V3_Blocks.cu` | — | — | — | — | — | — | 单 block |
| `V4.cu` | 0.385 ms | 28.11 | 24.91 | 74.93 | 3.55 | 66.55 | 结果正确 |
| `V5.cu` | 0.550 ms | — | — | — | — | — | 结果有偏差，待查 |
| `V6.cu` | 0.455 ms | — | — | — | — | — | 结果有偏差，待查 |
| `V7.cu` | 0.487 ms | — | — | — | — | — | 结果有偏差，待查 |

几个结论：

1. V0–V3 是单 block 设计，threads = N/2 超过 1024 上限，kernel 静默失败，只能跑小 N。要测大 N 必须用 V4 以后的多 block 版本。
2. V4 结果正确，V5–V7 输出和 CPU 参考不一致，说明展开/shuffle 的实现里有 bug，不是性能问题是正确性问题。
3. V4 的 DRAM 只有 24.91%，L2 Hit 仅 3.55%——数据量太大（64MB）基本走 HBM，但带宽没打满，还有优化空间（比如 V5–V7 的多阶段归约）。

---

## 6. SoftMax

| 版本 | 干了什么 | 耗时 (N=8192, C=4096) |
| ------ | --------- | ------ |
| `cpu.cu` | CPU 版，max→exp→sum→normalize，三趟 | — |
| `v0.cu` | GPU naive，每行一个线程 | 5.245 ms |
| `v1.cu` | online softmax，一趟扫完 | 3.946 ms |
| `v2.cu` | warp-level online softmax | 1.240 ms |

实测环境：RTX 5060 Laptop，8192 行 × 4096 列，cudaEvent 计时 + Nsight Compute。

| 版本 | 耗时 (cudaEvent) | 耗时 (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `v0.cu` (per-thread) | 5.245 ms | 7.41 ms | 45.01 | 18.57 | 77.37 | 74.21 | 23.71 |
| `v1.cu` (online) | 3.946 ms | 6.96 ms | 41.96 | 17.59 | 73.91 | 77.72 | 23.38 |
| `v2.cu` (warp-online) | 1.240 ms | 1.04 ms | 77.51 | 77.51 | 0 | 17.77 | 84.73 |

online 就是一边找 max 一边累 sum，旧的自动打折缩水。FlashAttention 用的就是这个。

v2 把一行丢给一个 warp（32 线程），用 `__shfl_down_sync` 做 warp 内归约，比 v1 快约 6×。关键指标差异：

1. v2 的 **L1 Hit Rate = 0%**——warp shuffle 直接在寄存器间交换数据，根本不走 L1 cache，这是它快的核心原因。
2. v0/v1 的 Occupancy 只有 23%（每行一个线程，大部分 SM 闲着），v2 跑到 84.73%。
3. v2 的 DRAM 直接打到 77.51%，v0/v1 只有 18%——v2 真正吃到了带宽墙，再优化就得从访存模式入手。

---

## 7. RMSNorm

| 版本 | 干了什么 | 耗时 (B=64, T=1024, C=2048) |
| ------ | --------- | ------ |
| `CPU.cu` | CPU baseline | — |
| `V0.cu` | 每行一个线程 | 16.984 ms |
| `V1.cu` | 每行一个 block，shared memory 归约 | 26.286 ms |
| `V2.cu` | 多行共享 shared memory，分段归约 | 4.477 ms |

实测环境：RTX 5060 Laptop，B=64, T=1024, C=2048，cudaEvent 计时 + Nsight Compute。

| 版本 | 耗时 (cudaEvent) | 耗时 (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `V0.cu` (per-thread) | 16.984 ms | 22.08 ms | 71.01 | 29.39 | 26.98 | 85.09 | 96.87 |
| `V1.cu` (shared 归约) | 26.286 ms | — | — | — | — | — | — |
| `V2.cu` (多行共享) | 4.477 ms | 4.18 ms | 86.57 | 86.57 | 24.64 | 33.59 | 49.64 |

RMSNorm 每个 token 独立算 `rms = sqrt(mean(x²))`。瓶颈在归约。

V0 的 Occupancy 高达 96.87% 但最慢——因为每行一个线程，每个线程串行扫 C=2048 个元素，Warp Cycles Per Issued 高达 255.6 cycles，大部分时间在等访存。V2 用 warp 归约（一个 warp 处理一行），DRAM 直接打到 86.57%，快到 V0 的 1/5。V1 用 shared memory 归约反而比 V0 慢，同步开销盖过了访存收益。

---

## 8. 矩阵乘法 GEMM V1→V4（已跑 Nsight）

实测环境：RTX 5060 Laptop，CUDA 12.x。1024³ 用 Nsight Compute 跑的，4096³/8192³ 用 cudaEvent 计时（跳过 CPU 验证，不然太慢）。

### 1024³（Nsight Compute）

| 版本 | 干了什么 | 耗时 | vs V1 | DRAM 利用率 | SM 利用率 |
| ------ | ------ | ------ | ------- | ------------ | ---------- |
| `v1.cu` | naive，全局内存反复读 | 2.133 ms | 1.00× | 0.90% | 91.82% |
| `v2.cu` | shared memory 分块 32×32 | 1.858 ms | 1.15× | 0.89% | 84.50% |
| `v3.cu` | 寄存器分块 8×8 | 1.220 ms | 1.75× | 1.73% | 20.23% |
| `v4_double_buffer.cu` | 双缓冲，加载和计算重叠 | 1.722 ms | 1.24× | 1.22% | 82.31% |

### 4096³（cudaEvent）

| 版本 | 耗时 | vs V1 | 等效带宽 |
| ------ | ------ | ------- | -------- |
| `v1.cu` | 211.332 ms | 1.00× | 计算密集 |
| `v2.cu` | 137.458 ms | 1.54× | — |
| `v3.cu` | 50.021 ms | 4.23× | — |
| `v4_double_buffer.cu` | 111.064 ms | 1.90× | — |

### 8192³（cudaEvent）

| 版本 | 耗时 | vs V1 |
| ------ | ------ | ------- |
| `v1.cu` | 4070.164 ms | 1.00× |
| `v2.cu` | 3467.456 ms | 1.17× |
| `v3.cu` | 1676.842 ms | 2.43× |
| `v4_double_buffer.cu` | 1869.763 ms | 2.18× |

几个有意思的事：

1. 1024³ 时 DRAM 全部不到 2% —— 三个矩阵加起来 12 MB，5060 的 L2 有 32 MB，全在缓存里。所以 V2 共享内存没怎么快（只 1.15×）。
2. 4096³ 开始，V3 的优势拉大到 4.2×，因为寄存器分块真正减少了全局内存访问次数，这时候数据已经装不下 L2，带宽瓶颈开始显现。
3. 8192³ 时所有版本都进入 HBM-bound，V2 只剩 1.17×（共享内存分块对带宽帮助有限），V3 还有 2.43×，V4 双缓冲反超 V2，说明大矩阵下加载/计算重叠终于有收益。
4. 带宽拐点在 1024→4096 之间。4096³ 三个矩阵约 192 MB，远超 32 MB L2，从此所有版本都被 HBM 带宽卡住。

优化思路就是一步步减少访存：全局 → 共享 → 寄存器 → 加载和计算叠起来跑。

---

## 9. Tensor Core GEMM（wmma）

实测环境：RTX 5060 Laptop，CUDA 12.x，单精度。REPS=100 取平均，跳过 CPU 验证（4096 以上 CPU 太慢）。

### 4096³

| kernel | 耗时 | TFLOPS | vs cuBLAS |
| ------ | ------ | ------ | --------- |
| k1（naive wmma） | 16513.59 us | 8.3 | 24.1% |
| k2 | 20609.69 us | 6.7 | 19.3% |
| k3 | 19172.82 us | 7.2 | 20.7% |
| k4 | 10970.97 us | 12.5 | 36.2% |
| k5（最终版） | 6387.46 us | 21.5 | 62.2% |
| cuBLAS | 3973.23 us | 34.6 | 100.0% |

### 8192³

| kernel | 耗时 | TFLOPS | vs cuBLAS |
| ------ | ------ | ------ | --------- |
| k1 | 505922.18 us | 2.2 | 6.4% |
| k2 | 195579.59 us | 5.6 | 16.5% |
| k3 | 168555.24 us | 6.5 | 19.2% |
| k4 | 97860.35 us | 11.2 | 33.1% |
| k5 | 56073.47 us | 19.6 | 57.7% |
| cuBLAS | 32346.58 us | 34.0 | 100.0% |

结论：cuBLAS 4096→8192 从 34.6 掉到 34.0 TFLOPS，几乎不掉，说明 cuBLAS 在 8192 还没到带宽墙；而手写 k5 从 21.5 掉到 19.6，掉了约 9%，说明手写版的访存模式比 cuBLAS 差，大矩阵下被带宽拖得更狠。

---

## 还差的事

- [x] GEMM V1→V4 实测（RTX 5060 + Nsight Compute）
- [x] 4096 / 8192 大矩阵测一次，看看什么时候触发带宽瓶颈
- [x] Reduce V0→V7 跑一遍（cudaEvent + Nsight Compute）
- [x] SoftMax / RMSNorm 补实测数字 + Nsight 指标
- [x] Tensor Core GEMM
- [x] Nsight Compute 权限解锁（注册表 RmProfilingAdminOnly=0）
- [ ] Reduce V5–V7 结果不对，需要查 bug（正确性问题，不是性能问题）
- [ ] GEMM 4096³/8192³ 的 Nsight 指标（目前只有 cudaEvent 计时）
