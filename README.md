![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM%2C32MB%20L2)-brightgreen)
![NSight Verified](https://img.shields.io/badge/Profiled%20by-NSight%20Compute-blue)

---

# CUDA Kernel Implementations

单卡RTX 5060 Laptop上手写CUDA kernel——从vector add一路到Tensor Core WMMA。每个kernel都有V0→final迭代链，用Nsight Compute在每一步定位硬件瓶颈。

**Reduce V0→V7 是所有后续项目的方法论蓝图**——这里练出来的partial+reduce模式直接复用到FlashAttention backward，Reduce V5踩过的「block太少→smem overhead反而拖慢」这条rule指导了FP8 PagedAttention的scale粒度选择：

| Version | 症状 | NCU定位的根因 | 解法 |
| ------- | ---- | ------------- | ---- |
| V0 | 死慢 | 串行规约，0 SM occupancy | 改warp-parallel |
| V1 | 还是慢 | non-coalesced reads | 调整读顺序 |
| V3 | bank conflict | smem stride导致4个thread撞同一个bank | 加padding |
| V5 | BW%卡在15% | smem atomic_add 4-way bank conflict | 改两阶段规约 |
| V6 | BW%→85% | 消除atomic竞争 + register存partial | 每个warp把partial写到不同bank，再跨warp规约 |
| V7 | 伪double buffering | 两个buffer轮着用但prefetch时机错了，反而更慢 | 修正prefetch cadence |

其他坑：

- register spill到local memory直接让latency翻倍（`Register/register_spill.cu`）
- 伪cp.async double buffering——看起来轮了两个buffer但prefetch时机不对，反而比单buffer慢（`GEMM/gemm_double_buffer.cu`）
- WMMA fragment布局和文档写的不一样——K/N stride搞反了，编译过了但结果是错的，花了一天才debug出来

## 1. Vector Add & Grid Stride

| 文件 | 说明 |
| ------ | -------- |
| `CUDA--vector_addtion/saxpy.cu` | 第一个kernel，SAXPY |
| `CUDA--vector_addtion/vectorAdd.cu` | 向量加法 |
| `CUDA.vscode/Grid_stride_Loop.cu` | Grid-Stride Loop——thread数量比数组小也能跑 |
| `CUDA.vscode/Index_check.cu` | `blockIdx`和`threadIdx`怎么映射到数据 |
| `CUDA.vscode/Thread and Grid Indexing.cu` | 1D/2D/3D thread组织 |
| `CUDA.vscode/GPU显存.cu` | `cudaMalloc`、`cudaMemcpy`、`cudaFree` |

公式：`blockIdx × blockDim + threadIdx`。

---

## 2. Registers（寄存器）

| 文件 | 说明 |
| ------ | -------- |
| `Register/register_overflow.cu` | register spill到local memory的性能影响 |
| `Register/register_spill.cu` | 变量太多→放不进register→spill→变慢 |
| `Register/register_opt.cu` | 减少变量避免spill |

一个SM总共65536个register。当`threads × registers per thread`超过这个数，spill到local memory（其实在HBM）就会性能跳水。

---

## 3. Shared Memory & Memory Access

| 文件 | 说明 |
| ------ | -------- |
| `内存管理/合并访问/coalesced_access.cu` | Coalesced access——同warp读连续地址最快 |
| `内存管理/向量化加载/main.cu` | `float4`一次读128 bit |
| `内存管理/向量化加载/vector_alignment.cu` | 没对齐会多一次load |
| `内存管理/共享内存Bank冲突优化/usaul_confict.cu` | Bank conflict：多thread撞同一个bank→串行化 |
| `内存管理/内存访问优化实战/matrix_transpose_optimization.cu` | 矩阵转置——用shared memory把列写改成行写 |
| `内存管理/内存访问优化实战/coding_convention_optimization.cu` | 源码写法不同，编译器优化结果差很多 |

Bank conflict发生在≥2个thread（32个里）落到同一个bank地址——一次访问变多次。两种解法：padding（加一列）或swizzle（XOR打散）。

---

## 4. Warp Primitives

| 文件 | 说明 |
| ------ | -------- |
| `CUDA.vscode/Warp/warp_shuffle.cu` | `__shfl_down_sync`——warp内数据交换 |
| `内存管理/.Warp级原语/Warp_match.cu` | `__match_any_sync`——找值相同的thread |
| `内存管理/.Warp级原语/Warp_vote.cu` | `__any_sync / __all_sync`——warp投票 |
| `内存管理/.Warp级原语/Warp_width.cu` | `__shfl_xor_sync`——蝶形交换 |
| `内存管理/.Warp级原语/__activemask()and__syncwarp().cu` | `__activemask()` + `__syncwarp()` |

Warp是CUDA最小执行单位——32个thread锁在一起。Shuffle在register层面传数据，比shared memory快一步。

---

## 5. Reduce V0→V7

| Version | 改了什么 | Latency (N=16M) | 备注 |
| ------ | --------- | ------ | ---- |
| `V0.cu` | 最简单的global memory stride，non-coalesced | — | Single block，N>2048静默失败 |
| `V1.cu` | 调整读顺序，coalesced access | — | Single block |
| `V2_share.cu` | 数据先搬到shared memory再规约 | — | Single block |
| `V3_Blocks.cu` | 每个block规约一部分，host再加起来 | — | Single block |
| `V4.cu` | warp内不用`__syncthreads`——省同步开销 | 0.545 ms | 结果正确 |
| `V5.cu` | Warp shuffle干掉最后32步shared memory | 0.498 ms | 正确（bug修了） |
| `V6.cu` | 两阶段规约消除atomicAdd竞争 | 0.449 ms | 正确（bug修了） |
| `V7.cu` | float4向量化 + 两阶段规约 | 0.592 ms | 正确（bug修了） |

RTX 5060 Laptop上测，N = 1<<24 = 16M，cudaEvent计时 + Nsight Compute profiling。

| Version | Latency | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% | 备注 |
| ------ | ------ | ------- | ----- | ------- | ------- | ---------- | ---- |
| `V0.cu` | — | — | — | — | — | — | Single block，N>2048静默失败 |
| `V1.cu` | — | — | — | — | — | — | Single block |
| `V2_share.cu` | — | — | — | — | — | — | Single block |
| `V3_Blocks.cu` | — | — | — | — | — | — | Single block |
| `V4.cu` | 0.545 ms | 28.11 | 24.91 | 74.93 | 3.55 | 66.55 | 正确 |
| `V5.cu` | 0.498 ms | — | — | — | — | — | 正确（bug修了） |
| `V6.cu` | 0.449 ms | — | — | — | — | — | 正确（bug修了） |
| `V7.cu` | 0.592 ms | — | — | — | — | — | 正确（bug修了） |

**关键发现**：

1. V0–V3都是single-block设计；N/2超过1024 limit时kernel静默失败，只能跑小N。V4+改成multi-block才能跑大N。
2. V5–V7都有bug并修了：
   - **V5**：512个thread全执行warp shuffle然后`if (tid % 32 == 0) atomicAdd`——所有16个warp的lane 0都加了垃圾smem值。修法：shuffle包在`if (tid < 32)`里，只让`tid==0`做atomicAdd
   - **V6/V7**：`reduce_final`只read前512个partial；block count>512时后面的数据被丢了。修法：每个thread `for (i = tid; i < num_blocks; i += BLOCK_SIZE)` 遍历所有partial
3. V4 DRAM只有24.91%，L2 Hit只有3.55%——数据太大（64 MB）主要走HBM但带宽没饱和。V6两阶段规约最快（0.449ms）。V7用float4向量化反而最慢——alignment overhead抵消了向量化收益（memory-bound场景）。

---

## 6. SoftMax

| Version | 改了什么 | Latency (N=8192, C=4096) |
| ------ | --------- | ------ |
| `cpu.cu` | CPU baseline，max→exp→sum→normalize三遍扫描 | — |
| `v0.cu` | GPU naive，一个thread一行 | 5.245 ms |
| `v1.cu` | Online softmax，单趟扫描 | 3.946 ms |
| `v2.cu` | Warp级online softmax | 1.240 ms |

RTX 5060 Laptop上测，8192行 × 4096列，cudaEvent计时 + Nsight Compute。

| Version | Latency (cudaEvent) | Latency (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `v0.cu` (per-thread) | 5.245 ms | 7.41 ms | 45.01 | 18.57 | 77.37 | 74.21 | 23.71 |
| `v1.cu` (online) | 3.946 ms | 6.96 ms | 41.96 | 17.59 | 73.91 | 77.72 | 23.38 |
| `v2.cu` (warp-online) | 1.240 ms | 1.04 ms | 77.51 | 77.51 | 0 | 17.77 | 84.73 |

Online就是一边找max一边累加sum——旧值自动打折。FlashAttention用的就是这个思路。

v2给每行分配一个warp（32 thread），用`__shfl_down_sync`做warp级reduction——比v1快了约6倍。关键指标差异：

1. v2 **L1 Hit Rate = 0%**——warp shuffle直接在register里传数据，L1 cache完全被绕过。这就是它快的原因。
2. v0/v1 Occupancy只有23%（一个thread一行，大部分SM闲着）；v2到了84.73%。
3. v2 DRAM到了77.51%，v0/v1只有18%——v2真正是带宽受限，再优化得从访问模式入手。

---

## 7. RMSNorm

| Version | 改了什么 | Latency (B=64, T=1024, C=2048) |
| ------ | --------- | ------ |
| `CPU.cu` | CPU baseline | — |
| `V0.cu` | 一个thread一行 | 16.984 ms |
| `V1.cu` | 一个block一行，shared memory规约 | 26.286 ms |
| `V2.cu` | 多行共享shared memory，分段规约 | 4.477 ms |

RTX 5060 Laptop上测，B=64, T=1024, C=2048，cudaEvent计时 + Nsight Compute。

| Version | Latency (cudaEvent) | Latency (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `V0.cu` (per-thread) | 16.984 ms | 22.08 ms | 71.01 | 29.39 | 26.98 | 85.09 | 96.87 |
| `V1.cu` (shared reduction) | 26.286 ms | — | — | — | — | — | — |
| `V2.cu` (multi-row share) | 4.477 ms | 4.18 ms | 86.57 | 86.57 | 24.64 | 33.59 | 49.64 |

RMSNorm对每个token独立算`rms = sqrt(mean(x²))`。瓶颈在reduction。

V0 Occupancy 96.87%但最慢——一个thread一行串行扫描C=2048个元素，Warp Cycles Per Issued到255.6 cycles，大部分时间等内存。V2用warp reduction（一个warp一行），DRAM到86.57%，比V0快5倍。V1用shared memory规约反而比V0慢——同步开销抵消了内存访问的收益。

---

## 8. GEMM V1→V4（Nsight Compute实测）

RTX 5060 Laptop，CUDA 12.x。所有size都用Nsight Compute测（ncu计时包含profiling overhead，比cudaEvent长）。

### 1024³ (Nsight Compute)

| Version | 改了什么 | Latency | vs V1 | vs cuBLAS FP32 | DRAM % | SM % |
| ------ | ------ | ------ | ------- | -------------- | ------------ | ---------- |
| `v1.cu` | Naive，反复读global memory | 2.133 ms | 1.00× | 18.5% | 0.90% | 91.82% |
| `v2.cu` | Shared memory tiling, 32×32 | 1.858 ms | 1.15× | 21.2% | 0.89% | 84.50% |
| `v3.cu` | Register tiling, 8×8 | 1.220 ms | 1.75× | 32.3% | 1.73% | 20.23% |
| `v4_double_buffer.cu` | Double buffering，load和compute重叠 | 1.722 ms | 1.24× | 22.9% | 1.22% | 82.31% |
| cuBLAS FP32 | 官方baseline | 0.394 ms | 5.41× | 100% | — | — |

### 4096³ (Nsight Compute)

| Version | cudaEvent | ncu Latency | vs cuBLAS FP32 | SM% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | --------- | -------- | -------------- | --- | ----- | ------- | ------- | ---------- |
| `v1.cu` | 185.5 ms | 275.8 ms | 12.4% | 76.84 | 8.27 | 94.98 | 49.36 | 66.66 |
| `v2.cu` | 113.7 ms | 179.6 ms | 20.2% | 81.12 | 12.69 | 0 | 49.39 | 66.66 |
| `v3.cu` | 48.7 ms | 70.1 ms | 47.3% | 32.97 | 6.15 | 14.93 | 66.52 | 32.83 |
| `v4_double_buffer.cu` | 115.1 ms | 178.6 ms | 20.0% | 81.57 | 13.15 | 0 | 49.39 | 66.65 |
| cuBLAS FP32 | 23.0 ms | — | 100% | — | — | — | — | — |

### 8192³ (Nsight Compute)

| Version | cudaEvent | ncu Latency | vs cuBLAS FP32 | SM% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | --------- | -------- | -------------- | --- | ----- | ------- | ------- | ---------- |
| `v1.cu` | 1547.6 ms | 2200.2 ms | 18.9% | 77.04 | 8.23 | 94.99 | 49.65 | 66.66 |
| `v2.cu` | 951.6 ms | 1432.9 ms | 30.8% | 81.33 | 12.62 | 0 | 49.67 | 66.66 |
| `v3.cu` | 382.0 ms | 549.9 ms | 76.7% | 33.63 | 8.57 | 19.05 | 42.32 | 33.23 |
| `v4_double_buffer.cu` | 961.8 ms | 1424.8 ms | 30.5% | 81.81 | 12.69 | 0 | 49.67 | 66.66 |
| cuBLAS FP32 | 293.0 ms | — | 100% | — | — | — | — | — |

### Nsight Systems系统级分析（v3 @ 4096³）

NSYS timeline，不是单kernel视角：

| Phase | Latency | % | 备注 |
| ------ | ------ | ---- | ---- |
| cudaMalloc | 296 ms | 54.3% | 3次分配4096³矩阵，总共192 MB |
| cudaEventSynchronize | 169 ms | 31.0% | 等kernel跑完 |
| cudaMemcpy H2D | 76 ms | 14.0% | A+B总共128 MB从host搬到device |
| cudaLaunchKernel | 1.4 ms | 0.3% | Kernel launch开销可以忽略 |
| matmul_v3 (GPU) | 169 ms | 100% GPU | 单个kernel实例，没有overlap |

观察：

1. **Kernel launch开销只有0.3%**——不是瓶颈，不需要CUDA Graphs。
2. **H2D copy和kernel是串行的**——没有overlap。多stream可以在多batch场景下overlap copy和compute。
3. **Kernel本身169 ms**（ncu给70 ms因为它只测kernel不测host overhead；cudaEvent给48.7 ms）。

> cuBLAS FP32 baseline用`矩阵乘法/cublas_baseline_fp32.cu`测，WARMUP=10, REPS=100，cudaEvent计时。

更多观察：

1. 1024³时所有版本DRAM都不到2%——三个矩阵总共12 MB，L2有32 MB，全放cache里了。所以V2 shared memory提速不明显（只有1.15×）。
2. 4096³时V3的领先扩大到3.81×——register tiling真正减少了global memory访问次数。V3 SM只有32.97%（远低于V2的81%）但靠减少内存流量赢了——瓶颈在内存，SM闲着没事。
3. 8192³时所有版本L2 Hit降到42-50%（V3在4096时还有66.5%）——数据放不进cache了。V3还是领先4.05×，V2/V4差不多（1.63×）。V2和V4指标几乎一样（SM~81%, DRAM~12.7%, L1=0%, L2~49.7%）——这个size下double buffering没额外收益。
4. V1 L1 Hit到94.98%——naive实现反复读相同的行/列，数据留在L1里。但它DRAM只有8.27%，SM 76.84%——既没榨带宽也没榨算力，慢纯粹是访问模式烂。
5. V2/V4 L1 Hit = 0%——数据搬到shared memory做tiling后，L1不再cache global memory访问。这是正常的，不是bug。

优化路径是一步步减内存：global → shared → register → load和compute overlap。

### 为什么V4比V3慢——double buffering回归分析

直觉上double buffering应该比单buffer快，但V4（1.14 TFLOPS）差不多是V3（2.0 TFLOPS）的一半。NCU对比发现根因不是double buffering本身——V4的基础架构更弱。

| 指标 | V3 (register tiling) | V4 (double buffering) |
| ------ | ----------------- | ------------- |
| Tile size | 128×128 | 32×32 |
| 每thread算C元素 | 8×8 = 64 | 1×1 = 1 |
| Block threads | 256 | 1024 |
| Shared memory | 32 KB | 16 KB |
| SM utilization | 32.97% | **81.57%** |
| Occupancy | 32.83% | **66.65%** |
| DRAM utilization | 6.15% | **13.15%** |
| L1 Hit | 14.93% | **0%** |
| L2 Hit | 66.52% | 49.39% |

**根因1：tile小了4倍，没有register tiling（主要原因）**
V3的128×128 tile每个thread算64个C元素；shared memory block `As[128][8]`被16个行thread复用，`Bs[8][128]`被16个列thread复用。V4的32×32 tile每个thread只算1个C元素；shared memory数据用一次就丢——**零register reuse**。

注意V4 SM utilization（81.57%）比V3（32.97%）高得多但它更慢——V4的SM在做低效计算（每个FMA贡献1个C），V3的SM activity低但每个FMA贡献64个C。**高SM utilization ≠ 快——看每cycle的有效输出。**

**根因2：V4的double buffering是「伪double buffering」**

```cpp
As[read][...] = A[...];           // load下一tile
for (k) sum += As[comp][...];     // 算当前tile
__syncthreads();                   // 等所有thread
```

Load和compute在**同一个warp内是串行的**——32个thread要么load要么compute，永远不会同时做。`__syncthreads()`把整个block卡住，零overlap。真正的double buffering需要`cp.async`硬件异步copy让load和compute单元真的并行。V4手写的double buffering只是多占了shared memory，没hide latency。

**根因3：L1 Hit = 0%**
V4的小32×32 tile导致数据load完立刻从L1被evict；V3的128×128 tile有14.93% L1 hit rate。

**结论**：V4慢不是因为double buffering，而是因为V4用了32×32小tile + 没有register tiling + 伪double buffering。正确的进化路径是V3（128×128 register tiling）→ 加`cp.async` double buffering，不是在V2（32×32）上写手搓double buffering。

---

## 9. Tensor Core GEMM (WMMA)

RTX 5060 Laptop，CUDA 12.x，FP32累加。REPS=100取平均；4096³以上跳过CPU验证（太慢）。

### 4096³

| Kernel | Latency | TFLOPS | vs cuBLAS |
| ------ | ------ | ------ | --------- |
| k1 (naive wmma) | 16513.59 us | 8.3 | 24.1% |
| k2 | 20609.69 us | 6.7 | 19.3% |
| k3 | 19172.82 us | 7.2 | 20.7% |
| k4 | 10970.97 us | 12.5 | 36.2% |
| k5 (final) | 6387.46 us | 21.5 | 62.2% |
| cuBLAS | 3973.23 us | 34.6 | 100.0% |

### 8192³

| Kernel | Latency | TFLOPS | vs cuBLAS |
| ------ | ------ | ------ | --------- |
| k1 | 505922.18 us | 2.2 | 6.4% |
| k2 | 195579.59 us | 5.6 | 16.5% |
| k3 | 168555.24 us | 6.5 | 19.2% |
| k4 | 97860.35 us | 11.2 | 33.1% |
| k5 | 56073.47 us | 19.6 | 57.7% |
| cuBLAS | 32346.58 us | 34.0 | 100.0% |

cuBLAS在4096→8192之间只从34.6降到34.0 TFLOPS——几乎没降，说明cuBLAS在8192还没撞带宽墙。手写k5从21.5降到19.6（约9%），说明手写的访问模式不如cuBLAS，大size时被拖得更多。

---

## 10. Fundamentals

早期的索引、内存访问、内存层级的小例子。上面正规的Reduce / SoftMax / RMSNorm / GEMM / WMMA优先看。

| 文件 | 说明 |
| --- | --- |
| `fundamentals/2D/2Dgrid+2DBlock.cu` | 2D grid，灰度图 |
| `fundamentals/3D/3Dgrid+3DBlock.cu` | `blockIdx.z`当batch |
| `fundamentals/Constant Memory/Constant_memory.cu` | `__constant__` |
| `fundamentals/Shared Memory/` | bank conflict、padding、shared memory反转 |
| `fundamentals/Texture Memory/Texture_Memory.cu` | `__ldg` / `__restrict__` 只读cache |
| `fundamentals/matmul_tiled/matmul_tiled.cu` | `TILE=16` tiled matmul |
