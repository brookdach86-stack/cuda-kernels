# CUDA 算子开发学习笔记

> 从零手写 CUDA kernel，逐版优化的完整学习路径。
> 每个算子从 CPU/Naive GPU 起步，逐步引入 shared memory、warp shuffle、双缓冲等优化。

---

## 学习路线

```
向量加 → Grid/Block 索引 → 寄存器 → 共享内存 →
Bank 冲突 → 合并访问 → 向量化加载 →
Warp 原语 → Warp 规约 → Reduce 全流程 →
SoftMax (online) → RMSNorm →
矩阵乘 (naive → tile → register block → 双缓冲)
```

---

## 1. CUDA 基础：向量加、Grid Stride

| 文件 | 学什么 |
|------|--------|
| `CUDA--vector_addtion/saxpy.cu` | 第一个 CUDA kernel（SAXPY） |
| `CUDA--vector_addtion/vectorAdd.cu` | 向量加法 baseline |
| `CUDA.vscode/Grid_stride_Loop.cu` | Grid-Stride Loop：任意大小数组 |
| `CUDA.vscode/Index_check.cu` | blockIdx/threadIdx 索引计算 |
| `CUDA.vscode/Thread and Grid Indexing.cu` | 1D/2D/3D 线程组织 |
| `CUDA.vscode/GPU显存.cu` | 显存分配/拷贝/释放流程 |

**关键收获：** 理解 `blockIdx × blockDim + threadIdx` 的索引映射，Grid-Stride Loop 让一个 kernel 处理任意规模数据。

---

## 2. 寄存器与指令级优化

| 文件 | 学什么 |
|------|--------|
| `Register/register_demo.cu` | 寄存器 vs 局部内存 |
| `Register/register_spill.cu` | 寄存器溢出：变量太多 → 自动掉到 local memory |
| `Register/register_opt.cu` | 减少寄存器压力，避免 spill |

**关键收获：** 每个 SM 的寄存器有限（65536），线程数 × 每线程寄存器 = 总占用。寄存器溢出 → 性能暴跌。

---

## 3. 共享内存与矩阵访存优化

| 文件 | 学什么 |
|------|--------|
| `内存管理/合并访问/coalesced_access.cu` | 合并访存：对齐、连续访问最速 |
| `内存管理/向量化加载/main.cu` | float4 向量化加载：一次读 128bit |
| `内存管理/向量化加载/vector_alignment.cu` | 对齐对向量化加载的影响 |
| `内存管理/共享内存Bank冲突优化/usaul_confict.cu` | Bank 冲突：多线程同时访问同一 bank → 串行化 |
| `内存管理/内存访问优化实战/matrix_transpose_optimization.cu` | 转置优化：用 shared memory 解决非合并写入 |
| `内存管理/内存访问优化实战/coding_convention_optimization.cu` | 代码习惯对编译优化的影响 |

**关键收获：**
- 全局内存 → 合并访问提速 10×
- 共享内存 32 个 bank，同 bank 冲突 → 串行
- 转置写回全局时，用 shared memory 中转，把"列写"变"行写"

---

## 4. Warp 原语：线程束级通信

| 文件 | 学什么 |
|------|--------|
| `CUDA.vscode/Warp/warp_shuffle.cu` | `__shfl_down_sync`：warp 内数据搬运 |
| `内存管理/.Warp级原语/Warp_match.cu` | `__match_any_sync` |
| `内存管理/.Warp级原语/Warp_vote.cu` | `__any_sync/__all_sync`：warp 投票 |
| `内存管理/.Warp级原语/Warp_width.cu` | `__shfl_xor_sync`：蝴蝶交换 |
| `内存管理/.Warp级原语/__activemask()and__syncwarp().cu` | activemask + syncwarp 同步 |

**关键收获：** Warp（32 线程）是 CUDA 的最小执行单元。`__shfl_down_sync` 在寄存器层面交换数据，比 shared memory 更快。

---

## 5. Reduce 规约 V0→V7

> 完整优化链条：每版解决上一个版本的瓶颈。

| 版本 | 优化 | 关键技巧 |
|------|------|---------|
| `V0.cu` | Naive 全局内存 reduce | stride loop，但非合并访问 |
| `V1.cu` | 连续访问 reduce | 修改变量读取顺序 |
| `V2_share.cu` | Shared memory reduce | 数据先搬到 shared memory，在片内归约 |
| `V3_Blocks.cu` | 多 Block 规约 | 每个 block 算部分和，host 归约 |
| `V4.cu` | 消除 warp 内同步 | 前半用 `__syncthreads`，warp 内不用 |
| `V5.cu` | 模板参数展开 | template 减少分支 |
| `V6.cu` | 完全循环展开 | `#pragma unroll` |
| `V7.cu` | 两阶段归约 + warp shuffle | block 级归约 + warp 级 `__shfl_down_sync` |

**性能演进预估：**
```
V0:  1x     (全局内存 stride)
V2:  ~8x    (shared memory)
V3:  ~3x    (多 block 并行)
V7:  ~2x    (warp shuffle + 循环展开)
────────────────
V0→V7: ~50x
```

---

## 6. SoftMax：Online Softmax

| 版本 | 优化 |
|------|------|
| `cpu.cu` | CPU baseline：三趟扫描（max→sum→normalize） |
| `v0.cu` | Naive GPU：每行一个线程，全局内存来回读 |
| `v1.cu` | Shared memory：一行数据搬进 shared memory 再算 |
| `v2.cu` | Warp-level online softmax：一个 warp 处理一行 |

**关键收获：** SoftMax 需要 max→exp→sum→normalize 四步，online 算法一边做 max 一边累积 exp，一趟扫完。这正是 FlashAttention 的数学基础。

---

## 7. RMSNorm

| 版本 | 优化 |
|------|------|
| `CPU.cu` | CPU baseline |
| `V0.cu` | Naive GPU：每行一个线程 |
| `V1.cu` | 每行一个 block，shared memory 归约 |
| `V2.cu` | 多行共用一个 block 的 shared memory，分 factor 段归约 |

**关键收获：** RMSNorm = 每个 token 独立归一化。计算 `rms=sqrt(mean(x²))`，warp/block 级归约是性能关键。

---

## 8. 矩阵乘法 GEMM V0→V4

> 最完整的优化链条。

| 版本 | 优化 | 关键技巧 |
|------|------|---------|
| `v0_cpu.cu` | CPU baseline | 三重循环 O(N³) |
| `v1.cu` | Naive GPU | 每线程算一个 C 元素，全局内存反复读 |
| `v2.cu` | Shared memory 分块 | TILE_SIZE=32，A/B 分块搬进 shared memory |
| `v3.cu` | 寄存器分块 | TM×TN=8×8，每个线程算 64 个 C 元素 |
| `v4_double_buffer.cu` | 双缓冲 | 两套 shared memory 轮流用，加载与计算流水线重叠 |

**性能演进预估：**
```
v0 (CPU):  1x
v1 (GPU):  ~50x    (几千线程并行)
v2:        ~5x     (shared memory 消全局访存)
v3:        ~3x     (寄存器分块，减少 shared memory 访问)
v4:        ~1.3x   (双缓冲隐藏加载延迟)
────────────────
v0→v4: ~1000x
```

**核心思想：**
- V1→V2：把重复的全局内存读变成 shared memory 读写（带宽 ×10）
- V2→V3：每个线程多算几个元素，复用寄存器中的数据
- V3→V4：加载和计算流水线重叠，GPU 计算单元不空等

---

## TODO

- [ ] 各模块实测性能数据（A100/H100 benchmark）
- [ ] 补充 PyTorch C++ 扩展示例
- [ ] Tensor Core 版本 GEMM
- [ ] FlashAttention 简易 CUDA 实现

---

## 环境

- CUDA Toolkit 12.x
- Visual Studio 2022
- NVIDIA GPU (compute capability ≥ 7.0)

## 参考

- 《CUDA C++ Programming Guide》
- 《Professional CUDA C Programming》
- CSAPP Cache Lab
- OpenAI Triton Tutorial