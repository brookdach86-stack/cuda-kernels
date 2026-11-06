# CUDA Kernel Implementations

Collection of handwritten CUDA kernels: from vector addition up through register spilling, shared memory bank conflicts, warp shuffles, to Reduce / SoftMax / RMSNorm / GEMM / Tensor Core WMMA. Each kernel has a V0→final evolution chain with Nsight Compute measured metrics.

Tested on RTX 5060 Laptop (sm_120), 26 SMs, 32 MB L2.

## 1. Vector Add, Grid Stride

| File | Description |
| ------ | -------- |
| `CUDA--vector_addtion/saxpy.cu` | First kernel, SAXPY |
| `CUDA--vector_addtion/vectorAdd.cu` | Vector addition |
| `CUDA.vscode/Grid_stride_Loop.cu` | Grid-Stride Loop — arrays larger than thread count still work |
| `CUDA.vscode/Index_check.cu` | How `blockIdx` and `threadIdx` map to data |
| `CUDA.vscode/Thread and Grid Indexing.cu` | 1D/2D/3D thread organization |
| `CUDA.vscode/GPU显存.cu` | `cudaMalloc`, `cudaMemcpy`, `cudaFree` |

`blockIdx × blockDim + threadIdx` — this is the formula.

---

## 2. Registers

| File | Description |
| ------ | -------- |
| `Register/register_overflow.cu` | Performance impact of register spilling to local memory |
| `Register/register_spill.cu` | Too many variables → won't fit in registers → spill → slow |
| `Register/register_opt.cu` | Use fewer variables to avoid spilling |

An SM has 65536 registers total. When `threads × registers per thread` exceeds this, spilling to local memory (in HBM) causes a performance cliff.

---

## 3. Shared Memory & Memory Access

| File | Description |
| ------ | -------- |
| `内存管理/合并访问/coalesced_access.cu` | Coalesced access — consecutive addresses read by same warp are fastest |
| `内存管理/向量化加载/main.cu` | `float4` loads 128 bits at a time |
| `内存管理/向量化加载/vector_alignment.cu` | Misalignment forces an extra load |
| `内存管理/共享内存Bank冲突优化/usaul_confict.cu` | Bank conflict: multiple threads hit same bank → serialized |
| `内存管理/内存访问优化实战/matrix_transpose_optimization.cu` | Matrix transpose — use shared memory to convert column writes to row writes |
| `内存管理/内存访问优化实战/coding_convention_optimization.cu` | Different source forms lead to very different compiler optimization |

Bank conflict happens when ≥2 threads among 32 land on the same bank address — one access becomes multiple. Two fixes: padding (add an extra column) or swizzle (XOR to scatter).

---

## 4. Warp Primitives

| File | Description |
| ------ | -------- |
| `CUDA.vscode/Warp/warp_shuffle.cu` | `__shfl_down_sync` — data exchange within a warp |
| `内存管理/.Warp级原语/Warp_match.cu` | `__match_any_sync` — find threads with same value |
| `内存管理/.Warp级原语/Warp_vote.cu` | `__any_sync / __all_sync` — warp voting |
| `内存管理/.Warp级原语/Warp_width.cu` | `__shfl_xor_sync` — butterfly exchange |
| `内存管理/.Warp级原语/__activemask()and__syncwarp().cu` | `__activemask()` + `__syncwarp()` |

Warp is the smallest CUDA execution unit — 32 threads locked together. Shuffles move data at register level, one step faster than shared memory.

---

## 5. Reduce V0→V7

| Version | What Changed | Latency (N=16M) | Notes |
| ------ | --------- | ------ | ---- |
| `V0.cu` | Simplest, global memory stride, non-coalesced | — | Single block, N>2048 fails silently |
| `V1.cu` | Fixed read order, coalesced access | — | Single block |
| `V2_share.cu` | Data moved to shared memory before reduction | — | Single block |
| `V3_Blocks.cu` | Each block reduces a partition, host sums them | — | Single block |
| `V4.cu` | No `__syncthreads` within warp — saves sync overhead | 0.545 ms | Correct result |
| `V5.cu` | Warp shuffle eliminates the last 32 steps of shared memory | 0.498 ms | Correct (bug fixed) |
| `V6.cu` | Two-stage reduction eliminates atomicAdd contention | 0.449 ms | Correct (bug fixed) |
| `V7.cu` | float4 vectorization + two-stage reduction | 0.592 ms | Correct (bug fixed) |

Tested on RTX 5060 Laptop, N = 1<<24 = 16M, cudaEvent timing + Nsight Compute.

| Version | Latency | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% | Notes |
| ------ | ------ | ------- | ----- | ------- | ------- | ---------- | ---- |
| `V0.cu` | — | — | — | — | — | — | Single block, N>2048 fails |
| `V1.cu` | — | — | — | — | — | — | Single block |
| `V2_share.cu` | — | — | — | — | — | — | Single block |
| `V3_Blocks.cu` | — | — | — | — | — | — | Single block |
| `V4.cu` | 0.545 ms | 28.11 | 24.91 | 74.93 | 3.55 | 66.55 | Correct |
| `V5.cu` | 0.498 ms | — | — | — | — | — | Correct (bug fixed) |
| `V6.cu` | 0.449 ms | — | — | — | — | — | Correct (bug fixed) |
| `V7.cu` | 0.592 ms | — | — | — | — | — | Correct (bug fixed) |

Key findings:

1. V0–V3 are single-block designs; threads = N/2 exceeds the 1024 limit, kernel fails silently, only works for small N. V4+ multi-block versions needed for large N.
2. All V5–V7 bugs fixed:
   - **V5**: All 512 threads executed warp shuffle, then `if (tid % 32 == 0) atomicAdd` — so lane 0 of all 16 warps added garbage smem values. Fix: wrap shuffle in `if (tid < 32)`, only `tid==0` does atomicAdd.
   - **V6/V7**: `reduce_final` read only the first 512 partials by tid; when block count > 512, remaining data was dropped. Fix: each thread loops `for (i = tid; i < num_blocks; i += BLOCK_SIZE)` over all partials.
3. V4 DRAM only 24.91%, L2 Hit only 3.55% — data is too large (64 MB) and mostly goes to HBM, but bandwidth isn't saturated. V6 two-stage reduction is fastest (0.449ms). V7 float4 vectorization is actually slowest — alignment overhead cancels vectorization gains in this memory-bound scenario.

---

## 6. SoftMax

| Version | What Changed | Latency (N=8192, C=4096) |
| ------ | --------- | ------ |
| `cpu.cu` | CPU baseline, max→exp→sum→normalize, three passes | — |
| `v0.cu` | GPU naive, one thread per row | 5.245 ms |
| `v1.cu` | Online softmax, single pass | 3.946 ms |
| `v2.cu` | Warp-level online softmax | 1.240 ms |

Tested on RTX 5060 Laptop, 8192 rows × 4096 columns, cudaEvent timing + Nsight Compute.

| Version | Latency (cudaEvent) | Latency (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `v0.cu` (per-thread) | 5.245 ms | 7.41 ms | 45.01 | 18.57 | 77.37 | 74.21 | 23.71 |
| `v1.cu` (online) | 3.946 ms | 6.96 ms | 41.96 | 17.59 | 73.91 | 77.72 | 23.38 |
| `v2.cu` (warp-online) | 1.240 ms | 1.04 ms | 77.51 | 77.51 | 0 | 17.77 | 84.73 |

Online means accumulating sum while finding the max — old values are automatically discounted. This is exactly what FlashAttention uses.

v2 assigns one row to one warp (32 threads), using `__shfl_down_sync` for warp-level reduction — ~6× faster than v1. Key metric differences:

1. v2 **L1 Hit Rate = 0%** — warp shuffle exchanges data directly between registers, L1 cache is bypassed entirely. This is why it's fast.
2. v0/v1 Occupancy only 23% (one thread per row, most SMs idle); v2 reaches 84.73%.
3. v2 DRAM hits 77.51%, v0/v1 only 18% — v2 is truly bandwidth-bound, further optimization must target access patterns.

---

## 7. RMSNorm

| Version | What Changed | Latency (B=64, T=1024, C=2048) |
| ------ | --------- | ------ |
| `CPU.cu` | CPU baseline | — |
| `V0.cu` | One thread per row | 16.984 ms |
| `V1.cu` | One block per row, shared memory reduction | 26.286 ms |
| `V2.cu` | Multiple rows share shared memory, segmented reduction | 4.477 ms |

Tested on RTX 5060 Laptop, B=64, T=1024, C=2048, cudaEvent timing + Nsight Compute.

| Version | Latency (cudaEvent) | Latency (ncu) | Memory% | DRAM% | L1 Hit% | L2 Hit% | Occupancy% |
| ------ | ------ | ------ | ------- | ----- | ------- | ------- | ---------- |
| `V0.cu` (per-thread) | 16.984 ms | 22.08 ms | 71.01 | 29.39 | 26.98 | 85.09 | 96.87 |
| `V1.cu` (shared reduction) | 26.286 ms | — | — | — | — | — | — |
| `V2.cu` (multi-row share) | 4.477 ms | 4.18 ms | 86.57 | 86.57 | 24.64 | 33.59 | 49.64 |

RMSNorm computes `rms = sqrt(mean(x²))` per token independently. The bottleneck is reduction.

V0 has Occupancy of 96.87% but is slowest — one thread per row serially scans C=2048 elements, Warp Cycles Per Issued reaches 255.6 cycles, mostly waiting on memory. V2 uses warp reduction (one warp per row), DRAM hits 86.57%, 5× faster than V0. V1 with shared memory reduction is actually slower than V0 — synchronization overhead outweighs the memory benefit.

---

## 8. GEMM V1→V4 (Nsight Compute measured)

Tested on RTX 5060 Laptop, CUDA 12.x. All sizes measured with Nsight Compute (ncu timing includes profiling overhead, longer than cudaEvent).

### 1024³ (Nsight Compute)

| Version | What Changed | Latency | vs V1 | vs cuBLAS FP32 | DRAM % | SM % |
| ------ | ------ | ------ | ------- | -------------- | ------------ | ---------- |
| `v1.cu` | Naive, reads global memory repeatedly | 2.133 ms | 1.00× | 18.5% | 0.90% | 91.82% |
| `v2.cu` | Shared memory tiling 32×32 | 1.858 ms | 1.15× | 21.2% | 0.89% | 84.50% |
| `v3.cu` | Register tiling 8×8 | 1.220 ms | 1.75× | 32.3% | 1.73% | 20.23% |
| `v4_double_buffer.cu` | Double buffering, overlap load and compute | 1.722 ms | 1.24× | 22.9% | 1.22% | 82.31% |
| cuBLAS FP32 | Official baseline | 0.394 ms | 5.41× | 100% | — | — |

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

### Nsight Systems system-level analysis (v3 @ 4096³)

NSYS timeline, not just single kernel:

| Phase | Latency | % | Notes |
| ------ | ------ | ---- | ---- |
| cudaMalloc | 296 ms | 54.3% | 3 allocations, 4096³ matrices total 192 MB |
| cudaEventSynchronize | 169 ms | 31.0% | Waiting for kernel to finish |
| cudaMemcpy H2D | 76 ms | 14.0% | A+B total 128 MB from host to device |
| cudaLaunchKernel | 1.4 ms | 0.3% | Kernel launch overhead is negligible |
| matmul_v3 (GPU) | 169 ms | 100% GPU | Single kernel instance, no overlap |

Observations:

1. **Kernel launch overhead is only 0.3%** — not a bottleneck, no need for CUDA Graphs.
2. **H2D copy and kernel are serialized**, no overlap. Multi-stream could overlap copy and compute in multi-batch scenarios.
3. **Kernel itself 169 ms** (ncu gives 70 ms because it only measures the kernel, not host overhead; cudaEvent gives 48.7 ms).

> cuBLAS FP32 baseline measured with `矩阵乘法/cublas_baseline_fp32.cu`, WARMUP=10, REPS=100, cudaEvent timing.

Interesting observations:

1. At 1024³, DRAM is under 2% across all versions — three matrices total 12 MB, L2 has 32 MB, everything fits in cache. So V2 shared memory doesn't speed things up much (only 1.15×).
2. At 4096³, V3's lead widens to 3.81× because register tiling truly reduces global memory access count. V3 SM is only 32.97% (much lower than V2's 81%), but it wins by reducing memory traffic — bottleneck is memory, idle SM is fine.
3. At 8192³ all versions' L2 Hit drops to 42-50% (V3 had 66.5% at 4096), data no longer fits in cache. V3 still leads at 4.05×, V2/V4 are roughly equal (1.63×). V2 and V4 metrics are nearly identical (SM~81%, DRAM~12.7%, L1=0%, L2~49.7%) — double buffering gives no extra benefit at this size.
4. V1 L1 Hit reaches 94.98% — naive implementation reads the same rows/columns repeatedly, data stays in L1. But its DRAM is only 8.27%, SM 76.84% — it saturates neither bandwidth nor compute, slow purely due to bad access patterns.
5. V2/V4 L1 Hit = 0% — after data is moved to shared memory via tiling, L1 no longer caches global memory accesses. This is normal, not a bug.

The optimization path is step-by-step memory reduction: global → shared → register → overlap load and compute.

### Why V4 is slower than V3: double buffer regression analysis

Intuitively double buffering should be faster than single buffering, but V4 (1.14 TFLOPS) is almost half of V3 (2.0 TFLOPS). NCU comparison shows the root cause isn't double buffering itself — V4's base architecture is weaker.

| Metric | V3 (register tiling) | V4 (double buffering) |
| ------ | ----------------- | ------------- |
| Tile size | 128×128 | 32×32 |
| C elements per thread | 8×8 = 64 | 1×1 = 1 |
| Block threads | 256 | 1024 |
| Shared memory | 32 KB | 16 KB |
| SM utilization | 32.97% | **81.57%** |
| Occupancy | 32.83% | **66.65%** |
| DRAM utilization | 6.15% | **13.15%** |
| L1 Hit | 14.93% | **0%** |
| L2 Hit | 66.52% | 49.39% |

**Root cause 1: tile is 4× smaller, no register tiling (primary)**

V3's 128×128 tile has each thread computing 64 C elements; shared memory block `As[128][8]` is reused by 16 row threads, `Bs[8][128]` by 16 column threads. V4's 32×32 tile has each thread computing only 1 C element; shared memory data is used once and discarded — **zero register reuse**.

Note V4 SM utilization (81.57%) is much higher than V3 (32.97%), but it's slower — V4's SM is doing inefficient compute (each FMA contributes 1 C), V3's SM has lower activity but each FMA contributes 64 C. **High SM utilization ≠ fast — measure effective output per cycle.**

**Root cause 2: V4's double buffering is "fake double buffering"**

```cpp
As[read][...] = A[...];           // load next tile
for (k) sum += As[comp][...];     // compute current tile
__syncthreads();                   // wait for all threads
```

Load and compute are **serial within the same warp** — all 32 threads either load or compute, never both simultaneously. `__syncthreads()` stalls the whole block, zero overlap. Real double buffering needs `cp.async` hardware async copy to let load and compute units truly run in parallel. V4's hand-written double buffering just doubles shared memory usage without hiding latency.

**Root cause 3: L1 Hit = 0%**

V4's small 32×32 tile means data is evicted from L1 immediately after loading; V3's 128×128 tile has 14.93% L1 hit rate.

**Conclusion**: V4 is slow not because of double buffering, but because V4 uses 32×32 small tiles + no register tiling + fake double buffering. The correct evolution path is V3 (128×128 register tiling) → add `cp.async` double buffering, not hand-written double buffering on V2 (32×32).

---

## 9. Tensor Core GEMM (WMMA)

Tested on RTX 5060 Laptop, CUDA 12.x, FP32 accumulation. REPS=100 averaged; CPU validation skipped above 4096³ (too slow).

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

cuBLAS drops from 34.6 to 34.0 TFLOPS between 4096→8192 — barely any drop, meaning cuBLAS hasn't hit bandwidth wall yet at 8192. Hand-written k5 drops from 21.5 to 19.6 (~9%), showing hand-written access patterns are worse than cuBLAS and get dragged down more at large sizes.

## 10. Fundamentals

Early indexing, memory access, memory hierarchy small examples. Formal Reduce / SoftMax / RMSNorm / GEMM / WMMA above take precedence.

| File | Description |
| --- | --- |
| `fundamentals/2D/2Dgrid+2DBlock.cu` | 2D grid for grayscale |
| `fundamentals/3D/3Dgrid+3DBlock.cu` | `blockIdx.z` as batch |
| `fundamentals/Constant Memory/Constant_memory.cu` | `__constant__` |
| `fundamentals/Shared Memory/` | bank conflict, padding, shared memory reversal |
| `fundamentals/Texture Memory/Texture_Memory.cu` | `__ldg` / `__restrict__` read-only cache |
| `fundamentals/matmul_tiled/matmul_tiled.cu` | `TILE=16` tiled matmul |
