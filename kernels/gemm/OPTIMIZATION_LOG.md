# NVFP4 GEMM Optimization Log

## Target
Optimize block-scaled FP4 GEMM kernel for NVIDIA B200 (Blackwell).

## Baseline Profile Analysis

NCU profiling of the baseline kernel revealed several critical bottlenecks:

| Issue | Value | Impact |
|-------|-------|--------|
| Grid Size | 48-56 blocks for 148 SMs | Only 0.13 waves/SM |
| Memory Stalls | 88.4% of cycles waiting on L1TEX | Warps starved for data |
| Occupancy | 6.23% achieved (18.75% theoretical) | Limited by registers (136) & shared memory |
| Pipelining | `num_ab_stage = 1` | No latency hiding |
| IPC | 0.04 executed | 99% of cycles have no eligible warps |

**Root Cause**: With only 1 pipeline stage, the kernel issues a TMA load and immediately waits for it to complete before computing. There's no overlap between memory loads and computation.

---

## Optimization 1: Double Buffering (num_ab_stage = 2)

### Change
```python
# Before
num_ab_stage = 1

# After  
num_ab_stage = 2
```

### Implementation
Restructured the mainloop for proper software pipelining:

1. **Prologue**: Pre-load first `num_ab_stage` (2) tiles into shared memory buffers
2. **Main loop**: 
   - Wait for current tile data
   - Compute on current tile
   - Release buffer
   - Issue async TMA load for next tile (overlaps with next iteration's wait)

```python
# Prologue: Fill the pipeline
for prologue_k in cutlass.range_constexpr(num_ab_stage):
    if prologue_k < k_tile_cnt:
        ab_empty = ab_producer.acquire_and_advance()
        cute.copy(tma_atom_a, tAgA[(None, prologue_k)], tAsA[(None, ab_empty.index)], ...)
        # ... load B, SFA, SFB

# Main loop: Overlap load with compute
for k_tile in range(k_tile_cnt):
    ab_full = ab_consumer.wait_and_advance()
    
    # Copy scale factors to TMEM, compute GEMM
    ...
    
    ab_full.release()  # Release buffer first
    
    # Issue next load (async, overlaps with next iteration)
    next_k = k_tile + num_ab_stage
    if next_k < k_tile_cnt:
        ab_empty = ab_producer.acquire_and_advance()
        cute.copy(...)  # Async TMA load
```

### Results

| Benchmark | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| M=128, N=7168, K=16384 | 96.8 µs | 64.9 µs | **33%** |
| M=128, N=4096, K=7168 | 51.6 µs | 38.3 µs | **26%** |
| M=128, N=7168, K=2048 | 37.6 µs | 32.9 µs | **12.5%** |

### Why It Works
- With 2 buffers, while computing on tile K, tile K+1's data is already loaded
- Reduces memory stall time by pre-fetching next tile
- Larger K dimensions benefit more (more tiles = more pipelining opportunities)

---

## Failed Optimizations

### Attempt: Smaller M Tile (128 → 64)
**Goal**: Double grid size for better SM utilization

**Result**: ❌ `OpError: expects the M-mode to be 128, but got 64`

**Reason**: `MmaMXF4NVF4Op` instruction requires M=128.

### Attempt: Smaller N Tile (128 → 64)  
**Goal**: Double grid size for better SM utilization

**Result**: ❌ TMA shape mismatch error for scale factor B

**Reason**: Block-scaled GEMM requires N=128 for proper scale factor TMA layout.

---

## Speed of Light Analysis

From task.yml, the theoretical speed of light times are:
```
M=128, N=7168, K=16384: 8.994 µs
M=128, N=4096, K=7168:  2.354 µs
M=128, N=7168, K=2048:  1.333 µs
```

Current performance vs SOL:
| Benchmark | Current | SOL | Gap |
|-----------|---------|-----|-----|
| K=16384 | 64.9 µs | 8.99 µs | 7.2x |
| K=7168 | 38.3 µs | 2.35 µs | 16.3x |
| K=2048 | 32.9 µs | 1.33 µs | 24.7x |

---

## Optimization 2: Triple Buffering (num_ab_stage = 3)

### Change
```python
# Before
num_ab_stage = 2

# After  
num_ab_stage = 3
```

### Results
| Benchmark | Double Buffer | Triple Buffer | Speedup |
|-----------|---------------|---------------|---------|
| M=128, N=7168, K=16384 | 64.9 µs | 54.4 µs | **16%** |
| M=128, N=4096, K=7168 | 38.3 µs | 33.4 µs | **13%** |
| M=128, N=7168, K=2048 | 32.9 µs | 31.9 µs | **3%** |

---

## Optimization 3: Quad Buffering (num_ab_stage = 4)

### Results
| Benchmark | Triple Buffer | Quad Buffer | Speedup |
|-----------|---------------|-------------|---------|
| M=128, N=7168, K=16384 | 54.4 µs | 50.1 µs | **8%** |
| M=128, N=4096, K=7168 | 33.4 µs | 31.1 µs | **7%** |
| M=128, N=7168, K=2048 | 31.9 µs | 31.4 µs | **2%** |

---

## Optimization 4: 5-Stage Buffering (num_ab_stage = 5) - BEST

### Results
| Benchmark | Quad Buffer | 5-Stage | Speedup |
|-----------|-------------|---------|---------|
| M=128, N=7168, K=16384 | 50.1 µs | **49.3 µs** | **2%** |
| M=128, N=4096, K=7168 | 31.1 µs | **30.8 µs** | **1%** |
| M=128, N=7168, K=2048 | 31.4 µs | **30.9 µs** | **2%** |

**6-stage testing**: No improvement, likely hitting diminishing returns.

---

## Total Optimization Progress

| Benchmark | Baseline | Final (5-stage) | **Total Speedup** |
|-----------|----------|-----------------|-------------------|
| M=128, N=7168, K=16384 | 96.8 µs | **49.3 µs** | **1.96x** |
| M=128, N=4096, K=7168 | 51.6 µs | **30.8 µs** | **1.68x** |
| M=128, N=7168, K=2048 | 37.6 µs | **30.9 µs** | **1.22x** |

---

## Failed Optimizations

### Attempt: Smaller M Tile (128 → 64)
**Goal**: Double grid size for better SM utilization

**Result**: ❌ `OpError: expects the M-mode to be 128, but got 64`

**Reason**: `MmaMXF4NVF4Op` instruction requires M=128.

### Attempt: Smaller N Tile (128 → 64)  
**Goal**: Double grid size for better SM utilization

**Result**: ❌ TMA shape mismatch error for scale factor B

**Reason**: Block-scaled GEMM requires N=128 for proper scale factor TMA layout.

### Attempt: Smaller K Tile (256 → 128)
**Goal**: Double grid size for better SM utilization

**Result**: ❌ TMA shape mismatch error for scale factors

**Reason**: Block-scaled GEMM with sf_vec_size=16 requires K=256 tile for proper alignment.

### Attempt: Loop Restructuring (Issue TMA before compute)
**Goal**: Better overlap of TMA loads with compute

**Result**: ❌ Timeout (likely deadlock or hang)

**Reason**: The producer-consumer pipeline semantics require release before the next acquire.

### Attempt: Split-K with Workspace Buffer
**Goal**: Increase grid size by splitting K dimension, using workspace for partial sums

**Implementation**:
- Divided K tiles across `split_k_slices` blocks (tested 2 and 4)
- Each block writes FP32 partial sums to workspace[split_k_idx, :, :]
- Host-side reduction: `workspace.sum(dim=split_k).to(fp16)` + copy to output

**Results** (split_k=4):
| Benchmark | 5-Stage | Split-K=4 | Change |
|-----------|---------|-----------|--------|
| K=16384 | 49.3 µs | 81.6 µs | **+65% slower** |
| K=7168 | 30.8 µs | 47.5 µs | **+54% slower** |
| K=2048 | 30.9 µs | 67.7 µs | **+119% slower** |

**Results** (split_k=2):
| Benchmark | 5-Stage | Split-K=2 | Change |
|-----------|---------|-----------|--------|
| K=16384 | 49.3 µs | 55.2 µs | **+12% slower** |
| K=7168 | 30.8 µs | 39.0 µs | **+27% slower** |
| K=2048 | 30.9 µs | 43.1 µs | **+39% slower** |

**Why It Failed**:
1. **Memory traffic**: Workspace is FP32 (2x memory write vs FP16 output)
2. **Reduction overhead**: Extra kernel/memop to sum workspace slices
3. **Less compute per block**: Fewer K tiles per block = lower arithmetic intensity
4. **Small K penalty**: For K=2048 (8 tiles), split_k=4 means only 2 tiles/block

---

### Attempt: Atomic FP16 Split-K (No Workspace)

**Goal**: Eliminate workspace overhead by using atomic FP16 adds directly to output

**Implementation**:
- Extended grid z-dimension by split_k factor
- Each block computes k_tile_start/k_tile_end from bidz
- Zero output before kernel launch
- Use `atom.add.noftz.f16` PTX instruction for atomic epilogue

**Issues Encountered**:
1. **Pointer access**: CuTe tensors don't expose individual element pointers easily
   - `tTR_gC[elem_idx].iterator` returns `Float16` value, not address
   - Need different approach to get memory addresses for atomics
2. **Infrastructure overhead**: Even with split_k=1, the extra calculations 
   (k_tile_start/k_tile_end from bidz) added measurable overhead:
   - K=16384: 50.0µs vs 49.3µs (+1.4%)

**Conclusion**: 
Atomic Split-K requires a different implementation approach:
- May need custom epilogue that directly computes addresses
- Or use a TMA-based atomic store if available
- The overhead from grid partitioning logic negates benefits for small split factors

**Verdict**: ❌ Reverted. The overhead of workspace allocation and reduction significantly outweighs the parallelism benefit.

---

## Speed of Light Analysis

From task.yml, the theoretical speed of light times are:
```
M=128, N=7168, K=16384: 8.994 µs
M=128, N=4096, K=7168:  2.354 µs
M=128, N=7168, K=2048:  1.333 µs
```

Current performance vs SOL:
| Benchmark | Current | SOL | Gap |
|-----------|---------|-----|-----|
| K=16384 | 49.3 µs | 8.99 µs | 5.5x |
| K=7168 | 30.8 µs | 2.35 µs | 13.1x |
| K=2048 | 30.9 µs | 1.33 µs | 23.2x |

---

## Optimization 5: Constexpr Parameter Optimization (4th Revision)

### Change
Refactored the kernel to use `Constexpr` parameters for `num_ab_stage` instead of global variables. This allows the compiler to generate more optimized code.

```python
# Before: global variable
num_ab_stage = 5  # Used directly in kernel

# After: Constexpr parameter
@cute.kernel
def kernel(..., num_ab_stage: cutlass.Constexpr[int]):
    # num_ab_stage is now a compile-time constant
```

### Results
| Benchmark | 3rd Revision | 4th Revision | Speedup |
|-----------|--------------|--------------|---------|
| M=128, N=7168, K=16384 | 54.3 µs | **49.3 µs** | **9.2%** |
| M=128, N=4096, K=7168 | 33.5 µs | **30.8 µs** | **8.1%** |
| M=128, N=7168, K=2048 | 31.9 µs | **30.9 µs** | **3.1%** |

### Why It Works
- Constexpr parameters are baked into the compiled kernel as constants
- Enables better loop unrolling and dead code elimination
- Reduces register pressure from storing runtime values
- Allows compiler to optimize shared memory allocation

---

## Updated Speed of Light Analysis

Current performance vs SOL (after 4th revision):
| Benchmark | Current | SOL | Gap |
|-----------|---------|-----|-----|
| K=16384 | 49.3 µs | 8.99 µs | 5.5x |
| K=7168 | 30.8 µs | 2.35 µs | 13.1x |
| K=2048 | 30.9 µs | 1.33 µs | 23.2x |

---

## Potential Future Optimizations

1. **Split-K parallelism**
   - Split K dimension across multiple blocks
   - Requires atomic reduction in epilogue
   - Would significantly increase grid size (48 → 192+ blocks)

2. **Persistent kernel with tile scheduler**
   - Use `StaticPersistentTileScheduler` 
   - Blocks loop over multiple output tiles
   - Better SM utilization when grid is small

3. **TMA Store for Epilogue**
   - Replace SIMT stores with `CopyBulkTensorTileS2GOp`
   - Could reduce epilogue overhead

4. **Warp specialization**
   - Dedicated warps for TMA loads vs compute
   - Could enable true load/compute overlap

5. **Register pressure reduction**
   - Current: 136 registers/thread limits occupancy to 18.75%
   - May require algorithmic changes

---

## 5th Revision Attempts (Failed)

Attempted many major restructures, but none improved over 4th revision:

### 1. 2-CTA Cooperative MMA
- **Approach**: Use `CtaGroup.TWO` for cooperative MMA across 2 CTAs
- **Changes**: Modified MMA op, TMA atoms, TMEM allocator
- **Result**: ❌ Layout mismatch errors - SMEM and CTA V-map shapes incompatible
- **Error**: `expected top-level shape equivalence between the SMEM layout and the CTA V-map`

### 2. K Tile Size Reduction (128 instead of 256)
- **Approach**: Reduce K tile to fit more blocks per SM
- **Changes**: `mma_tiler_mnk = (128, 128, 128)`
- **Result**: ❌ Layout mismatch for SFA TMA atom
- **Reason**: Block-scaled GEMM requires specific K tile sizes

### 3. Thread Block Clusters with TMA Multicast
- **Approach**: Use cluster=(1,2,1) with TMA multicast for A matrix
- **Changes**: Cluster layout, TMA atoms with `CopyBulkTensorTileG2SMulticastOp`
- **Result**: ❌ API mismatch - `make_tiled_tma_atom_A` doesn't support `num_multicast`
- **Note**: Would need to use lower-level `make_tiled_tma_atom` instead

### 4. Simple Clusters Without Multicast
- **Approach**: Just cluster=(1,2,1) without multicast
- **Result**: ❌ No improvement (same performance)

### 5. Main Loop Unrolling by 2
- **Approach**: Manually unroll main loop to process 2 K tiles per iteration
- **Result**: ❌ No improvement (same performance)

### 6. Adaptive Pipeline Stages (3/4/5 based on K)
- **Approach**: Fewer stages for small K to reduce overhead
- **Results**: 
  - 3 stages for K=2048: ❌ Worse (31.9 vs 31.1 µs)
  - 4 stages for K=7168: ❌ Worse (31.1 vs 30.8 µs)
  - 6 stages for K=16384: ❌ Slightly worse (49.5 vs 49.4 µs)

### 7. TMEM Allocation Reduction
- **Approach**: Reduce `num_tmem_alloc_cols` from 512 to 256
- **Result**: ❌ No improvement

### 8. 2 Accumulator Stages
- **Approach**: `num_acc_stage = 2` for overlapped accumulator access
- **Result**: ❌ High variance, unstable (range 49.2-366 µs)

### 9. 7-Stage Pipelining
- **Approach**: Increase from 5 to 7 stages
- **Result**: ❌ Timeout (too much shared memory)

---

## Root Cause Analysis

The fundamental bottleneck is **insufficient parallelism**:

1. **Grid Size**: Only 48-56 blocks for 148 SMs (0.33 waves per SM)
2. **Block Limit**: Shared memory (184KB) limits to 1 block per SM
3. **Memory Bound**: Analysis shows ~17% of peak bandwidth achieved
4. **Cache Efficient**: L1 hit rate 98.44% is excellent

### Memory Traffic Analysis (K=16384 benchmark)
| Component | Size |
|-----------|------|
| A matrix | 1 MB |
| B matrix | 59 MB |
| SFA | 128 KB |
| SFB | 7 MB |
| C output | 1.8 MB |
| **Total** | **~68 MB** |

B200 bandwidth: ~8 TB/s → Theoretical time: 68MB / 8TB/s = **8.5 µs**

This matches the SOL of 8.99 µs, confirming the kernel is memory-bound.

Current time 49.4 µs = 68 MB / 49.4 µs = **1.38 TB/s** (17% of peak)

### To close the gap requires:
1. **TMA Multicast** - Share A matrix loads across blocks (all blocks use same A)
2. **Better scheduling** - Improve memory access patterns for B matrix
3. **Split-K with atomic reduction** - Increase grid size

---

## TMA Multicast Attempt (Failed - Timeout)

Attempted to implement TMA multicast using pattern from `examples/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`:

### Key Components Required:
1. **Cluster shape**: `cluster_shape_mnk = (1, 2, 1)` for 2 CTAs sharing N dimension
2. **Multicast TMA atoms**: `CopyBulkTensorTileG2SMulticastOp` for A and SFA
3. **Multicast masks**: `cpasync.create_tma_multicast_mask()` with correct `mcast_mode`
4. **Cluster-aware TMA partition**: `cpasync.tma_partition()` with cluster coordinates

### Why It Failed:
The implementation **timed out (deadlock)** because:
1. **Pipeline changes required**: Need `PipelineTmaUmma` instead of current pipeline
2. **Producer group handling**: Must calculate `num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1`
3. **Index handling**: Use `ab_empty.count` instead of loop index for TMA coordinates
4. **CTA coordination**: Only leader CTA issues certain operations (`is_leader_cta` pattern)

### Correct Pattern (from fp16_gemm_1.py):
```python
# Pipeline with multicast awareness
num_mcast_ctas_a = cute.size(cta_layout_vmnk.shape[2])
num_mcast_ctas_b = cute.size(cta_layout_vmnk.shape[1])
num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1

ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=ab_stages,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_tma_producer
    ),
    tx_count=num_tma_copy_bytes,
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    cta_layout_vmnk=cta_layout_vmnk,  # Critical: pass cluster layout
).make_participants()

# In main loop
for _ in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
    ab_empty = ab_producer.acquire_and_advance()
    cute.copy(
        tma_atom_a,
        tAgA[(None, ab_empty.count)],  # Use count, not loop index
        tAsA[(None, ab_empty.index)],
        tma_bar_ptr=ab_empty.barrier,
        mcast_mask=tma_mcast_mask_a,
    )
```

### Future Work:
A full TMA multicast implementation requires:
1. Restructuring the pipeline initialization
2. Proper CTA coordination and leader selection
3. Handling the different iteration patterns with `ab_empty.count`

---

## Full TMA Multicast Attempt with PipelineTmaUmma (Segfault)

Attempted a complete implementation following `fp16_gemm_1.py` exactly:

### Changes Made:
1. Added `cluster_shape_mnk = (1, 2, 1)` configuration
2. Used `CopyBulkTensorTileG2SMulticastOp` for A and SFA TMA atoms
3. Created `cta_layout_vmnk` from cluster shape
4. Passed `cta_layout_vmnk` to `PipelineTmaUmma.create()`
5. Added `tmem_dealloc_mbar_ptr` to SharedStorage
6. Implemented `is_leader_cta` pattern for compute gating
7. Used `ab_empty.count` for global tensor indexing
8. Added `tmem.relinquish_alloc_permit()` before epilogue
9. Added `ab_producer.tail()` and `acc_producer.tail()` cleanup
10. Used `pipeline.sync(barrier_id=1)` before TMEM free

### Result: **Segfault**

The implementation causes a segfault, likely due to:
- Block-scaled GEMM's complex scale factor handling (S2T copies)
- Incompatibility between multicast and block-scaled SF tensor partitioning
- Leader CTA selection logic may not work correctly for this kernel structure

---

## Prefetch Stages Pattern Attempt (Segfault)

Tried using `cutlass.range(k_tile_cnt, prefetch_stages=num_ab_stage - 2)` with a unified loop (no separate prologue/main loop) following the pattern from `tour_to_sol_gemm.ipynb`:

### Result: **Segfault**

The block-scaled GEMM kernel has specific pipeline requirements that don't work with the standard prefetch pattern. The separate prologue/main loop structure is necessary.

---

## Additional Stage Testing

| Stages | K=16384 | K=7168 | K=2048 | Result |
|--------|---------|--------|--------|--------|
| 5 | 49.3µs | 30.8µs | 31.0µs | **Best** |
| 6 (large K) | 49.4µs | - | - | No improvement |
| 4 (med/small K) | - | 31.2µs | 31.4µs | 1-2% worse |

5 stages remains optimal for all K sizes.

---

## Final Performance

**Current Best (5th Revision = 4th Revision):**
| Benchmark | Time | Improvement vs Baseline |
|-----------|------|------------------------|
| K=16384 | 49.3µs | ~9% faster |
| K=7168 | 30.8µs | ~8% faster |
| K=2048 | 31.0µs | ~2.5% faster |

---

## Remaining Optimization Opportunities

1. **Split-K with atomic reduction**: Would increase grid size from 56 to 56*split_k blocks, better utilizing 148 SMs
2. **Custom PTX**: Fine-grained control over memory operations and instruction scheduling
3. **Stream-K**: More sophisticated work distribution across SMs
4. **Cooperative Groups**: Alternative cluster synchronization patterns

The main bottleneck remains **low parallelism** (56 blocks for 148 SMs = 38% utilization). Addressing this requires either Split-K (with atomic reduction overhead) or a more sophisticated TMA multicast implementation that properly handles block-scaled GEMMs.

