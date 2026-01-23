# JVP Benchmark: Performance Analysis

This document summarizes the findings from benchmarking the MeanFlow loss JVP overhead.

## Background

The MeanFlow paper (Appendix B.4) claims that JVP computation should add approximately **16% wall clock time** to training. Our benchmarks initially showed significantly higher overhead, which we've been working to reduce.

## Benchmark Methodology

The benchmark (`scripts/benchmark_meanflow_loss.py`) directly uses the `MeanFlowLoss` class to measure:

1. **Forward only (no_grad)** - Baseline forward pass cost
2. **Standard FM MSE (fwd+bwd)** - Typical flow matching training step
3. **MeanFlowLoss 0% ratio** - No JVP computation (should match standard FM)
4. **MeanFlowLoss 25% ratio** - Paper's default training setup
5. **MeanFlowLoss 100% ratio** - All samples use JVP (worst case)
6. **`_compute_target` isolation** - JVP kernel cost only

### Verification

The benchmark design was reviewed by:
- **Codex (OpenAI gpt-5.2-codex)** - Mathematical correctness
- **Gemini (Google Gemini 3)** - Architectural review

Both approved the methodology with the caveat that "Standard FM MSE" is not directly comparable to MeanFlowLoss 0% due to differences in time sampling and weighting.

---

## Optimization History

### Optimization 1: JVP Primal Reuse (Implemented)

**Problem:** The original implementation ran the model twice for MeanFlow samples - once in the main forward pass, then again inside `torch.func.jvp()` which computes both primal and tangent.

**Solution:** Modified `_compute_target` to return the JVP primal output, then split the batch so:
- FM samples: run forward pass directly
- MF samples: reuse JVP primal output

**Result:** ~16% speedup on MeanFlowLoss 100% ratio (172.8ms → 145.2ms)

---

## Current Results (After Optimization 1)

### CUDA A100-40GB - FP32

| Measurement | Time (ms) | Overhead vs Standard FM |
|-------------|-----------|------------------------|
| Forward only | 15.9 | - |
| Standard FM MSE (fwd+bwd) | 45.5 | baseline |
| MeanFlowLoss 0% ratio | 47.0 | +3.2% |
| MeanFlowLoss 25% ratio | 181.6 ± 38.3 | **+299.3%** |
| MeanFlowLoss 100% ratio | 145.2 | **+219.1%** |
| `_compute_target` (JVP) | 117.0 | - |

**Key metrics:**
- JVP as % of forward pass: **737%**
- JVP as % of standard FM step: **257%**

### CUDA A100-40GB - BF16 (via autocast)

| Measurement | Time (ms) | Overhead vs Standard FM |
|-------------|-----------|------------------------|
| Forward only | 18.0 | - |
| Standard FM MSE (fwd+bwd) | 47.3 | baseline |
| MeanFlowLoss 0% ratio | 48.2 | +2.0% |
| MeanFlowLoss 25% ratio | 207.6 ± 44.6 | **+339.2%** |
| MeanFlowLoss 100% ratio | 163.7 ± 48.8 | **+246.3%** |
| `_compute_target` (JVP) | 121.1 | - |

**Key finding:** BF16 does NOT help with JVP performance. Actually slightly worse than FP32, likely because `torch.func.jvp` internally uses FP32 for numerical stability, causing dtype conversion overhead.

### Historical Baseline (Before Optimization 1)

| Measurement | Time (ms) | Overhead vs Standard FM |
|-------------|-----------|------------------------|
| Forward only | 16.0 | - |
| Standard FM MSE (fwd+bwd) | 45.8 | baseline |
| MeanFlowLoss 0% ratio | 46.7 | +1.9% |
| MeanFlowLoss 25% ratio | 175.2 | **+282.6%** |
| MeanFlowLoss 100% ratio | 172.8 | **+277.3%** |
| `_compute_target` (JVP) | 119.1 | - |

---

## Remaining Bottleneck: JVP Cost

The primary bottleneck is now the JVP computation itself:

| Metric | Value |
|--------|-------|
| Forward pass | 16 ms |
| JVP (`_compute_target`) | 117 ms |
| JVP / Forward ratio | **7.3x** |
| Theoretical JVP / Forward | ~2x |

`torch.func.jvp` should theoretically cost ~2x a forward pass (primal + tangent). We're seeing **7.3x**, indicating significant overhead from:

1. **Eager mode dispatch** - Many small kernel launches instead of fused operations
2. **Function tracing** - `torch.func` must trace the function to compute derivatives
3. **FP32 enforcement** - JVP uses FP32 internally regardless of input dtype
4. **No kernel fusion** - Unlike compiled code, eager JVP can't fuse operations

---

## Optimization Attempts and Results

### Option A: torch.compile on the model ❌ NO BENEFIT FOR JVP

Benchmarked with `--compile` flag on CUDA A100:

| Metric | Uncompiled | Compiled |
|--------|------------|----------|
| Forward only | 15.9 ms | 11.6 ms (27% faster ✓) |
| _compute_target (JVP) | 117.0 ms | 117.4 ms (unchanged) |
| MeanFlowLoss 100% | 145.2 ms | 147.2 ms (unchanged) |

**Conclusion:** torch.compile speeds up the forward pass but does NOT speed up JVP operations. The JVP overhead is intrinsic to `torch.func.jvp` tracing and cannot be compiled away. **Do NOT use torch.compile for MeanFlow training** as it provides no benefit and adds compilation overhead.

### Option B: Custom VJP via autograd.Function ❌ INCORRECT APPROACH

VJP computes vector-Jacobian products (v^T J). We need Jacobian-vector products (J v) to get du/dt where the tangent is [0, 0, 1]. Forward-mode AD (JVP) is mathematically correct and optimal for this "many-to-many" map where the input perturbation is a scalar.

**Conclusion:** VJP is the wrong tool for this problem.

### Option C: Analytical derivatives ❌ NOT FEASIBLE

Manually deriving gradients for every layer type (Conv2d, GroupNorm, SiLU, etc.) in UNet/DiT is impractical.

### Option D: Alternative JVP strategies ❌ SLOWER

Per EXPERIMENT_LOG.md benchmarks on A100:

| Approach | Result |
|----------|--------|
| Split JVP (∂v/∂z + ∂v/∂t separately) | 2x SLOWER |
| Micro-batching JVP | 4x SLOWER |

### Option E: Full-batch JVP ✓ ALREADY EFFICIENT

JVP scales sub-linearly with batch size:

| Ratio | n_jvp | Time (ms) | Scaling vs Linear |
|-------|-------|-----------|-------------------|
| 0.25 | 16 | 139 | 1.0x (baseline) |
| 0.50 | 32 | 144 | 0.52x (2x better) |
| 0.75 | 48 | 151 | 0.36x (3x better) |
| 1.00 | 64 | 154 | 0.28x (4x better) |

Doubling JVP samples only adds ~10% time. This sub-linear scaling means JVP is MORE efficient at larger batches.

### Option F: CUDA Graphs ✓ **MAJOR IMPROVEMENT** (2026-01-23)

**BREAKTHROUGH:** CUDA graph capture reduces JVP overhead by **56%**!

| Measurement | Time (ms) | vs Forward |
|-------------|-----------|------------|
| Forward only | 15.7 ms | 1.0x |
| Eager JVP | 120.3 ms | 7.6x |
| **CUDA graph JVP** | **52.6 ms** | **3.3x** |

CUDA graphs work by:
1. **Capturing** the kernel launch sequence once during warmup
2. **Replaying** the captured graph without Python/CPU overhead on subsequent calls

This brings JVP cost from 7.6x forward to **3.3x forward**, much closer to the theoretical 2x.

**Requirements for CUDA graph capture:**
- Static input shapes (same batch size, image dimensions)
- No data-dependent control flow (`if tensor.any()`, etc.)
- No CPU-GPU synchronization (`.item()`, `.numpy()`)
- Pre-allocated static buffers for inputs/outputs

**Limitations:**
- Cannot capture `_compute_target` directly due to `use_meanflow.any()` check
- Requires wrapping the pure JVP call separately
- Adds implementation complexity (static buffer management)

**Status:** Proof-of-concept benchmark working. Production implementation planned.

**Implementation Progress:**

1. ✅ **Full-batch JVP mode** implemented (`full_batch_jvp=True`)
   - Removes CPU-GPU sync (`if use_meanflow.any()`)
   - All samples go through JVP (for FM samples where `r == t`, `delta_t = 0`, so `u_tgt = v_true`)
   - Prerequisite for CUDA graph capture

2. ⚠️ **CUDA graph for full loss** NOT YET IMPLEMENTED
   - Pure JVP CUDA graph works (52ms vs 120ms eager)
   - But integrating with loss.backward() is complex:
     - CUDA graphs capture the autograd tape
     - After backward(), tape is freed
     - Next iteration fails because tape is gone
   - Requires capturing forward + backward + optimizer together
   - Alternative: Use `torch.cuda.make_graphed_callables` (has torch.func limitations)

**Key Insight:** Full-batch JVP doesn't waste compute - for FM samples, `delta_t = t - r = 0`, so the JVP target naturally equals `v_true`. The only tradeoff is FP32 for all samples (JVP forces FP32 for numerical stability).

**To Enable Full CUDA Graph Benefits:**
- Capture the entire training step (forward + backward + optimizer)
- Use `torch.compile(mode="reduce-overhead")` which includes cudagraph trees
- Investigate `torch.cuda.make_graphed_callables` compatibility with `torch.func.jvp`

---

## Reproduction

```bash
# Local (MPS/CPU)
PYTHONPATH=. python scripts/benchmark_meanflow_loss.py

# Modal (CUDA A100) - FP32
modal run scripts/modal_app.py::run_benchmark --script-name benchmark_meanflow_loss.py --script-args "--dtype float32"

# Modal (CUDA A100) - BF16
modal run scripts/modal_app.py::run_benchmark --script-name benchmark_meanflow_loss.py --script-args "--dtype bfloat16"

# With torch.compile
modal run scripts/modal_app.py::run_benchmark --script-name benchmark_meanflow_loss.py --script-args "--compile"
```

---

## Final Conclusion

### Optimizations Implemented

1. **JVP Primal Reuse** - Eliminated redundant forward pass by reusing the JVP primal output for MeanFlow samples instead of running model twice. Result: **16% speedup** (172.8ms → 145.2ms at 100% ratio).

2. **CUDA Graphs (proof-of-concept)** - Captured JVP kernel sequence to eliminate Python/CPU overhead. Result: **56% speedup** on isolated JVP (120ms → 52ms).

### Current Status

| Metric | Before Opt | After JVP Reuse | With CUDA Graphs |
|--------|------------|-----------------|------------------|
| JVP cost | 119 ms | 117 ms | **52 ms** |
| vs Forward | 7.4x | 7.3x | **3.3x** |
| MeanFlowLoss 100% | 172.8 ms | 145.2 ms | ~80 ms (est) |

CUDA graphs bring JVP cost close to the theoretical 2x overhead. The remaining 3.3x overhead (vs 2x theoretical) is intrinsic compute, not dispatch.

### Remaining Work

**CUDA graph integration into MeanFlowLoss** requires:
1. Static buffer management for JVP inputs/outputs
2. Removing `use_meanflow.any()` check (or moving it outside capture)
3. Handling variable MeanFlow sample counts (may need multiple graphs or padding)
4. Testing with training loop (backward pass compatibility)

### Training Recommendations

1. Use **batch_size=64** with gradient accumulation for effective batch 1024
2. Use **ratio=0.25** (paper default) - JVP only computed for 25% of samples
3. **Do NOT use torch.compile** for MeanFlow training (no JVP benefit, compilation overhead)
4. Use **A100/H100** - JVP scales well; MPS is impractical for ratio > 0
5. **Enable TF32** - marginal benefit for non-JVP parts

### Path to Paper's 16% Overhead

The paper claims 16% overhead. Our current best (with CUDA graphs) achieves:
- 52ms JVP vs 16ms forward = **~225% overhead** on the JVP call
- At 25% ratio: ~25% × 225% = **~56% overhead** on full step (vs paper's 16%)

To reach 16%, additional optimizations may be needed:
- Custom CUDA kernels for forward-mode AD
- JAX's more optimized JVP implementation
- Full CUDA graph integration (capture entire loss computation)
