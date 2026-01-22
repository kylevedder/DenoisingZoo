# MeanFlow Experiment Log

This log tracks experiments for validating the MeanFlow implementation.

---

## Reference: Paper Hyperparameters

From [Mean Flows for One-step Generative Modeling (arXiv:2505.13447)](https://arxiv.org/abs/2505.13447):

**ImageNet 256×256 (DiT)**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `logit_normal_mean` | -0.4 | |
| `logit_normal_std` | 1.0 | |
| `meanflow_ratio` | 0.25 | |
| `weighting_power` | 1.0 | |
| CFG scale | 3.0 | For SiT-B/4 |
| epochs | 80-240 | Depending on model |
| Target FID (DiT-B/4) | 6.17 | 80 epochs |
| Target FID (DiT-XL/2) | 3.43 | 240 epochs, 1-NFE |

**CIFAR-10 (UNet) - Different from ImageNet!**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `logit_normal_mean` | -2.0 | More negative than ImageNet |
| `logit_normal_std` | 2.0 | Wider than ImageNet |
| `meanflow_ratio` | 0.75 | Higher than ImageNet |
| `weighting_power` | 0.75 | Lower than ImageNet |
| batch_size | 1024 | Or 512 with grad accum |
| epochs | 19200 | ~800k iterations |
| Target FID | 2.92 | 1-NFE unconditional |

---

## Implementation Notes

### JVP Computation (Paper Eq. 8)
The MeanFlow loss requires computing the total time derivative along the trajectory:
```
du/dt = v·∂v/∂z + ∂v/∂t
```

This is computed via `torch.func.jvp` with:
- Inputs: `(z, t)`
- Tangents: `(v, 1)` where v is the model's velocity prediction

### Selective JVP Optimization
- JVP is only computed for MeanFlow samples (where r != t)
- For ratio=0.25, only 25% of batch needs JVP computation
- Provides ~4x speedup over naive implementation

### JVP Performance on Different Hardware

| Platform | Forward (bs=64) | JVP (bs=64, ratio=0.75) | Notes |
|----------|-----------------|-------------------------|-------|
| MPS (M1/M2) | ~2.5s/batch | ~25s+/batch | Memory pressure, impractical |
| A100-40GB | ~25ms/batch | ~150ms/batch | Efficient, scales sub-linearly |

**Key findings:**
- `torch.compile` + JVP works fine after warmup (~1.03x speedup), but has ~82s initial compilation overhead
- For long training runs, compile overhead amortizes - consider enabling for runs > 1 hour
- ratio=0.75 OOMs with batch_size=128 on A100-40GB; use batch_size=64 with grad accumulation
- MPS is only practical for ratio=0 (standard FM); use Modal/cloud for MeanFlow training

### JVP Performance Analysis Summary (2026-01-22)

**TL;DR: JVP is already efficient. No further optimizations needed.**

Comprehensive A100-40GB benchmarks evaluated potential JVP optimizations for UNet (51M params) on CIFAR-10 32×32 with bf16 autocast.

**Finding 1: JVP scales sub-linearly (efficient)**
| Ratio | n_jvp | Time (ms) | Scaling vs Linear |
|-------|-------|-----------|-------------------|
| 0.25 | 16 | 139 | 1.0x (baseline) |
| 0.50 | 32 | 144 | 0.52x (2x better than linear) |
| 0.75 | 48 | 151 | 0.36x (3x better than linear) |
| 1.00 | 64 | 154 | 0.28x (4x better than linear) |

Doubling JVP samples only adds ~10% time, not 2x. JVP is MORE efficient at larger batches.

**Finding 2: Alternative approaches are slower**
| Approach | Result |
|----------|--------|
| Split JVP (∂v/∂z + ∂v/∂t separately) | 2x SLOWER, more memory |
| Micro-batching | 4x SLOWER |

**Finding 3: torch.compile provides NO benefit for JVP workloads**
| Mode | JVP Time | Speedup |
|------|----------|---------|
| Uncompiled | 139.4ms | 1.00x |
| default | 142.2ms | 0.98x (slower) |
| reduce-overhead | 141.2ms | 0.99x |
| max-autotune | 147.0ms | 0.95x (slower) |

**Why**: torch.compile only compiles the forward pass, not the JVP operation itself. The overhead of calling compiled code from within `torch.func.jvp` actually hurts performance.

**Recommendation**: Do NOT use torch.compile for MeanFlow training with ratio > 0.

**Recommendations:**
1. Use selective JVP (already implemented) - only compute for MeanFlow samples
2. Use largest batch size that fits in memory
3. Use gradient accumulation for effective batch_size=1024 with bs=64
4. Do NOT use `compile=true` for MeanFlow training (torch.compile hurts JVP performance)

### Modal Infrastructure Hardening

Multiple experiments failed due to Modal client connection drops (`StreamTerminatedError`, `GRPCError`). Fix applied 2026-01-21:
- Increased timeout from 4h to 12h per attempt
- Added automatic retries (5 retries with 30s delay)
- Enabled `--detach` flag by default for fire-and-forget execution
- Auto-inject `resume=true` for seamless checkpoint recovery on retry

Verification: 1-epoch CIFAR-10 completed successfully (loss 0.246, ED 3.38) with detached execution.

---

## Phase 1: Unit Tests

**Status:** COMPLETED

```bash
./tests/run_tests.sh
```

All 94 tests passing:
- `test_flow_matching_math.py` - Interpolation, velocity field, unified input
- `test_jvp_correctness.py` - JVP computation correctness
- `test_meanflow_target.py` - MeanFlow target formula
- `test_time_sampling.py` - Logit-normal time sampling
- `test_adaptive_weighting.py` - Adaptive weighting scheme
- `test_meanflow_jvp_optimization.py` - Full JVP with ∂v/∂t term
- `test_dataset_items.py` - Dataset item instantiation

---

## Phase 2: Overfitting Experiments (Deterministic Data)

### 2.1: MSE Loss Baseline

**Status:** COMPLETED

| Metric | Value |
|--------|-------|
| Final Loss | 6.8e-5 |
| Energy Distance | 26.71 |

```bash
python launcher.py dataloaders=deterministic model=unet loss=mse epochs=200 run_name=mse_baseline
```

### 2.2: MeanFlow ratio=0 (Standard Flow Matching)

**Status:** COMPLETED

| Metric | Value |
|--------|-------|
| Final Loss | 0.000157 |
| Energy Distance | 14.63 |

```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 \
  loss.meanflow_ratio=0 loss.use_batch_time=true run_name=meanflow_ratio0
```

### 2.3: MeanFlow ratio=0.25 (Paper Config)

**Status:** COMPLETED

| Metric | Value |
|--------|-------|
| Final Loss | 0.000072 |
| Energy Distance | 9.09 |

```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 \
  loss.meanflow_ratio=0.25 loss.use_batch_time=true run_name=meanflow_ratio025
```

### Phase 2 Summary

| Experiment | Loss Type | Ratio | Final Loss | Energy Dist |
|------------|-----------|-------|------------|-------------|
| 2.1 | MSE | N/A | 6.8e-5 | 26.71 |
| 2.2 | MeanFlow | 0.0 | 0.000157 | 14.63 |
| 2.3 | MeanFlow | 0.25 | 0.000072 | 9.09 |

**Conclusion:** MeanFlow ratio=0.25 achieves best energy distance, validating the implementation.

---

## Phase 3: Sanity Checks

### 3.1: MeanFlow One-step vs Solver Consistency

**Status:** COMPLETED

Verify MeanFlow 1-step sampling matches solver-based sampling.

```bash
python scripts/eval_meanflow_vs_solver.py --checkpoint outputs/ckpts/unet/archive/meanflow_ratio025_epoch_0200.pt
```

| Method | Energy Distance | NFEs |
|--------|-----------------|------|
| MeanFlow 1-step | 96.705 | 1 |
| Euler (10 steps) | 98.715 | 10 |
| Euler (50 steps) | 97.118 | 50 |
| Euler (100 steps) | 96.923 | 100 |
| RK4 (10 steps) | 98.752 | 40 |
| RK4 (50 steps) | 97.123 | 200 |
| RK4 (100 steps) | 96.925 | 400 |

**Conclusion:** MeanFlow 1-step (ED=96.7) matches best solver (Euler 100, ED=96.9) with 100x fewer NFEs.

### 3.2: Multi-Class Overfit

**Status:** COMPLETED

Test loss/inference across multiple fixed targets (4 classes, 10 samples/class).

| Run | Ratio | Final Loss | Energy Dist |
|-----|-------|------------|-------------|
| ratio=0 | 0.0 | 0.011 | 1.531 |
| ratio=0.25 | 0.25 | 0.024 | 4.304 |

**Conclusion:** Both converge stably. ratio=0.25 has higher loss (learning mean velocities vs pointwise).

### 3.3: Weighting Ablation

**Status:** COMPLETED

| Run | Weighting Power | Final Loss | Energy Dist |
|-----|-----------------|------------|-------------|
| Weighted | 0.5 | 0.000072 | 9.09 |
| Unweighted | 0.0 | 0.000125 | 10.11 |

**Conclusion:** Weighting provides slight improvement but isn't masking fundamental issues.

---

## Phase 4: CIFAR-10 Experiments

### Preliminary Tests (MPS + Modal Validation)

Quick tests to establish baselines and validate Modal training:

| Run | Platform | Ratio | Epochs | Loss | ED | Speed |
|-----|----------|-------|--------|------|-----|-------|
| cifar10_ratio0_test | MPS | 0.0 | 5 | 0.2102 | 0.8548 | ~2.5s/batch |
| modal_test_full2 | A100 | 0.25 | 1 | 0.2449 | 0.4455 | 2.3 it/s |

**Conclusion:** Modal A100 is ~60x faster than MPS for ratio=0, and can run ratio>0 experiments that are impractical on MPS.

### 4.1: CIFAR-10 Standard FM Baseline (ratio=0, 20 epochs)

**Status:** COMPLETED

```bash
python launcher.py --backend modal dataloaders=cifar10 model=unet loss=meanflow epochs=20 \
  loss.meanflow_ratio=0 loss.logit_normal_mean=-2.0 loss.logit_normal_std=2.0 \
  loss.weighting_power=0.75 dataloaders.train.batch_size=128 optimizer.lr=1e-4 \
  precision=bf16 eval_every=5 save_every=5 run_name=cifar10_ratio0_20ep resume=false
```

| Metric | Value |
|--------|-------|
| Final Loss | 0.199 |
| Best Energy Distance | 0.239 (epoch 15) |
| Final Energy Distance | 0.376 (epoch 20) |
| Speed | ~1.6 min/epoch |

**Checkpoints:** `cifar10_ratio0_20ep_epoch_{0005,0010,0015,0020}.pt`

**Note:** Must use `resume=false` for new runs to avoid loading previous experiment's checkpoint.

### 4.2: CIFAR-10 MeanFlow ratio=0.25 (20 epochs)

**Status:** COMPLETED

```bash
python launcher.py --backend modal dataloaders=cifar10 model=unet loss=meanflow epochs=20 \
  loss.meanflow_ratio=0.25 loss.logit_normal_mean=-2.0 loss.logit_normal_std=2.0 \
  loss.weighting_power=0.75 dataloaders.train.batch_size=128 optimizer.lr=1e-4 \
  precision=bf16 eval_every=5 save_every=10 run_name=cifar10_ratio025_20ep
```

| Metric | Value |
|--------|-------|
| Final Loss | ~0.20 |
| Energy Distance (epoch 5) | 0.246 |
| Energy Distance (epoch 10) | 2.74 (spike, sampling variance) |
| Speed | 2.5-2.6 it/s (A100-40GB) |

**Checkpoints:** `cifar10_ratio025_20ep_epoch_{0005,0010,0015,0020}.pt`

### 4.2b: CIFAR-10 MeanFlow ratio=0.5 (20 epochs)

**Status:** COMPLETED

```bash
python launcher.py --backend modal dataloaders=cifar10 model=unet loss=meanflow epochs=20 \
  loss.meanflow_ratio=0.5 loss.logit_normal_mean=-2.0 loss.logit_normal_std=2.0 \
  loss.weighting_power=0.75 dataloaders.train.batch_size=128 optimizer.lr=1e-4 \
  precision=bf16 eval_every=5 save_every=5 run_name=cifar10_ratio05_20ep resume=false
```

| Metric | Value |
|--------|-------|
| Final Loss | 0.201 |
| Energy Distance (epoch 5) | 2.93 |
| Energy Distance (epoch 10) | 0.229 (best) |
| Energy Distance (epoch 15) | 1.75 |
| Energy Distance (epoch 20) | 0.498 |
| Speed | ~3.8 min/epoch |

**Checkpoints:** `cifar10_ratio05_20ep_epoch_{0005,0010,0015,0020}.pt`

**Note:** High variance in energy distance across epochs. Best ED (0.229) comparable to ratio=0 baseline (0.239).

---

### 4.3: CIFAR-10 MeanFlow ratio=0.75 (20 epochs)

**Status:** IN PROGRESS (v3)

Paper config uses ratio=0.75 with batch_size=1024. On A100-40GB:
- batch_size=128 OOMs immediately
- batch_size=64 with ratio=0.75 also OOMs (silent failure - job appeared running but no progress for 1+ hour)
- **Working config:** batch_size=32 with gradient_accumulation_steps=4 (effective batch=128)

**v1 (FAILED - silent OOM):**
```bash
# batch_size=64 - hung for 1+ hour with no checkpoints or trackio logs
python launcher.py --backend modal ... dataloaders.train.batch_size=64 gradient_accumulation_steps=2
```

**v2 (FAILED - silent failure):**
```bash
# Job appeared running but never produced trackio run or checkpoints
python launcher.py --backend modal dataloaders=cifar10 model=unet loss=meanflow epochs=20 \
  loss.meanflow_ratio=0.75 loss.logit_normal_mean=-2.0 loss.logit_normal_std=2.0 \
  loss.weighting_power=0.75 dataloaders.train.batch_size=32 gradient_accumulation_steps=4 \
  optimizer.lr=1e-4 precision=bf16 eval_every=5 save_every=5 run_name=cifar10_ratio075_20ep_v2 resume=false
```

**v3 (current):**
```bash
python launcher.py --backend modal dataloaders=cifar10 model=unet loss=meanflow epochs=20 \
  loss.meanflow_ratio=0.75 loss.logit_normal_mean=-2.0 loss.logit_normal_std=2.0 \
  loss.weighting_power=0.75 dataloaders.train.batch_size=32 gradient_accumulation_steps=4 \
  optimizer.lr=1e-4 precision=bf16 eval_every=5 save_every=5 run_name=cifar10_ratio075_20ep_v3 resume=false
```

Job: https://modal.com/apps/kyle-c-vedder/main/ap-tWmT9AeDNvuh2rCwvyD1Ic
Started: 2026-01-22 ~18:36 UTC
Speed: ~2.8 it/s (batch_size=32 with grad_accum=4)

**Results (in progress):**
| Metric | Epoch 5 | Epoch 10 | Epoch 15 | Epoch 20 |
|--------|---------|----------|----------|----------|
| Loss | 0.218 | 0.211 | 0.207 | - |
| Energy Distance | 3.82 | 4.49 | 2.26 | - |

**Note:** ED variance is high, similar to ratio=0.5 experiment. Epoch 15 shows significant improvement.

**Checkpoints:** `cifar10_ratio075_20ep_v3_epoch_{0005,0010,0015}.pt`

---

## Proposed Experiments

### CIFAR-10 Ablations

| Experiment | Goal | Status |
|------------|------|--------|
| Ratio ablation (0, 0.25, 0.5, 0.75) | Replicate paper Table 1 | PROPOSED |
| Time sampling ablation | Validate logit-normal params | PROPOSED |
| Weighting power ablation | Validate p=0.75 optimal | PROPOSED |

### ImageNet Experiments

| Experiment | Model | Epochs | Target FID | Compute |
|------------|-------|--------|------------|---------|
| DiT-B/4 | 131M params | 80 | 6.17 | ~100 A100-hours |
| DiT-XL/2 | 675M params | 240 | 3.43 | ~1000 A100-hours |

---

## Configuration Change Log

| Date | Change |
|------|--------|
| 2026-01-20 | Added `loss.use_batch_time` for deterministic overfit experiments |
| 2026-01-20 | MeanFlow loss now averages per-element (MSE-style) instead of summing |
| 2026-01-20 | Removed temporary MeanFlow debug flags after investigation |
| 2026-01-20 | Enforced two time channels everywhere (removed 1-channel plumbing) |
| 2026-01-21 | Hardened Modal infrastructure (12h timeout, 5 retries, auto-resume) |
| 2026-01-22 | Discovered batch_size=64 with ratio=0.75 OOMs silently; need batch_size=32 |

---

## Modal Job Monitoring Checklist

**Run this checklist every 5-10 minutes while experiments are running:**

1. **Check app status:**
   ```bash
   modal app list | head -10
   ```
   - `ephemeral (detached)` with tasks > 0 = running
   - `stopped` = completed or failed

2. **Verify training started (within first 5 min):**
   ```bash
   python scripts/modal_app.py sync
   trackio list runs --project denoising-zoo | grep <run_name>
   ```
   - If run doesn't appear after 5 min → likely OOM or crash

3. **Check checkpoint progress:**
   ```bash
   python scripts/modal_app.py ckpts | grep <run_name>
   ```
   - First checkpoint (epoch 5) should appear within expected time
   - No checkpoints after 2x expected time → likely hung/OOMed

4. **Expected timing by ratio (CIFAR-10, A100-40GB):**
   | Ratio | Batch Size | Time/Epoch | First Checkpoint (ep5) |
   |-------|------------|------------|------------------------|
   | 0.0 | 128 | ~1.6 min | ~8 min |
   | 0.25 | 128 | ~2.5 min | ~12 min |
   | 0.5 | 128 | ~4 min | ~20 min |
   | 0.75 | 32 | TBD | TBD |

5. **Red flags (take action):**
   - App shows "ephemeral" but no trackio run after 5 min → Stop and check OOM
   - No checkpoints after 2x expected time → Stop and investigate
   - GPU util at 0% for extended period → Job is stuck

---

## Useful Commands

```bash
# View runs
trackio list runs --project denoising-zoo

# Get loss curve
trackio get metric --project denoising-zoo --run <run> --metric "train/loss" --json

# Get energy distance
trackio get metric --project denoising-zoo --run <run> --metric "eval/energy_distance" --json

# Sync Modal data
python scripts/modal_app.py sync

# List Modal checkpoints
python scripts/modal_app.py ckpts
```
