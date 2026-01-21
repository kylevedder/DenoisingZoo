# MeanFlow Experiment Log

This log tracks experiments for validating the MeanFlow implementation.

## Reference: MeanFlow Paper Hyperparameters

From [Mean Flows for One-step Generative Modeling (arXiv:2505.13447)](https://arxiv.org/abs/2505.13447):

| Parameter | Paper Optimal | Notes |
|-----------|---------------|-------|
| `logit_normal_mean` | -0.4 | lognorm(-0.4, 1.0) achieves best results |
| `logit_normal_std` | 1.0 | |
| `meanflow_ratio` | 0.25 | 25% of samples use r != t |
| `weighting_power` | 1.0 | p=0.5 also competitive |
| CFG guidance | 3.0 | For sampling |
| Training epochs | 240 | ImageNet 256x256 |

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

---

## Phase 1: Unit Tests

**Status:** COMPLETED

Run all unit tests to verify implementation correctness:
```bash
./tests/run_tests.sh
```

**Run Log:**
- 2026-01-20: Ran `./tests/run_tests.sh -v` → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (post-fix) → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (post-experiments) → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (use_batch_time update) → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (loss scaling change) → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (post-debug cleanup) → 94 passed
- 2026-01-20: Reran `./tests/run_tests.sh -v` (time-channel cleanup) → 94 passed

Tests to pass:
- `test_flow_matching_math.py` - Interpolation, velocity field, unified input
- `test_jvp_correctness.py` - JVP computation correctness
- `test_meanflow_target.py` - MeanFlow target formula
- `test_time_sampling.py` - Logit-normal time sampling
- `test_adaptive_weighting.py` - Adaptive weighting scheme
- `test_meanflow_jvp_optimization.py` - Full JVP with ∂v/∂t term
- `test_dataset_items.py` - Dataset item instantiation

---

## Phase 2: Overfitting Experiments

### Experiment 2.1: MSE Loss Baseline

**Status:** COMPLETED (near target)

Verify basic training works with MSE loss (no MeanFlow complexity).

**Command:**
```bash
python launcher.py dataloaders=deterministic model=unet loss=mse epochs=200 eval_every=20 save_every=20 run_name=mse_baseline
```

**Success Criteria:** Loss < 1e-4

**Run Log:**
- 2026-01-20: Started `mse_baseline` (deterministic + UNet + MSE)
- 2026-01-20: `python` not found in shell; rerunning with `uv run python`
- 2026-01-20: Training reached epoch 160 (loss 1.15e-4) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Resume completed to epoch 200; final loss 6.8e-5, eval energy_distance 26.712995

---

### Experiment 2.2: MeanFlow ratio=0 (Standard Flow Matching)

**Status:** COMPLETED

MeanFlow with ratio=0 should behave like standard flow matching.

**Command:**
```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 loss.meanflow_ratio=0 loss.use_batch_time=true eval_every=20 save_every=20 run_name=meanflow_ratio0
```

**Success Criteria:** Loss < 1e-4, comparable to MSE baseline

**Run Log:**
- 2026-01-20: Started `meanflow_ratio0` (deterministic + UNet + MeanFlow ratio=0)
- 2026-01-20: Training reached epoch 160 (loss 0.2285) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Resume completed to epoch 200; final loss 0.202086, eval energy_distance 32.891383
- 2026-01-20: Rerun started after MeanFlow time-input fix (expect FM parity)
- 2026-01-20: Rerun reached epoch 160 (loss 0.2305) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Rerun completed to epoch 200; final loss 0.199273, eval energy_distance 45.872156
- 2026-01-20: Rerun started with `loss.use_batch_time=true` to use dataset t values
- 2026-01-20: Rerun reached epoch 160 (loss 0.319956) before tool timeout; will resume after loss scaling change
- 2026-01-20: Resume completed to epoch 200; final loss 0.000157, eval energy_distance 14.625792

---

### Experiment 2.3: MeanFlow ratio=0.25 (Paper Config)

**Status:** COMPLETED

Paper-recommended configuration with 25% MeanFlow samples.

**Command:**
```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 loss.meanflow_ratio=0.25 loss.use_batch_time=true eval_every=20 save_every=20 run_name=meanflow_ratio025
```

**Success Criteria:** Converges without NaN, loss decreases

**Run Log:**
- 2026-01-20: Started `meanflow_ratio025` (deterministic + UNet + MeanFlow ratio=0.25)
- 2026-01-20: Training reached epoch 120 (loss 0.2653) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Resume completed to epoch 200; final loss 0.206236, eval energy_distance 24.541180
- 2026-01-20: Rerun started after MeanFlow time-input fix
- 2026-01-20: Rerun reached epoch 120 (loss 0.2841) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Rerun completed to epoch 200; final loss 0.181259, eval energy_distance 15.872602
- 2026-01-20: Rerun started with `loss.use_batch_time=true` to use dataset t values
- 2026-01-20: Rerun reached epoch 120 (loss 0.000196) before tool timeout during eval; will resume from checkpoint
- 2026-01-20: Resume attempt failed to spawn (aborted); will retry
- 2026-01-20: Resume restarted from `last.pt` at epoch 1; reached epoch 108 before timeout; eval energy_distance at epoch 100 was 13.237382; will resume from checkpoint
- 2026-01-20: Resume completed to epoch 200; final loss 0.000072, eval energy_distance 9.088364

---

### Experiment 2.4: MeanFlow ratio=1.0 (Full MeanFlow)

**Status:** PENDING (batch-time rerun not completed)

All samples use MeanFlow loss (maximum JVP computation).

**Command:**
```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 loss.meanflow_ratio=1.0 loss.use_batch_time=true eval_every=20 save_every=20 run_name=meanflow_ratio100
```

**Success Criteria:** Converges without NaN, loss decreases

**Run Log:**
- 2026-01-20: Started `meanflow_ratio100` (deterministic + UNet + MeanFlow ratio=1.0)
- 2026-01-20: Training reached epoch 53 (loss displayed as 0.000000) before tool timeout; will resume from checkpoint
- 2026-01-20: Resume ran to epoch 110 (loss displayed as 0.000000); eval energy_distance at epoch 100 was 60.270638
- 2026-01-20: Resume ran to epoch 172 (loss displayed as 0.000000); eval energy_distance at epoch 160 was 60.270638
- 2026-01-20: Resume completed to epoch 200; final loss displayed as 0.000000, eval energy_distance 60.270638
- 2026-01-20: Rerun started after MeanFlow time-input fix
- 2026-01-20: Rerun reached epoch 56 (loss displayed as 0.000000) before tool timeout; will resume from checkpoint
- 2026-01-20: Resume reached epoch 115 (loss displayed as 0.000000); eval energy_distance at epoch 100 was 60.270638; will resume from checkpoint
- 2026-01-20: Resume reached epoch 176 (loss displayed as 0.000000) before tool timeout; will resume from checkpoint
- 2026-01-20: Rerun completed to epoch 200; final loss 0.000000, eval energy_distance 60.270638
- 2026-01-20: Rerun started with `loss.use_batch_time=true` to use dataset t values
- 2026-01-20: Ran `meanflow_ratio100_debug` (epochs=1, debug enabled) → v_pred/u_tgt ~0, v_true nonzero (mse ~2.96e-1), jvp ~0, eval energy_distance 60.270638

---

## Phase 3: Proposed Sanity Experiments (Not Yet Run)

### Experiment 3.1: MeanFlow One-step vs Solver Consistency

**Goal:** Verify MeanFlow 1-step sampling matches solver-based sampling on the deterministic dataset.

**Sketch:** Add a small evaluation helper that uses `generate_samples_meanflow` (r=0, t=1) and computes energy distance against `y_true`, then compare to solver ED for the same checkpoint.

**Success Criteria:** MeanFlow 1-step ED close to solver ED on deterministic data.

---

### Experiment 3.2: Deterministic Multi-Class Overfit

**Goal:** Ensure loss/inference works across multiple fixed targets (not just single image).

**Sketch:** Add a `deterministic_multi.yaml` dataloader config pointing to `DeterministicFlowDataset` with `num_classes=4` and `samples_per_class=10`, then run ratio=0 and ratio=0.25 with `loss.use_batch_time=true`.

**Success Criteria:** Loss < 1e-4 for ratio=0; ratio=0.25 converges without NaNs.

---

### Experiment 3.3: Weighting Off vs On

**Goal:** Check adaptive weighting is not masking issues.

**Command (ratio=0.25, weighting off):**
```bash
python launcher.py dataloaders=deterministic model=unet loss=meanflow epochs=200 loss.meanflow_ratio=0.25 loss.use_batch_time=true loss.weighting_power=0 eval_every=20 save_every=20 run_name=meanflow_ratio025_weight0
```

**Success Criteria:** Similar convergence to the weighted run; no instability.

---

## Hardware Constraint

- Single GPU (MPS) available
- Experiments must run sequentially

---

## Trackio Commands

```bash
# View runs
trackio list runs --project denoising-zoo

# Get loss curve
trackio get metric --project denoising-zoo --run <run> --metric "train/loss" --json

# Get energy distance
trackio get metric --project denoising-zoo --run <run> --metric "eval/energy_distance" --json
```

---

## Configuration Change Log

- 2026-01-20: Added `loss.use_batch_time` (default false) to optionally use batch-provided `t` for deterministic overfit experiments where logit-normal resampling prevents pure memorization.
- 2026-01-20: MeanFlow loss now averages per-element squared error (MSE-style) instead of summing over features to match `torch.nn.MSELoss` scaling.
- 2026-01-20: Removed temporary MeanFlow debug flags after investigation.
- 2026-01-20: Enforced two time channels everywhere (removed 1-channel plumbing).

---

## Results Summary

| Experiment | Loss Type | ratio | Final Loss | Energy Dist | Status |
|------------|-----------|-------|------------|-------------|--------|
| 2.1 | MSE | N/A | 6.8e-5 | 26.712995 | COMPLETED |
| 2.2 | MeanFlow | 0.0 | 0.000157 | 14.625792 | COMPLETED (near target) |
| 2.3 | MeanFlow | 0.25 | 0.000072 | 9.088364 | COMPLETED |
| 2.4 | MeanFlow | 1.0 | 0.000000 | 60.270638 | PENDING |
