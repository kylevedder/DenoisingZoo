# Loss Function Investigation Instructions

This document provides context for reviewing the MeanFlow loss implementation for correctness.

## Papers

1. **MeanFlow**: "Mean Flows for One-step Generative Modeling" (arXiv 2505.13447)
   - Paper: https://arxiv.org/abs/2505.13447
   - HTML: https://arxiv.org/html/2505.13447v1

2. **Flow Matching**: "Flow Matching for Generative Modeling" (Lipman et al.)
   - Paper: https://arxiv.org/abs/2210.02747

## Key Equations from MeanFlow Paper

### Interpolation (Standard Flow Matching)
```
z_t = (1 - t) * x + t * y
```
Where `x` is noise (source), `y` is data (target), `t ∈ [0, 1]`.

### Velocity Field
```
v = y - x  (constant, time-independent)
```

### Average Velocity Definition (Eq. 3)
```
u(z_t, r, t) = (1/(t-r)) ∫_r^t v(z_τ, τ) dτ
```

### MeanFlow Identity (Eq. 6)
```
u(z_t, r, t) = v(z_t, t) - (t - r) * (du/dt)
```

### JVP Formula (Eq. 8)
The total time derivative along the trajectory:
```
du/dt = v · ∂u/∂z + ∂u/∂t
```
This is computed via JVP with tangent vector `[v, 1]` for inputs `[z, t]`.

### Training Loss (Eq. 9-10)
```
L(θ) = E[||u_θ(z_t, r, t) - sg(u_tgt)||²]

u_tgt = v_t - (t - r) * (v_t · ∂v/∂z + ∂v/∂t)
```
Where `sg()` is stop-gradient.

### Time Sampling
- `t` sampled from logit-normal distribution: `t = sigmoid(N(μ, σ))`
- `r` sampled uniformly from `[0, t]`
- A fraction (`meanflow_ratio`) of samples use `r ≠ t`; the rest use `r = t` (standard FM)

### Adaptive Weighting
```
w = 1 / (||Δ||² + c)^p
```
Applied with stop-gradient.

## Implementation Files

### Main Loss Implementation
**File**: `losses/meanflow_loss.py`

Key sections to review:
- Lines 56-66: Logit-normal time sampling
- Lines 98-107: Time `t` and `r` sampling logic
- Lines 109-119: Interpolation `z_t`, `z_r`, and velocity `v_true`
- Lines 135-170: JVP computation and MeanFlow target
- Lines 175-189: Adaptive weighting and final loss

### Tests
- `tests/test_meanflow_jvp_optimization.py` - JVP correctness tests
- `tests/test_meanflow_target.py` - MeanFlow target formula tests
- `tests/test_flow_matching_math.py` - Basic flow matching math
- `tests/test_time_sampling.py` - Logit-normal sampling tests
- `tests/test_adaptive_weighting.py` - Weighting scheme tests

## Specific Things to Verify

### 1. JVP Computation (CRITICAL)
The implementation computes:
```python
def model_fn_mf(z: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
    unified = make_unified_flow_matching_input(z, t_input)
    return self._model(unified)

tangent_t = torch.ones_like(t_mf)
v_t_mf, jvp_mf = torch.func.jvp(
    model_fn_mf,
    (z_t_mf, t_mf),
    (v_t_mf_tangent, tangent_t),  # tangents: [v, 1]
)
```

**Question**: Does this correctly compute `v · ∂v/∂z + ∂v/∂t` as per Eq. 8?

### 2. Model Parameterization
The paper describes `u_θ(z, r, t)` taking both time endpoints.
The implementation uses `v_θ(z, t)` with single time input.

The loss trains `v_θ(z_r, r)` to match `u_tgt` computed from `v_θ(z_t, t)`.

**Question**: Is this parameterization equivalent? Does it correctly learn the average velocity?

### 3. Inference Procedure
For one-step generation, the paper implies:
```
x_1 = x_0 + u_θ(x_0, r=0, t=1)
```

With a `v_θ(z, t)` model, what should inference be?
- `x_1 = x_0 + v_θ(x_0, t=0)`?
- `x_1 = x_0 + v_θ(x_0, t=1)`?
- Something else?

### 4. Stop Gradient Placement
The implementation applies `detach()` to `u_tgt_mf`:
```python
u_tgt[mf_mask] = u_tgt_mf.detach()
```

**Question**: Is this the correct place for stop-gradient per Eq. 9?

### 5. Loss Computation Location
The loss is computed as:
```python
v_r = self._model(unified_r)  # Prediction at (z_r, r)
diff = v_r - u_tgt            # Target computed from (z_t, t)
loss = (weights * sq_error).mean()
```

**Question**: Should the prediction be at `(z_r, r)` or `(z_t, t)`?

## Known Concerns

1. **Model takes (z, t) not (z, r, t)**: The paper's formulation has the model explicitly condition on both `r` and `t`. Our implementation uses a standard velocity model that only sees one time value. This might be intentional (simpler architecture) but needs verification.

2. **Inference procedure unclear**: I haven't verified how one-step generation should work with this parameterization.

## How to Run Tests
```bash
# Run all loss-related tests
python -m pytest tests/test_meanflow_*.py tests/test_flow_matching_math.py tests/test_time_sampling.py tests/test_adaptive_weighting.py -v

# Run a quick training test
python launcher.py dataloaders=synthetic model=unet loss=meanflow epochs=1 run_name=test trackio.enabled=false
```

## Summary Checklist

- [ ] JVP formula matches Eq. 8
- [ ] MeanFlow target formula matches Eq. 6
- [ ] Time/r sampling matches paper Section 4.3
- [ ] Adaptive weighting matches paper
- [ ] Stop gradient placement is correct
- [ ] Model parameterization (z,t) vs (z,r,t) is valid
- [ ] Inference procedure is clear and correct
