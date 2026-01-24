# Design: BF16 Mixed-Precision Training

## Overview

This document describes the implementation plan for bfloat16 (bf16) mixed-precision training in DenoisingZoo, following the official MeanFlow implementation which uses bf16 for weights.

## Motivation

1. **Memory efficiency**: BF16 uses 16 bits vs 32 bits, reducing memory footprint by ~50%
2. **Throughput**: Modern GPUs (Ampere+) have dedicated BF16 tensor cores
3. **Numerical stability**: BF16 has same exponent range as FP32 (unlike FP16), no GradScaler needed
4. **Alignment**: Official MeanFlow implementation uses bf16

## Verification Sign-off

- [x] **Gemini**: Approved with minor feedback on optimizer state handling
- [x] **Codex**: Identified critical issues (addressed in v2 of this design)

### Critical Issues Identified by Codex

1. **bf16 weights + fp32 JVP = dtype mismatch**: If model weights are bf16 and autocast is disabled for JVP with fp32 inputs, PyTorch raises dtype errors (`mat1 and mat2 must have the same dtype`)

2. **GroupNorm32 incompatible with bf16 weights**: The `GroupNorm32` class casts inputs to float32 but expects float32 parameters. With bf16 parameters, this causes `mixed dtype: expect parameter to have scalar type of Float`

3. **Solver/eval paths lack autocast**: If model is bf16 but solver/eval call with fp32 inputs, operations will fail

## Two Approaches

Based on reviewer feedback, we present two approaches:

### Approach A: Standard Mixed Precision (Recommended)

**Keep weights in fp32, use autocast for bf16 compute.**

This is the standard PyTorch AMP pattern and avoids all dtype conflicts.

```python
# Weights stay fp32
model = build_model(cfg, device)  # fp32 weights

# Autocast handles bf16 compute during forward pass
with torch.autocast(device_type, dtype=torch.bfloat16):
    loss = criterion(model(x), y)  # ops run in bf16 where safe

# JVP: disable autocast, run in fp32 (weights are already fp32)
with torch.autocast(device_type, enabled=False):
    _, jvp = torch.func.jvp(u_func, inputs, tangents)  # fp32
```

**Pros**:
- No dtype conflicts anywhere
- JVP naturally runs in fp32 (weights are fp32)
- Solver/eval work without changes
- Standard pattern, well-tested

**Cons**:
- Slightly higher memory (fp32 weights + bf16 activations)
- Optimizer states are fp32 (typical anyway)

### Approach B: True BF16 Weights (Advanced)

**Cast weights to bf16, keep norm layers in fp32, use autocast everywhere.**

This matches the "bf16 for weights" claim but requires careful handling.

```python
# Cast model to bf16, except norm layers
model = build_model(cfg, device)
for m in model.modules():
    if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
        m.float()  # Keep norms in fp32
    else:
        m.to(torch.bfloat16)

# Autocast everywhere (training, eval, solver)
with torch.autocast(device_type, dtype=torch.bfloat16):
    loss = criterion(model(x), y)

# JVP: two options
# Option 1: Run JVP in bf16 (faster but potentially noisier)
# Option 2: Maintain fp32 model copy for JVP (slower but precise)
```

**Pros**:
- Lower memory footprint
- Potentially faster (fewer dtype conversions)

**Cons**:
- Complex: must track which layers are which dtype
- Must wrap solver/eval in autocast
- JVP in bf16 may introduce noise, or need fp32 model copy
- GroupNorm32 pattern breaks, need different approach

## Recommended Approach: A (Standard Mixed Precision)

For this codebase, **Approach A is recommended** because:

1. JVP stability is critical for MeanFlow training
2. Solver/eval don't need changes
3. Memory savings from bf16 weights are marginal vs fp32 weights + bf16 activations
4. Less complexity, fewer edge cases

## Design (Approach A)

### 1. Fix device_type Detection

**File**: `helpers.py`

**Current** (buggy):
```python
def build_precision_settings(precision: str, device: torch.device) -> PrecisionSettings:
    ...
    # Wrong: hardcodes "cpu" for non-CUDA devices
    device_type = "cuda" if device.type == "cuda" else "cpu"
```

**Proposed**:
```python
def build_precision_settings(precision: str, device: torch.device) -> PrecisionSettings:
    p = precision.lower()
    # Correct device_type for autocast
    if device.type in ("cuda", "mps"):
        device_type = device.type
    else:
        device_type = "cpu"

    if p in {"fp32", "float32"}:
        return PrecisionSettings(None, False, device_type)
    if p in {"bf16", "bfloat16"}:
        return PrecisionSettings(torch.bfloat16, False, device_type)
    if p in {"fp16", "float16", "half"}:
        if device.type != "cuda":
            raise ValueError("fp16 is only supported on CUDA devices")
        return PrecisionSettings(torch.float16, True, "cuda")
    raise ValueError(f"Unknown precision: {precision}")
```

This ensures:
- `torch.autocast(device_type="mps")` works on Apple Silicon
- fp16 is explicitly rejected on non-CUDA (it requires GradScaler which doesn't work on MPS/CPU)

### 2. Keep Model in fp32 (No Change)

The current code already keeps models in fp32:

```python
def build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = instantiate(cfg.model)
    model.to(device)  # device only, dtype stays fp32
    return model
```

No change needed. Autocast handles bf16 compute.

### 3. JVP Precision (Already Correct)

The current MeanFlow loss implementation is already correct for Approach A:

```python
# losses/meanflow_loss.py, line ~247
# Disable autocast for numerical stability during JVP
device_type = z_t_mf.device.type
with torch.amp.autocast(device_type, enabled=False):
    z_float = z_t_mf.float()
    r_float = r_mf.float()
    t_float = t_mf.float()
    ...
    _, jvp_result = torch.func.jvp(u_func, primals, tangents)
```

With fp32 weights, disabling autocast and using float inputs works correctly.

### 4. Configuration

**File**: `configs/train.yaml` (already exists)

```yaml
precision: fp32  # choices: fp32, bf16
                 # fp16 requires CUDA
```

Document that:
- `fp32`: No autocast, all ops in fp32
- `bf16`: Autocast to bf16 where safe, fp32 weights, fp32 JVP
- `fp16`: CUDA only, uses GradScaler

### 5. Add Autocast to Evaluation (Optional Enhancement)

For potential speedups in eval, wrap evaluation in autocast:

**File**: `helpers.py`

```python
def evaluate_epoch_energy_distance(
    ...,
    precision_settings: PrecisionSettings | None = None,
):
    device_type = precision_settings.device_type if precision_settings else "cpu"
    autocast_dtype = precision_settings.autocast_dtype if precision_settings else None

    with torch.autocast(device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        # existing eval code
```

This is optional since eval is typically fast, but provides consistency.

## Implementation Order

1. **Phase 1: Bug Fix** (no behavior change for fp32)
   - Fix device_type detection in `helpers.py`
   - Add fp16 device validation

2. **Phase 2: Testing**
   - Add bf16 variants to existing tests
   - Verify JVP correctness in bf16 autocast context
   - Test on MPS device

3. **Phase 3: Documentation**
   - Update CLAUDE.md with bf16 usage guidance
   - Document precision trade-offs

## Testing Strategy

### Unit Tests

```python
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_meanflow_loss_precision(precision, device):
    """Loss values should be finite and similar across precisions."""
    settings = build_precision_settings(precision, device)

    with torch.autocast(settings.device_type, dtype=settings.autocast_dtype,
                        enabled=settings.autocast_dtype is not None):
        loss = criterion(batch, device)

    assert torch.isfinite(loss)
    # bf16 loss should be within 1% of fp32 loss

@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_jvp_correctness_precision(precision, device):
    """JVP should match finite differences."""
    # Reference computed in fp32
    # bf16 test uses larger epsilon and looser tolerance
```

### Integration Tests

```python
def test_bf16_training_smoke():
    """Run 10 steps of bf16 training, verify loss is finite and doesn't explode."""
    # Fixed seed for reproducibility
    # Assert loss < 10 * initial_loss (doesn't explode)
    # Assert all losses are finite

def test_bf16_mps_autocast():
    """Verify autocast works on MPS with bf16."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    # Run forward pass under autocast, check output dtype
```

### Tolerances

```python
# fp32 tests
assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

# bf16 tests (relaxed due to ~3 decimal digits precision)
assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)

# JVP finite-difference tests
# fp32: epsilon=1e-4
# bf16: epsilon=1e-3 (larger to account for precision)
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MPS bf16 edge cases | Test thoroughly, document known issues |
| JVP noise in bf16 context | JVP runs in fp32, only autocast context is bf16 |
| Eval precision differences | Optional: add autocast to eval for consistency |

## Open Questions (Resolved)

1. **Should we support FP16?**
   - Answer: Yes, but CUDA-only with GradScaler. Error on MPS/CPU.

2. **Cast weights to bf16 or use autocast only?**
   - Answer: Autocast only (Approach A). Simpler, avoids dtype conflicts.

3. **JVP in bf16?**
   - Answer: No. Keep JVP in fp32 for stability. With fp32 weights, this is automatic.

## References

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [BF16 vs FP16 Comparison](https://cloud.google.com/tpu/docs/bfloat16)
- [PyTorch Mixed Precision Best Practices](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
