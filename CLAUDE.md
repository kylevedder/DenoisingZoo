# CLAUDE.md

## MANDATORY: Consult Codex + Gemini

**For ALL major decisions, code changes, and plans: consult both Codex and Gemini BEFORE declaring complete.**

This is not optional. These tools catch mistakes, verify math, and provide independent review. Run them in parallel:

```bash
echo "Your detailed prompt with full code context" | codex exec
NODE_NO_WARNINGS=1 gemini "Your detailed prompt with full code context"
```

**What requires verification:**
- Any non-trivial code change
- Architecture decisions
- Mathematical implementations (loss functions, JVP, gradients)
- Performance optimizations
- Bug fixes that touch core logic

**Provide full context** - complete code, equations, what it should do, edge cases. Address feedback from BOTH tools before proceeding. If they disagree, investigate.

---

## CRITICAL: Experiment Monitoring

**NEVER stop monitoring experiments until explicitly told to stop.**

When running Modal experiments:
1. Continuously poll status (`modal app list`, sync trackio, check checkpoints)
2. Update `EXPERIMENT_LOG.md` with results as they come in
3. Start next experiment immediately when one completes
4. Report failures promptly and restart

## Project Overview

Flow Matching research codebase with MeanFlow implementation for single-step generation.

## Experiment Log

**Always keep `EXPERIMENT_LOG.md` up to date** with configs, commands, results, and observations.

## Device Strategy

- **MPS**: Only for quick unit tests and micro-tests that don't need performance
- **Modal + CUDA**: All real training and performance-sensitive work

```bash
# Real training - always use Modal
python launcher.py --backend modal run_name=exp dataloaders=cifar10 model=unet loss=meanflow

# Recommended CUDA settings for MeanFlow
python launcher.py --backend modal run_name=exp loss=meanflow loss.full_batch_jvp=true loss.use_cuda_graph=true precision=bf16
```

**WARNING: Do NOT use `torch.compile` with MeanFlow/JVP** - adds compile overhead without speedup, some modes incompatible.

## Code Quality

Write clean, direct code. No defensive programming.

- **Typing**: `list[int]`, `dict[str, Any]`, `x: int | None = None`
- **Files**: Use `pathlib.Path`, never `os.path`
- **No** unnecessary `hasattr()`, defensive try-except, or overly cautious None checks
- **No** `__init__.py` files unless explicitly asked
- Prefer inheritance over Protocols

## Setup

```bash
./scripts/setup-hooks.sh  # Install pre-commit hooks (runs tests before commit)
```

## Commands

### Training

```bash
# Every run needs run_name - use descriptive names: cifar10_unet_lr1e4, dit_b_bs64_ep100
python launcher.py run_name=my_experiment

# Modal (preferred for real training)
python launcher.py --backend modal run_name=exp dataloaders=cifar10 model=unet

# Resume
python launcher.py run_name=my_experiment resume=true
```

### Modal Operations

```bash
python scripts/modal_app.py sync      # Sync trackio logs
python scripts/modal_app.py list      # List runs
python scripts/modal_app.py ckpts     # List checkpoints
python scripts/modal_app.py download unet/last.pt
```

### Trackio

```bash
trackio show --project denoising-zoo
trackio list runs --project denoising-zoo
python scripts/trackio_delete.py --list  # Always list before deleting
```

## Architecture

### Input Pattern
Models receive state + time fused: `(B, C, H, W)` + `(B, 2)` → `(B, C+2, H, W)`

### MeanFlow
Single-step generation: `x = noise + model(noise, r=0, t=1)` (1 NFE)

### Key Files
- `losses/meanflow_loss.py`: MeanFlow loss with JVP
- `configs/`: Hydra YAML configs
- `models/unet/`, `models/dit/`: Model implementations

### Checkpoints
- Live: `outputs/ckpts/<arch>/last.pt`
- Archive: `outputs/ckpts/<arch>/archive/<run_name>_epoch_XXXX.pt`
