# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DenoisingZoo is a research codebase for Flow Matching—a generative modeling technique where neural networks learn velocity fields to transform between distributions via ODE integration. Includes MeanFlow implementation for single-step generation.

## Experiment Log

**IMPORTANT: Always keep `EXPERIMENT_LOG.md` up to date.**

When running experiments:
1. **Before starting**: Add a new experiment section with config, command, and expected outcomes
2. **During training**: Update results as they come in (loss values, metrics)
3. **After completion**: Record final results, observations, and any issues encountered
4. **When changing configs**: Log parameter changes with reasoning in the "Configuration Change Log" section

The experiment log tracks:
- Hyperparameter configurations and their sources (e.g., paper references)
- Training commands for reproducibility
- Results tables with loss/metrics progression
- Observations, issues, and next steps

This is critical for tracking research progress and debugging training issues.

## Development Environment

**Primary development is on Apple Silicon (MPS).** Always use MPS for training and testing to utilize hardware acceleration and save wallclock time. Default device configs are set to `mps`. When writing training loops or test scripts, always move tensors and models to the appropriate device:

```python
device = torch.device("mps")
model = Model().to(device)
data = data.to(device)
```

## Commands

### Training

**Every training run requires a `run_name`** to identify it in trackio and checkpoints:

```bash
# Basic training (run_name is required!)
python launcher.py run_name=my_experiment

# With Hydra config overrides
python launcher.py run_name=celeba_cnn_v1 dataloaders=celeba model=cnn epochs=100

# CIFAR-10 with UNet and MeanFlow loss
python launcher.py run_name=cifar10_meanflow dataloaders=cifar10 model=unet loss=meanflow epochs=100

# ImageNet with DiT (requires pre-computed latents or VAE)
python launcher.py run_name=imagenet_dit_b dataloaders=imagenet model=dit_b loss=meanflow epochs=80

# Resume from checkpoint
python launcher.py run_name=my_experiment resume=true

# Eval-only mode (run_name still required but won't create new trackio run)
python launcher.py run_name=eval_run eval_checkpoint=outputs/ckpts/mlp/archive/run_YYYYMMDD_HHMMSS_epoch_0010.pt
```

**Run naming conventions:**
- Use descriptive names: `cifar10_unet_meanflow_lr1e4`
- Include key hyperparameters: `dit_b_bs64_ep100`
- Version experiments: `celeba_v1`, `celeba_v2`

### Remote Training (Modal)
```bash
modal token new  # one-time auth
python launcher.py --backend modal dataloaders=celeba model=cnn
```

### Experiment Tracking (Trackio)
Trackio is enabled by default. Each run is identified by the required `run_name` parameter.

**Metrics logged per epoch:**
- `epoch`, `train/loss`, `eval/energy_distance`
- On MPS: `mps/allocated_mb`, `mps/driver_mb`, `gpu/util_pct`, `gpu/freq_mhz`, `gpu/power_w`, `gpu/temp_c`

GPU metrics are collected via async streaming from macmon (install: `brew install macmon`). The monitor samples every 500ms and reports rolling averages over a 5-second window.

```bash
# View dashboard (launches in browser)
trackio show --project denoising-zoo

# Disable tracking for a run
python launcher.py run_name=test trackio.enabled=false
```

**CLI for reading results:**
```bash
trackio list projects                                      # list all projects
trackio list runs --project denoising-zoo                  # list runs
trackio list metrics --project denoising-zoo --run <run>   # list metrics in a run
trackio get metric --project denoising-zoo --run <run> --metric "train/loss" --json
trackio get metric --project denoising-zoo --run <run> --metric "eval/energy_distance" --json
trackio get metric --project denoising-zoo --run <run> --metric "gpu/util_pct" --json
```

**Deleting runs:**
```bash
# List all runs
python scripts/trackio_delete.py --list

# Delete runs matching a regex pattern (with confirmation)
python scripts/trackio_delete.py "test_.*"

# Preview what would be deleted
python scripts/trackio_delete.py "old_experiment" --dry-run
```

**Storage:** `~/.cache/huggingface/trackio/`

### Visualization
```bash
cd visualizers && ./vis_all.sh
```

## Architecture

### Unified Input Pattern
All models receive state and time fused into a single tensor:
- Dense data: `(B, D)` + `(B, 1)` → `(B, D+1)` (time concatenated as feature)
- Images: `(B, C, H, W)` + `(B, 1)` → `(B, C+1, H, W)` (time as extra channel)

Implementation: `dataloaders/base_dataloaders.py::make_unified_flow_matching_input()`

### Data Flow
```
Dataset → BaseItem(unified_input, target, t, input, raw_source, raw_target)
    ↓
DataLoader (DictDatasetAdapter)
    ↓
train.py: loss = criterion(model(unified_input), target)  # or MeanFlowLoss
    ↓
Solver (Euler/RK4): integrates dx/dt = v(x,t) from t=0 to t=1
    ↓
Evaluation: Energy Distance / FID between predicted and ground truth distributions
```

### MeanFlow (Single-Step Generation)
MeanFlow enables one-step generation by learning mean velocity fields:
```python
# Standard flow matching requires ODE integration
x = solver.solve(noise)  # Multiple NFEs

# MeanFlow: single forward pass
x = noise - model(noise, t=1)  # 1 NFE
```

Key files:
- `losses/meanflow_loss.py`: MeanFlow loss with JVP computation
- `helpers_cfg.py`: CFG sampling utilities
- `evaluation/sample.py`: Sample generation (multi-step and 1-NFE)

### Hydra Configuration
- Main config: `configs/train.yaml`
- Config groups: `dataloaders/`, `model/`, `loss/`
- Override syntax: `python launcher.py dataloaders=celeba model=cnn precision=bf16`

### Key Directories
- `models/flow_matching/`: MLP and CNN velocity field models
- `models/unet/`: UNet (~51M params for CIFAR-10)
- `models/dit/`: DiT (Diffusion Transformer) for ImageNet latent space
- `models/vae/`: SD VAE wrapper for latent-space training
- `dataloaders/`: Dataset implementations (kmeans, celeba, cifar10, imagenet)
- `solvers/`: ODE integrators (Euler, RK4)
- `losses/`: Loss functions (MSE, MeanFlow)
- `evaluation/`: FID computation and sample generation
- `visualizers/`: Animation generation scripts
- `configs/`: Hydra YAML configs

### Model Variants
| Model | Params | Use Case |
|-------|--------|----------|
| MLP | <1M | 2D synthetic data |
| SmallCNN | ~0.5M | Small images |
| UNet | ~51M | CIFAR-10 (32×32 pixel space) |
| DiT-S | ~33M | Small latent experiments |
| DiT-B | ~131M | ImageNet latent (32×32×4) |
| DiT-L | ~458M | ImageNet latent |
| DiT-XL | ~675M | ImageNet latent (matches paper) |

### Checkpoints
- Live: `outputs/ckpts/<arch>/last.pt`
- Archive: `outputs/ckpts/<arch>/archive/<run_name>_epoch_XXXX.pt`
- Checkpoints embed full resolved config for reproducibility

### Adding New Components
Extend by subclassing and adding a Hydra config:
- Models: subclass `nn.Module`, add config in `configs/model/`
- Datasets: subclass from `dataloaders/base_dataloaders.py`, add config in `configs/dataloaders/`
- Solvers: subclass `BaseSolver` from `solvers/base_solver.py`
- Losses: add to `losses/`, create config in `configs/loss/`

## Dependencies for Full MeanFlow Pipeline

```bash
# Core
pip install torch torchvision hydra-core omegaconf tqdm trackio

# FID evaluation
pip install clean-fid

# SD VAE (for ImageNet latent training)
pip install diffusers safetensors

# Optional: Modal for remote training
pip install modal
```
