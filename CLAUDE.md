# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DenoisingZoo is a research codebase for Flow Matching—a generative modeling technique where neural networks learn velocity fields to transform between distributions via ODE integration. Includes MeanFlow implementation for single-step generation.

## Development Environment

**Primary development is on Apple Silicon (MPS).** Always use MPS for training and testing to utilize hardware acceleration and save wallclock time. Default device configs are set to `mps`. When writing training loops or test scripts, always move tensors and models to the appropriate device:

```python
device = torch.device("mps")
model = Model().to(device)
data = data.to(device)
```

## Commands

### Training
```bash
# Local training (auto-detects device: CUDA → MPS → CPU)
python launcher.py

# Explicit device
python launcher.py --device mps

# With Hydra config overrides
python launcher.py dataloaders=celeba model=cnn epochs=100

# CIFAR-10 with UNet and MeanFlow loss
python launcher.py dataloaders=cifar10 model=unet loss=meanflow epochs=100

# ImageNet with DiT (requires pre-computed latents or VAE)
python launcher.py dataloaders=imagenet model=dit_b loss=meanflow epochs=80

# Resume from checkpoint
python launcher.py resume=true

# Eval-only mode
python launcher.py eval_checkpoint=outputs/ckpts/mlp/archive/run_YYYYMMDD_HHMMSS_epoch_0010.pt
```

### Remote Training (Modal)
```bash
modal token new  # one-time auth
python launcher.py --backend modal dataloaders=celeba model=cnn
```

### Experiment Tracking (Trackio)
Trackio is enabled by default. Metrics logged per epoch:
- `epoch`, `train/loss`, `eval/energy_distance`
- On MPS: `mps/allocated_mb`, `mps/driver_mb`, `gpu/util_pct`, `gpu/freq_mhz`, `gpu/power_w`, `gpu/temp_c`

For GPU metrics, install macmon: `brew install macmon`

```bash
# View dashboard (launches in browser)
trackio show --project denoising-zoo

# Disable tracking for a run
python launcher.py trackio.enabled=false
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
