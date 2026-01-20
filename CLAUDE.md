# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DenoisingZoo is a research codebase for Flow Matching—a generative modeling technique where neural networks learn velocity fields to transform between distributions via ODE integration.

## Commands

### Training
```bash
# Local training (auto-detects device: CUDA → MPS → CPU)
python launcher.py

# Explicit device
python launcher.py --device mps

# With Hydra config overrides
python launcher.py dataloaders=celeba model=cnn epochs=100

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

### Visualization
```bash
cd visualizers && ./vis_all.sh
```

## Architecture

### Unified Input Pattern
All models receive state and time fused into a single tensor:
- Dense data: `(B, D)` + `(B, 1)` → `(B, D+1)` (time concatenated as feature)
- Images: `(B, C, H, W)` + `(B, 1)` → `(B, C+1, H, W)` (time as extra channel)

Implementation: `dataloaders/base_dataloaders.py::make_unified_input()`

### Data Flow
```
Dataset → BaseItem(unified_input, target, t, input)
    ↓
DataLoader (DictDatasetAdapter)
    ↓
train.py: loss = criterion(model(unified_input), target)
    ↓
Solver (Euler/RK4): integrates dx/dt = v(x,t) from t=0 to t=1
    ↓
Evaluation: Energy Distance between predicted and ground truth distributions
```

### Hydra Configuration
- Main config: `configs/train.yaml`
- Config groups: `dataloaders/`, `model/`, `loss/`
- Override syntax: `python launcher.py dataloaders=celeba model=cnn precision=bf16`

### Key Directories
- `models/flow_matching/`: MLP and CNN velocity field models
- `models/vqvae/`: VQVAE model and losses
- `dataloaders/`: Dataset implementations (kmeans synthetic 2D, celeba images)
- `solvers/`: ODE integrators (Euler, RK4)
- `visualizers/`: Animation generation scripts
- `configs/`: Hydra YAML configs

### Checkpoints
- Live: `outputs/ckpts/<arch>/last.pt`
- Archive: `outputs/ckpts/<arch>/archive/<run_name>_epoch_XXXX.pt`
- Checkpoints embed full resolved config for reproducibility

### Adding New Components
Extend by subclassing and adding a Hydra config:
- Models: subclass `nn.Module`, add config in `configs/model/`
- Datasets: subclass from `dataloaders/base_dataloaders.py`, add config in `configs/dataloaders/`
- Solvers: subclass `BaseSolver` from `solvers/base_solver.py`
