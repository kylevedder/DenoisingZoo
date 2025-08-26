# Denoising Zoo

This is a clean, simple, hackable codebase for research on denoising models. From this, we hope you can gain a deeper understanding of how these models work and develop new research ideas.

<p>
  <img src="docs/kmeans_flow.gif" alt="K-means Flow Animation" width="400" />
  <img src="docs/kmeans_particles.gif" alt="K-means Particles" width="400" />
</p>

## Getting Started

Prerequisites:
- uv (for venv and dependency management)
- Python 3.12
- Optional: Modal CLI (`pip install modal`) if you want to run on remote NVIDIA GPUs

The launcher handles creating a local `.venv/` via uv and syncing dependencies automatically.

## Usage

### Local (auto-detect device)
```bash
python launcher.py
```

Explicit device:
```bash
python launcher.py --device mps   # Apple Silicon
python launcher.py --device cuda  # NVIDIA
python launcher.py --device cpu
```

Hydra overrides (examples):
```bash
python launcher.py epochs=2 eval_every=1
python launcher.py dataloaders=celeba model=cnn
```

Notes:
- Checkpoints are saved per-architecture at `outputs/ckpts/<arch>/last.pt`.
- `eval_every` controls how often evaluation runs each epoch.

### Modal (remote NVIDIA GPU, streaming logs)
Authenticate once:
```bash
modal token new
```

Run (CelebA + CNN example):
```bash
python launcher.py --backend modal dataloaders=celeba model=cnn
```

Optional dataset persistence across runs (recommended):
```bash
modal volume create training-data
# optional preload from local machine:
# modal volume put training-data data/celeba
```


## Launcher usage

Use the Python launcher to set up the venv (via uv) and run training locally or on Modal.

### Local (auto-detect device)

```bash
python launcher.py
```

Explicit device:

```bash
python launcher.py --device mps
python launcher.py --device cuda
```

Pass Hydra overrides (training config):

```bash
python launcher.py epochs=2 eval_every=1
python launcher.py dataloaders=celeba model=cnn
```

### Modal (remote NVIDIA GPU, logs stream)

One-time auth:

```bash
modal token new
```

Run on GPU (CelebA + CNN example):

```bash
python launcher.py --backend modal dataloaders=celeba model=cnn
```

Tip: First run may download CelebA. For persistence across runs, create a Modal volume and (optionally) preload the dataset:

```bash
modal volume create training-data
# optional (from local machine):
# modal volume put training-data data/celeba
```

