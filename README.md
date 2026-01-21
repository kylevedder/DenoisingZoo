# Denoising Zoo

A clean, simple, hackable codebase for research on denoising models.

## Papers Implemented

- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2022)
- **MeanFlow**: [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) - single-step generation via mean velocity fields
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (Peebles & Xie, 2023)

## Models

| Model | Params | Use Case |
|-------|--------|----------|
| MLP | <1M | 2D synthetic data |
| SmallCNN | ~0.5M | Small images |
| UNet | ~51M | CIFAR-10 (32×32 pixel space) |
| DiT-S | ~33M | Small latent experiments |
| DiT-B | ~131M | ImageNet latent (32×32×4) |
| DiT-L | ~458M | ImageNet latent |
| DiT-XL | ~675M | ImageNet latent |

## Datasets

- `kmeans`: 2D synthetic Gaussian mixtures
- `celeba`: CelebA faces (64×64)
- `cifar10`: CIFAR-10 (32×32)
- `imagenet`: ImageNet (latent space via SD VAE)

<p>
  <img src="docs/kmeans_flow.gif" alt="K-means Flow Animation" width="400" />
  <img src="docs/kmeans_particles.gif" alt="K-means Particles" width="400" />
</p>

*Left: Learned velocity field evolving over time t∈[0,1]. Right: Particles transported from Gaussian noise (t=0) to target Gaussian mixture (t=1) via ODE integration.*

## Getting Started

Prerequisites:
- uv (for venv and dependency management)
- Python 3.12

The launcher handles creating a local `.venv/` via uv and syncing dependencies automatically.

## Usage

```bash
python launcher.py run_name=my_experiment
```

Device selection:
```bash
python launcher.py run_name=exp --device mps   # Apple Silicon
python launcher.py run_name=exp --device cuda  # NVIDIA
python launcher.py run_name=exp --device cpu
```

Hydra overrides:
```bash
python launcher.py run_name=exp epochs=100 eval_every=10
python launcher.py run_name=exp dataloaders=celeba model=cnn
python launcher.py run_name=exp dataloaders=cifar10 model=unet loss=meanflow
```

Checkpoints are saved at `outputs/ckpts/<arch>/last.pt`.

## Modal (Remote GPU)

One-time auth:
```bash
modal token new
```

Run on remote NVIDIA GPU:
```bash
python launcher.py --backend modal run_name=exp dataloaders=celeba model=cnn
```

Optional dataset persistence:
```bash
modal volume create training-data
```

## Visualization

Generate animations from a trained checkpoint:

```bash
cd visualizers && ./vis_all.sh [CKPT_PATH] [CFG_PATH]
```

Outputs to `outputs/vis/`:
- `kmeans_flow.gif` / `.mp4` - velocity field visualization
- `particles.gif` - particle transport (Euler solver)
- `particles_rk4.gif` - particle transport (RK4 solver)

Individual visualizers:
```bash
# Velocity field animation
python visualizers/kmeans_field_vis.py --ckpt outputs/ckpts/mlp/last.pt --out field.gif

# Particle transport
python visualizers/kmeans_particles_vis.py --ckpt outputs/ckpts/mlp/last.pt --out particles.gif --solver rk4
```
