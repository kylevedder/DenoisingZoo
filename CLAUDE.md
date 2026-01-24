# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Experiment Monitoring Behavior

**NEVER stop monitoring and running experiments until the user explicitly tells you to stop.**

When running experiments on Modal or any remote compute:
1. **Continuously poll for status** - Check `modal app list`, sync trackio data, check checkpoints
2. **Update EXPERIMENT_LOG.md** with results as they come in
3. **Start the next experiment** immediately when one completes
4. **Report issues promptly** - If a job fails, investigate and restart
5. **Do not wait for user prompts** - Proactively monitor and manage the experiment queue

This is research work - experiments take time and need active management. Stopping prematurely wastes GPU hours and delays results.

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

## Code Quality Standards

**Write clean, direct code. Avoid defensive programming patterns that obscure intent.**

### Typing
- Use lowercase generics: `list[int]`, `tuple[str, ...]`, `dict[str, Any]`
- Use `| None` not `Optional[T]`: `def foo(x: int | None = None)`
- Imports should be absolute from the repo root

### File Operations
- Use `pathlib.Path`, never `os.path` operations
- Example: `Path("data") / "file.txt"` not `os.path.join("data", "file.txt")`

### Control Flow
- **No unnecessary `hasattr()` checks.** If an attribute should exist, access it directly. If you're unsure about the interface, fix the interface.
- **No defensive try-except in straight-line code.** Don't wrap code that should always succeed. Let errors propagate with clear tracebacks.
- **No overly cautious None checks.** If a value should never be None at that point, don't check for it.

```python
# BAD - defensive garbage
def process(model):
    if hasattr(model, 'encoder'):
        try:
            result = model.encoder(x)
        except Exception:
            result = None
    return result

# GOOD - direct and clear
def process(model):
    return model.encoder(x)
```

### Class Design
- Prefer inheritance over Protocols. Python's Protocol/structural typing is a mess—use explicit base classes for shared interfaces.

### Project Hygiene
- Do not create `__init__.py` files unless explicitly asked

## AI-Assisted Verification (Codex + Gemini)

**CRITICAL: Before declaring any plan or code change complete, get verification from both Codex and Gemini.**

Two CLI tools are available for external validation:
- **Codex** (`codex`) - OpenAI Codex 5.2 High Reasoning
- **Gemini** (`gemini`) - Google Gemini 3 (1M context)

Both tools serve the same purpose: verifying math, code correctness, and design decisions. Use both for independent verification.

### Required Verification Workflow

**For all non-trivial changes:**
1. Write initial implementation or plan
2. Run **both Codex and Gemini in parallel** with the same verification prompt
3. Address feedback from both before declaring done
4. If they disagree, investigate and resolve

**Provide detailed context.** These agents work best with clear, complete information. Include:
- The full code being reviewed (not just snippets)
- Relevant equations or paper references
- What the code is supposed to do
- Any constraints or edge cases to consider
- Related code that interacts with the changes

The more context you provide, the more effective their verification will be.

### Invocation

**Codex** requires non-interactive mode via `codex exec` with stdin:
```bash
echo "Your prompt here" | codex exec
```

**Gemini** uses positional arguments:
```bash
NODE_NO_WARNINGS=1 gemini "Your prompt here"
```

### Example Prompts

```bash
# Mathematical derivations and paper implementations
echo "Review this MeanFlow loss implementation against Equation 7 from the paper: [paste code]" | codex exec
NODE_NO_WARNINGS=1 gemini "Review this MeanFlow loss implementation against Equation 7 from the paper: [paste code]"

# Gradient/JVP/VJP verification
echo "Verify the JVP computation in this loss function matches the chain rule derivation" | codex exec
NODE_NO_WARNINGS=1 gemini "Verify the JVP computation in this loss function matches the chain rule derivation"

# Architectural review
echo "Review this training loop refactor for correctness and edge cases" | codex exec
NODE_NO_WARNINGS=1 gemini "Review this training loop refactor for correctness and edge cases"

# Code review
echo "Review these changes to the dataloader. Are there any bugs or missing cases?" | codex exec
NODE_NO_WARNINGS=1 gemini "Review these changes to the dataloader. Are there any bugs or missing cases?"
```

### Sign-off Checklist

Before declaring a task complete:
- [ ] Both Codex and Gemini reviewed the changes
- [ ] Both tools' feedback has been addressed
- [ ] If disagreements exist between tools, investigate and resolve

## Commands

### Git Hooks

Pre-commit hooks run unit tests before each commit. Install with:

```bash
./scripts/setup-hooks.sh
```

This installs `hooks/pre-commit` to `.git/hooks/`. The hook:
- Runs all unit tests via pytest
- Blocks commit if any test fails
- Can be skipped with `git commit --no-verify` (use sparingly)

To uninstall: `rm .git/hooks/pre-commit`

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

### torch.compile (Kernel Fusion)

Enable `torch.compile` for potential speedups via kernel fusion:

```bash
python launcher.py run_name=my_experiment compile=true

# With specific compile mode
python launcher.py run_name=my_experiment compile=true compile_mode=max-autotune
```

**Compile modes:**
- `default` - balanced compilation (recommended starting point)
- `reduce-overhead` - reduces Python overhead, good for small batches
- `max-autotune` - tries more kernel configs, slower compile but potentially faster runtime

**Notes:**
- Default is `compile=false` (off)
- First batch is slower due to compilation warmup
- Works with MeanFlow loss (JVP is compatible with compiled models)
- Compiled models are automatically unwrapped for type checks and checkpoint naming

**When to use:** Try `compile=true` when GPU utilization is low or training is CPU-bound. May help fuse small kernels on MPS.

### Mixed Precision (BF16)

**BF16 is the recommended precision for MeanFlow experiments.** It provides ~2x memory reduction with minimal precision loss.

```bash
# Recommended for MeanFlow training
python launcher.py run_name=my_experiment precision=bf16 loss=meanflow

# Works on CUDA, MPS, and CPU
python launcher.py run_name=my_experiment precision=bf16 device=mps
```

**Precision options:**
- `fp32` - Full precision (default, safest)
- `bf16` - BFloat16 mixed precision (recommended for MeanFlow)
- `fp16` - Float16 with GradScaler (CUDA only, not recommended)

**How it works:**
- Model weights stay in fp32
- Forward/backward passes use bf16 via `torch.autocast`
- JVP computation in MeanFlow loss automatically runs in fp32 for numerical stability
- No GradScaler needed (bf16 has same dynamic range as fp32)

**Device support:**
- **CUDA**: Requires Ampere or newer GPU (A100, A10G, RTX 30xx+). Pre-Ampere GPUs will raise an error.
- **MPS**: Fully supported on Apple Silicon
- **CPU**: Supported but slow (useful for debugging)

**Why bf16 over fp16:**
- bf16 has same exponent range as fp32 (no gradient underflow issues)
- No GradScaler complexity
- Better numerical stability for JVP/higher-order derivatives

### CUDA Performance Optimizations

On CUDA devices, the following optimizations are automatically enabled:
- **TF32 matmul** (`torch.set_float32_matmul_precision("high")`) - uses TensorFloat-32 on Ampere+ GPUs for ~3x faster matmuls with minimal precision loss
- **cuDNN benchmark** (`torch.backends.cudnn.benchmark = True`) - auto-tunes convolution algorithms for specific input sizes

These are enabled automatically when `device=cuda` - no config needed.

### MeanFlow JVP Performance (CUDA)

MeanFlow training requires computing Jacobian-vector products (JVP) which adds overhead compared to standard flow matching. On CUDA, enable hybrid CUDA graph mode for best performance:

```bash
# Recommended CUDA settings for MeanFlow (add to loss config or command line)
python launcher.py run_name=exp loss=meanflow loss.full_batch_jvp=true loss.use_cuda_graph=true precision=bf16
```

**Performance comparison (A100, UNet, batch=64):**

| Mode | Time | vs Standard FM |
|------|------|----------------|
| Standard FM (baseline) | 43ms | - |
| MeanFlow (default selective) | 145ms | +237% |
| MeanFlow (CUDA graph + BF16) | 81ms | **+72%** |
| At 25% ratio (paper default) | ~52ms | **~19%** |

**Options:**
- `full_batch_jvp=true` - Compute JVP for all samples (removes CPU-GPU sync)
- `use_cuda_graph=true` - Capture JVP in CUDA graph (47% speedup, requires full_batch_jvp)

**Notes:**
- CUDA graph mode requires static batch sizes
- MPS does not support CUDA graphs - use default selective mode
- The ~19% overhead at 25% ratio is close to the paper's claimed 16%

### Remote Training (Modal)
```bash
modal token new  # one-time auth
python launcher.py --backend modal run_name=my_experiment dataloaders=celeba model=cnn
```

**Syncing trackio logs during and after training:**

Trackio logs are stored in a Modal volume. The volume is automatically committed every 5 minutes during training, so you can sync progress while a long run is still active:

```bash
# In a separate terminal, sync trackio data from Modal to local
python scripts/modal_app.py sync

# List runs stored in Modal volume
python scripts/modal_app.py list

# View synced data locally
trackio show --project denoising-zoo
```

For long runs, periodically run `sync` to pull the latest metrics. The sync merges Modal data with your local trackio database.

**Downloading checkpoints from Modal:**

Checkpoints are automatically persisted to the Modal volume during training. To download them locally:

```bash
# List available checkpoints
python scripts/modal_app.py ckpts

# Download a specific checkpoint
python scripts/modal_app.py download unet/last.pt                    # to outputs/ckpts/unet/last.pt
python scripts/modal_app.py download unet/archive/run_epoch_0050.pt  # archived checkpoint
```

Note: Large checkpoints (500MB+) may hit Modal's response size limits. For very large models, consider mounting the Modal volume locally.

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

**IMPORTANT: Always run `--list` first before deleting.** Deletions are permanent and cannot be undone. List all runs to see exact names before constructing a delete pattern.

```bash
# ALWAYS list first to see exact run names
python scripts/trackio_delete.py --list

# Preview what would be deleted (use --dry-run before actual delete)
python scripts/trackio_delete.py "old_experiment" --dry-run

# Delete runs matching a regex pattern (with confirmation)
python scripts/trackio_delete.py "test_.*"
```

**Storage:** `~/.cache/huggingface/trackio/`

### Visualization
```bash
cd visualizers && ./vis_all.sh
```

## Architecture

### Unified Input Pattern
All models receive state and time fused into a single tensor:
- Dense data: `(B, D)` + `(B, 2)` → `(B, D+2)` (time concatenated as features)
- Images: `(B, C, H, W)` + `(B, 2)` → `(B, C+2, H, W)` (time as extra channels)

For standard flow matching, the second time input is fixed to `1`.

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

# MeanFlow: single forward pass (u(z, r, t) with r=0, t=1)
x = noise + model(noise, r=0, t=1)  # 1 NFE
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

**Cleaning up test/debug checkpoints:**

UNet checkpoints are ~580MB each and accumulate quickly. Periodically clean up test experiments:

```bash
# Check disk usage
du -sh outputs/ckpts/*/archive/

# List unique experiment names
ls outputs/ckpts/unet/archive/ | sed 's/_epoch_[0-9]*\.pt$//' | sort -u

# Count checkpoints and size per experiment
for run in $(ls outputs/ckpts/unet/archive/ | sed 's/_epoch_[0-9]*\.pt$//' | sort -u); do
  count=$(ls outputs/ckpts/unet/archive/${run}_epoch_*.pt 2>/dev/null | wc -l | tr -d ' ')
  size=$(du -ch outputs/ckpts/unet/archive/${run}_epoch_*.pt 2>/dev/null | tail -1 | cut -f1)
  printf "%-30s %3s ckpts  %s\n" "$run" "$count" "$size"
done

# Delete checkpoints for a specific experiment
rm outputs/ckpts/unet/archive/test_*.pt
rm outputs/ckpts/unet/archive/debug_*.pt
```

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
