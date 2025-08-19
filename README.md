# Denoising Zoo

## macOS (Apple Silicon) setup with uv

This project is set up to run natively on macOS with Apple Silicon using the Metal (MPS) backend in PyTorch. We use `uv` for fast, reproducible Python environments.

### Prerequisites
- macOS on Apple Silicon (M1/M2/M3)
- Homebrew installed
  - If you don't have it:
    ```bash
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
    ```
- uv installed
  ```bash
  brew install uv
  ```

### Quick start (recommended)
The `train.sh` script prepares the environment, syncs dependencies, auto-detects MPS, and runs training.

```bash
chmod +x ./train.sh        # one-time
./train.sh                 # runs with MPS if available, else CPU

# Examples
./train.sh epochs=1        # 1 epoch
./train.sh device=cpu      # force CPU
```

On startup you will see diagnostics like:
```
[train.sh] torch version: 2.8.0
[train.sh] mps available: True
[train.sh] selected device: mps
...
torch version: 2.8.0
mps backend present: True
mps available: True
selected device: mps
sample tensor device: mps:0
```

### Manual environment commands (optional)
If you prefer to manage the environment yourself:

```bash
# Create & activate a Python 3.12 venv managed by uv
uv venv --python 3.12 .venv
source .venv/bin/activate

# Reproduce the environment from lockfile
uv sync --no-install-project

# Run training
python train.py device=mps
# or
uv run python train.py device=mps
```

### Configuration
Training is configured via Hydra. Defaults in `configs/train.yaml`. You can override via CLI:

```bash
./train.sh epochs=10 batch_size=1024
./train.sh device=cpu
```

### Notes
- MPS works only on macOS natively (not inside Docker). This repo is configured for native macOS.
- You may see a warning about `pin_memory` on MPS; it is benign.
- Dependencies are defined in `pyproject.toml` and locked in `uv.lock` for reproducibility.

