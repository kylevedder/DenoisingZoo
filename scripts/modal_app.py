"""
Modal application for remote training on NVIDIA GPUs.
"""

import os
import sys
import subprocess
import modal


# Enable Modal client output early so image build logs are streamed locally
modal.enable_output()

app = modal.App("denoisingzoo-training")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Build steps first
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        # Install non-Torch deps at build time
        "hydra-core>=1.3,<2.0",
        "omegaconf>=2.3,<3.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "matplotlib>=3.8",
    )
    .run_commands(
        # Install CUDA-enabled torch/vision versions available on cu124 index
        "pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6,<2.7' 'torchvision>=0.21,<0.22'"
    )
    .run_commands("mkdir -p /root/.cache/torch/hub/checkpoints")
    # Then add local sources last so no build steps follow (faster dev loop)
    .add_local_dir("configs", "/root/app/configs", copy=True)
    .add_local_dir("models", "/root/app/models", copy=True)
    .add_local_dir("dataloaders", "/root/app/dataloaders", copy=True)
    .add_local_dir("metal_fallbacks", "/root/app/metal_fallbacks", copy=True)
    .add_local_dir("solvers", "/root/app/solvers", copy=True)
    .add_local_dir("visualizers", "/root/app/visualizers", copy=True)
    .add_local_file("train.py", "/root/app/train.py", copy=True)
    .add_local_file("helpers.py", "/root/app/helpers.py", copy=True)
    .add_local_file("pyproject.toml", "/root/app/pyproject.toml", copy=True)
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
)
def train_on_modal(extra_args: list[str]) -> None:
    """Run training on Modal with NVIDIA GPU."""
    # Workdir contains our code baked into the image
    os.chdir("/root/app")

    # Verify GPU availability
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    os.makedirs("outputs/ckpts", exist_ok=True)

    cmd = ["python", "train.py", "device=cuda", *extra_args]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_modal_training(extra_args: list[str]) -> None:
    """Helper function to run Modal training from launcher.py."""
    # Rely on Modal's default auth (e.g., ~/.modal.toml). If missing, Modal will prompt.
    print("Starting Modal training job...")
    with app.run():
        # Call remote; logs will stream because modal.enable_output() is enabled
        train_on_modal.remote(extra_args)
    print("Modal training job completed!")


if __name__ == "__main__":
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else []
    run_modal_training(extra_args)


@app.local_entrypoint()
def main(*extra_args: str) -> None:
    """CLI entrypoint so `modal run scripts/modal_app.py -- ...` streams logs."""
    train_on_modal.remote(list(extra_args))
