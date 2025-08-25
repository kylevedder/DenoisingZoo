"""
Modal application for remote training on NVIDIA GPUs.
"""

import os
import sys
from pathlib import Path
from typing import List

import modal


# Create a Modal stub for the training application
stub = modal.Stub("denoisingzoo-training")


@stub.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "hydra-core>=1.3,<2.0", 
        "omegaconf>=2.3,<3.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "matplotlib>=3.8",
    )
    .run_commands(
        "mkdir -p /root/.cache/torch/hub/checkpoints",
    ),
    gpu=modal.gpu.A100(),  # Use A100 GPU
    timeout=3600,  # 1 hour timeout
    secrets=[
        modal.Secret.from_name("modal-training-secrets")  # For any API keys or secrets
    ],
    volumes={
        "/data": modal.Volume.from_name("training-data"),  # Persistent storage for checkpoints
    }
)
def train_on_modal(extra_args: List[str]) -> None:
    """Run training on Modal with NVIDIA GPU."""
    import torch
    import subprocess
    import sys
    
    # Set device to CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Verify GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Copy local files to Modal
    local_files = [
        "train.py",
        "helpers.py", 
        "pyproject.toml",
        "configs/",
        "models/",
        "dataloaders/",
        "solvers/",
    ]
    
    for file_path in local_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                subprocess.run(["cp", "-r", file_path, "/root/"], check=True)
            else:
                subprocess.run(["cp", file_path, "/root/"], check=True)
    
    # Change to root directory where files were copied
    os.chdir("/root")
    
    # Create outputs directory
    os.makedirs("outputs/ckpts", exist_ok=True)
    
    # Run training with CUDA device
    cmd = ["python", "train.py", "device=cuda"] + extra_args
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Copy results back to persistent volume
        if os.path.exists("outputs"):
            subprocess.run(["cp", "-r", "outputs", "/data/"], check=True)
            print("Training results saved to persistent volume")
            
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        raise


@stub.local_entrypoint()
def main(extra_args: List[str] = None):
    """Main entrypoint for Modal training."""
    if extra_args is None:
        extra_args = []
    
    print("Starting Modal training job...")
    train_on_modal.remote(extra_args)
    print("Modal training job completed!")


def run_modal_training(extra_args: List[str]) -> None:
    """Helper function to run Modal training from launcher.py."""
    # Set up Modal token if not already set
    if not os.getenv("MODAL_TOKEN_ID") and not os.getenv("MODAL_TOKEN_SECRET"):
        print("Modal credentials not found. Please set up Modal authentication:")
        print("1. Run: modal token new")
        print("2. Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables")
        sys.exit(1)
    
    # Run the Modal app
    with stub.run():
        main(extra_args)


if __name__ == "__main__":
    # Allow running directly: python modal_app.py [extra_args...]
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else []
    run_modal_training(extra_args)