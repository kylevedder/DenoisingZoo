#!/usr/bin/env python3
"""
Launcher script for training jobs with support for local and Modal backends.
Replaces the original train.sh script with Python-based logic.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch


def check_uv_available() -> bool:
    """Check if uv is available in the system."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_venv() -> None:
    """Set up virtual environment using uv."""
    if not check_uv_available():
        print("Error: 'uv' is not installed. Install via: pip install uv")
        sys.exit(1)
    
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run(["uv", "venv", "--python", "3.12", ".venv"], check=True)
    
    # Sync dependencies
    print("Syncing dependencies...")
    subprocess.run(["uv", "sync", "--no-install-project"], check=True)


def detect_device() -> str:
    """Detect the best available device for training."""
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        return "cuda"
    
    # Check MPS availability (Apple Silicon)
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    print(f"MPS available: {mps_available}")
    if mps_available:
        return "mps"
    
    # Fallback to CPU
    print("No GPU available, using CPU")
    return "cpu"


def run_local_training(device: str, extra_args: List[str]) -> None:
    """Run training locally with the specified device."""
    print(f"Running local training on device: {device}")
    
    # Activate virtual environment and run training
    if os.name == "nt":  # Windows
        activate_script = ".venv\\Scripts\\activate"
        cmd = ["cmd", "/c", f"{activate_script} && python train.py device={device}"] + extra_args
    else:  # Unix-like
        activate_script = ".venv/bin/activate"
        cmd = ["bash", "-c", f"source {activate_script} && python train.py device={device} {' '.join(extra_args)}"]
    
    subprocess.run(cmd, check=True)


def run_modal_training(extra_args: List[str]) -> None:
    """Run training on Modal with NVIDIA GPU."""
    print("Running training on Modal with NVIDIA GPU...")
    
    # Import modal here to avoid dependency issues when running locally
    try:
        import modal
    except ImportError:
        print("Error: modal is not installed. Run: uv sync")
        sys.exit(1)
    
    # This will be handled by the modal_app.py module
    from modal_app import run_modal_training
    run_modal_training(extra_args)


def main():
    parser = argparse.ArgumentParser(
        description="Launch training jobs with support for local and Modal backends"
    )
    parser.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="Backend to use for training (default: local)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for local training (default: auto)"
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments to pass to train.py"
    )
    
    args = parser.parse_args()
    
    # Set up virtual environment
    setup_venv()
    
    if args.backend == "modal":
        run_modal_training(args.extra_args)
    else:
        # Local training
        device = args.device
        if device == "auto":
            device = detect_device()
        run_local_training(device, args.extra_args)


if __name__ == "__main__":
    main()