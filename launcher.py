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
from typing import List

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
    activate_script = ".venv/bin/activate"
    cmd = [
        "bash",
        "-c",
        f"source {activate_script} && python train.py device={device} {' '.join(extra_args)}",
    ]

    subprocess.run(cmd, check=True)


def run_modal_training(extra_args: List[str]) -> None:
    """Run training on Modal with NVIDIA GPU (streams logs)."""
    print("Running training on Modal with NVIDIA GPU...")

    # Prefer Modal CLI for reliable log streaming
    try:
        subprocess.run(["modal", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'modal' CLI not found. Install with: pip install modal")
        sys.exit(1)

    cli_args = " ".join(extra_args)
    activate_script = ".venv/bin/activate"
    cmd = [
        "bash",
        "-c",
        f"source {activate_script} && modal run scripts/modal_app.py -- {cli_args}",
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Launch training jobs with support for local and Modal backends"
    )
    parser.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="Backend to use for training (default: local)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for local training (default: auto)",
    )
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to pass to train.py"
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
