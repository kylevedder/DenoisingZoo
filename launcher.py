#!/usr/bin/env python3
"""
Launcher script for training jobs with support for local and Modal backends.
Replaces the original train.sh script with Python-based logic.
"""

import argparse
import subprocess
import shlex
import sys
from datetime import datetime
from pathlib import Path

import torch

from helpers import has_mps_backend, is_mps_available

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
    print(f"MPS backend present: {has_mps_backend()}")
    mps_available = is_mps_available()
    print(f"MPS available: {mps_available}")
    if mps_available:
        return "mps"

    # Fallback to CPU
    print("No GPU available, using CPU")
    return "cpu"


def run_local_training(device: str, extra_args: list[str], run_name: str) -> None:
    """Run training locally with the specified device."""
    print(f"Running local training on device: {device}")

    # Activate virtual environment and run training
    activate_script = ".venv/bin/activate"
    args_with_run = list(extra_args)
    if not any(
        str(a).startswith("run_name=") or str(a).startswith("+run_name=")
        for a in args_with_run
    ):
        # Use Hydra append syntax to allow adding keys to struct configs
        args_with_run.append(f"+run_name={run_name}")
    extra = " ".join(shlex.quote(a) for a in args_with_run)
    cmd = [
        "bash",
        "-c",
        f"source {activate_script} && python train.py device={device} {extra}",
    ]

    subprocess.run(cmd, check=True)


def run_modal_training(extra_args: list[str], run_name: str, detach: bool = True) -> None:
    """Run training on Modal with NVIDIA GPU.

    Args:
        extra_args: Hydra config overrides to pass to train.py
        run_name: Name for this training run (used for checkpoints/logging)
        detach: If True, run detached so job continues if client disconnects
    """
    print("Running training on Modal with NVIDIA GPU...")

    # Prefer Modal CLI for reliable execution
    try:
        subprocess.run(["modal", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'modal' CLI not found. Install with: pip install modal")
        sys.exit(1)

    args_with_run = list(extra_args)
    if not any(
        str(a).startswith("run_name=") or str(a).startswith("+run_name=")
        for a in args_with_run
    ):
        args_with_run.append(f"+run_name={run_name}")
    cli_args = " ".join(shlex.quote(a) for a in args_with_run)
    activate_script = ".venv/bin/activate"

    # Use --detach for fire-and-forget execution that survives client disconnect
    detach_flag = "--detach" if detach else ""
    cmd = [
        "bash",
        "-c",
        f"source {activate_script} && modal run {detach_flag} scripts/modal_app.py -- {cli_args}",
    ]

    if detach:
        print("Job will run detached - you can close this terminal and it will continue.")
        print(f"Monitor progress: python scripts/modal_app.py sync && trackio show --project denoising-zoo")

    subprocess.run(cmd, check=True)

    if detach:
        print("\nJob submitted successfully!")
        print("Sync trackio data:  python scripts/modal_app.py sync")
        print("List checkpoints:   python scripts/modal_app.py ckpts")


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
        "--no-detach",
        action="store_true",
        help="For Modal backend: stream logs instead of running detached (job dies if you disconnect)",
    )
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to pass to train.py"
    )

    args = parser.parse_args()

    # Set up virtual environment
    setup_venv()

    # Generate a default run name timestamp if not provided by user
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.backend == "modal":
        run_modal_training(args.extra_args, run_name, detach=not args.no_detach)
    else:
        # Local training
        device = args.device
        if device == "auto":
            device = detect_device()
        run_local_training(device, args.extra_args, run_name)


if __name__ == "__main__":
    main()
