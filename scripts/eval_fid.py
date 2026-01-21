"""FID evaluation script for generative models.

Computes FID against CIFAR-10 training set using clean-fid.
Supports both MeanFlow 1-step and multi-step solver sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)
from models.unet import UNet
from solvers.euler_solver import EulerSolver
from hydra.utils import instantiate


def sample_meanflow_1step(
    model: torch.nn.Module,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate samples using MeanFlow 1-step (r=0, t=1).

    Args:
        model: Trained velocity field model.
        noise: Starting noise of shape (B, C, H, W).
        device: Device to use.

    Returns:
        Generated samples of shape (B, C, H, W).
    """
    B = noise.shape[0]

    # Build time input for r=0, t=1
    r_tensor = torch.zeros(B, 1, device=device, dtype=noise.dtype)
    t_tensor = torch.ones(B, 1, device=device, dtype=noise.dtype)
    time_input = make_time_input(t_tensor, r=r_tensor)

    # Build unified input
    unified = make_unified_flow_matching_input(noise, time_input)

    # MeanFlow 1-step: x1 = x0 + (t - r) * v = x0 + 1 * v
    with torch.no_grad():
        v = model(unified)
    return noise + v


def sample_with_solver(
    model: torch.nn.Module,
    noise: torch.Tensor,
    solver: EulerSolver,
) -> torch.Tensor:
    """Generate samples using ODE solver.

    Args:
        model: Trained velocity field model.
        noise: Starting noise of shape (B, C, H, W).
        solver: ODE solver instance.

    Returns:
        Generated samples of shape (B, C, H, W).
    """
    with torch.no_grad():
        result = solver.solve(noise)
    return result.final_state


def generate_samples_to_dir(
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    sample_shape: tuple[int, int, int],
    use_solver: bool = False,
    solver_steps: int = 50,
) -> None:
    """Generate samples and save to directory.

    Args:
        model: Trained velocity field model.
        device: Device to use.
        output_dir: Directory to save samples.
        num_samples: Number of samples to generate.
        batch_size: Batch size for generation.
        sample_shape: Shape of each sample (C, H, W).
        use_solver: If True, use ODE solver; else use MeanFlow 1-step.
        solver_steps: Number of solver steps (only if use_solver=True).
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = None
    if use_solver:
        solver = EulerSolver(model, num_steps=solver_steps)

    sample_idx = 0
    pbar = tqdm(total=num_samples, desc="Generating samples")

    while sample_idx < num_samples:
        current_batch = min(batch_size, num_samples - sample_idx)
        noise = torch.randn(current_batch, *sample_shape, device=device)

        if use_solver:
            samples = sample_with_solver(model, noise, solver)
        else:
            samples = sample_meanflow_1step(model, noise, device)

        # Clamp to valid range and save
        samples = samples.clamp(-1, 1)

        for i in range(current_batch):
            # Convert from [-1, 1] to [0, 1] for saving
            img = (samples[i] + 1) / 2
            save_image(img, output_dir / f"{sample_idx:05d}.png")
            sample_idx += 1
            pbar.update(1)

    pbar.close()


def compute_fid_score(
    generated_dir: Path,
    dataset_name: str = "cifar10",
    dataset_res: int = 32,
    dataset_split: str = "train",
    mode: str = "clean",
    device: torch.device | None = None,
) -> float:
    """Compute FID score against reference dataset.

    Args:
        generated_dir: Directory containing generated images.
        dataset_name: Reference dataset name (e.g., "cifar10").
        dataset_res: Resolution of reference dataset.
        dataset_split: Split of reference dataset ("train" or "test").
        mode: FID computation mode ("clean" recommended).
        device: Device to use for feature extraction.

    Returns:
        FID score.
    """
    # clean-fid uses "cuda" string, not torch.device
    device_str = "cuda" if device is not None and device.type == "cuda" else "cpu"
    if device is not None and device.type == "mps":
        # MPS not directly supported, fall back to CPU
        device_str = "cpu"

    score = fid.compute_fid(
        fdir1=str(generated_dir),
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        mode=mode,
        device=torch.device(device_str),
        verbose=True,
    )
    return score


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (model, config dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Create model with same config as training
    model = UNet(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 3),
        base_channels=model_config.get("base_channels", 128),
        time_channels=model_config.get("time_channels", 2),
        channel_mult=model_config.get("channel_mult", [1, 2, 2, 2]),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        attention_resolutions=model_config.get("attention_resolutions", [16]),
        dropout=model_config.get("dropout", 0.1),
        num_heads=model_config.get("num_heads", 4),
        input_resolution=model_config.get("input_resolution", 32),
        use_separate_time_embeds=model_config.get("use_separate_time_embeds", True),
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Compute FID for trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of samples to generate (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use",
    )
    parser.add_argument(
        "--use-solver",
        action="store_true",
        help="Use ODE solver instead of MeanFlow 1-step",
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=50,
        help="Number of solver steps (only if --use-solver)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated samples (uses temp dir if not specified)",
    )
    parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="Keep generated samples after computing FID",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Loaded model from epoch {config.get('epoch', 'unknown')}")

    # Determine sample shape from model config
    model_config = config.get("model", {})
    sample_shape = (
        model_config.get("in_channels", 3),
        model_config.get("input_resolution", 32),
        model_config.get("input_resolution", 32),
    )
    print(f"Sample shape: {sample_shape}")

    # Generate samples
    method = "solver" if args.use_solver else "meanflow_1step"
    print(f"\nGenerating {args.num_samples} samples using {method}...")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="fid_samples_")
        output_dir = Path(temp_dir)

    generate_samples_to_dir(
        model=model,
        device=device,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sample_shape=sample_shape,
        use_solver=args.use_solver,
        solver_steps=args.solver_steps,
    )

    # Compute FID
    print(f"\nComputing FID against CIFAR-10 train set...")
    fid_score = compute_fid_score(
        generated_dir=output_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="clean",
        device=device,
    )
    print(f"\n{'=' * 60}")
    print(f"FID Score ({method}): {fid_score:.2f}")
    print(f"{'=' * 60}")

    # Cleanup
    if not args.keep_samples and not args.output_dir:
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleaned up temporary directory")
    else:
        print(f"Samples saved to: {output_dir}")


if __name__ == "__main__":
    main()
