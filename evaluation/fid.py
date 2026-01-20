"""FID (Fréchet Inception Distance) evaluation utilities.

Uses clean-fid library for accurate FID computation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from cleanfid import fid as cleanfid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False


def compute_fid(
    model: nn.Module,
    sample_fn: Callable[[], torch.Tensor],
    num_samples: int,
    dataset_name: str = "cifar10",
    dataset_split: str = "train",
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    seed: int = 42,
) -> float:
    """Compute FID between generated samples and a reference dataset.

    Args:
        model: The generative model (not directly used, for API consistency)
        sample_fn: Function that returns a batch of samples (B, C, H, W) in [-1, 1]
        num_samples: Number of samples to generate for FID
        dataset_name: Name of reference dataset ("cifar10", "imagenet", etc.)
        dataset_split: Split of reference dataset ("train", "test")
        device: Device for computation
        batch_size: Batch size for feature extraction
        seed: Random seed

    Returns:
        FID score (lower is better)
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )

    # Generate samples and save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "samples"
        sample_dir.mkdir()

        print(f"Generating {num_samples} samples for FID evaluation...")
        samples_generated = 0

        while samples_generated < num_samples:
            # Get batch of samples
            samples = sample_fn()

            # Convert from [-1, 1] to [0, 255] uint8
            samples = (samples.clamp(-1, 1) + 1) * 127.5
            samples = samples.to(torch.uint8).cpu().numpy()

            # Save each sample as PNG
            for i, sample in enumerate(samples):
                if samples_generated >= num_samples:
                    break

                # Convert from (C, H, W) to (H, W, C)
                img = sample.transpose(1, 2, 0)

                # Handle grayscale
                if img.shape[2] == 1:
                    img = img.squeeze(2)

                img_path = sample_dir / f"{samples_generated:06d}.png"
                Image.fromarray(img).save(img_path)
                samples_generated += 1

        print(f"Computing FID against {dataset_name} {dataset_split}...")

        # Compute FID using clean-fid
        fid_score = cleanfid.compute_fid(
            str(sample_dir),
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            mode="clean",
            batch_size=batch_size,
        )

        return fid_score


def compute_fid_from_samples(
    samples: torch.Tensor,
    dataset_name: str = "cifar10",
    dataset_split: str = "train",
    batch_size: int = 64,
) -> float:
    """Compute FID from pre-generated samples.

    Args:
        samples: Generated samples tensor of shape (N, C, H, W) in [-1, 1]
        dataset_name: Name of reference dataset
        dataset_split: Split of reference dataset
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "samples"
        sample_dir.mkdir()

        print(f"Saving {len(samples)} samples for FID evaluation...")

        # Convert from [-1, 1] to [0, 255] uint8
        samples = (samples.clamp(-1, 1) + 1) * 127.5
        samples = samples.to(torch.uint8).cpu().numpy()

        # Save each sample as PNG
        for i, sample in enumerate(tqdm(samples, desc="Saving samples")):
            # Convert from (C, H, W) to (H, W, C)
            img = sample.transpose(1, 2, 0)

            # Handle grayscale
            if img.shape[2] == 1:
                img = img.squeeze(2)

            img_path = sample_dir / f"{i:06d}.png"
            Image.fromarray(img).save(img_path)

        print(f"Computing FID against {dataset_name} {dataset_split}...")

        fid_score = cleanfid.compute_fid(
            str(sample_dir),
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            mode="clean",
            batch_size=batch_size,
        )

        return fid_score


def compute_fid_between_folders(
    folder1: str | Path,
    folder2: str | Path,
    batch_size: int = 64,
) -> float:
    """Compute FID between two folders of images.

    Args:
        folder1: Path to first folder of images
        folder2: Path to second folder of images
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )

    return cleanfid.compute_fid(
        str(folder1),
        str(folder2),
        mode="clean",
        batch_size=batch_size,
    )


def compute_inception_stats(
    folder: str | Path,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Inception statistics (mu, sigma) for a folder of images.

    Args:
        folder: Path to folder of images
        batch_size: Batch size for feature extraction

    Returns:
        Tuple of (mu, sigma) arrays for FID computation
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "clean-fid is required. Install with: pip install clean-fid"
        )

    return cleanfid.get_folder_features(
        str(folder),
        batch_size=batch_size,
    )
