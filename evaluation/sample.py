"""Sample generation utilities for flow matching and MeanFlow models."""

from __future__ import annotations

from typing import Iterator
import math

import torch
from torch import nn
from tqdm import tqdm

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)
from model_contracts import TimeChannelModule


def generate_samples(
    model: nn.Module,
    solver: object,
    num_samples: int,
    sample_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int = 64,
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Generate samples using ODE integration.

    Args:
        model: Velocity field model
        solver: ODE solver (Euler, RK4, etc.) with solve(x0) method
        num_samples: Total number of samples to generate
        sample_shape: Shape of each sample (C, H, W) or (D,)
        device: Device to generate on
        batch_size: Batch size for generation
        seed: Random seed for reproducibility

    Yields:
        Batches of generated samples of shape (B, ...)
    """
    model.eval()

    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)
    else:
        rng = None

    num_batches = math.ceil(num_samples / batch_size)
    samples_generated = 0

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            # Determine batch size (might be smaller for last batch)
            current_batch = min(batch_size, num_samples - samples_generated)

            # Sample from source distribution (standard Gaussian)
            x0 = torch.randn(
                (current_batch, *sample_shape),
                device=device,
                generator=rng,
            )

            # Integrate ODE from t=0 to t=1
            result = solver.solve(x0)
            samples = result.final_state

            yield samples
            samples_generated += current_batch


def generate_samples_meanflow(
    model: nn.Module,
    num_samples: int,
    sample_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int = 64,
    seed: int | None = None,
    r: float = 0.0,
    t: float = 1.0,
) -> Iterator[torch.Tensor]:
    """Generate samples using single-step MeanFlow.

    MeanFlow enables one-step generation: x = z + (t - r) * u(z, r, t)

    Args:
        model: MeanFlow velocity field model
        num_samples: Total number of samples to generate
        sample_shape: Shape of each sample (C, H, W) or (D,)
        device: Device to generate on
        batch_size: Batch size for generation
        seed: Random seed for reproducibility
        r: Start time (default: 0.0)
        t: End time (default: 1.0)

    Yields:
        Batches of generated samples of shape (B, ...)
    """
    model.eval()

    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)
    else:
        rng = None

    num_batches = math.ceil(num_samples / batch_size)
    samples_generated = 0

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples (MeanFlow 1-NFE)"):
            # Determine batch size (might be smaller for last batch)
            current_batch = min(batch_size, num_samples - samples_generated)

            # Sample from source distribution (standard Gaussian)
            z = torch.randn(
                (current_batch, *sample_shape),
                device=device,
                generator=rng,
            )

            if not isinstance(model, TimeChannelModule):
                raise ValueError("MeanFlow sampling requires TimeChannelModule model.")

            # Build time tensor for (r, t)
            r_tensor = torch.full(
                (current_batch, 1), r, device=device, dtype=z.dtype
            )
            t_tensor = torch.full(
                (current_batch, 1), t, device=device, dtype=z.dtype
            )
            time_input = make_time_input(t_tensor, r=r_tensor)

            # Build unified input
            unified = make_unified_flow_matching_input(z, time_input)

            # Single-step generation: x = z + (t - r) * u(z, r, t)
            # For r=0, t=1: x = z + u(z, 0, 1)
            v = model(unified)
            samples = z + (t - r) * v

            yield samples
            samples_generated += current_batch


def collect_samples(
    sample_iterator: Iterator[torch.Tensor],
    num_samples: int | None = None,
) -> torch.Tensor:
    """Collect samples from an iterator into a single tensor.

    Args:
        sample_iterator: Iterator yielding sample batches
        num_samples: If provided, truncate to this many samples

    Returns:
        Tensor of shape (N, ...) containing all samples
    """
    all_samples = []
    total = 0

    for batch in sample_iterator:
        all_samples.append(batch.cpu())
        total += batch.shape[0]
        if num_samples is not None and total >= num_samples:
            break

    samples = torch.cat(all_samples, dim=0)

    if num_samples is not None:
        samples = samples[:num_samples]

    return samples


def samples_to_images(
    samples: torch.Tensor,
    clip: bool = True,
) -> torch.Tensor:
    """Convert model output samples to uint8 images.

    Assumes samples are in [-1, 1] range.

    Args:
        samples: Tensor of shape (N, C, H, W) in [-1, 1]
        clip: Whether to clip values to [-1, 1] before conversion

    Returns:
        Tensor of shape (N, C, H, W) as uint8 in [0, 255]
    """
    if clip:
        samples = samples.clamp(-1, 1)

    # Convert from [-1, 1] to [0, 255]
    samples = (samples + 1) * 127.5
    samples = samples.to(torch.uint8)

    return samples
