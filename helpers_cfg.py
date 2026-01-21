"""Classifier-Free Guidance (CFG) utilities.

Implements CFG sampling where the velocity is computed as:
    v_cfg = w * v_cond + (1 - w) * v_uncond

where w is the guidance scale.
"""

from __future__ import annotations

import torch
from torch import nn

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)
from constants import CFG_NULL_LABEL, TIME_CHANNELS_REQUIRED
from model_contracts import TimeChannelModule
from solvers.base_solver import BaseSolver


def cfg_sample_step(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float = 1.0,
    null_label: int = CFG_NULL_LABEL,
    r: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute CFG-guided velocity at time t.

    Args:
        model: Model that takes (unified_input, labels) -> velocity
        x: Current state (B, C, H, W)
        t: Time tensor (B,), (B, 1), or (B, 2)
        labels: Class labels (B,)
        guidance_scale: CFG scale w (default: 1.0 = no guidance)
        null_label: Label value for unconditional (default: CFG_NULL_LABEL)

    Returns:
        CFG-weighted velocity (B, C, H, W)
    """
    if not isinstance(model, TimeChannelModule):
        raise ValueError("CFG sampling requires TimeChannelModule model.")

    if t.dim() == 1:
        t = t.unsqueeze(-1)
    if r is not None and r.dim() == 1:
        r = r.unsqueeze(-1)

    # Create unified input
    time_input = make_time_input(t, r=r)
    unified = make_unified_flow_matching_input(x, time_input)

    if guidance_scale == 1.0:
        # No guidance needed - just conditional
        return model(unified, labels)

    # Double batch for conditional + unconditional
    unified_double = torch.cat([unified, unified], dim=0)
    null_labels = torch.full_like(labels, null_label)
    labels_double = torch.cat([labels, null_labels], dim=0)

    # Single forward pass for both
    v_both = model(unified_double, labels_double)
    v_cond, v_uncond = v_both.chunk(2, dim=0)

    # CFG combination
    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

    return v_cfg


class CFGWrapper(nn.Module):
    """Wrapper that adds CFG to any class-conditional model.

    Makes the model callable as model(unified_input) with CFG applied,
    useful for integration with existing solvers.
    """

    def __init__(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        guidance_scale: float = 1.0,
        null_label: int = CFG_NULL_LABEL,
    ) -> None:
        super().__init__()
        self.model = model
        self.labels = labels
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        """Forward with CFG applied."""
        from dataloaders.base_dataloaders import make_ununified_flow_matching_input

        if not isinstance(self.model, TimeChannelModule):
            raise ValueError("CFGWrapper requires TimeChannelModule model.")
        ununified = make_ununified_flow_matching_input(
            unified_input, num_time_channels=TIME_CHANNELS_REQUIRED
        )
        x = ununified.x
        t = ununified.t

        return cfg_sample_step(
            self.model,
            x,
            t,
            self.labels,
            guidance_scale=self.guidance_scale,
            null_label=self.null_label,
        )


def generate_samples_cfg(
    model: nn.Module,
    labels: torch.Tensor,
    sample_shape: tuple[int, ...],
    device: torch.device,
    solver: BaseSolver,
    guidance_scale: float = 1.0,
    null_label: int = CFG_NULL_LABEL,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate samples with CFG using an ODE solver.

    Args:
        model: Class-conditional model
        labels: Class labels (B,)
        sample_shape: Shape of each sample (C, H, W)
        device: Device for generation
        solver: ODE solver (will use wrapped model)
        guidance_scale: CFG scale
        null_label: Label for unconditional
        seed: Random seed

    Returns:
        Generated samples (B, C, H, W)
    """
    B = labels.shape[0]

    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)
    else:
        rng = None

    # Sample initial noise
    x0 = torch.randn((B, *sample_shape), device=device, generator=rng)

    # Wrap model with CFG
    cfg_model = CFGWrapper(model, labels, guidance_scale, null_label)

    result = solver.solve_with_model(cfg_model, x0)
    samples = result.final_state

    return samples


def generate_samples_meanflow_cfg(
    model: nn.Module,
    labels: torch.Tensor,
    sample_shape: tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 1.0,
    null_label: int = CFG_NULL_LABEL,
    seed: int | None = None,
    r: float = 0.0,
    t: float = 1.0,
) -> torch.Tensor:
    """Generate samples with CFG using single-step MeanFlow.

    MeanFlow 1-NFE generation: x = z + (t - r) * u_cfg(z, r, t)

    Args:
        model: Class-conditional MeanFlow model
        labels: Class labels (B,)
        sample_shape: Shape of each sample (C, H, W)
        device: Device for generation
        guidance_scale: CFG scale
        null_label: Label for unconditional
        seed: Random seed
        r: Start time (default: 0.0)
        t: End time (default: 1.0)

    Returns:
        Generated samples (B, C, H, W)
    """
    B = labels.shape[0]

    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)
    else:
        rng = None

    # Sample initial noise
    z = torch.randn((B, *sample_shape), device=device, generator=rng)

    # Create time tensor
    if not isinstance(model, TimeChannelModule):
        raise ValueError("MeanFlow CFG sampling requires TimeChannelModule model.")
    r_tensor = torch.full((B, 1), r, device=device, dtype=z.dtype)
    t_tensor = torch.full((B, 1), t, device=device, dtype=z.dtype)

    # Get CFG velocity
    v_cfg = cfg_sample_step(
        model,
        z,
        t_tensor,
        labels,
        guidance_scale=guidance_scale,
        null_label=null_label,
        r=r_tensor,
    )

    # Single-step generation
    samples = z + (t - r) * v_cfg

    return samples
