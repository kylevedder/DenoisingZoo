"""MeanFlow loss implementation.

Based on "Mean Flows for One-step Generative Modeling" (arXiv 2505.13447).
Enables single-step generation by learning the mean velocity field.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)
from model_contracts import TimeChannelModule


class MeanFlowLoss(nn.Module):
    """MeanFlow loss for training mean velocity field models.

    Key idea: Train u(z_t, r, t) to predict the mean velocity from time r to t,
    enabling one-step generation by setting r=0, t=1.

    Args:
        model: The velocity field model v(unified_input) -> velocity
        meanflow_ratio: Fraction of samples with r != t (default: 0.25)
        logit_normal_mean: Mean of logit-normal time distribution (default: 0.0)
        logit_normal_std: Std of logit-normal time distribution (default: 1.0)
        weighting_power: Power p for adaptive weighting w = 1/(||Δ||² + c)^p (default: 0.5)
        weighting_const: Constant c for adaptive weighting (default: 1e-4)
        eps: Small epsilon for numerical stability (default: 1e-5)
        use_batch_time: If True, prefer batch-provided t when available
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        meanflow_ratio: float = 0.25,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        weighting_power: float = 0.5,
        weighting_const: float = 1e-4,
        eps: float = 1e-5,
        use_batch_time: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self.meanflow_ratio = meanflow_ratio
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.weighting_power = weighting_power
        self.weighting_const = weighting_const
        self.eps = eps
        self.use_batch_time = use_batch_time

    def set_model(self, model: nn.Module) -> None:
        """Set the model for loss computation (called by training loop)."""
        self._model = model

    def _sample_logit_normal(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample from logit-normal distribution, clipped to (eps, 1-eps)."""
        # Sample from normal, then apply sigmoid
        z = torch.randn(shape, device=device, dtype=dtype)
        z = z * self.logit_normal_std + self.logit_normal_mean
        t = torch.sigmoid(z)
        # Clip to avoid exact 0 or 1
        t = t.clamp(self.eps, 1 - self.eps)
        return t

    def _get_batch_time(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = batch["t"]
        if device is not None:
            t = t.to(device, non_blocking=True)
        t = t.to(dtype=dtype)
        if t.dim() == 1:
            t = t.view(-1, 1)
        else:
            t = t.reshape(t.shape[0], -1)
        if t.shape[0] != batch_size:
            raise ValueError(
                f"Batch time size {t.shape[0]} does not match batch size {batch_size}"
            )
        if t.shape[1] != 1:
            raise ValueError("Batch time must have a single column.")
        return t

    def _select_time(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device | None,
        dtype: torch.dtype,
        sample_device: torch.device,
    ) -> torch.Tensor:
        if self.use_batch_time and "t" in batch:
            return self._get_batch_time(batch, batch_size, device, dtype)
        return self._sample_logit_normal((batch_size, 1), sample_device, dtype)

    def _select_interpolated_state(
        self,
        batch: dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        device: torch.device | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.use_batch_time and "input" in batch:
            z_t = batch["input"]
            if device is not None:
                z_t = z_t.to(device, non_blocking=True)
            return z_t.to(dtype=dtype)
        t_b = self._broadcast_time(t, x)
        return (1 - t_b) * x + t_b * y

    def _build_time_pred(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
        use_meanflow: torch.Tensor,
    ) -> torch.Tensor:
        time_pred = make_time_input(t)
        if use_meanflow.any():
            time_pred = time_pred.clone()
            mf_mask = use_meanflow.squeeze(-1)
            time_pred[mf_mask, 0:1] = r[mf_mask]
            time_pred[mf_mask, 1:2] = t[mf_mask]
        return time_pred

    def _compute_target(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_true: torch.Tensor,
        use_meanflow: torch.Tensor,
    ) -> torch.Tensor:
        if not use_meanflow.any():
            return v_true

        # We need to compute full JVP: v · ∂v/∂z + ∂v/∂t
        # This requires differentiating w.r.t. both z and t with tangents [v, 1]
        # Only compute for MeanFlow samples to save compute (~4x speedup at ratio=0.25)
        mf_mask = use_meanflow.squeeze(-1)  # (B,) bool
        z_t_mf = z_t[mf_mask]
        t_mf = t[mf_mask]
        r_mf = r[mf_mask]

        def model_fn_mf(z: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
            time_input = make_time_input(t_input)
            unified = make_unified_flow_matching_input(z, time_input)
            return self._model(unified)

        with torch.no_grad():
            v_t_mf_tangent = model_fn_mf(z_t_mf, t_mf)

        tangent_t = torch.ones_like(t_mf)
        v_t_mf, jvp_mf = torch.func.jvp(
            model_fn_mf,
            (z_t_mf, t_mf),
            (v_t_mf_tangent, tangent_t),
        )

        delta_t_mf = self._broadcast_time(t_mf - r_mf, v_t_mf)
        u_tgt_mf = v_t_mf - delta_t_mf * jvp_mf

        u_tgt = v_true.clone()
        u_tgt[mf_mask] = u_tgt_mf.detach()
        return u_tgt

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute MeanFlow loss.

        Args:
            batch: Dictionary containing:
                - raw_source: Noise samples x ~ N(0,1) of shape (B, ...)
                - raw_target: Data samples y of shape (B, ...)
            device: Device to run computation on

        Returns:
            Scalar loss tensor
        """
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        # Extract source (noise) and target (data) from batch
        x = batch["raw_source"]  # (B, ...) noise
        y = batch["raw_target"]  # (B, ...) data

        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        B = x.shape[0]
        dtype = x.dtype

        # Sample time t from logit-normal (or use batch-provided t)
        t = self._select_time(batch, B, device, dtype, x.device)

        # Determine which samples use r != t (MeanFlow) vs r = t (standard FM)
        use_meanflow = torch.rand(B, 1, device=x.device, dtype=dtype) < self.meanflow_ratio

        # Sample r uniformly in [0, t] for MeanFlow samples
        # For standard FM, r = t
        r_uniform = torch.rand(B, 1, device=x.device, dtype=dtype) * t
        r = torch.where(use_meanflow, r_uniform, t)

        if not isinstance(self._model, TimeChannelModule):
            raise ValueError(
                "MeanFlowLoss requires model to inherit from TimeChannelModule."
            )

        # Compute interpolated state at time t
        # z_t = (1 - t) * x + t * y
        z_t = self._select_interpolated_state(batch, x, y, t, device, dtype)

        # Ground truth velocity
        v_true = y - x

        # Compute model predictions.
        # For standard FM samples, time input should be (t, 1).
        # For MeanFlow samples, time input should be (r, t).
        time_pred = self._build_time_pred(t, r, use_meanflow)
        unified_t = make_unified_flow_matching_input(z_t, time_pred)

        # Always compute u(z_t, r, t) for MeanFlow samples and v(z_t, t) for FM samples
        v_pred = self._model(unified_t)

        # Compute target for MeanFlow samples
        # From paper Eq. 6: u = v_t - (t - r) * du/dt
        # From paper Eq. 8: du/dt = v · ∂u/∂z + ∂u/∂t (total derivative along trajectory)
        # So: u_tgt = v_t - (t - r) * (v_t · ∂v/∂z + ∂v/∂t)

        u_tgt = self._compute_target(z_t, t, r, v_true, use_meanflow)

        # Compute loss with adaptive weighting
        diff = v_pred - u_tgt
        sq_error = (diff ** 2).flatten(1).mean(dim=1)  # (B,)

        # Adaptive weighting: w = 1 / (||delta||² + c)^p
        # where delta = u_tgt - v_true
        if self.weighting_power > 0:
            # Weight based on the MeanFlow correction magnitude
            with torch.no_grad():
                delta = (u_tgt.detach() - v_true).flatten(1)
                delta_sq = (delta ** 2).sum(dim=1)
                weights = 1.0 / (delta_sq + self.weighting_const) ** self.weighting_power
                weights = weights / weights.mean()  # Normalize to preserve scale
        else:
            weights = torch.ones(B, device=x.device, dtype=dtype)

        loss = (weights * sq_error).mean()

        return loss

    def _broadcast_time(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Broadcast time tensor to match shape of like tensor."""
        # t is (B, 1), like is (B, C, H, W) or (B, D)
        if like.dim() == 2:
            return t  # Already (B, 1)
        elif like.dim() == 4:
            # (B, 1) -> (B, 1, 1, 1)
            return t.view(-1, 1, 1, 1)
        else:
            # Generic case
            shape = [t.shape[0]] + [1] * (like.dim() - 1)
            return t.view(*shape)



class MeanFlowLossWrapper(nn.Module):
    """Wrapper that presents MeanFlow loss with (pred, target) interface for compatibility.

    This is a simplified version for use with existing training infrastructure.
    Does NOT compute JVP - just standard flow matching with logit-normal time sampling.
    Use MeanFlowLoss directly for full MeanFlow training.
    """

    def __init__(
        self,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard MSE loss - use MeanFlowLoss for full MeanFlow."""
        return F.mse_loss(pred, target)
