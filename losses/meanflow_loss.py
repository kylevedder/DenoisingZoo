"""MeanFlow loss implementation.

Based on "Mean Flows for One-step Generative Modeling" (arXiv 2505.13447).
Enables single-step generation by learning the mean velocity field.

Matches official implementation: https://github.com/Gsunshine/py-meanflow
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)
from model_contracts import TimeChannelModule, is_time_channel_module


class MeanFlowLoss(nn.Module):
    """MeanFlow loss for training mean velocity field models.

    Key idea: Train u(z_t, r, t) to predict the mean velocity from time r to t,
    enabling one-step generation by setting r=0, t=1.

    Args:
        model: The velocity field model v(unified_input) -> velocity
        meanflow_ratio: Fraction of samples with r != t (default: 0.25)
        logit_normal_mean_t: Mean of logit-normal distribution for t (default: 0.0)
        logit_normal_std_t: Std of logit-normal distribution for t (default: 1.0)
        logit_normal_mean_r: Mean of logit-normal distribution for r (default: 0.0)
        logit_normal_std_r: Std of logit-normal distribution for r (default: 1.0)
        weighting_power: Power p for adaptive weighting w = 1/(||error||² + c)^p (default: 0.5)
        weighting_const: Constant c for adaptive weighting (default: 1e-4)
        eps: Small epsilon for numerical stability (default: 1e-5)
        use_batch_time: If True, prefer batch-provided t when available
        full_batch_jvp: If True, always compute JVP on full batch (enables CUDA graph capture)
        use_cuda_graph: If True, capture JVP in CUDA graph (requires full_batch_jvp=True, CUDA only)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        meanflow_ratio: float = 0.25,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        logit_normal_mean_t: float | None = None,
        logit_normal_std_t: float | None = None,
        logit_normal_mean_r: float | None = None,
        logit_normal_std_r: float | None = None,
        weighting_power: float = 0.5,
        weighting_const: float = 1e-4,
        eps: float = 1e-5,
        use_batch_time: bool = False,
        full_batch_jvp: bool = False,
        use_cuda_graph: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self.meanflow_ratio = meanflow_ratio
        # Support both old API (single mean/std) and new API (separate t/r)
        self.logit_normal_mean_t = (
            logit_normal_mean_t if logit_normal_mean_t is not None else logit_normal_mean
        )
        self.logit_normal_std_t = (
            logit_normal_std_t if logit_normal_std_t is not None else logit_normal_std
        )
        self.logit_normal_mean_r = (
            logit_normal_mean_r if logit_normal_mean_r is not None else logit_normal_mean
        )
        self.logit_normal_std_r = (
            logit_normal_std_r if logit_normal_std_r is not None else logit_normal_std
        )
        self.weighting_power = weighting_power
        self.weighting_const = weighting_const
        self.eps = eps
        self.use_batch_time = use_batch_time
        self.full_batch_jvp = full_batch_jvp
        self.use_cuda_graph = use_cuda_graph

        # CUDA graph state (lazy initialized on first forward pass)
        self._graphed_jvp_func = None
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_batch_size: int = -1
        self._cuda_graph_data_shape: tuple | None = None
        # Static buffers for CUDA graph (initialized by _capture_cuda_graph)
        self._static_z: torch.Tensor | None = None
        self._static_r: torch.Tensor | None = None
        self._static_t: torch.Tensor | None = None
        self._static_v: torch.Tensor | None = None
        self._static_tang_r: torch.Tensor | None = None
        self._static_tang_t: torch.Tensor | None = None
        self._static_primal: torch.Tensor | None = None
        self._static_tangent: torch.Tensor | None = None

    def set_model(self, model: nn.Module) -> None:
        """Set the model for loss computation (called by training loop)."""
        self._model = model
        # Reset CUDA graph when model changes
        self._graphed_jvp_func = None

    def _sample_logit_normal(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        mean: float,
        std: float,
    ) -> torch.Tensor:
        """Sample from logit-normal distribution, clipped to (eps, 1-eps)."""
        z = torch.randn(shape, device=device, dtype=dtype)
        z = z * std + mean
        t = torch.sigmoid(z)
        t = t.clamp(self.eps, 1 - self.eps)
        return t

    def _sample_two_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (t, r) following official MeanFlow implementation.

        Strategy (v0 from paper):
        1. Sample t and r independently from logit-normal distributions
        2. Sort so t >= r (swap if needed)
        3. With probability (1 - ratio), set r = t (standard FM)

        Returns:
            t: End time tensor of shape (B, 1)
            r: Start time tensor of shape (B, 1)
        """
        # Step 1: Sample two independent timesteps from logit-normal
        t = self._sample_logit_normal(
            (batch_size, 1), device, dtype, self.logit_normal_mean_t, self.logit_normal_std_t
        )
        r = self._sample_logit_normal(
            (batch_size, 1), device, dtype, self.logit_normal_mean_r, self.logit_normal_std_r
        )

        # Step 2: Ensure t >= r by sorting
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # Step 3: With probability (1 - ratio), set r = t (standard FM)
        prob = torch.rand(batch_size, 1, device=device, dtype=dtype)
        mask = prob < (1 - self.meanflow_ratio)
        r = torch.where(mask, t, r)

        return t, r

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select (t, r) time pair for the batch.

        Returns:
            t: End time tensor of shape (B, 1)
            r: Start time tensor of shape (B, 1)
        """
        if self.use_batch_time and "t" in batch:
            t = self._get_batch_time(batch, batch_size, device, dtype)
            # When using batch time, r = t (standard FM behavior)
            return t, t.clone()
        return self._sample_two_timesteps(batch_size, sample_device, dtype)

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Compute MeanFlow target and prediction using JVP.

        Following official implementation, computes:
            u_tgt = v - (t - r) * du/dt

        where du/dt is the total derivative along the flow trajectory:
            du/dt = ∂u/∂z · v + ∂u/∂r · 0 + ∂u/∂t · 1

        The JVP is computed with 3 inputs (z, r, t) and tangents (v, 0, 1).

        Returns:
            u_tgt: Target tensor for all samples (MF samples have JVP-computed target)
            mf_pred_info: None if no MF samples, else (mf_mask, u_pred_mf) where
                u_pred_mf is the JVP primal output (with gradients attached)
        """
        if not use_meanflow.any():
            return v_true, None

        # Only compute for MeanFlow samples to save compute
        mf_mask = use_meanflow.squeeze(-1)  # (B,) bool
        z_t_mf = z_t[mf_mask]
        t_mf = t[mf_mask]
        r_mf = r[mf_mask]
        v_true_mf = v_true[mf_mask]

        # Model function takes (z, r, t) as separate inputs
        # This allows proper JVP computation with tangents (v, 0, 1)
        def u_func(z: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            # Build time input as (r, t) - our format
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z, time_input)
            return self._model(unified)

        # Compute tangent for z: velocity v = dz/dt
        # We use v_true (ground truth velocity) as the tangent
        tangent_z = v_true_mf
        tangent_r = torch.zeros_like(r_mf)  # dr/dt = 0 (start time fixed)
        tangent_t = torch.ones_like(t_mf)  # dt/dt = 1

        # Disable autocast for numerical stability during JVP
        # (higher-order differentiation can be unstable in float16)
        device_type = z_t_mf.device.type
        with torch.amp.autocast(device_type, enabled=False):
            # Cast to float32 for JVP if needed
            z_float = z_t_mf.float()
            r_float = r_mf.float()
            t_float = t_mf.float()
            tangent_z_float = tangent_z.float()
            tangent_r_float = tangent_r.float()
            tangent_t_float = tangent_t.float()

            u_pred_mf, dudt_mf = torch.func.jvp(
                u_func,
                (z_float, r_float, t_float),
                (tangent_z_float, tangent_r_float, tangent_t_float),
            )

            # Compute target: u_tgt = v - (t - r) * du/dt
            delta_t_mf = self._broadcast_time(t_float - r_float, u_pred_mf)
            u_tgt_mf = (v_true_mf.float() - delta_t_mf * dudt_mf).to(v_true.dtype)

        u_tgt = v_true.clone()
        u_tgt[mf_mask] = u_tgt_mf.detach()

        # Return prediction with gradients attached (NOT detached) for use in loss
        return u_tgt, (mf_mask, u_pred_mf.to(v_true.dtype))

    def _jvp_forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure JVP forward computation (capture-safe).

        Returns:
            u_pred: JVP primal output
            dudt: JVP tangent output
            delta_t: Time difference broadcasted to match u_pred shape
        """
        def u_func(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return self._model(unified)

        tangent_z = v
        tangent_r = torch.zeros_like(r)
        tangent_t = torch.ones_like(t)

        u_pred, dudt = torch.func.jvp(
            u_func,
            (z, r, t),
            (tangent_z, tangent_r, tangent_t),
        )

        delta_t = self._broadcast_time(t - r, u_pred)
        return u_pred, dudt, delta_t

    def _compute_target_full_batch(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MeanFlow target and prediction using full-batch JVP.

        This version always runs JVP on ALL samples, avoiding CPU-GPU sync
        and enabling CUDA graph capture. For FM samples (r == t), delta_t = 0,
        so the target naturally equals v_true.

        Returns:
            u_tgt: Target tensor for all samples
            u_pred: Prediction tensor (JVP primal) for all samples
        """
        device = z_t.device
        dtype = v_true.dtype

        # Eager mode (CUDA graph optimization removed for now - requires capturing backward)
        # See JVP_BENCHMARK_PROBLEMS.md for details on why raw CUDA graph capture doesn't work
        device_type = z_t.device.type
        with torch.amp.autocast(device_type, enabled=False):
            z_float = z_t.float()
            r_float = r.float()
            t_float = t.float()
            v_float = v_true.float()

            u_pred, dudt, delta_t = self._jvp_forward(z_float, r_float, t_float, v_float)
            u_tgt = (v_float - delta_t * dudt).to(dtype)

        return u_tgt.detach(), u_pred.to(dtype)

    def _capture_cuda_graph(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_true: torch.Tensor,
    ) -> None:
        """Capture CUDA graph for JVP computation.

        Creates static buffers and captures the JVP computation graph.
        Must be called before using _compute_target_cuda_graph_hybrid.
        """
        device = z_t.device
        batch_size = z_t.shape[0]
        data_shape = z_t.shape

        # Create static buffers (all float32 for JVP)
        self._static_z = z_t.float().clone()
        self._static_r = r.float().clone()
        self._static_t = t.float().clone()
        self._static_v = v_true.float().clone()
        self._static_tang_r = torch.zeros_like(r).float()
        self._static_tang_t = torch.ones_like(t).float()

        # Pure JVP function for capture
        def jvp_func(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return self._model(unified)

        # Warmup
        for _ in range(3):
            torch.func.jvp(
                jvp_func,
                (self._static_z, self._static_r, self._static_t),
                (self._static_v, self._static_tang_r, self._static_tang_t),
            )
            torch.cuda.synchronize()

        # Capture graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._static_primal, self._static_tangent = torch.func.jvp(
                jvp_func,
                (self._static_z, self._static_r, self._static_t),
                (self._static_v, self._static_tang_r, self._static_tang_t),
            )

        self._cuda_graph_batch_size = batch_size
        self._cuda_graph_data_shape = data_shape

    def _compute_target_cuda_graph_hybrid(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MeanFlow target using CUDA-graphed JVP (detached).

        This is the hybrid approach:
        1. Use CUDA-graphed JVP to compute dudt (detached, no backward)
        2. Compute target = v - delta_t * dudt (detached)
        3. Caller runs separate forward pass for prediction (with gradients)

        Returns:
            u_tgt: Detached target tensor
        """
        device = z_t.device
        dtype = v_true.dtype

        # Check if we need to (re)capture the graph
        if (
            self._cuda_graph is None
            or self._cuda_graph_batch_size != z_t.shape[0]
            or self._cuda_graph_data_shape != z_t.shape
        ):
            self._capture_cuda_graph(z_t, t, r, v_true)

        # Copy data into static buffers
        self._static_z.copy_(z_t.float())
        self._static_r.copy_(r.float())
        self._static_t.copy_(t.float())
        self._static_v.copy_(v_true.float())

        # Replay graph
        self._cuda_graph.replay()

        # Compute target from graphed outputs (all detached)
        delta_t = self._broadcast_time(self._static_t - self._static_r, self._static_primal)
        u_tgt = (self._static_v - delta_t * self._static_tangent).to(dtype)

        return u_tgt.detach()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute MeanFlow loss with optimized batch splitting.

        This implementation eliminates redundant forward passes by:
        1. Running the model forward ONLY for standard FM samples
        2. Reusing the JVP primal output for MeanFlow samples

        Following official implementation:
        1. Sample t and r using logit-normal + sorting (see _sample_two_timesteps)
        2. Compute target using JVP with tangents (v, 0, 1)
        3. Apply error-based adaptive weighting: w = 1/(||error||² + c)^p

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

        # Sample (t, r) following official implementation:
        # - Both sampled from logit-normal, sorted so t >= r
        # - With prob (1-ratio), r = t (standard FM)
        t, r = self._select_time(batch, B, device, dtype, x.device)

        # Determine which samples use MeanFlow (r != t) vs standard FM (r == t)
        use_meanflow = (r != t).any(dim=1, keepdim=True)
        mf_mask = use_meanflow.squeeze(-1)  # (B,) bool
        fm_mask = ~mf_mask  # (B,) bool - standard flow matching samples

        if not is_time_channel_module(self._model):
            raise ValueError(
                "MeanFlowLoss requires model to inherit from TimeChannelModule."
            )

        # Compute interpolated state at time t
        # z_t = (1 - t) * x + t * y
        z_t = self._select_interpolated_state(batch, x, y, t, device, dtype)

        # Ground truth velocity: v = y - x (points from noise to data)
        v_true = y - x

        if self.use_cuda_graph and z_t.device.type == "cuda":
            # Hybrid CUDA graph mode: graphed JVP (target) + separate forward (pred)
            # This achieves 47% speedup over eager full_batch_jvp mode
            # See JVP_BENCHMARK_PROBLEMS.md for details
            u_tgt = self._compute_target_cuda_graph_hybrid(z_t, t, r, v_true)

            # Separate forward pass for prediction (with gradients for backward)
            time_pred = torch.cat([r, t], dim=1)
            unified_t = make_unified_flow_matching_input(z_t, time_pred)
            u_pred = self._model(unified_t)

        elif self.full_batch_jvp:
            # Full-batch JVP mode: compute JVP on ALL samples
            # This avoids CPU-GPU sync and enables CUDA graph capture
            # For FM samples (r == t), delta_t = 0, so u_tgt = v_true naturally
            u_tgt, u_pred = self._compute_target_full_batch(z_t, t, r, v_true)
        else:
            # Selective JVP mode (default): only compute JVP for MeanFlow samples
            # Build time input: (r, t) for all samples
            time_pred = torch.cat([r, t], dim=1)
            unified_t = make_unified_flow_matching_input(z_t, time_pred)

            # Compute target and get JVP primal for MeanFlow samples
            # From paper Eq. 7: u_tgt = v - (t - r) * du/dt
            u_tgt, mf_pred_info = self._compute_target(z_t, t, r, v_true, use_meanflow)

            # Allocate prediction tensor
            u_pred = torch.empty_like(v_true)

            # Forward pass ONLY for standard FM samples (if any exist)
            if fm_mask.any():
                unified_fm = unified_t[fm_mask]
                u_pred_fm = self._model(unified_fm)
                u_pred[fm_mask] = u_pred_fm

            # Use JVP primal for MeanFlow samples (if any exist)
            if mf_pred_info is not None:
                mf_mask_from_jvp, u_pred_mf = mf_pred_info
                u_pred[mf_mask_from_jvp] = u_pred_mf

        # Compute squared error per sample (sum over spatial dims, not mean)
        # This matches official: loss.sum(dim=(1, 2, 3))
        diff = u_pred - u_tgt
        sq_error = (diff**2).flatten(1).sum(dim=1)  # (B,) - sum, not mean

        # Adaptive weighting following official implementation:
        # adp_wt = (loss + eps)^p; loss = loss / adp_wt
        # This gives: loss * (loss + eps)^(-p), effectively a powered L2 metric
        # With p=0.5: loss / sqrt(loss) = sqrt(loss) (Euclidean distance)
        if self.weighting_power > 0:
            with torch.no_grad():
                adp_wt = (sq_error.detach() + self.weighting_const) ** self.weighting_power
            loss = (sq_error / adp_wt).mean()
        else:
            loss = sq_error.mean()

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
