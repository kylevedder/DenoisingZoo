"""Tests for MeanFlow loss correctness with time channels."""

import torch
import torch.nn as nn

from losses.meanflow_loss import MeanFlowLoss


class LinearTimeModel(nn.Module):
    """Deterministic model with known derivatives."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.time_channels = 2

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        x = unified_input[:, : self.feature_dim]
        times = unified_input[:, self.feature_dim :]
        r = times[:, 0:1]
        t = times[:, 1:2]
        return 2.0 * x + 3.0 * r.expand_as(x) + 5.0 * t.expand_as(x)


def test_meanflow_loss_matches_manual_linear_model():
    torch.manual_seed(123)
    B = 6
    D = 4

    model = LinearTimeModel(feature_dim=D)
    loss_fn = MeanFlowLoss(model=model, meanflow_ratio=1.0, weighting_power=0.0)

    x = torch.randn(B, D)
    y = torch.randn(B, D)
    batch = {"raw_source": x, "raw_target": y}

    rng_state = torch.random.get_rng_state()
    loss = loss_fn(batch)

    # Recompute with the same RNG state for deterministic comparison
    torch.random.set_rng_state(rng_state)
    z = torch.randn((B, 1), device=x.device, dtype=x.dtype)
    z = z * loss_fn.logit_normal_std + loss_fn.logit_normal_mean
    t = torch.sigmoid(z).clamp(loss_fn.eps, 1 - loss_fn.eps)

    use_meanflow = torch.rand(B, 1, device=x.device, dtype=x.dtype) < loss_fn.meanflow_ratio
    r_uniform = torch.rand(B, 1, device=x.device, dtype=x.dtype) * t
    r = torch.where(use_meanflow, r_uniform, t)

    z_t = (1 - t) * x + t * y

    # Model prediction u(z_t, r, t)
    u_pred = 2.0 * z_t + 3.0 * r + 5.0 * t

    # v(z_t, t) = u(z_t, r=t, t) = 2*z_t + 8*t
    v_t = 2.0 * z_t + 8.0 * t
    jvp = 2.0 * v_t + 8.0

    u_tgt = v_t - (t - r) * jvp
    diff = u_pred - u_tgt
    expected_loss = (diff ** 2).sum(dim=1).mean()

    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_meanflow_loss_requires_two_time_channels():
    class SingleTimeModel(nn.Module):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.time_channels = 1

        def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
            return unified_input[:, : self.feature_dim]

    model = SingleTimeModel(feature_dim=3)
    loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.5)
    batch = {
        "raw_source": torch.randn(2, 3),
        "raw_target": torch.randn(2, 3),
    }

    try:
        _ = loss_fn(batch)
        assert False, "Expected MeanFlowLoss to raise with time_channels < 2"
    except RuntimeError as exc:
        assert "time_channels" in str(exc)
