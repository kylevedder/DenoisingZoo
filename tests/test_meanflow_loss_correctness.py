"""Tests for MeanFlow loss correctness with time channels."""

import torch

from losses.meanflow_loss import MeanFlowLoss
from model_contracts import TimeChannelModule


class LinearTimeModel(TimeChannelModule):
    """Deterministic model with known derivatives."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        x = unified_input[:, : self.feature_dim]
        times = unified_input[:, self.feature_dim :]
        r = times[:, 0:1]
        t = times[:, 1:2]
        return 2.0 * x + 3.0 * r.expand_as(x) + 5.0 * t.expand_as(x)


def test_meanflow_loss_matches_manual_linear_model():
    """Test MeanFlow loss against manual computation with a linear model.

    LinearTimeModel: u(z, r, t) = 2*z + 3*r + 5*t

    For this model:
    - ∂u/∂z = 2  (so ∂u/∂z · v = 2*v where v = y - x)
    - ∂u/∂r = 3
    - ∂u/∂t = 5

    With JVP tangents (v, 0, 1):
    du/dt = ∂u/∂z · v + ∂u/∂r · 0 + ∂u/∂t · 1 = 2*v + 5

    Target: u_tgt = v - (t - r) * du/dt = v - (t - r) * (2*v + 5)
    """
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
    # Following new _sample_two_timesteps logic:
    # 1. Sample t from logit-normal
    # 2. Sample r from logit-normal
    # 3. Sort so t >= r
    # 4. With prob (1-ratio), set r = t (but ratio=1.0, so this never happens)
    torch.random.set_rng_state(rng_state)

    # Sample t from logit-normal
    z_t_sample = torch.randn((B, 1), device=x.device, dtype=x.dtype)
    z_t_sample = z_t_sample * loss_fn.logit_normal_std_t + loss_fn.logit_normal_mean_t
    t_sample = torch.sigmoid(z_t_sample).clamp(loss_fn.eps, 1 - loss_fn.eps)

    # Sample r from logit-normal
    z_r_sample = torch.randn((B, 1), device=x.device, dtype=x.dtype)
    z_r_sample = z_r_sample * loss_fn.logit_normal_std_r + loss_fn.logit_normal_mean_r
    r_sample = torch.sigmoid(z_r_sample).clamp(loss_fn.eps, 1 - loss_fn.eps)

    # Sort so t >= r
    t = torch.maximum(t_sample, r_sample)
    r = torch.minimum(t_sample, r_sample)

    # With prob (1 - ratio), set r = t (ratio=1.0, so skip)
    prob = torch.rand(B, 1, device=x.device, dtype=x.dtype)
    mask = prob < (1 - loss_fn.meanflow_ratio)
    r = torch.where(mask, t, r)

    # Compute interpolated state
    z_t_state = (1 - t) * x + t * y

    # Ground truth velocity
    v_true = y - x

    # Model prediction u(z_t, r, t)
    u_pred = 2.0 * z_t_state + 3.0 * r + 5.0 * t

    # Compute JVP manually for linear model
    # du/dt = ∂u/∂z · v + ∂u/∂r · 0 + ∂u/∂t · 1 = 2*v + 0 + 5 = 2*v + 5
    dudt = 2.0 * v_true + 5.0

    # Target: u_tgt = v - (t - r) * du/dt
    u_tgt = v_true - (t - r) * dudt

    # Loss: sum of squared errors (not mean over features), then mean over batch
    diff = u_pred - u_tgt
    expected_loss = (diff ** 2).sum(dim=1).mean()

    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_meanflow_loss_requires_two_time_channels():
    class SingleTimeModel(TimeChannelModule):
        def __init__(self, feature_dim: int) -> None:
            super().__init__(time_channels=1)
            self.feature_dim = feature_dim

        def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
            return unified_input[:, : self.feature_dim]

    try:
        _ = SingleTimeModel(feature_dim=3)
        assert False, "Expected model construction to raise with time_channels < 2"
    except ValueError as exc:
        assert "time_channels" in str(exc)
