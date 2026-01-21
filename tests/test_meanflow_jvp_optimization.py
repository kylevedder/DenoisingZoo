"""Tests for MeanFlow JVP correctness.

Verifies that the JVP computation matches paper Eq. 8: du/dt = v·∂v/∂z + ∂v/∂t
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.func import jvp

from dataloaders.base_dataloaders import (
    make_time_input,
    make_unified_flow_matching_input,
)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self, in_channels: int, hidden_channels: int, time_channels: int = 2):
        super().__init__()
        if time_channels != 2:
            raise ValueError("time_channels must be 2")
        self.time_channels = time_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + time_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_jvp(model, z_t, t):
    """Compute full JVP: v·∂v/∂z + ∂v/∂t (paper Eq. 8)."""

    def model_fn(z: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
        time_input = make_time_input(t_input)
        unified = make_unified_flow_matching_input(z, time_input)
        return model(unified)

    with torch.no_grad():
        v_t_initial = model_fn(z_t, t)

    tangent_t = torch.ones_like(t)
    v_t, jvp_result = jvp(
        model_fn,
        (z_t, t),
        (v_t_initial.detach(), tangent_t),
    )

    return v_t, jvp_result


def compute_jvp_finite_diff(model, z_t, t, eps=1e-4):
    """Finite difference reference for full JVP."""

    def model_fn(z: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        time_input = make_time_input(t_val)
        unified = make_unified_flow_matching_input(z, time_input)
        return model(unified)

    with torch.no_grad():
        v_t = model_fn(z_t, t)

        # ∂v/∂z · v
        jvp_z = (model_fn(z_t + eps * v_t, t) - model_fn(z_t - eps * v_t, t)) / (2 * eps)

        # ∂v/∂t
        jvp_t = (model_fn(z_t, t + eps) - model_fn(z_t, t - eps)) / (2 * eps)

        jvp_fd = jvp_z + jvp_t

    return v_t, jvp_fd


class TestJVPCorrectness:
    """Test JVP computation correctness."""

    def test_mlp_jvp_vs_finite_diff(self):
        """Verify JVP matches finite differences for MLP."""
        torch.manual_seed(42)
        model = SimpleMLP(in_dim=4, hidden_dim=32, out_dim=2)
        z_t = torch.randn(8, 2)
        t = torch.rand(8, 1) * 0.8 + 0.1

        v_t, jvp_result = compute_jvp(model, z_t, t)
        v_t_fd, jvp_fd = compute_jvp_finite_diff(model, z_t, t)

        assert torch.allclose(v_t, v_t_fd, atol=1e-6)
        assert torch.allclose(jvp_result, jvp_fd, atol=1e-2)

    def test_cnn_jvp_vs_finite_diff(self):
        """Verify JVP matches finite differences for CNN."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16, time_channels=2)
        z_t = torch.randn(4, 3, 8, 8)
        t = torch.rand(4, 1) * 0.8 + 0.1

        v_t, jvp_result = compute_jvp(model, z_t, t)
        v_t_fd, jvp_fd = compute_jvp_finite_diff(model, z_t, t)

        assert torch.allclose(v_t, v_t_fd, atol=1e-6)
        assert torch.allclose(jvp_result, jvp_fd, atol=1e-2)

    def test_time_derivative_is_nonzero(self):
        """Verify ∂v/∂t contributes to JVP (not just ∂v/∂z)."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16, time_channels=2)
        model.eval()
        z_t = torch.randn(4, 3, 8, 8)
        t = torch.rand(4, 1) * 0.5 + 0.25

        # Compute ∂v/∂t separately
        def model_fn(z, t_val):
            time_input = make_time_input(t_val)
            return model(make_unified_flow_matching_input(z, time_input))

        with torch.no_grad():
            eps = 1e-4
            dvdt = (model_fn(z_t, t + eps) - model_fn(z_t, t - eps)) / (2 * eps)

        assert dvdt.norm() > 0.1, "∂v/∂t should be significant"

    def test_gradient_flow(self):
        """Verify gradients flow through JVP."""
        torch.manual_seed(42)
        model = SimpleMLP(in_dim=4, hidden_dim=32, out_dim=2)
        z_t = torch.randn(4, 2, requires_grad=True)
        t = torch.rand(4, 1) * 0.8 + 0.1

        v_t, jvp_result = compute_jvp(model, z_t, t)
        loss = (v_t ** 2 + jvp_result ** 2).sum()
        loss.backward()

        assert z_t.grad is not None
        assert z_t.grad.abs().sum() > 0

    def test_meanflow_loss_integration(self):
        """Test MeanFlowLoss computes without error."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16, time_channels=2)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=1.0)

        batch = {
            'raw_source': torch.randn(4, 3, 8, 8),
            'raw_target': torch.randn(4, 3, 8, 8),
        }

        loss = loss_fn(batch)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss > 0

        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
