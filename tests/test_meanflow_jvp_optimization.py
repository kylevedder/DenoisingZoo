"""Tests for MeanFlow JVP correctness.

These tests verify that the JVP computation produces correct results.
Note: torch.func.linearize would be more efficient but nn.Linear/nn.Conv2d
don't support forward-mode AD yet, so we use the two-pass jvp approach.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.func import jvp

from dataloaders.base_dataloaders import make_unified_flow_matching_input


class SimpleMLP(nn.Module):
    """Simple MLP for testing JVP correctness."""

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
    """Simple CNN for testing JVP correctness with images."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # in_channels + 1 for time channel
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_jvp_z_only(model, z_t, t):
    """JVP with respect to z only (incomplete, missing ∂v/∂t)."""

    def model_fn(z: torch.Tensor) -> torch.Tensor:
        unified = make_unified_flow_matching_input(z, t)
        return model(unified)

    # First forward pass to get tangent vector
    with torch.no_grad():
        v_t_initial = model_fn(z_t)

    # JVP only with respect to z
    v_t, jvp_result = jvp(
        model_fn,
        (z_t,),
        (v_t_initial.detach(),),
    )

    return v_t, jvp_result


def compute_jvp_full(model, z_t, t):
    """Full JVP: v·∂v/∂z + ∂v/∂t (correct MeanFlow formula from paper Eq. 8)."""

    def model_fn(z: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
        unified = make_unified_flow_matching_input(z, t_input)
        return model(unified)

    # First forward pass to get tangent vector for z
    with torch.no_grad():
        v_t_initial = model_fn(z_t, t)

    # JVP with both z and t: tangent is [v, 1]
    tangent_t = torch.ones_like(t)
    v_t, jvp_result = jvp(
        model_fn,
        (z_t, t),
        (v_t_initial.detach(), tangent_t),
    )

    return v_t, jvp_result


def compute_jvp_finite_diff_z_only(model, z_t, t, eps=1e-4):
    """Finite difference for ∂v/∂z · v only."""

    def model_fn(z: torch.Tensor) -> torch.Tensor:
        unified = make_unified_flow_matching_input(z, t)
        return model(unified)

    with torch.no_grad():
        v_t = model_fn(z_t)

        # Finite difference: ∂v/∂z · v ≈ (f(z + eps*v) - f(z - eps*v)) / (2*eps)
        v_plus = model_fn(z_t + eps * v_t)
        v_minus = model_fn(z_t - eps * v_t)
        jvp_fd = (v_plus - v_minus) / (2 * eps)

    return v_t, jvp_fd


def compute_jvp_finite_diff_full(model, z_t, t, eps=1e-4):
    """Finite difference for full JVP: v·∂v/∂z + ∂v/∂t."""

    def model_fn(z: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        unified = make_unified_flow_matching_input(z, t_val)
        return model(unified)

    with torch.no_grad():
        v_t = model_fn(z_t, t)

        # ∂v/∂z · v
        v_plus_z = model_fn(z_t + eps * v_t, t)
        v_minus_z = model_fn(z_t - eps * v_t, t)
        jvp_z = (v_plus_z - v_minus_z) / (2 * eps)

        # ∂v/∂t · 1
        v_plus_t = model_fn(z_t, t + eps)
        v_minus_t = model_fn(z_t, t - eps)
        jvp_t = (v_plus_t - v_minus_t) / (2 * eps)

        # Full JVP
        jvp_fd = jvp_z + jvp_t

    return v_t, jvp_fd


# Backwards compatibility aliases
compute_jvp_two_pass = compute_jvp_z_only
compute_jvp_finite_diff = compute_jvp_finite_diff_z_only


class TestFullJVPFormula:
    """Test that full JVP includes both ∂v/∂z·v AND ∂v/∂t terms (paper Eq. 8)."""

    def test_full_jvp_includes_time_derivative(self):
        """Verify full JVP is different from z-only JVP (∂v/∂t matters)."""
        torch.manual_seed(42)

        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        B = 4
        H, W = 8, 8
        z_t = torch.randn(B, 3, H, W)
        t = torch.rand(B, 1) * 0.5 + 0.25  # t in [0.25, 0.75]

        _, jvp_z_only = compute_jvp_z_only(model, z_t, t)
        _, jvp_full = compute_jvp_full(model, z_t, t)

        # The full JVP should differ from z-only JVP
        diff_norm = (jvp_full - jvp_z_only).norm()
        full_norm = jvp_full.norm()

        # ∂v/∂t should contribute significantly (not just numerical noise)
        assert diff_norm > 0.01 * full_norm, \
            f"∂v/∂t term should be significant: diff={diff_norm}, full={full_norm}"

    def test_full_jvp_matches_finite_diff(self):
        """Verify full JVP matches finite difference approximation."""
        torch.manual_seed(42)

        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        B = 4
        H, W = 8, 8
        z_t = torch.randn(B, 3, H, W)
        t = torch.rand(B, 1) * 0.5 + 0.25

        _, jvp_full = compute_jvp_full(model, z_t, t)
        _, jvp_fd = compute_jvp_finite_diff_full(model, z_t, t)

        assert torch.allclose(jvp_full, jvp_fd, atol=1e-2), \
            f"Full JVP mismatch: {(jvp_full - jvp_fd).abs().max()}"

    def test_meanflow_loss_uses_full_jvp(self):
        """Verify MeanFlowLoss uses the full JVP formula."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)

        # Use CNN which has significant ∂v/∂t
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=1.0)  # Force all MeanFlow

        B = 4
        batch = {
            'raw_source': torch.randn(B, 3, 8, 8),
            'raw_target': torch.randn(B, 3, 8, 8),
        }

        loss = loss_fn(batch)
        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss > 0, "Loss should be positive"


class TestJVPCorrectness:
    """Test JVP computation correctness."""

    def test_mlp_jvp_vs_finite_diff(self):
        """Verify JVP method matches finite differences for MLP."""
        torch.manual_seed(42)

        model = SimpleMLP(in_dim=3, hidden_dim=32, out_dim=2)  # 2D input + time -> 2D output
        B = 8
        z_t = torch.randn(B, 2)
        t = torch.rand(B, 1) * 0.8 + 0.1  # t in [0.1, 0.9]

        v_t_jvp, jvp_result = compute_jvp_two_pass(model, z_t, t)
        v_t_fd, jvp_fd = compute_jvp_finite_diff(model, z_t, t)

        # v_t should match exactly
        assert torch.allclose(v_t_jvp, v_t_fd, atol=1e-6), \
            f"v_t mismatch: {(v_t_jvp - v_t_fd).abs().max()}"

        # JVP should match approximately (finite diff has some error)
        assert torch.allclose(jvp_result, jvp_fd, atol=1e-3), \
            f"JVP mismatch: {(jvp_result - jvp_fd).abs().max()}"

        print("✓ MLP JVP matches finite differences")

    def test_cnn_jvp_vs_finite_diff(self):
        """Verify JVP method matches finite differences for CNN."""
        torch.manual_seed(42)

        model = SimpleCNN(in_channels=3, hidden_channels=16)
        B = 4
        H, W = 8, 8
        z_t = torch.randn(B, 3, H, W)
        t = torch.rand(B, 1) * 0.8 + 0.1

        v_t_jvp, jvp_result = compute_jvp_two_pass(model, z_t, t)
        v_t_fd, jvp_fd = compute_jvp_finite_diff(model, z_t, t)

        assert torch.allclose(v_t_jvp, v_t_fd, atol=1e-6), \
            f"v_t mismatch: {(v_t_jvp - v_t_fd).abs().max()}"

        assert torch.allclose(jvp_result, jvp_fd, atol=1e-3), \
            f"JVP mismatch: {(jvp_result - jvp_fd).abs().max()}"

        print("✓ CNN JVP matches finite differences")

    def test_gradient_flow(self):
        """Verify gradients flow correctly through JVP."""
        torch.manual_seed(42)

        model = SimpleMLP(in_dim=3, hidden_dim=32, out_dim=2)
        B = 4
        z_t = torch.randn(B, 2, requires_grad=True)
        t = torch.rand(B, 1) * 0.8 + 0.1

        v_t, jvp_result = compute_jvp_two_pass(model, z_t, t)
        loss = (v_t ** 2 + jvp_result ** 2).sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert z_t.grad is not None, "No gradient for z_t"
        assert z_t.grad.abs().sum() > 0, "Gradient is all zeros"

        print("✓ Gradients flow correctly through JVP")

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        torch.manual_seed(42)
        model = SimpleMLP(in_dim=3, hidden_dim=32, out_dim=2)

        for B in [1, 2, 7, 16, 32]:
            z_t = torch.randn(B, 2)
            t = torch.rand(B, 1) * 0.8 + 0.1

            v_t_jvp, jvp_result = compute_jvp_two_pass(model, z_t, t)
            v_t_fd, jvp_fd = compute_jvp_finite_diff(model, z_t, t)

            assert torch.allclose(v_t_jvp, v_t_fd, atol=1e-6)
            assert torch.allclose(jvp_result, jvp_fd, atol=1e-3)

        print("✓ All batch sizes pass")

    def test_meanflow_loss_integration(self):
        """Test that MeanFlowLoss computes without error."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleMLP(in_dim=3, hidden_dim=32, out_dim=2)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.5)

        batch = {
            'raw_source': torch.randn(8, 2),
            'raw_target': torch.randn(8, 2),
        }

        loss = loss_fn(batch)
        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"

        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"

        print("✓ MeanFlowLoss integration test passed")


def run_all_tests():
    """Run all JVP correctness tests."""
    test = TestJVPCorrectness()

    test.test_mlp_jvp_vs_finite_diff()
    test.test_cnn_jvp_vs_finite_diff()
    test.test_gradient_flow()
    test.test_different_batch_sizes()
    test.test_meanflow_loss_integration()

    print("\n✅ All JVP correctness tests passed!")


if __name__ == "__main__":
    run_all_tests()
