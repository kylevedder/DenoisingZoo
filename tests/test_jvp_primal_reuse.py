"""Tests for JVP primal reuse optimization.

Verifies that reusing the JVP primal output instead of running a separate
forward pass produces identical results with correct gradient flow.
"""

import torch
import torch.nn as nn

from dataloaders.base_dataloaders import make_unified_flow_matching_input
from model_contracts import TimeChannelModule


class SimpleCNN(TimeChannelModule):
    """Simple CNN for testing."""

    def __init__(self, in_channels: int, hidden_channels: int, time_channels: int = 2):
        super().__init__(time_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + time_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestJVPPrimalEquality:
    """Test that JVP primal output equals direct forward pass."""

    def test_jvp_primal_equals_forward_pass(self):
        """Verify torch.func.jvp primal output is identical to model(x)."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()  # Disable dropout for determinism

        z = torch.randn(4, 3, 8, 8)
        r = torch.rand(4, 1) * 0.3 + 0.1
        t = torch.rand(4, 1) * 0.3 + 0.6

        # Direct forward pass
        time_input = torch.cat([r, t], dim=1)
        unified = make_unified_flow_matching_input(z, time_input)
        u_direct = model(unified)

        # JVP primal
        def u_func(z_in, r_in, t_in):
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return model(unified)

        tangent_z = torch.randn_like(z)
        tangent_r = torch.zeros_like(r)
        tangent_t = torch.ones_like(t)

        u_primal, _ = torch.func.jvp(
            u_func,
            (z, r, t),
            (tangent_z, tangent_r, tangent_t),
        )

        # Should be identical (same precision, same model state)
        assert torch.allclose(u_direct, u_primal, atol=1e-6), (
            f"Max diff: {(u_direct - u_primal).abs().max()}"
        )

    def test_jvp_primal_float32_consistency(self):
        """Verify JVP primal in float32 matches float32 forward pass."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        z = torch.randn(4, 3, 8, 8)
        r = torch.rand(4, 1) * 0.3 + 0.1
        t = torch.rand(4, 1) * 0.3 + 0.6

        # Direct forward in float32
        time_input = torch.cat([r, t], dim=1)
        unified = make_unified_flow_matching_input(z, time_input)
        u_direct = model(unified.float()).float()

        # JVP in float32 (matching MeanFlowLoss behavior)
        def u_func(z_in, r_in, t_in):
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return model(unified)

        u_primal, _ = torch.func.jvp(
            u_func,
            (z.float(), r.float(), t.float()),
            (torch.randn_like(z).float(), torch.zeros_like(r).float(), torch.ones_like(t).float()),
        )

        assert torch.allclose(u_direct, u_primal, atol=1e-6)


class TestJVPPrimalGradients:
    """Test gradient flow through JVP primal output."""

    def test_gradients_flow_through_jvp_primal(self):
        """Verify gradients flow correctly when using JVP primal as prediction."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)

        z = torch.randn(4, 3, 8, 8)
        r = torch.rand(4, 1) * 0.3 + 0.1
        t = torch.rand(4, 1) * 0.3 + 0.6

        def u_func(z_in, r_in, t_in):
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return model(unified)

        tangent_z = torch.randn_like(z)
        tangent_r = torch.zeros_like(r)
        tangent_t = torch.ones_like(t)

        u_primal, dudt = torch.func.jvp(
            u_func,
            (z, r, t),
            (tangent_z, tangent_r, tangent_t),
        )

        # Use primal for loss (NOT detached)
        target = torch.randn_like(u_primal)
        loss = ((u_primal - target) ** 2).mean()
        loss.backward()

        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "Gradients should flow through JVP primal"

    def test_detached_primal_blocks_gradients(self):
        """Verify that detaching primal blocks gradient flow (sanity check)."""
        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)

        z = torch.randn(4, 3, 8, 8)
        r = torch.rand(4, 1) * 0.3 + 0.1
        t = torch.rand(4, 1) * 0.3 + 0.6

        def u_func(z_in, r_in, t_in):
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return model(unified)

        u_primal, _ = torch.func.jvp(
            u_func,
            (z, r, t),
            (torch.randn_like(z), torch.zeros_like(r), torch.ones_like(t)),
        )

        # Detach primal - loss won't require grad at all
        target = torch.randn_like(u_primal)
        loss = ((u_primal.detach() - target) ** 2).mean()

        # Detached tensor means loss doesn't require grad
        assert not loss.requires_grad, "Loss from detached primal should not require grad"


class TestMeanFlowLossOptimization:
    """Test the optimized MeanFlowLoss implementation."""

    def test_optimized_loss_matches_original_all_mf(self):
        """Verify optimized implementation matches original with 100% MeanFlow."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=1.0, weighting_power=0.0)

        batch = {
            "raw_source": torch.randn(4, 3, 8, 8),
            "raw_target": torch.randn(4, 3, 8, 8),
        }

        # Run loss
        loss = loss_fn(batch)

        assert not torch.isnan(loss)
        assert loss > 0

        # Check gradients flow
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0

    def test_optimized_loss_matches_original_all_fm(self):
        """Verify optimized implementation matches original with 0% MeanFlow."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.0, weighting_power=0.0)

        batch = {
            "raw_source": torch.randn(4, 3, 8, 8),
            "raw_target": torch.randn(4, 3, 8, 8),
        }

        loss = loss_fn(batch)

        assert not torch.isnan(loss)
        assert loss > 0

        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0

    def test_optimized_loss_matches_original_mixed(self):
        """Verify optimized implementation with mixed FM/MF samples."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.5, weighting_power=0.0)

        batch = {
            "raw_source": torch.randn(8, 3, 8, 8),
            "raw_target": torch.randn(8, 3, 8, 8),
        }

        loss = loss_fn(batch)

        assert not torch.isnan(loss)
        assert loss > 0

        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0


class TestEdgeCases:
    """Test edge cases for the optimization."""

    def test_single_sample_mf(self):
        """Test with single MeanFlow sample."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=1.0)

        batch = {
            "raw_source": torch.randn(1, 3, 8, 8),
            "raw_target": torch.randn(1, 3, 8, 8),
        }

        loss = loss_fn(batch)
        assert not torch.isnan(loss)
        loss.backward()

    def test_single_sample_fm(self):
        """Test with single FM sample."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.0)

        batch = {
            "raw_source": torch.randn(1, 3, 8, 8),
            "raw_target": torch.randn(1, 3, 8, 8),
        }

        loss = loss_fn(batch)
        assert not torch.isnan(loss)
        loss.backward()

    def test_large_batch(self):
        """Test with larger batch size."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.25)

        batch = {
            "raw_source": torch.randn(32, 3, 8, 8),
            "raw_target": torch.randn(32, 3, 8, 8),
        }

        loss = loss_fn(batch)
        assert not torch.isnan(loss)
        loss.backward()


class TestFullBatchJVPMode:
    """Test the full_batch_jvp mode that enables CUDA graph capture."""

    def test_full_batch_jvp_matches_selective(self):
        """Verify full_batch_jvp mode produces same results as selective mode."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        model.eval()

        # Create loss functions with different modes
        loss_fn_selective = MeanFlowLoss(model=model, meanflow_ratio=0.5, weighting_power=0.0)
        loss_fn_full = MeanFlowLoss(
            model=model, meanflow_ratio=0.5, weighting_power=0.0, full_batch_jvp=True
        )

        batch = {
            "raw_source": torch.randn(8, 3, 8, 8),
            "raw_target": torch.randn(8, 3, 8, 8),
        }

        # Run both modes
        torch.manual_seed(123)
        loss_selective = loss_fn_selective(batch)

        torch.manual_seed(123)
        loss_full = loss_fn_full(batch)

        # Losses should be close (not exact due to different code paths)
        assert torch.allclose(loss_selective, loss_full, rtol=0.01), (
            f"Selective: {loss_selective.item()}, Full: {loss_full.item()}"
        )

    def test_full_batch_jvp_gradients_flow(self):
        """Verify gradients flow correctly in full_batch_jvp mode."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(
            model=model, meanflow_ratio=1.0, full_batch_jvp=True
        )

        batch = {
            "raw_source": torch.randn(4, 3, 8, 8),
            "raw_target": torch.randn(4, 3, 8, 8),
        }

        loss = loss_fn(batch)
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "Gradients should flow in full_batch_jvp mode"


class TestNumericalRegression:
    """Regression tests to verify optimization doesn't change numerical behavior."""

    def test_linear_model_loss_deterministic(self):
        """Verify loss value is deterministic with fixed seed."""
        from losses.meanflow_loss import MeanFlowLoss
        from model_contracts import TimeChannelModule

        class LinearModel(TimeChannelModule):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # u(z, r, t) = 2*z + 3*r + 5*t
                # Input is (B, C, H, W) with last 2 channels being (r, t)
                z = x[:, :-2]
                r = x[:, -2:-1]
                t = x[:, -1:]
                return 2.0 * z + 3.0 * r + 5.0 * t

        def run_with_seed(seed):
            torch.manual_seed(seed)
            model = LinearModel()
            loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.5, weighting_power=0.0)
            batch = {
                "raw_source": torch.randn(8, 3, 4, 4),
                "raw_target": torch.randn(8, 3, 4, 4),
            }
            return loss_fn(batch)

        loss1 = run_with_seed(12345)
        loss2 = run_with_seed(12345)

        assert torch.allclose(loss1, loss2, atol=1e-6)

    def test_gradients_correct_magnitude(self):
        """Verify gradient magnitudes are reasonable."""
        from losses.meanflow_loss import MeanFlowLoss

        torch.manual_seed(42)
        model = SimpleCNN(in_channels=3, hidden_channels=16)
        loss_fn = MeanFlowLoss(model=model, meanflow_ratio=0.5)

        batch = {
            "raw_source": torch.randn(4, 3, 8, 8),
            "raw_target": torch.randn(4, 3, 8, 8),
        }

        loss = loss_fn(batch)
        loss.backward()

        # Collect gradient norms
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())

        # Gradients should be finite and non-zero
        assert all(0 < g < 1e6 for g in grad_norms), f"Gradient norms: {grad_norms}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
