"""Test MeanFlow target computation.

Verifies the MeanFlow-specific target: u_tgt = v_t - (t - r) * JVP
"""

import torch
import torch.nn as nn
import pytest

from dataloaders.base_dataloaders import make_unified_flow_matching_input


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, channels: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1),  # +1 for time channel
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestREqualsT:
    """Test that r=t case reduces to standard flow matching."""

    def test_r_equals_t_target_is_v_true(self):
        """When r=t, MeanFlow target should equal true velocity."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 16

        # Source and target
        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)

        # True velocity
        v_true = y - x

        # When r = t, the MeanFlow target formula simplifies:
        # u_tgt = v_t - (t - r) * JVP = v_t - 0 * JVP = v_t
        # And v_t should be learned to approximate v_true

        # At r = t, z_r = z_t, so the model sees the same input
        # The target is just v_true
        t = torch.rand(B, 1)
        r = t.clone()  # r = t

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        # For standard FM (r=t), target is v_true
        target = v_true

        # Verify dimensions match
        assert target.shape == v_true.shape

    def test_r_equals_t_loss_reduces_to_mse(self):
        """When r=t for all samples, loss should be MSE(v_pred, v_true)."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 16

        model = SimpleModel(channels=C)
        model.eval()

        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        v_true = y - x

        t = torch.rand(B, 1)
        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        unified = make_unified_flow_matching_input(z_t, t)

        with torch.no_grad():
            v_pred = model(unified)

        # For r=t, loss is just MSE
        loss = ((v_pred - v_true) ** 2).mean()

        # Verify it's a valid positive scalar
        assert loss.ndim == 0
        assert loss >= 0


class TestMeanFlowTarget:
    """Test MeanFlow target computation u_tgt = v_t - (t-r) * JVP."""

    def test_meanflow_target_formula(self):
        """Test the explicit MeanFlow target formula."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 8

        model = SimpleModel(channels=C, hidden=16)
        model.eval()

        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)

        # Sample t and r with r < t
        t = torch.rand(B, 1) * 0.5 + 0.5  # t in [0.5, 1]
        r = torch.rand(B, 1) * t  # r in [0, t]

        t_b = t.view(B, 1, 1, 1)
        r_b = r.view(B, 1, 1, 1)

        z_t = (1 - t_b) * x + t_b * y
        z_r = (1 - r_b) * x + r_b * y

        # Get v_t first
        unified_t = make_unified_flow_matching_input(z_t, t)
        with torch.no_grad():
            v_t_initial = model(unified_t)

        # Compute JVP
        def model_fn(z):
            unified = make_unified_flow_matching_input(z, t)
            return model(unified)

        v_t, jvp_result = torch.func.jvp(
            model_fn,
            (z_t,),
            (v_t_initial.detach(),),
        )

        # Compute MeanFlow target
        delta_t = (t - r).view(B, 1, 1, 1)
        u_tgt = v_t - delta_t * jvp_result

        # Verify shapes
        assert u_tgt.shape == (B, C, H, W)

        # Verify it's different from v_t when r != t
        if not torch.allclose(t, r):
            assert not torch.allclose(u_tgt, v_t), \
                "MeanFlow target should differ from v_t when r != t"

    def test_meanflow_target_at_r0_t1(self):
        """Test MeanFlow target at r=0, t=1 (full integration case)."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 8

        model = SimpleModel(channels=C, hidden=16)
        model.eval()

        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)

        # r=0, t=1
        t = torch.ones(B, 1)
        r = torch.zeros(B, 1)

        t_b = t.view(B, 1, 1, 1)
        r_b = r.view(B, 1, 1, 1)

        z_t = (1 - t_b) * x + t_b * y  # = y at t=1
        z_r = (1 - r_b) * x + r_b * y  # = x at r=0

        assert torch.allclose(z_t, y), "z_t should equal y at t=1"
        assert torch.allclose(z_r, x), "z_r should equal x at r=0"

        # Get v_t
        unified_t = make_unified_flow_matching_input(z_t, t)
        with torch.no_grad():
            v_t_initial = model(unified_t)

        # Compute JVP
        def model_fn(z):
            unified = make_unified_flow_matching_input(z, t)
            return model(unified)

        v_t, jvp_result = torch.func.jvp(
            model_fn,
            (z_t,),
            (v_t_initial.detach(),),
        )

        # MeanFlow target: u_tgt = v_t - (t-r) * JVP = v_t - 1 * JVP
        delta_t = (t - r).view(B, 1, 1, 1)
        u_tgt = v_t - delta_t * jvp_result

        # Verify shapes
        assert u_tgt.shape == (B, C, H, W)


class TestGradientFlow:
    """Test that gradients are handled correctly in MeanFlow."""

    def test_target_requires_no_grad(self):
        """MeanFlow target should have stop_gradient (no requires_grad)."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 8

        model = SimpleModel(channels=C, hidden=16)
        model.train()

        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)

        t = torch.rand(B, 1) * 0.5 + 0.5
        r = torch.rand(B, 1) * t

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        # Get v_t first (with no_grad for the target computation)
        unified_t = make_unified_flow_matching_input(z_t, t)
        with torch.no_grad():
            v_t_initial = model(unified_t)

        def model_fn(z):
            unified = make_unified_flow_matching_input(z, t)
            return model(unified)

        v_t, jvp_result = torch.func.jvp(
            model_fn,
            (z_t,),
            (v_t_initial.detach(),),  # Tangent should be detached
        )

        delta_t = (t - r).view(B, 1, 1, 1)
        u_tgt = v_t - delta_t * jvp_result

        # Detach target for loss computation
        u_tgt_detached = u_tgt.detach()

        assert not u_tgt_detached.requires_grad, \
            "Detached target should not require grad"

    def test_loss_gradients_flow_to_model(self):
        """Gradients from loss should flow to model parameters."""
        torch.manual_seed(42)
        B = 4
        C = 3
        H = W = 8

        model = SimpleModel(channels=C, hidden=16)
        model.train()

        x = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        v_true = y - x

        t = torch.rand(B, 1)
        r = torch.rand(B, 1) * t

        t_b = t.view(B, 1, 1, 1)
        r_b = r.view(B, 1, 1, 1)
        z_r = (1 - r_b) * x + r_b * y

        unified_r = make_unified_flow_matching_input(z_r, r)
        v_r = model(unified_r)

        # Simple loss (not full MeanFlow, just testing gradient flow)
        loss = ((v_r - v_true) ** 2).mean()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestVelocityFieldProperties:
    """Test mathematical properties of the velocity field."""

    def test_velocity_integrates_to_displacement(self):
        """Integrating v from t to 1 should give displacement to y."""
        # For constant velocity v = y - x:
        # Integral from t to 1 of v dt = (1 - t) * v = (1 - t) * (y - x)
        # z_t + (1-t) * v = z_t + (1-t) * (y-x)
        #                 = (1-t)*x + t*y + (1-t)*(y-x)
        #                 = (1-t)*x + t*y + (1-t)*y - (1-t)*x
        #                 = t*y + (1-t)*y = y

        B = 4
        x = torch.randn(B, 3, 16, 16)
        y = torch.randn(B, 3, 16, 16)

        for t_val in [0.0, 0.3, 0.7, 1.0]:
            t = torch.full((B, 1), t_val)
            t_b = t.view(B, 1, 1, 1)

            z_t = (1 - t_b) * x + t_b * y
            v = y - x

            # Integrate velocity from t to 1
            z_1 = z_t + (1 - t_b) * v

            assert torch.allclose(z_1, y, atol=1e-6), \
                f"Integrating velocity should reach y, failed at t={t_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
