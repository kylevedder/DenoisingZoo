"""Test flow matching mathematical foundations.

Verifies interpolation, velocity field, and unified input handling.
"""

import torch
import pytest

from dataloaders.base_dataloaders import (
    make_unified_flow_matching_input,
    make_ununified_flow_matching_input,
)


class TestInterpolation:
    """Test linear interpolation z_t = (1-t)*x + t*y."""

    def test_interpolation_at_t0_equals_x(self):
        """At t=0, z_t should equal x (source)."""
        B = 4
        x = torch.randn(B, 3, 32, 32)
        y = torch.randn(B, 3, 32, 32)
        t = torch.zeros(B, 1)

        # z_t = (1-t)*x + t*y at t=0 should equal x
        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        assert torch.allclose(z_t, x), "At t=0, z_t should equal x"

    def test_interpolation_at_t1_equals_y(self):
        """At t=1, z_t should equal y (target)."""
        B = 4
        x = torch.randn(B, 3, 32, 32)
        y = torch.randn(B, 3, 32, 32)
        t = torch.ones(B, 1)

        # z_t = (1-t)*x + t*y at t=1 should equal y
        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        assert torch.allclose(z_t, y), "At t=1, z_t should equal y"

    def test_interpolation_at_t05_is_midpoint(self):
        """At t=0.5, z_t should be the midpoint."""
        B = 4
        x = torch.randn(B, 3, 32, 32)
        y = torch.randn(B, 3, 32, 32)
        t = torch.full((B, 1), 0.5)

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y
        expected = (x + y) / 2

        assert torch.allclose(z_t, expected), "At t=0.5, z_t should be midpoint"

    def test_interpolation_2d_vectors(self):
        """Test interpolation for 2D data (B, D)."""
        B = 4
        D = 16
        x = torch.randn(B, D)
        y = torch.randn(B, D)
        t = torch.zeros(B, 1)

        z_0 = (1 - t) * x + t * y
        assert torch.allclose(z_0, x), "At t=0, z_t should equal x (2D case)"

        t = torch.ones(B, 1)
        z_1 = (1 - t) * x + t * y
        assert torch.allclose(z_1, y), "At t=1, z_t should equal y (2D case)"


class TestVelocityField:
    """Test constant velocity v = y - x."""

    def test_velocity_is_difference(self):
        """Velocity should be v = y - x."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randn(4, 3, 32, 32)

        v = y - x

        # Verify velocity takes us from x to y in unit time
        # z_1 = x + 1 * v should equal y
        z_1 = x + v
        assert torch.allclose(z_1, y, atol=1e-6), "x + v should equal y"

    def test_velocity_is_time_independent(self):
        """Velocity is constant (same at all times)."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randn(4, 3, 32, 32)

        v = y - x

        # The velocity at any interpolated point should still be y - x
        for t_val in [0.0, 0.3, 0.5, 0.8, 1.0]:
            t = torch.full((4, 1), t_val)
            t_b = t.view(4, 1, 1, 1)
            z_t = (1 - t_b) * x + t_b * y

            # Derivative of z_t w.r.t. t is always y - x
            dz_dt = y - x
            assert torch.allclose(dz_dt, v), f"Velocity should be constant at t={t_val}"


class TestUnifiedInput:
    """Test unified input packing and unpacking."""

    def test_pack_unpack_2d_roundtrip(self):
        """Pack and unpack 2D input preserves values."""
        B = 4
        D = 16
        x = torch.randn(B, D)
        t = torch.rand(B, 1)

        unified = make_unified_flow_matching_input(x, t)

        # Should be (B, D+1)
        assert unified.shape == (B, D + 1), f"Expected shape {(B, D+1)}, got {unified.shape}"

        # Unpack
        result = make_ununified_flow_matching_input(unified)

        assert torch.allclose(result.x, x), "Unpacked x should match original"
        assert torch.allclose(result.t, t), "Unpacked t should match original"

    def test_pack_unpack_4d_roundtrip(self):
        """Pack and unpack 4D (image) input preserves values."""
        B = 4
        C = 3
        H = W = 32
        x = torch.randn(B, C, H, W)
        t = torch.rand(B, 1)

        unified = make_unified_flow_matching_input(x, t)

        # Should be (B, C+1, H, W)
        assert unified.shape == (B, C + 1, H, W), f"Expected shape {(B, C+1, H, W)}, got {unified.shape}"

        # Unpack
        result = make_ununified_flow_matching_input(unified)

        assert torch.allclose(result.x, x), "Unpacked x should match original"
        assert torch.allclose(result.t, t, atol=1e-5), "Unpacked t should match original"

    def test_time_channel_is_constant(self):
        """Time channel in 4D unified input should be spatially constant."""
        B = 4
        C = 3
        H = W = 32
        x = torch.randn(B, C, H, W)
        t = torch.rand(B, 1)

        unified = make_unified_flow_matching_input(x, t)

        # Extract time channel
        t_channel = unified[:, -1, :, :]  # (B, H, W)

        # Should be constant across spatial dimensions
        for b in range(B):
            t_slice = t_channel[b]  # (H, W)
            assert torch.allclose(
                t_slice, t[b, 0].expand(H, W)
            ), f"Time channel should be constant for batch {b}"

    def test_batch_size_mismatch_raises(self):
        """Mismatched batch sizes should raise ValueError."""
        x = torch.randn(4, 3, 32, 32)
        t = torch.rand(2, 1)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            make_unified_flow_matching_input(x, t)

    def test_unsupported_dimension_raises(self):
        """Unsupported tensor dimensions should raise ValueError."""
        x = torch.randn(4, 3, 16, 16, 8)  # 5D tensor
        t = torch.rand(4, 1)

        with pytest.raises(ValueError, match="Unsupported"):
            make_unified_flow_matching_input(x, t)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_source(self):
        """Source = zeros should work correctly."""
        B = 4
        x = torch.zeros(B, 3, 32, 32)
        y = torch.randn(B, 3, 32, 32)
        t = torch.rand(B, 1)

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        # Should equal t * y when x = 0
        expected = t_b * y
        assert torch.allclose(z_t, expected), "z_t should be t*y when x=0"

    def test_same_source_target(self):
        """When x = y, z_t = x for all t."""
        B = 4
        x = torch.randn(B, 3, 32, 32)
        y = x.clone()
        t = torch.rand(B, 1)

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        assert torch.allclose(z_t, x), "When x=y, z_t should equal x for all t"

    def test_large_values(self):
        """Test with large tensor values."""
        B = 2
        x = torch.randn(B, 3, 32, 32) * 1000
        y = torch.randn(B, 3, 32, 32) * 1000
        t = torch.tensor([[0.0], [1.0]])

        t_b = t.view(B, 1, 1, 1)
        z_t = (1 - t_b) * x + t_b * y

        assert torch.allclose(z_t[0], x[0]), "At t=0, should equal x"
        assert torch.allclose(z_t[1], y[1]), "At t=1, should equal y"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
