"""Test adaptive weighting scheme.

Verifies w = 1/(err + c)^p weighting for MeanFlow loss.
"""

import torch
import pytest


def compute_adaptive_weights(
    sq_error: torch.Tensor,
    power: float = 0.5,
    const: float = 1e-4,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute adaptive weights: w = 1/(err + c)^p."""
    weights = 1.0 / (sq_error + const) ** power
    if normalize:
        weights = weights / weights.mean()
    return weights


class TestSmallErrorHighWeight:
    """Test that small errors get high weights."""

    def test_zero_error_gets_max_weight(self):
        """Zero error should get weight ≈ 1/c^p."""
        sq_error = torch.tensor([0.0, 1.0, 10.0])
        const = 1e-4
        power = 0.5

        weights = compute_adaptive_weights(sq_error, power, const, normalize=False)

        expected_at_zero = 1.0 / (const ** power)
        assert torch.isclose(weights[0], torch.tensor(expected_at_zero)), \
            f"Zero error weight {weights[0]} != expected {expected_at_zero}"

    def test_small_error_higher_weight(self):
        """Smaller errors should get higher weights."""
        sq_error = torch.tensor([0.01, 0.1, 1.0, 10.0])

        weights = compute_adaptive_weights(sq_error, normalize=False)

        # Weights should be monotonically decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], \
                f"Weight at {i} should be > weight at {i+1}"

    def test_large_error_small_weight(self):
        """Large errors should get small weights."""
        sq_error = torch.tensor([0.001, 100.0])

        weights = compute_adaptive_weights(sq_error, normalize=False)

        assert weights[1] < weights[0] / 10, \
            "Large error should have much smaller weight"


class TestWeightNormalization:
    """Test weight normalization preserves scale."""

    def test_normalized_weights_mean_one(self):
        """After normalization, mean(w) should equal 1."""
        sq_error = torch.rand(100) * 10

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5), \
            f"Normalized weights mean {weights.mean()} != 1"

    def test_normalization_preserves_relative_ordering(self):
        """Normalization should preserve relative weight ordering."""
        sq_error = torch.tensor([0.1, 1.0, 10.0])

        weights_raw = compute_adaptive_weights(sq_error, normalize=False)
        weights_norm = compute_adaptive_weights(sq_error, normalize=True)

        # Check ordering is preserved
        assert (weights_raw[0] > weights_raw[1]) == (weights_norm[0] > weights_norm[1])
        assert (weights_raw[1] > weights_raw[2]) == (weights_norm[1] > weights_norm[2])

    def test_normalization_sum(self):
        """Sum of normalized weights should equal batch size."""
        B = 32
        sq_error = torch.rand(B) * 10

        weights = compute_adaptive_weights(sq_error, normalize=True)

        expected_sum = float(B)
        assert torch.isclose(weights.sum(), torch.tensor(expected_sum), atol=1e-3), \
            f"Sum {weights.sum()} != expected {expected_sum}"


class TestGradientDetachment:
    """Test that weights don't backpropagate gradients."""

    def test_weights_should_be_detached(self):
        """Weights computation should use detached errors."""
        sq_error = torch.rand(10, requires_grad=True) * 10

        with torch.no_grad():
            weights = compute_adaptive_weights(sq_error.detach(), normalize=True)

        assert not weights.requires_grad, \
            "Weights should not require gradients"

    def test_weighted_loss_still_backprops(self):
        """Loss with detached weights should still backprop through prediction."""
        B = 10
        pred = torch.randn(B, 4, requires_grad=True)
        target = torch.randn(B, 4)

        # Compute squared error
        sq_error = ((pred - target) ** 2).sum(dim=1)

        # Weights are detached
        with torch.no_grad():
            weights = compute_adaptive_weights(sq_error.detach(), normalize=True)

        # Weighted loss
        loss = (weights * sq_error).mean()
        loss.backward()

        assert pred.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(pred.grad == 0), "Gradients should be non-zero"


class TestPowerParameter:
    """Test effect of power parameter p."""

    def test_power_zero_uniform_weights(self):
        """Power=0 should give uniform weights."""
        sq_error = torch.tensor([0.01, 0.1, 1.0, 10.0])

        weights = compute_adaptive_weights(sq_error, power=0.0, normalize=False)

        # All weights should be 1
        assert torch.allclose(weights, torch.ones_like(weights)), \
            "Power=0 should give uniform weights"

    def test_higher_power_more_extreme(self):
        """Higher power should create more extreme weight differences."""
        sq_error = torch.tensor([0.01, 10.0])

        weights_p05 = compute_adaptive_weights(sq_error, power=0.5, normalize=False)
        weights_p10 = compute_adaptive_weights(sq_error, power=1.0, normalize=False)

        ratio_p05 = weights_p05[0] / weights_p05[1]
        ratio_p10 = weights_p10[0] / weights_p10[1]

        assert ratio_p10 > ratio_p05, \
            "Higher power should create larger weight ratio"

    def test_negative_power_inverts_weighting(self):
        """Negative power would invert weighting (not typically used)."""
        sq_error = torch.tensor([0.1, 10.0])

        weights_pos = compute_adaptive_weights(sq_error, power=0.5, normalize=False)
        weights_neg = compute_adaptive_weights(sq_error, power=-0.5, normalize=False)

        # With negative power, larger error gets larger weight
        assert weights_neg[1] > weights_neg[0], \
            "Negative power should give larger errors higher weights"


class TestConstantParameter:
    """Test effect of constant parameter c."""

    def test_larger_constant_smoother_weights(self):
        """Larger c should make weights more uniform."""
        sq_error = torch.tensor([0.0, 0.01, 0.1, 1.0])

        weights_small_c = compute_adaptive_weights(sq_error, const=1e-6, normalize=True)
        weights_large_c = compute_adaptive_weights(sq_error, const=1.0, normalize=True)

        std_small_c = weights_small_c.std()
        std_large_c = weights_large_c.std()

        assert std_large_c < std_small_c, \
            "Larger constant should produce more uniform weights"

    def test_constant_prevents_division_by_zero(self):
        """Constant should prevent inf weights at zero error."""
        sq_error = torch.tensor([0.0])

        weights = compute_adaptive_weights(sq_error, const=1e-4, normalize=False)

        assert torch.isfinite(weights).all(), \
            "Constant should prevent infinite weights"


class TestBatchBehavior:
    """Test adaptive weighting across batches."""

    def test_batch_independent_weights(self):
        """Each sample should get weight based on its own error."""
        B = 32
        sq_error = torch.rand(B) * 10

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert weights.shape == (B,), \
            f"Expected shape {(B,)}, got {weights.shape}"

    def test_outlier_gets_small_weight(self):
        """Outlier (large error) should be downweighted."""
        # 9 normal errors + 1 outlier
        sq_error = torch.cat([
            torch.full((9,), 0.1),
            torch.tensor([100.0]),
        ])

        weights = compute_adaptive_weights(sq_error, normalize=True)

        # Outlier should have smallest weight
        assert weights[-1] < weights[:-1].min(), \
            "Outlier should have smallest weight"

    def test_all_equal_errors_uniform_weights(self):
        """If all errors equal, weights should be uniform."""
        sq_error = torch.full((10,), 1.0)

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert torch.allclose(weights, torch.ones_like(weights)), \
            "Equal errors should give uniform weights"


class TestNumericalStability:
    """Test numerical stability of weighting scheme."""

    def test_very_small_errors(self):
        """Should handle very small errors."""
        sq_error = torch.tensor([1e-10, 1e-8, 1e-6])

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert torch.isfinite(weights).all(), \
            "Should handle very small errors"

    def test_very_large_errors(self):
        """Should handle very large errors."""
        sq_error = torch.tensor([1e6, 1e8, 1e10])

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert torch.isfinite(weights).all(), \
            "Should handle very large errors"

    def test_mixed_scale_errors(self):
        """Should handle errors across many scales."""
        sq_error = torch.tensor([1e-8, 1e-4, 1.0, 1e4, 1e8])

        weights = compute_adaptive_weights(sq_error, normalize=True)

        assert torch.isfinite(weights).all(), \
            "Should handle mixed scale errors"
        # Smallest error should still have largest weight
        assert weights[0] > weights[-1], \
            "Smallest error should have largest weight"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
