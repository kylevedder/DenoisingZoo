"""Test time sampling distributions.

Verifies logit-normal time sampling and its properties.
"""

import torch
import pytest
import math


class TestLogitNormalSampling:
    """Test logit-normal distribution sampling."""

    def sample_logit_normal(
        self,
        shape: tuple,
        mean: float = 0.0,
        std: float = 1.0,
        eps: float = 1e-5,
        seed: int = 42,
    ) -> torch.Tensor:
        """Sample from logit-normal distribution."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(shape, generator=gen)
        z = z * std + mean
        t = torch.sigmoid(z)
        t = t.clamp(eps, 1 - eps)
        return t

    def test_samples_in_valid_range(self):
        """All samples should be in (eps, 1-eps)."""
        eps = 1e-5
        samples = self.sample_logit_normal((10000,), eps=eps)

        assert samples.min() >= eps, f"Min {samples.min()} < eps"
        assert samples.max() <= 1 - eps, f"Max {samples.max()} > 1-eps"

    def test_no_exact_zero_or_one(self):
        """Samples should never be exactly 0 or 1."""
        samples = self.sample_logit_normal((10000,))

        assert (samples > 0).all(), "Some samples are exactly 0"
        assert (samples < 1).all(), "Some samples are exactly 1"

    def test_symmetric_distribution_mean_zero(self):
        """With mean=0, distribution should be symmetric around 0.5."""
        samples = self.sample_logit_normal((100000,), mean=0.0, std=1.0)

        sample_mean = samples.mean().item()
        # Should be close to 0.5 (not exactly due to clamping effects)
        assert abs(sample_mean - 0.5) < 0.02, \
            f"Mean {sample_mean} not close to 0.5 for symmetric distribution"

    def test_positive_mean_shifts_distribution(self):
        """Positive mean should shift distribution toward 1."""
        samples_centered = self.sample_logit_normal((10000,), mean=0.0, seed=42)
        samples_shifted = self.sample_logit_normal((10000,), mean=1.0, seed=42)

        assert samples_shifted.mean() > samples_centered.mean(), \
            "Positive mean should shift distribution toward higher values"

    def test_negative_mean_shifts_distribution(self):
        """Negative mean should shift distribution toward 0."""
        samples_centered = self.sample_logit_normal((10000,), mean=0.0, seed=42)
        samples_shifted = self.sample_logit_normal((10000,), mean=-1.0, seed=42)

        assert samples_shifted.mean() < samples_centered.mean(), \
            "Negative mean should shift distribution toward lower values"

    def test_larger_std_spreads_distribution(self):
        """Larger std should create more uniform-like distribution."""
        samples_narrow = self.sample_logit_normal((10000,), std=0.5)
        samples_wide = self.sample_logit_normal((10000,), std=2.0)

        # Narrow distribution should be more concentrated around 0.5
        assert samples_narrow.std() < samples_wide.std(), \
            "Smaller std should have smaller variance in output"

    def test_small_std_concentrates_near_mean(self):
        """Small std should concentrate samples near sigmoid(mean)."""
        mean = 1.0  # sigmoid(1) ≈ 0.73
        samples = self.sample_logit_normal((10000,), mean=mean, std=0.1)

        expected_center = torch.sigmoid(torch.tensor(mean)).item()
        sample_mean = samples.mean().item()

        assert abs(sample_mean - expected_center) < 0.05, \
            f"Small std should concentrate samples near {expected_center}"


class TestUniformSampling:
    """Test uniform time sampling for comparison."""

    def test_uniform_in_range(self):
        """Uniform samples should be in [0, 1]."""
        samples = torch.rand(10000)
        assert samples.min() >= 0
        assert samples.max() <= 1

    def test_uniform_mean_is_half(self):
        """Uniform distribution should have mean ≈ 0.5."""
        samples = torch.rand(100000)
        assert abs(samples.mean().item() - 0.5) < 0.01


class TestClampingBehavior:
    """Test clamping at boundaries."""

    def test_clamping_prevents_exact_boundaries(self):
        """Clamping should prevent exact 0 or 1."""
        # Create samples that would be at boundaries without clamping
        z = torch.tensor([-100.0, 100.0])  # Extreme values
        t = torch.sigmoid(z)

        # Without clamping, these would be effectively 0 and 1 (or exactly due to float precision)
        assert t[0] < 1e-10
        assert t[1] >= 1 - 1e-10  # May be exactly 1.0 due to float32 precision

        # After clamping
        eps = 1e-5
        t_clamped = t.clamp(eps, 1 - eps)
        assert t_clamped[0] == eps
        assert t_clamped[1] == 1 - eps

    def test_clamping_preserves_interior_values(self):
        """Clamping should not affect interior values."""
        z = torch.linspace(-3, 3, 100)
        t = torch.sigmoid(z)

        eps = 1e-5
        t_clamped = t.clamp(eps, 1 - eps)

        # Interior values should be unchanged
        interior_mask = (t > eps) & (t < 1 - eps)
        assert torch.allclose(t[interior_mask], t_clamped[interior_mask])


class TestTimeDistributionProperties:
    """Test properties relevant to flow matching training."""

    def sample_logit_normal(
        self,
        shape: tuple,
        mean: float = 0.0,
        std: float = 1.0,
        eps: float = 1e-5,
        seed: int = 42,
    ) -> torch.Tensor:
        """Sample from logit-normal distribution."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(shape, generator=gen)
        z = z * std + mean
        t = torch.sigmoid(z)
        t = t.clamp(eps, 1 - eps)
        return t

    def test_logit_normal_avoids_extremes(self):
        """Logit-normal should naturally avoid 0 and 1."""
        samples = self.sample_logit_normal((10000,), mean=0.0, std=1.0)

        # Count samples in extreme regions
        near_zero = (samples < 0.01).sum()
        near_one = (samples > 0.99).sum()

        # With std=1, there should be relatively few extreme samples
        assert near_zero < 500, f"Too many samples ({near_zero}) near 0"
        assert near_one < 500, f"Too many samples ({near_one}) near 1"

    def test_batch_sampling(self):
        """Verify batch sampling produces correct shapes."""
        B = 32
        samples = self.sample_logit_normal((B, 1))

        assert samples.shape == (B, 1)
        assert (samples > 0).all()
        assert (samples < 1).all()

    def test_different_seeds_different_samples(self):
        """Different seeds should produce different samples."""
        samples1 = self.sample_logit_normal((100,), seed=42)
        samples2 = self.sample_logit_normal((100,), seed=123)

        assert not torch.allclose(samples1, samples2), \
            "Different seeds should produce different samples"

    def test_same_seed_same_samples(self):
        """Same seed should reproduce same samples."""
        samples1 = self.sample_logit_normal((100,), seed=42)
        samples2 = self.sample_logit_normal((100,), seed=42)

        assert torch.allclose(samples1, samples2), \
            "Same seed should produce same samples"


class TestTimeSamplingVsUniform:
    """Compare logit-normal to uniform sampling."""

    def sample_logit_normal(
        self,
        shape: tuple,
        mean: float = 0.0,
        std: float = 1.0,
        eps: float = 1e-5,
        seed: int = 42,
    ) -> torch.Tensor:
        """Sample from logit-normal distribution."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(shape, generator=gen)
        z = z * std + mean
        t = torch.sigmoid(z)
        t = t.clamp(eps, 1 - eps)
        return t

    def test_logit_normal_concentrates_more_at_center(self):
        """Logit-normal (std=1) should have more mass near center than uniform."""
        n = 100000
        logit_samples = self.sample_logit_normal((n,), std=1.0)

        gen = torch.Generator().manual_seed(42)
        uniform_samples = torch.rand(n, generator=gen)

        # Count samples in middle region [0.4, 0.6]
        logit_center = ((logit_samples > 0.4) & (logit_samples < 0.6)).sum()
        uniform_center = ((uniform_samples > 0.4) & (uniform_samples < 0.6)).sum()

        # Logit-normal should have more mass in center
        assert logit_center > uniform_center, \
            "Logit-normal should concentrate more at center"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
