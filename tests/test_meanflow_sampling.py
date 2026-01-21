"""Tests for MeanFlow sampling logic."""

import torch

from evaluation.sample import generate_samples_meanflow
from model_contracts import TimeChannelModule


class TimeAwareModel(TimeChannelModule):
    """Simple model that depends on r and t."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        x = unified_input[:, : self.feature_dim]
        times = unified_input[:, self.feature_dim :]
        r = times[:, 0:1]
        t = times[:, 1:2]
        return torch.ones_like(x) * (r + 2.0 * t)


def test_generate_samples_meanflow_uses_r_and_t():
    device = torch.device("cpu")
    model = TimeAwareModel(feature_dim=4)
    seed = 7

    num_samples = 8
    batch_size = 8
    sample_shape = (4,)
    r = 0.2
    t = 0.8

    iterator = generate_samples_meanflow(
        model,
        num_samples=num_samples,
        sample_shape=sample_shape,
        device=device,
        batch_size=batch_size,
        seed=seed,
        r=r,
        t=t,
    )
    samples = next(iterator)

    rng = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((batch_size, *sample_shape), device=device, generator=rng)
    u = torch.ones_like(z) * (r + 2.0 * t)
    expected = z + (t - r) * u

    assert torch.allclose(samples, expected)
