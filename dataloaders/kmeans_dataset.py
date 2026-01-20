from __future__ import annotations

from dataclasses import dataclass
import torch

from dataloaders.base_dataloaders import (
    BaseDataset,
    BaseItem,
    make_unified_flow_matching_input,
)


@dataclass
class KMeansItem(BaseItem):
    raw_source: torch.Tensor
    raw_target: torch.Tensor


class GaussianKMeansDataset(BaseDataset):
    def __init__(
        self,
        centroids: list[list[float]],
        length: int,
        source_std: float = 1.0,
        target_std: float = 0.1,
        seed: int = 69,
    ) -> None:
        self._length = int(length)
        self._centroids = torch.tensor(centroids)
        self._source_std = float(source_std)
        self._target_std = float(target_std)
        self._rng = torch.Generator().manual_seed(seed)

        if self._centroids.ndim != 2:
            raise ValueError("centroids must be a 2D sequence with shape (k, dim)")
        self._sample_size = int(self._centroids.shape[1])

        # Keep centroids on default device/dtype; sampling also uses defaults.

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> BaseItem:
        del index
        x = self._generate_source()
        t = self._generate_time()
        y = self._generate_target()

        # Linearly interpolate between source and target
        x_t = x * (1 - t) + y * t

        # Regression target is the velocity, i.e. diff between the source and
        # target, irrespective of the time
        pred_target = y - x

        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), t.unsqueeze(0)
        ).squeeze(0)
        return KMeansItem(
            raw_source=x,
            raw_target=pred_target,
            t=t,
            input=x_t,
            target=pred_target,
            unified_input=unified,
        )

    # Private single-sample generators -------------------------------------------------
    def _generate_source(self) -> torch.Tensor:
        sample = torch.randn(self._sample_size, generator=self._rng) * self._source_std
        return sample

    def _generate_target(self) -> torch.Tensor:
        num_means = self._centroids.shape[0]
        component = int(torch.randint(0, num_means, (1,), generator=self._rng).item())
        idx = torch.tensor([component], dtype=torch.long)
        mean = torch.index_select(self._centroids, 0, idx)[0]
        sample = (
            torch.randn(self._sample_size, generator=self._rng) * self._target_std
            + mean
        )
        return sample

    def _generate_time(self) -> torch.Tensor:
        t = torch.rand(1, generator=self._rng)
        return t
