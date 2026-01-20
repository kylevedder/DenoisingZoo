from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import datasets, transforms  # type: ignore[import]
from torch.utils.data import Subset

from dataloaders.base_dataloaders import (
    BaseDataset,
    BaseItem,
    make_time_input,
    make_unified_flow_matching_input,
)


@dataclass
class CIFAR10Item(BaseItem):
    raw_source: torch.Tensor
    raw_target: torch.Tensor


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset producing flow-matching training tuples.

    For each image y (target), we sample a Gaussian noise image x (source),
    a time t ~ Uniform[0, 1], and return:
      - input: x_t = (1 - t) * x + t * y
      - target: v = y - x

    Args:
      root: directory to store/download CIFAR-10
      train: if True, use training set; else use test set
      download: whether to download the dataset if missing
      seed: RNG seed for sampling noise and times
      subset_length: if provided, use only first N samples
    """

    def __init__(
        self,
        root: str = "data/cifar10",
        train: bool = True,
        download: bool = True,
        seed: int = 42,
        subset_length: int | None = None,
    ) -> None:
        # CIFAR-10 is 32x32x3, normalize to [-1, 1]
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=self._transform,
            download=download,
        )

        # Shape is (3, 32, 32)
        self._sample_shape = (3, 32, 32)

        self._rng = torch.Generator().manual_seed(seed)

        if subset_length is not None:
            n = min(int(subset_length), len(self._dataset))
            self._dataset = Subset(self._dataset, list(range(n)))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> BaseItem:
        y_img, _ = self._dataset[index]
        # Ensure float tensor
        y: torch.Tensor = y_img.to(dtype=torch.float32)

        # Source noise x ~ N(0, 1)
        x = torch.randn(self._sample_shape, generator=self._rng, dtype=torch.float32)

        # Time t ~ U[0, 1] (shape [1])
        t = torch.rand(1, generator=self._rng, dtype=torch.float32)

        # Interpolate and target velocity
        x_t = x * (1 - t) + y * t
        v = y - x

        time_input = make_time_input(t.unsqueeze(0))
        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), time_input
        ).squeeze(0)
        return CIFAR10Item(
            raw_source=x, raw_target=y, t=t, input=x_t, target=v, unified_input=unified
        )
