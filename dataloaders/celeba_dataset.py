from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import datasets, transforms  # type: ignore[import]

from dataloaders.base_dataloaders import BaseDataset, BaseItem


@dataclass
class CelebAItem(BaseItem):
    raw_source: torch.Tensor
    raw_target: torch.Tensor


class CelebADataset(BaseDataset):
    """CelebA-based dataset producing flow-matching training tuples.

    For each image y (target), we sample a Gaussian noise image x (source),
    a time t ~ Uniform[0, 1], and return:
      - input: x_t = (1 - t) * x + t * y
      - target: v = y - x

    Args:
      root: directory to store/download CelebA
      split: one of {"train", "valid", "test", "all"}
      image_size: final square size to which images are resized
      download: whether to download the dataset if missing
      to_grayscale: convert to single-channel if true
      normalize: if true, normalize to [-1, 1] per channel with mean 0.5, std 0.5
      seed: RNG seed for sampling noise and times
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 128,
        download: bool = False,
        to_grayscale: bool = False,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        self._root = Path(root)

        tfs = [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
        ]
        if to_grayscale:
            tfs.append(transforms.Grayscale(num_output_channels=1))
        tfs.append(transforms.ToTensor())
        if normalize:
            # Normalize images to [-1, 1] range per channel
            num_channels = 1 if to_grayscale else 3
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            tfs.append(transforms.Normalize(mean=mean, std=std))

        self._transform = transforms.Compose(tfs)
        self._dataset = datasets.CelebA(
            root=str(self._root),
            split=split,
            target_type="attr",
            transform=self._transform,
            download=download,
        )

        # Determine sample shape (C, H, W) from a probe
        sample0, _ = self._dataset[0]
        self._sample_shape = sample0.shape  # type: ignore[assignment]

        self._rng = torch.Generator().manual_seed(seed)

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

        return CelebAItem(raw_source=x, raw_target=y, t=t, input=x_t, target=v)
