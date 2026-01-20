from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms  # type: ignore[import]
from torch.utils.data import Dataset, Subset


class CelebASimpleDataset(Dataset):
    """Simple CelebA dataset for autoencoder training.

    Returns images only, without flow matching interpolation.
    Images are normalized to [-1, 1] range.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 128,
        download: bool = False,
        to_grayscale: bool = False,
        normalize: bool = True,
        subset_length: int | None = None,
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

        if subset_length is not None:
            n = min(int(subset_length), len(self._dataset))
            self._dataset = Subset(self._dataset, list(range(n)))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        img, _ = self._dataset[index]
        return img.to(dtype=torch.float32)
