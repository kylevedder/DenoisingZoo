"""ImageNet dataset for flow matching training.

Supports both pixel-space and latent-space (with pre-trained VAE) training.
Includes class labels for classifier-free guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms  # type: ignore[import]

from dataloaders.base_dataloaders import (
    BaseDataset,
    BaseItem,
    make_time_input,
    make_unified_flow_matching_input,
)
from constants import CFG_NULL_LABEL


@dataclass
class ImageNetItem(BaseItem):
    """ImageNet flow matching item with class label."""
    raw_source: torch.Tensor
    raw_target: torch.Tensor
    label: int


class ImageNetDataset(BaseDataset):
    """ImageNet dataset for flow matching training.

    For each image y (target) with label c, we sample:
      - Gaussian noise x (source)
      - Time t ~ Uniform[0, 1]

    And return:
      - input: x_t = (1 - t) * x + t * y
      - target: v = y - x
      - label: class index (0-999)

    Args:
        root: Path to ImageNet data (expects train/ and val/ subdirs)
        split: "train" or "val"
        image_size: Size to resize images to (default: 256)
        center_crop_size: Size for center crop before resize (default: 256)
        seed: Random seed for sampling
        subset_length: If provided, use only first N samples
        label_dropout: Probability of dropping label (for CFG training, default: 0.0)
        null_label: Label value to use when dropped (default: CFG_NULL_LABEL)
    """

    NUM_CLASSES = 1000

    def __init__(
        self,
        root: str = "data/imagenet",
        split: str = "train",
        image_size: int = 256,
        center_crop_size: int | None = None,
        seed: int = 42,
        subset_length: int | None = None,
        label_dropout: float = 0.0,
        null_label: int = CFG_NULL_LABEL,
    ) -> None:
        self._root = Path(root)
        self._split = split
        self._image_size = image_size
        self._label_dropout = label_dropout
        self._null_label = null_label

        if center_crop_size is None:
            center_crop_size = image_size

        # Build transforms
        if split == "train":
            # Training: random resized crop + horizontal flip
            self._transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            # Validation: center crop + resize
            self._transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        # Load ImageNet dataset
        split_dir = "train" if split == "train" else "val"
        self._dataset = datasets.ImageFolder(
            root=str(self._root / split_dir),
            transform=self._transform,
        )

        self._sample_shape = (3, image_size, image_size)
        self._rng = torch.Generator().manual_seed(seed)

        if subset_length is not None:
            n = min(int(subset_length), len(self._dataset))
            self._dataset = Subset(self._dataset, list(range(n)))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> ImageNetItem:
        y_img, label = self._dataset[index]
        y: torch.Tensor = y_img.to(dtype=torch.float32)

        # Source noise x ~ N(0, 1)
        x = torch.randn(self._sample_shape, generator=self._rng, dtype=torch.float32)

        # Time t ~ U[0, 1]
        t = torch.rand(1, generator=self._rng, dtype=torch.float32)

        # Interpolate and velocity
        x_t = x * (1 - t) + y * t
        v = y - x

        # Label dropout for CFG training
        if self._label_dropout > 0:
            if torch.rand(1, generator=self._rng).item() < self._label_dropout:
                label = self._null_label

        time_input = make_time_input(t.unsqueeze(0))
        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), time_input
        ).squeeze(0)

        return ImageNetItem(
            raw_source=x,
            raw_target=y,
            t=t,
            input=x_t,
            target=v,
            unified_input=unified,
            label=label,
        )


class ImageNetLatentDataset(BaseDataset):
    """ImageNet dataset in VAE latent space.

    Loads pre-computed VAE latents for efficient training.
    If latents don't exist, they are computed on-the-fly (slow).

    Args:
        root: Path to ImageNet data
        latent_root: Path to pre-computed latents (optional)
        vae: Pre-trained VAE for encoding (required if latent_root is None)
        split: "train" or "val"
        image_size: Original image size (for VAE encoding)
        latent_size: Latent spatial size (typically image_size // 8)
        seed: Random seed
        subset_length: Limit dataset size
        label_dropout: Probability of dropping label
        null_label: Label value when dropped
    """

    NUM_CLASSES = 1000

    def __init__(
        self,
        root: str = "data/imagenet",
        latent_root: str | None = None,
        vae: object | None = None,
        split: str = "train",
        image_size: int = 256,
        latent_size: int = 32,
        latent_channels: int = 4,
        seed: int = 42,
        subset_length: int | None = None,
        label_dropout: float = 0.0,
        null_label: int = 1000,
    ) -> None:
        self._root = Path(root)
        self._latent_root = Path(latent_root) if latent_root else None
        self._vae = vae
        self._split = split
        self._label_dropout = label_dropout
        self._null_label = null_label

        self._sample_shape = (latent_channels, latent_size, latent_size)
        self._rng = torch.Generator().manual_seed(seed)

        # If we have pre-computed latents, use them
        if self._latent_root is not None and self._latent_root.exists():
            self._use_precomputed = True
            self._setup_precomputed_latents()
        else:
            # Fall back to on-the-fly encoding
            self._use_precomputed = False
            if self._vae is None:
                raise ValueError(
                    "Either latent_root with pre-computed latents or vae must be provided"
                )
            self._setup_image_dataset(image_size)

        if subset_length is not None:
            n = min(int(subset_length), self._length)
            self._length = n

    def _setup_precomputed_latents(self) -> None:
        """Setup dataset for pre-computed latents."""
        split_dir = "train" if self._split == "train" else "val"
        latent_dir = self._latent_root / split_dir

        # Find all .pt files with latents
        self._latent_files = sorted(latent_dir.glob("**/*.pt"))
        self._length = len(self._latent_files)

        # Load labels file if exists
        labels_path = latent_dir / "labels.pt"
        if labels_path.exists():
            self._labels = torch.load(labels_path, weights_only=True)
        else:
            # Extract labels from directory structure
            self._labels = None

    def _setup_image_dataset(self, image_size: int) -> None:
        """Setup underlying image dataset for on-the-fly encoding."""
        split_dir = "train" if self._split == "train" else "val"

        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._image_dataset = datasets.ImageFolder(
            root=str(self._root / split_dir),
            transform=transform,
        )
        self._length = len(self._image_dataset)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> ImageNetItem:
        if self._use_precomputed:
            # Load pre-computed latent
            latent_path = self._latent_files[index]
            data = torch.load(latent_path, weights_only=True)
            y = data["latent"].to(dtype=torch.float32)
            label = int(data.get("label", 0))
        else:
            # Encode on-the-fly
            img, label = self._image_dataset[index]
            img = img.unsqueeze(0)  # Add batch dim

            # Encode with VAE
            with torch.no_grad():
                y = self._vae.encode(img).squeeze(0)

        # Source noise in latent space
        x = torch.randn(self._sample_shape, generator=self._rng, dtype=torch.float32)

        # Time
        t = torch.rand(1, generator=self._rng, dtype=torch.float32)

        # Interpolate and velocity
        x_t = x * (1 - t) + y * t
        v = y - x

        # Label dropout
        if self._label_dropout > 0:
            if torch.rand(1, generator=self._rng).item() < self._label_dropout:
                label = self._null_label

        time_input = make_time_input(t.unsqueeze(0))
        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), time_input
        ).squeeze(0)

        return ImageNetItem(
            raw_source=x,
            raw_target=y,
            t=t,
            input=x_t,
            target=v,
            unified_input=unified,
            label=label,
        )
