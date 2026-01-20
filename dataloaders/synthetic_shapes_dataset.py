"""Simple synthetic dataset for overfit smoke tests.

Generates geometric shapes (circles, squares) on solid backgrounds.
No I/O required - everything is generated in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from dataloaders.base_dataloaders import (
    BaseDataset,
    BaseItem,
    make_unified_flow_matching_input,
)


@dataclass
class SyntheticShapesItem(BaseItem):
    raw_source: torch.Tensor
    raw_target: torch.Tensor


class SyntheticShapesDataset(Dataset):
    """Synthetic dataset generating colored shapes on solid backgrounds.

    Each sample is deterministically generated from its index and the seed.
    Returns images normalized to [-1, 1] range (same as CelebA).
    """

    def __init__(
        self,
        num_samples: int = 16,
        image_size: int = 64,
        num_channels: int = 3,
        seed: int = 42,
        mode: str = "shapes",  # "shapes" or "solid" for solid colors only
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.seed = seed
        self.mode = mode

        # Pre-generate all images for consistency and speed
        self._images = self._generate_all()

    def _generate_all(self) -> torch.Tensor:
        """Generate all images upfront."""
        gen = torch.Generator().manual_seed(self.seed)
        images = []

        for _ in range(self.num_samples):
            img = self._generate_one(gen)
            images.append(img)

        return torch.stack(images, dim=0)

    def _generate_one(self, gen: torch.Generator) -> torch.Tensor:
        """Generate a single image with a random shape."""
        H = W = self.image_size
        C = self.num_channels

        # Random background color in [-1, 1]
        bg_color = torch.rand(C, 1, 1, generator=gen) * 2 - 1
        img = bg_color.expand(C, H, W).clone()

        if self.mode == "solid":
            return img

        # Random foreground color
        fg_color = torch.rand(C, generator=gen) * 2 - 1

        # Random shape type: 0=circle, 1=square, 2=horizontal stripe, 3=vertical stripe
        shape_type = torch.randint(0, 4, (1,), generator=gen).item()

        # Random position and size
        cx = torch.rand(1, generator=gen).item() * 0.6 + 0.2  # center x in [0.2, 0.8]
        cy = torch.rand(1, generator=gen).item() * 0.6 + 0.2  # center y in [0.2, 0.8]
        size = torch.rand(1, generator=gen).item() * 0.3 + 0.15  # size in [0.15, 0.45]

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H).view(1, H, 1).expand(1, H, W)
        x_coords = torch.linspace(0, 1, W).view(1, 1, W).expand(1, H, W)

        if shape_type == 0:  # Circle
            dist = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2).sqrt()
            mask = (dist < size).float()
        elif shape_type == 1:  # Square
            mask_x = (x_coords - cx).abs() < size
            mask_y = (y_coords - cy).abs() < size
            mask = (mask_x & mask_y).float()
        elif shape_type == 2:  # Horizontal stripe
            mask = ((y_coords - cy).abs() < size * 0.3).float()
        else:  # Vertical stripe
            mask = ((x_coords - cx).abs() < size * 0.3).float()

        # Apply foreground color where mask is 1
        mask = mask.expand(C, H, W)
        img = img * (1 - mask) + fg_color.view(C, 1, 1).expand(C, H, W) * mask

        return img

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._images[index].clone()


class SyntheticShapesFlowDataset(BaseDataset):
    """Flow matching dataset using synthetic shapes as targets.

    For each shape image y (target), samples Gaussian noise x (source),
    time t ~ Uniform[0,1], and returns:
      - input: x_t = (1 - t) * x + t * y
      - target: v = y - x (velocity)
      - unified_input: input with time channel appended
    """

    def __init__(
        self,
        num_samples: int = 16,
        image_size: int = 64,
        num_channels: int = 3,
        seed: int = 42,
    ) -> None:
        self._base = SyntheticShapesDataset(
            num_samples=num_samples,
            image_size=image_size,
            num_channels=num_channels,
            seed=seed,
        )
        self._rng = torch.Generator().manual_seed(seed + 1000)
        self._sample_shape = (num_channels, image_size, image_size)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> BaseItem:
        y = self._base[index]  # target image

        # Source noise x ~ N(0, 1)
        x = torch.randn(self._sample_shape, generator=self._rng, dtype=torch.float32)

        # Time t ~ U[0, 1]
        t = torch.rand(1, generator=self._rng, dtype=torch.float32)

        # Interpolate: x_t = (1 - t) * x + t * y
        x_t = x * (1 - t) + y * t

        # Velocity: v = y - x
        v = y - x

        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), t.unsqueeze(0)
        ).squeeze(0)

        return SyntheticShapesItem(
            input=x_t, t=t, target=v, unified_input=unified, raw_source=x, raw_target=y
        )
