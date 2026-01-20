"""Deterministic synthetic dataset for overfitting tests.

Creates k fixed images (one per class) with deterministic noise sources,
allowing loss to converge to zero when model memorizes the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from dataloaders.base_dataloaders import (
    BaseDataset,
    BaseItem,
    make_time_input,
    make_unified_flow_matching_input,
)


@dataclass
class DeterministicItem(BaseItem):
    """Item with class label and deterministic source/target."""
    raw_source: torch.Tensor
    raw_target: torch.Tensor
    label: int


class DeterministicFlowDataset(BaseDataset):
    """Deterministic dataset for flow matching overfit tests.

    Creates k fixed target images and k fixed source patterns.
    For each sample:
      - Picks a class k deterministically based on index
      - Uses fixed source x_k and target y_k for that class
      - Samples time t deterministically from a fixed set

    This allows the model to perfectly memorize the velocity field,
    driving training loss to zero.

    Args:
        num_classes: Number of distinct classes/images (default: 4)
        samples_per_class: Number of time samples per class (default: 10)
        image_size: Spatial size of images (default: 32)
        num_channels: Number of channels (default: 3)
        seed: Random seed for generating fixed images
        use_zero_source: If True, source is all zeros (simplest case)
    """

    def __init__(
        self,
        num_classes: int = 4,
        samples_per_class: int = 10,
        image_size: int = 32,
        num_channels: int = 3,
        seed: int = 42,
        use_zero_source: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.num_channels = num_channels
        self.use_zero_source = use_zero_source

        gen = torch.Generator().manual_seed(seed)

        # Generate k fixed target images (distinct patterns)
        self._targets = self._generate_targets(gen)

        # Generate k fixed source images (noise or zeros)
        if use_zero_source:
            self._sources = torch.zeros(
                num_classes, num_channels, image_size, image_size
            )
        else:
            self._sources = self._generate_sources(gen)

        # Pre-compute velocities v_k = y_k - x_k
        self._velocities = self._targets - self._sources

        # Generate fixed time values (uniform grid)
        self._times = torch.linspace(0.01, 0.99, samples_per_class)

        self._sample_shape = (num_channels, image_size, image_size)

    def _generate_targets(self, gen: torch.Generator) -> torch.Tensor:
        """Generate k distinct target images."""
        targets = []
        H = W = self.image_size
        C = self.num_channels

        for k in range(self.num_classes):
            # Each class gets a unique colored pattern
            img = torch.zeros(C, H, W)

            # Use different patterns for each class
            if k % 4 == 0:
                # Solid color
                color = torch.rand(C, generator=gen) * 2 - 1
                img = color.view(C, 1, 1).expand(C, H, W).clone()
            elif k % 4 == 1:
                # Horizontal gradient
                color = torch.rand(C, generator=gen) * 2 - 1
                gradient = torch.linspace(-1, 1, W).view(1, 1, W).expand(C, H, W)
                img = gradient * color.view(C, 1, 1)
            elif k % 4 == 2:
                # Vertical gradient
                color = torch.rand(C, generator=gen) * 2 - 1
                gradient = torch.linspace(-1, 1, H).view(1, H, 1).expand(C, H, W)
                img = gradient * color.view(C, 1, 1)
            else:
                # Checkerboard pattern
                color1 = torch.rand(C, generator=gen) * 2 - 1
                color2 = torch.rand(C, generator=gen) * 2 - 1
                check_size = max(4, H // 4)
                for i in range(0, H, check_size):
                    for j in range(0, W, check_size):
                        color = color1 if ((i // check_size) + (j // check_size)) % 2 == 0 else color2
                        img[:, i:i+check_size, j:j+check_size] = color.view(C, 1, 1)

            targets.append(img)

        return torch.stack(targets, dim=0)

    def _generate_sources(self, gen: torch.Generator) -> torch.Tensor:
        """Generate k fixed source images (noise patterns)."""
        sources = []
        for k in range(self.num_classes):
            # Each class gets a fixed noise pattern
            noise = torch.randn(
                self.num_channels, self.image_size, self.image_size,
                generator=gen
            )
            sources.append(noise)
        return torch.stack(sources, dim=0)

    def __len__(self) -> int:
        return self.num_classes * self.samples_per_class

    def __getitem__(self, index: int) -> DeterministicItem:
        # Determine class and time index
        class_idx = index // self.samples_per_class
        time_idx = index % self.samples_per_class

        # Get fixed source, target, velocity for this class
        x = self._sources[class_idx].clone()
        y = self._targets[class_idx].clone()
        v = self._velocities[class_idx].clone()

        # Get fixed time for this sample
        t = self._times[time_idx:time_idx+1].clone()

        # Compute interpolated state
        x_t = x * (1 - t) + y * t

        # Build unified input
        time_input = make_time_input(t.unsqueeze(0))
        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), time_input
        ).squeeze(0)

        return DeterministicItem(
            input=x_t,
            t=t,
            target=v,
            unified_input=unified,
            raw_source=x,
            raw_target=y,
            label=class_idx,
        )


class SingleImageDataset(BaseDataset):
    """Simplest possible overfit test: single fixed image.

    Source is zeros, target is a fixed image.
    v = y - 0 = y (constant velocity)
    x_t = t * y (linear interpolation from origin)

    The model should perfectly memorize: given (t*y, t), predict y.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 32,
        num_channels: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels

        gen = torch.Generator().manual_seed(seed)

        # Generate one fixed target image (colorful gradient)
        H = W = image_size
        C = num_channels

        # Create a distinctive pattern
        y_coord = torch.linspace(-1, 1, H).view(1, H, 1).expand(C, H, W)
        x_coord = torch.linspace(-1, 1, W).view(1, 1, W).expand(C, H, W)

        # Different pattern per channel
        self._target = torch.zeros(C, H, W)
        self._target[0] = y_coord[0]  # Red: vertical gradient
        if C > 1:
            self._target[1] = x_coord[0]  # Green: horizontal gradient
        if C > 2:
            self._target[2] = (y_coord[0] + x_coord[0]) / 2  # Blue: diagonal

        # Source is zeros
        self._source = torch.zeros(C, H, W)

        # Velocity is just the target
        self._velocity = self._target.clone()

        # Fixed time values (uniform grid)
        self._times = torch.linspace(0.01, 0.99, num_samples)

        self._sample_shape = (C, H, W)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> DeterministicItem:
        x = self._source.clone()
        y = self._target.clone()
        v = self._velocity.clone()

        t = self._times[index:index+1].clone()

        # x_t = t * y (since x = 0)
        x_t = y * t

        time_input = make_time_input(t.unsqueeze(0))
        unified = make_unified_flow_matching_input(
            x_t.unsqueeze(0), time_input
        ).squeeze(0)

        return DeterministicItem(
            input=x_t,
            t=t,
            target=v,
            unified_input=unified,
            raw_source=x,
            raw_target=y,
            label=0,
        )
