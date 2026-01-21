from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Lightweight CNN for image-based flow matching.

    Expects:
      - x: (batch, channels, height, width)
      - time inputs: two channels appended to the image

    Conditioning on time is performed by concatenating a broadcast time channel
    to the image input. The network predicts a per-pixel velocity with the same
    spatial shape as the input (channels, height, width).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        base_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        time_channels: int = 2,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if time_channels != 2:
            raise ValueError("time_channels must be 2")

        padding = kernel_size // 2
        layers: list[nn.Module] = []

        self.time_channels = time_channels

        # First layer takes time channels as extra input
        current_in = in_channels + time_channels
        current_out = base_channels
        layers.append(nn.Conv2d(current_in, current_out, kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(current_out, current_out, kernel_size, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))

        # Final projection to velocity field
        layers.append(
            nn.Conv2d(current_out, out_channels, kernel_size, padding=padding)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        if unified_input.dim() != 4:
            raise ValueError(
                f"unified_input must be rank-4 (B, C, H, W), got shape {tuple(unified_input.shape)}"
            )
        return self.net(unified_input)
