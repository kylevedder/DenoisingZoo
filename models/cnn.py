from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Lightweight CNN for image-based flow matching.

    Expects:
      - x: (batch, channels, height, width)
      - t: (batch, 1)

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
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        padding = kernel_size // 2
        layers: list[nn.Module] = []

        # First layer takes time channel as an extra input
        current_in = in_channels + 1
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"x must be rank-4 (B, C, H, W), got shape {tuple(x.shape)}"
            )
        if t.dim() != 2 or t.shape[1] != 1:
            raise ValueError(f"t must be rank-2 (B, 1), got shape {tuple(t.shape)}")
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x and t: {x.shape[0]} vs {t.shape[0]}"
            )

        t_map = t.to(dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
        t_map = t_map.expand(-1, 1, x.shape[2], x.shape[3])
        xt = torch.cat([x, t_map], dim=1)
        return self.net(xt)
