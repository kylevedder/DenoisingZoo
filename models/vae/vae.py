"""KL-regularized VAE following the Stable Diffusion architecture style.

This is a simpler implementation suitable for experimentation and overfit testing.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """VAE Encoder with ResNet blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(channels, out_channels))
                channels = out_channels
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(channels))

        # Middle
        self.mid_block1 = ResBlock(channels, channels)
        self.mid_block2 = ResBlock(channels, channels)

        # Output: mu and logvar
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 2 * latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder with ResNet blocks."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()

        # Start from the deepest level
        channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)

        # Middle
        self.mid_block1 = ResBlock(channels, channels)
        self.mid_block2 = ResBlock(channels, channels)

        # Upsampling blocks (reverse order)
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock(channels, out_ch))
                channels = out_ch
            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(Upsample(channels))

        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)  # Output in [-1, 1]


class VAE(nn.Module):
    """KL-regularized Variational Autoencoder.

    Architecture follows Stable Diffusion's AutoencoderKL style.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(reconstruction=recon, mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using mean of latent distribution."""
        mu, _ = self.encode(x)
        return self.decode(mu)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between N(mu, sigma) and N(0, 1)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
