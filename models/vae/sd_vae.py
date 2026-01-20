"""Stable Diffusion VAE wrapper for loading pre-trained models.

Uses HuggingFace diffusers to load pre-trained SD VAE (kl-f8 architecture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


@dataclass
class SDVAEOutput:
    """Output from SD VAE encode/decode."""
    latent: torch.Tensor
    reconstruction: torch.Tensor | None = None


class SDVAE(nn.Module):
    """Wrapper around pre-trained Stable Diffusion VAE.

    The SD VAE (kl-f8) has:
    - 8x spatial downsampling: 256x256 -> 32x32
    - 4 latent channels
    - Latent scale factor of 0.18215 (for SD 1.x/2.x)

    Args:
        model_id: HuggingFace model ID (default: "stabilityai/sd-vae-ft-mse")
        device: Device to load model on
        dtype: Data type for model (default: float32)
        freeze: Whether to freeze VAE weights (default: True for inference)
        scaling_factor: Latent scaling factor (default: 0.18215 for SD)
    """

    # Common model IDs
    SD_VAE_FT_MSE = "stabilityai/sd-vae-ft-mse"
    SD_VAE_FT_EMA = "stabilityai/sd-vae-ft-ema"
    SDXL_VAE = "stabilityai/sdxl-vae"

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__()

        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required to load SD VAE. "
                "Install with: pip install diffusers"
            )

        self.model_id = model_id
        self.scaling_factor = scaling_factor
        self._device = torch.device(device)
        self._dtype = dtype

        # Load pre-trained VAE
        print(f"Loading SD VAE from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.vae = self.vae.to(self._device)

        if freeze:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

        # Cache latent dimensions
        self.latent_channels = self.vae.config.latent_channels
        self.spatial_scale = 2 ** (len(self.vae.config.down_block_types))

        print(f"SD VAE loaded: {self.latent_channels} latent channels, "
              f"{self.spatial_scale}x downsampling")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_latent_shape(
        self, image_shape: tuple[int, int]
    ) -> tuple[int, int, int]:
        """Get latent shape for a given image size.

        Args:
            image_shape: (H, W) of input image

        Returns:
            (C, H_latent, W_latent) latent shape
        """
        H, W = image_shape
        return (
            self.latent_channels,
            H // self.spatial_scale,
            W // self.spatial_scale,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents.

        Args:
            x: Images of shape (B, 3, H, W) in [-1, 1]

        Returns:
            Latents of shape (B, 4, H//8, W//8) scaled by scaling_factor
        """
        x = x.to(device=self._device, dtype=self._dtype)

        # Encode to latent distribution
        posterior = self.vae.encode(x).latent_dist

        # Sample from posterior (deterministic mode for inference)
        latent = posterior.mode()

        # Apply scaling factor
        latent = latent * self.scaling_factor

        return latent

    @torch.no_grad()
    def encode_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents with sampling (for training).

        Args:
            x: Images of shape (B, 3, H, W) in [-1, 1]

        Returns:
            Latents of shape (B, 4, H//8, W//8) scaled by scaling_factor
        """
        x = x.to(device=self._device, dtype=self._dtype)

        posterior = self.vae.encode(x).latent_dist
        latent = posterior.sample()
        latent = latent * self.scaling_factor

        return latent

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            z: Latents of shape (B, 4, H_latent, W_latent) (already scaled)

        Returns:
            Images of shape (B, 3, H, W) in [-1, 1]
        """
        z = z.to(device=self._device, dtype=self._dtype)

        # Undo scaling factor
        z = z / self.scaling_factor

        # Decode
        image = self.vae.decode(z).sample

        return image

    def forward(
        self,
        x: torch.Tensor,
        return_recon: bool = False,
    ) -> SDVAEOutput:
        """Encode and optionally decode.

        Args:
            x: Images of shape (B, 3, H, W) in [-1, 1]
            return_recon: Whether to also return reconstruction

        Returns:
            SDVAEOutput with latent and optional reconstruction
        """
        latent = self.encode(x)

        recon = None
        if return_recon:
            recon = self.decode(latent)

        return SDVAEOutput(latent=latent, reconstruction=recon)


def load_sd_vae(
    model_id: str = "stabilityai/sd-vae-ft-mse",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SDVAE:
    """Convenience function to load pre-trained SD VAE.

    Args:
        model_id: HuggingFace model ID
        device: Device to load on
        dtype: Data type

    Returns:
        SDVAE instance ready for use
    """
    return SDVAE(model_id=model_id, device=device, dtype=dtype, freeze=True)
