from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class TrainResult:
    reconstruction: torch.Tensor
    reconstruction_loss: torch.Tensor
    vq_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass
class QuantizeResult:
    z_q: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor


@dataclass
class ForwardResult:
    input_image: torch.Tensor
    encoder_out: torch.Tensor
    pre_vq_out: torch.Tensor
    quant: QuantizeResult
    post_vq_out: torch.Tensor
    reconstruction: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hc = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, hc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, hc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, hc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hc = hidden_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hc, hc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, hc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hc, hc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # assume images normalized to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Straight-through vector quantization layer.

    Codebook contains `num_embeddings` vectors of dimension `embedding_dim`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)

        embedding = torch.randn(self.num_embeddings, self.embedding_dim)
        self.codebook = nn.Parameter(embedding)

    def forward(self, z_e: torch.Tensor) -> QuantizeResult:
        """Quantize latents.

        Args:
            z_e: encoder latents of shape (B, D, H, W) where D == embedding_dim

        Returns:
            z_q: quantized latents (B, D, H, W)
            codebook_indices: (B, H, W) long indices into the codebook
            vq_loss: scalar loss to add (codebook + commitment)
        """
        if z_e.dim() != 4 or z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected z_e shape (B, {self.embedding_dim}, H, W), got {tuple(z_e.shape)}"
            )

        B, D, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (BHW, D)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )  # (BHW, K)
        codes = torch.argmin(distances, dim=1)  # (BHW,)

        z_q_flat = self.codebook[codes]  # (BHW, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Loss terms
        codebook_loss = torch.mean((z_q.detach() - z_e) ** 2)
        commitment_loss = torch.mean((z_q - z_e.detach()) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        indices = codes.view(B, H, W)
        return QuantizeResult(z_q=z_q_st, indices=indices, vq_loss=vq_loss)


class VQVAE(nn.Module):
    """VQ-VAE model compatible with unified_input API.

    Expects `unified_input` with shape (B, C[, +1], H, W). If an extra channel is
    present (e.g., time), the first `in_channels` channels are used as the image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embedding_dim = int(embedding_dim)

        self.encoder = Encoder(
            in_channels=self.in_channels, hidden_channels=self.hidden_channels
        )
        self.pre_vq = nn.Conv2d(self.hidden_channels, self.embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.post_vq = nn.Conv2d(
            self.embedding_dim, self.hidden_channels, kernel_size=1
        )
        self.decoder = Decoder(
            out_channels=self.in_channels, hidden_channels=self.hidden_channels
        )

        # For reconstruction loss outside the model
        self.reconstruction_criterion = nn.L1Loss()

    def _extract_image_from_unified(self, unified_input: torch.Tensor) -> torch.Tensor:
        if unified_input.dim() != 4:
            raise ValueError(
                f"unified_input must be rank-4 (B, C, H, W), got shape {tuple(unified_input.shape)}"
            )
        if unified_input.shape[1] < self.in_channels:
            raise ValueError(
                f"unified_input has {unified_input.shape[1]} channels; expected at least {self.in_channels}"
            )
        if unified_input.shape[1] == self.in_channels:
            return unified_input
        return unified_input[:, : self.in_channels, ...]

    def forward(self, unified_input: torch.Tensor) -> ForwardResult:
        x = self._extract_image_from_unified(unified_input)
        enc = self.encoder(x)
        z_e = self.pre_vq(enc)
        q = self.quantizer(z_e)
        z_q = q.z_q
        h = self.post_vq(z_q)
        recon = self.decoder(h)
        return ForwardResult(
            input_image=x,
            encoder_out=enc,
            pre_vq_out=z_e,
            quant=q,
            post_vq_out=h,
            reconstruction=recon,
        )

    @torch.no_grad()
    def reconstruct(self, unified_input: torch.Tensor) -> torch.Tensor:
        return self.forward(unified_input).reconstruction

    def forward_with_losses(self, unified_input: torch.Tensor) -> TrainResult:
        fr = self.forward(unified_input)
        recon_loss = self.reconstruction_criterion(fr.reconstruction, fr.input_image)
        total_loss = recon_loss + fr.quant.vq_loss
        return TrainResult(
            reconstruction=fr.reconstruction,
            reconstruction_loss=recon_loss,
            vq_loss=fr.quant.vq_loss,
            total_loss=total_loss,
        )
