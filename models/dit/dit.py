"""Diffusion Transformer (DiT) for image generation.

Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
Implements DiT-S, DiT-B, DiT-L, DiT-XL variants.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from constants import TIME_CHANNELS_REQUIRED
from model_contracts import TimeChannelModule

from dataloaders.base_dataloaders import make_ununified_flow_matching_input


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embed class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        use_cfg = dropout_prob > 0
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # Extra embedding for null/dropped class
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg), hidden_size
        )

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Drop labels for CFG training."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero).

    Conditioning on both time and class through modulation parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim)

        # adaLN-Zero: 6 modulation parameters per block
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Get modulation parameters from conditioning
        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )

        # Attention with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 2 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(TimeChannelModule):
    """Diffusion Transformer for flow matching / diffusion models.

    Args:
        input_size: Spatial size of input (e.g., 32 for 32x32 latents)
        patch_size: Size of each patch (default: 2)
        in_channels: Number of input channels (default: 4 for VAE latents)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim multiplier
        num_classes: Number of classes for conditioning (default: 1000 for ImageNet)
        learn_sigma: Whether to predict variance (not used for flow matching)
        class_dropout_prob: Label dropout for CFG training
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        time_channels: int = TIME_CHANNELS_REQUIRED,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__(time_channels)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.time_channels = time_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Patch embedding
        self.x_embedder = nn.Conv2d(
            in_channels + time_channels,  # +T for time channels from unified input
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.num_patches = (input_size // patch_size) ** 2

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # Time and label embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights following DiT paper."""
        # Initialize transformer layers
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch embed like linear
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.x_embedder.bias)

        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-out final layer
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image.

        Args:
            x: (B, num_patches, patch_size^2 * out_channels)

        Returns:
            (B, out_channels, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(
        self,
        unified_input: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            unified_input: Input tensor (B, C+T, H, W) with time channels
            y: Class labels (B,) - optional, if None uses null class

        Returns:
            Velocity field (B, C, H, W)
        """
        # Extract time from unified input
        ununified = make_ununified_flow_matching_input(
            unified_input, num_time_channels=self.time_channels
        )
        t = ununified.t  # (B, T)

        B = unified_input.shape[0]

        # Patch embedding (keeps time channel in input)
        x = self.x_embedder(unified_input)  # (B, hidden_size, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)

        # Add positional embedding
        x = x + self.pos_embed

        # Get time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[1] == 1:
            t_emb = self.t_embedder(t.squeeze(-1))
        else:
            t_emb = None
            for idx in range(t.shape[1]):
                emb = self.t_embedder(t[:, idx])
                t_emb = emb if t_emb is None else t_emb + emb

        # Get label embedding
        if y is None:
            # Use null class for unconditional generation
            y = torch.full((B,), self.y_embedder.num_classes, device=x.device)
        y_emb = self.y_embedder(y, self.training)  # (B, hidden_size)

        # Combined conditioning
        c = t_emb + y_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify
        x = self.unpatchify(x)

        return x


# Model configurations (following DiT paper)
def DiT_S(**kwargs) -> DiT:
    """DiT-S: ~33M parameters"""
    return DiT(hidden_size=384, depth=12, num_heads=6, **kwargs)


def DiT_B(**kwargs) -> DiT:
    """DiT-B: ~131M parameters"""
    return DiT(hidden_size=768, depth=12, num_heads=12, **kwargs)


def DiT_L(**kwargs) -> DiT:
    """DiT-L: ~459M parameters"""
    return DiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)


def DiT_XL(**kwargs) -> DiT:
    """DiT-XL: ~676M parameters"""
    return DiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
