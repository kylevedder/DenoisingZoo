"""UNet architecture for diffusion/flow matching models.

Based on ADM (Dhariwal & Nichol, 2021) / Improved Diffusion (Nichol & Dhariwal, 2021).
~55M parameters with default settings for CIFAR-10.
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


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal position embeddings.

    Args:
        t: Time values of shape (B,) or (B, 1)
        dim: Embedding dimension (must be even)

    Returns:
        Embeddings of shape (B, dim)
    """
    if t.dim() == 2:
        t = t.squeeze(-1)

    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb_scale)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 for stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class Upsample(nn.Module):
    """2x nearest upsampling followed by conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """2x strided conv downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with time conditioning.

    Uses FiLM-style conditioning: out = (scale * out) + shift
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time embedding projection to scale and shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )

        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time conditioning
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with residual."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)

        # Scaled dot-product attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 2, 3)  # (B, heads, head_dim, HW)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


class UNet(TimeChannelModule):
    """UNet for diffusion/flow matching models.

    Architecture following ADM/Improved Diffusion:
    - Encoder path with residual blocks and downsampling
    - Middle block with attention
    - Decoder path with residual blocks and upsampling
    - Skip connections between encoder and decoder

    Default configuration yields ~55M parameters for CIFAR-10.

    Args:
        in_channels: Input image channels (default: 3)
        out_channels: Output channels (default: same as in_channels)
        base_channels: Base channel count (default: 128)
        channel_mult: Channel multipliers per resolution (default: [1, 2, 2, 2])
        num_res_blocks: Number of ResBlocks per resolution (default: 2)
        attention_resolutions: Resolutions at which to apply attention (default: [16])
        dropout: Dropout rate (default: 0.1)
        num_heads: Number of attention heads (default: 4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        base_channels: int = 128,
        time_channels: int = TIME_CHANNELS_REQUIRED,
        channel_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        dropout: float = 0.1,
        num_heads: int = 4,
        input_resolution: int = 32,
        use_separate_time_embeds: bool = True,
    ) -> None:
        super().__init__(time_channels)

        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_channels = time_channels
        self.channel_mult = list(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.input_resolution = input_resolution
        self.use_separate_time_embeds = use_separate_time_embeds

        time_emb_dim = base_channels * 4

        if use_separate_time_embeds:
            # Two separate time embedding pathways (MeanFlow official style)
            # Pathway 1: embed t (end time)
            self.time_embed_t = nn.Sequential(
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            # Pathway 2: embed (t - r) (duration)
            self.time_embed_duration = nn.Sequential(
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            # Projection from concatenated embeddings back to time_emb_dim
            self.time_embed_proj = nn.Linear(time_emb_dim * 2, time_emb_dim)
        else:
            # Legacy: single time embedding MLP that sums sinusoidal embeddings
            self.time_embed = nn.Sequential(
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        # Input projection (accounts for time channel in unified input)
        self.input_conv = nn.Conv2d(
            in_channels + time_channels, base_channels, 3, padding=1
        )

        # Track channels at each skip connection point
        self._skip_channels: list[int] = []

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        current_res = input_resolution

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            # Residual blocks for this level
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = ResBlock(ch, out_ch, time_emb_dim, dropout)
                level_blocks.append(block)
                ch = out_ch
                self._skip_channels.append(ch)

                # Add attention if at attention resolution
                if current_res in self.attention_resolutions:
                    level_blocks.append(SelfAttention(ch, num_heads))

            self.down_blocks.append(level_blocks)

            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                # Save skip channel BEFORE downsampling (for resolution matching)
                self._skip_channels.append(ch)
                self.downsamplers.append(Downsample(ch))
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block
        self.middle_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.middle_attn = SelfAttention(ch, num_heads)
        self.middle_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            out_ch = base_channels * mult

            # Residual blocks for this level
            # - If coming from a level that had downsampling, we have one extra skip
            level_blocks = nn.ModuleList()

            # Determine how many res blocks this decoder level needs
            # Last encoder level (level = len-1) has no downsample, so num_res_blocks skips
            # Other levels have num_res_blocks + 1 skips (including downsample)
            if level == len(channel_mult) - 1:
                num_blocks = num_res_blocks
            else:
                num_blocks = num_res_blocks + 1

            # First, handle upsampling from previous level (except for first decoder level)
            if level < len(channel_mult) - 1:
                self.upsamplers.append(Upsample(ch))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

            # Process res blocks (with skip connections)
            for i in range(num_blocks):
                skip_ch = self._skip_channels.pop()
                block = ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                level_blocks.append(block)
                ch = out_ch

                # Add attention if at attention resolution
                if current_res in self.attention_resolutions:
                    level_blocks.append(SelfAttention(ch, num_heads))

            self.up_blocks.append(level_blocks)

        # Output projection
        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Initialize output conv to zero for better training stability
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # Store skip channels for forward pass (rebuilt during forward)
        self._num_skip_per_level = [
            num_res_blocks + (1 if i < len(channel_mult) - 1 else 0)
            for i in range(len(channel_mult))
        ]

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            unified_input: Input tensor of shape (B, C+T, H, W) where
                the last T channels are time inputs.

        Returns:
            Velocity field of shape (B, C, H, W)
        """
        if unified_input.dim() != 4:
            raise ValueError(
                f"unified_input must be rank-4 (B, C, H, W), got shape {tuple(unified_input.shape)}"
            )

        # Extract time from unified input
        ununified = make_ununified_flow_matching_input(
            unified_input, num_time_channels=self.time_channels
        )
        t = ununified.t  # (B, T)

        # Time embedding
        t_emb = self._build_time_embedding(t)

        # Initial convolution (on full unified input including time channel)
        h = self.input_conv(unified_input)

        # Encoder - collect skip connections
        skips: list[torch.Tensor] = []
        for level, (blocks, downsampler) in enumerate(
            zip(self.down_blocks, self.downsamplers)
        ):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    skips.append(h)
                else:  # SelfAttention
                    h = block(h)

            if not isinstance(downsampler, nn.Identity):
                # Save skip BEFORE downsampling for resolution matching
                skips.append(h)
                h = downsampler(h)

        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        # Decoder - consume skip connections in reverse
        for level, (upsampler, blocks) in enumerate(
            zip(self.upsamplers, self.up_blocks)
        ):
            if not isinstance(upsampler, nn.Identity):
                h = upsampler(h)

            for block in blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                else:  # SelfAttention
                    h = block(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h

    def _build_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Build time embedding from time tensor.

        Args:
            t: Time tensor of shape (B, 2) where t[:, 0] is r (start time)
               and t[:, 1] is t (end time).

        Returns:
            Time embedding of shape (B, time_emb_dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        if self.use_separate_time_embeds:
            # MeanFlow official style: separate pathways for t and (t - r)
            if t.shape[1] != 2:
                raise ValueError(
                    f"use_separate_time_embeds=True requires time input of shape (B, 2), "
                    f"got shape {tuple(t.shape)}"
                )
            r = t[:, 0]  # start time
            t_end = t[:, 1]  # end time
            duration = t_end - r  # duration = t - r

            # Embed t (end time) through first pathway
            emb_t = sinusoidal_embedding(t_end, self.base_channels)
            emb_t = self.time_embed_t(emb_t)

            # Embed (t - r) (duration) through second pathway
            emb_duration = sinusoidal_embedding(duration, self.base_channels)
            emb_duration = self.time_embed_duration(emb_duration)

            # Concatenate and project back to time_emb_dim
            emb_concat = torch.cat([emb_t, emb_duration], dim=1)
            return self.time_embed_proj(emb_concat)
        else:
            # Legacy: sum sinusoidal embeddings and pass through single MLP
            if t.shape[1] == 1:
                base = sinusoidal_embedding(t, self.base_channels)
            else:
                base = None
                for idx in range(t.shape[1]):
                    emb = sinusoidal_embedding(t[:, idx], self.base_channels)
                    base = emb if base is None else base + emb
            return self.time_embed(base)
