from __future__ import annotations

import torch
from torch import nn

from constants import TIME_CHANNELS_REQUIRED
from model_contracts import TimeChannelModule


class MLP(TimeChannelModule):
    """
    Simple MLP for flow matching.

    - Expects `x` with shape (batch_size, feature_dim)
    - Expects unified input with two time channels (batch_size, feature_dim + 2)
    - Returns a tensor with shape (batch_size, feature_dim)

    The model concatenates time inputs to `x` along the feature dimension and passes
    the result through exactly two hidden layers with ReLU activations.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        time_channels: int = TIME_CHANNELS_REQUIRED,
    ) -> None:
        super().__init__(time_channels)
        input_dim = feature_dim + time_channels  # concatenate time inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, unified_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with unified input of shape (B, feature_dim + 2)."""
        if unified_input.dim() != 2:
            raise ValueError(
                f"unified_input must be rank-2 (B, n), got shape {tuple(unified_input.shape)}"
            )
        return self.net(unified_input)
