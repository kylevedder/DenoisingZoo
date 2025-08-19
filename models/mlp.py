from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """
    Simple MLP for flow matching.

    - Expects `x` with shape (batch_size, feature_dim)
    - Expects `t` with shape (batch_size, 1)
    - Returns a tensor with shape (batch_size, feature_dim)

    The model concatenates `t` to `x` along the feature dimension and passes
    the result through exactly two hidden layers with ReLU activations.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = feature_dim + 1  # concatenate time `t`
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, feature_dim)
            t: Tensor of shape (batch_size, 1)

        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        if x.dim() != 2:
            raise ValueError(f"x must be rank-2 (B, n), got shape {tuple(x.shape)}")
        if t.dim() != 2:
            raise ValueError(f"t must be rank-2 (B, 1), got shape {tuple(t.shape)}")
        if t.shape[1] != 1:
            raise ValueError(f"t's feature dimension must be 1, got {t.shape[1]}")
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x and t: {x.shape[0]} vs {t.shape[0]}"
            )

        t = t.to(dtype=x.dtype, device=x.device)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)
