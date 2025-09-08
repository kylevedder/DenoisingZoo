from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

# Import the forward result structure produced by VQVAE
from models.vqvae.vqvae import ForwardResult


class VQVAETrainingLoss(nn.Module):
    """Criterion compatible with train.py for VQ-VAE training.

    Expects the model to return a ForwardResult; computes:
      total = recon_weight * L(reconstruction, input_image) + vq_weight * vq_loss
    The target tensor provided by train.py is ignored (kept for API compatibility).
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        base_loss: str = "l1",
    ) -> None:
        super().__init__()
        self._recon_weight = float(recon_weight)
        self._vq_weight = float(vq_weight)
        if base_loss.lower() == "l1":
            self._recon_crit: nn.Module = nn.L1Loss()
        elif base_loss.lower() in {"l2", "mse"}:
            self._recon_crit = nn.MSELoss()
        else:
            raise ValueError("base_loss must be one of {'l1','l2','mse'}")

    def forward(
        self, pred: ForwardResult | torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        # If pred is a tensor (non-VQVAE models), fall back to standard loss against target
        if isinstance(pred, torch.Tensor):
            if target is None:
                raise ValueError("Target tensor must be provided when pred is a Tensor")
            return self._recon_crit(pred, target)

        # VQVAE path: use reconstruction vs input_image and add codebook/commitment losses
        recon_loss = self._recon_crit(pred.reconstruction, pred.input_image)
        total = self._recon_weight * recon_loss + self._vq_weight * pred.quant.vq_loss
        return total
