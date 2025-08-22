from __future__ import annotations

import torch


def pdist_compat(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Compatibility wrapper for torch.pdist with an MPS-friendly fallback.

    On MPS devices, torch.pdist is not implemented. This function computes the
    condensed upper-triangular distances using torch.cdist (which is supported)
    and returns a 1D tensor matching the values (order may differ, but most
    downstream metrics only use reductions like mean).
    """
    if x.device.type == "mps":
        n = x.shape[0]
        if n <= 1:
            return torch.empty(0, dtype=x.dtype, device=x.device)
        full = torch.cdist(x, x, p=p)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=x.device)
        return full[idx_i, idx_j]
    # Non-MPS: use native implementation
    return torch.pdist(x, p=p)
