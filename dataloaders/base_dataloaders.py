from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict

from torch.utils.data import Dataset
import torch


@dataclass
class BaseItem:
    input: torch.Tensor
    t: torch.Tensor
    target: torch.Tensor | None
    unified_input: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)


class BaseDataset(Dataset):
    """
    Base dataset that returns dataclass instances instead of dictionaries.

    Subclasses should implement __len__ and __getitem__ to return an instance
    of a dataclass (or any object) that implements SupportsToDict.
    """

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - abstract contract
        pass

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> BaseItem:  # pragma: no cover - abstract contract
        pass


class DictDatasetAdapter(Dataset):
    """
    Adapter dataset that converts dataclass items to dictionaries.
    """

    def __init__(self, source_dataset: BaseDataset):
        self._source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self._source_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._source_dataset[index]
        assert isinstance(item, BaseItem), f"Item must be a BaseItem, got {type(item)}"

        return item.to_dict()


def make_time_input(
    t: torch.Tensor,
    *,
    r: torch.Tensor | None = None,
    time_channels: int = 2,
    end_time: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Build time conditioning tensor for unified input.

    Args:
        t: Time tensor of shape (B,), (B, 1), or (B, K)
        r: Optional start time tensor (B,) or (B, 1) when time_channels=2
        time_channels: Number of time channels to produce (must be 2)
        end_time: Optional end time (float or tensor) when time_channels=2 and r is None

    Returns:
        Time tensor of shape (B, time_channels)
    """

    def _reshape_time(value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 0:
            raise ValueError("Time tensor must include a batch dimension.")
        return value.reshape(value.shape[0], -1)

    if time_channels != 2:
        raise ValueError("time_channels must be 2")

    t_flat = _reshape_time(t)
    if r is None:
        if t_flat.shape[1] == 2:
            return t_flat.to(dtype=t.dtype, device=t.device)
        if t_flat.shape[1] == 1:
            if end_time is None:
                end_value = torch.ones_like(t_flat)
            elif isinstance(end_time, torch.Tensor):
                end_value = _reshape_time(end_time).to(
                    dtype=t.dtype, device=t.device
                )
                if end_value.shape[0] != t_flat.shape[0]:
                    raise ValueError(
                        "end_time must match batch size when provided as a tensor."
                    )
                if end_value.shape[1] != 1:
                    raise ValueError("end_time tensor must have a single column.")
            else:
                end_value = torch.full_like(t_flat, float(end_time))
            return torch.cat([t_flat, end_value], dim=1).to(
                dtype=t.dtype, device=t.device
            )
        raise ValueError(
            "time_channels=2 expects t with 1 or 2 columns when r is None."
        )

    r_flat = _reshape_time(r).to(dtype=t.dtype, device=t.device)
    if t_flat.shape[1] != 1 or r_flat.shape[1] != 1:
        raise ValueError(
            "time_channels=2 expects r and t to have a single column each."
        )
    if r_flat.shape[0] != t_flat.shape[0]:
        raise ValueError(
            f"Batch size mismatch between r ({r_flat.shape[0]}) and t ({t_flat.shape[0]})"
        )
    return torch.cat([r_flat[:, :1], t_flat[:, :1]], dim=1)


def make_unified_flow_matching_input(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fuse inputs into a single tensor for model/solver consumption.

    Rules:
    - If x is (B, D): concatenate t as extra features to get (B, D+T)
    - If x is (B, C, H, W): broadcast t to (B, T, H, W) and concat on channel -> (B, C+T, H, W)
    - Else: raise ValueError for unsupported shapes
    """
    if x.shape[0] != t.shape[0]:
        raise ValueError(
            f"Batch size mismatch between x ({x.shape[0]}) and t ({t.shape[0]})"
        )
    # Normalize t to shape (B, T)
    t_batch = t.reshape(t.shape[0], -1).to(dtype=x.dtype, device=x.device)
    if x.dim() == 2:
        return torch.cat([x, t_batch], dim=1)
    if x.dim() == 4:
        # (B, 1, H, W)
        t_map = t_batch.view(t_batch.shape[0], t_batch.shape[1], 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3]
        )
        return torch.cat([x, t_map], dim=1)
    raise ValueError(f"Unsupported input tensor rank {x.dim()} for unified input")


@dataclass
class UnunifiedFlowMatchingInput:
    x: torch.Tensor
    t: torch.Tensor


def make_ununified_flow_matching_input(
    unified: torch.Tensor,
    num_time_channels: int = 2,
) -> UnunifiedFlowMatchingInput:
    """Invert make_unified_flow_matching_input.

    Requires num_time_channels=2 (r, t).

    - If unified is (B, D+T): returns x=(B, D) and t=(B, T)
    - If unified is (B, C+T, H, W): returns x=(B, C, H, W) and t=(B, T)
      where t is the spatial mean of the appended time channels (constant by construction).
    """
    if num_time_channels != 2:
        raise ValueError("num_time_channels must be 2")
    if unified.dim() == 2:
        if unified.shape[1] <= num_time_channels:
            raise ValueError(
                "Unified vector input must include data features and time channels."
            )
        x = unified[:, :-num_time_channels]
        t = unified[:, -num_time_channels:].contiguous()
        return UnunifiedFlowMatchingInput(x=x, t=t)
    if unified.dim() == 4:
        if unified.shape[1] <= num_time_channels:
            raise ValueError(
                "Unified image input must include data channels and time channels."
            )
        t_map = unified[:, -num_time_channels:, :, :]
        x = unified[:, :-num_time_channels, :, :]
        # Recover scalar t per-sample by averaging the constant time map
        t = t_map.mean(dim=(2, 3)).contiguous()
        return UnunifiedFlowMatchingInput(x=x, t=t)
    raise ValueError(
        f"Unsupported unified tensor rank {unified.dim()} for un-unification"
    )
