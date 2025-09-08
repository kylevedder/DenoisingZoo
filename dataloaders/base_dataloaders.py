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


def make_unified_input(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fuse inputs into a single tensor for model/solver consumption.

    Rules:
    - If x is (B, D): concatenate t as an extra feature to get (B, D+1)
    - If x is (B, C, H, W): broadcast t to (B, 1, H, W) and concat on channel -> (B, C+1, H, W)
    - Else: raise ValueError for unsupported shapes
    """
    if x.shape[0] != t.shape[0]:
        raise ValueError(
            f"Batch size mismatch between x ({x.shape[0]}) and t ({t.shape[0]})"
        )
    # Normalize t to shape (B, 1)
    t_batch = t.reshape(t.shape[0], -1)[:, :1].to(dtype=x.dtype, device=x.device)
    if x.dim() == 2:
        return torch.cat([x, t_batch], dim=1)
    if x.dim() == 4:
        # (B, 1, H, W)
        t_map = t_batch.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return torch.cat([x, t_map], dim=1)
    raise ValueError(f"Unsupported input tensor rank {x.dim()} for unified input")
