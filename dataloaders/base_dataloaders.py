from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

from torch.utils.data import Dataset
import torch


@dataclass
class BaseItem:
    input: torch.Tensor
    t: torch.Tensor
    target: torch.Tensor | None

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
