from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

from torch.utils.data import Dataset


@runtime_checkable
class SupportsToDict(Protocol):
    """Protocol for dataclass-like objects with a to_dict method."""

    def to_dict(self) -> Dict[str, Any]:
        ...


class DataclassMixin:
    """
    Mixin for Python dataclasses to provide a standard to_dict() implementation.

    This mixin assumes the inheriting class is a @dataclass. If it's not, a
    TypeError is raised when to_dict() is called.
    """

    def to_dict(self) -> Dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(
                f"{self.__class__.__name__} must be a dataclass to use DataclassMixin.to_dict()"
            )
        return asdict(self)


T = TypeVar("T", bound=SupportsToDict)


class BaseDataclassDataset(Dataset, Generic[T], ABC):
    """
    Base dataset that returns dataclass instances instead of dictionaries.

    Subclasses should implement __len__ and __getitem__ to return an instance
    of a dataclass (or any object) that implements SupportsToDict.
    """

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - abstract contract
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> T:  # pragma: no cover - abstract contract
        pass


class DictDatasetAdapter(Dataset):
    """
    Adapter dataset that converts dataclass items to dictionaries.

    This behaves like a typical PyTorch dataset that returns dictionaries,
    making it convenient to plug into standard training loops and collate
    functions. If the underlying item is already a dict, it is returned as-is.
    """

    def __init__(self, source_dataset: Dataset):
        self._source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self._source_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._source_dataset[index]

        # Prefer explicit to_dict if available
        if isinstance(item, SupportsToDict):
            value = item.to_dict()
            if not isinstance(value, dict):
                raise TypeError(
                    "to_dict() must return a dictionary for items used with DictDatasetAdapter"
                )
            return value

        # Fallback: convert plain dataclasses without a to_dict method
        if is_dataclass(item):
            return asdict(item)

        # If it's already a dict, pass through
        if isinstance(item, dict):
            return item

        raise TypeError(
            "Item must be a dataclass (optionally with to_dict) or a dict to be used with DictDatasetAdapter"
        )


