from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol

import torch


class VectorFieldModel(Protocol):
    def __call__(
        self, unified_input: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - protocol
        ...


@dataclass
class FlowSolveResult:
    final_state: torch.Tensor
    trajectory: list[torch.Tensor]
    times: list[float]


class BaseSolver(ABC):
    def __init__(
        self, model: VectorFieldModel, t_start: float = 0.0, t_end: float = 1.0
    ) -> None:
        self._model = model
        self._t_start = t_start
        self._t_end = t_end

    @abstractmethod
    def solve(
        self, initial_state: torch.Tensor
    ) -> FlowSolveResult:  # pragma: no cover - abstract contract
        pass
