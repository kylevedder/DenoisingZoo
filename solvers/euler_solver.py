from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from solvers.base_solver import BaseSolver, FlowSolveResult, VectorFieldModel


class EulerSolver(BaseSolver):
    """Fixed-step explicit Euler integrator for flow matching ODEs.

    Interprets the model output as instantaneous velocity dx/dt at time t.
    """

    def __init__(self, model: VectorFieldModel, num_steps: int = 10) -> None:
        super().__init__(model)
        self._num_steps = num_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

    @torch.no_grad()
    def solve(
        self,
        initial_state: torch.Tensor,
    ) -> FlowSolveResult:
        x = initial_state

        dt = float(self._t_end - self._t_start) / float(self._num_steps)
        t_schedule = np.linspace(self._t_start, self._t_end, self._num_steps + 1)

        trajectory: list[torch.Tensor] = [x.clone()]  # include initial
        times = t_schedule.tolist()

        for t in t_schedule:
            # Compute velocity v(x, t)
            t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
            v = self._model(x, t_tensor)

            # Euler update: x_{k+1} = x_k + dt * v(x_k, t_k)
            x = x + dt * v

            trajectory.append(x.clone())

        return FlowSolveResult(final_state=x, trajectory=trajectory, times=times)
