from __future__ import annotations

import numpy as np
import torch

from solvers.base_solver import BaseSolver, FlowSolveResult, VectorFieldModel
from dataloaders.base_dataloaders import make_unified_input


class RK4Solver(BaseSolver):
    """Fixed-step classic Runge–Kutta (RK4) integrator for flow matching ODEs.

    Interprets the model output as instantaneous velocity dx/dt at time t.
    """

    def __init__(self, model: VectorFieldModel, num_steps: int = 10) -> None:
        super().__init__(model)
        self._num_steps = int(num_steps)
        if self._num_steps <= 0:
            raise ValueError("num_steps must be positive")

    @torch.no_grad()
    def solve(self, initial_state: torch.Tensor) -> FlowSolveResult:
        x = initial_state

        dt = float(self._t_end - self._t_start) / float(self._num_steps)
        t_schedule = np.linspace(self._t_start, self._t_end, self._num_steps + 1)

        trajectory: list[torch.Tensor] = [x.clone()]
        times = t_schedule.tolist()

        for t in t_schedule:
            t0 = float(t)
            t1 = t0 + 0.5 * dt
            t2 = t0 + dt

            t0_tensor = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
            k1 = self._model(make_unified_input(x, t0_tensor))

            t1_tensor = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
            k2 = self._model(make_unified_input(x + 0.5 * dt * k1, t1_tensor))

            k3 = self._model(make_unified_input(x + 0.5 * dt * k2, t1_tensor))

            t2_tensor = torch.full((x.shape[0], 1), t2, device=x.device, dtype=x.dtype)
            k4 = self._model(make_unified_input(x + dt * k3, t2_tensor))

            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            trajectory.append(x.clone())

        return FlowSolveResult(final_state=x, trajectory=trajectory, times=times)
