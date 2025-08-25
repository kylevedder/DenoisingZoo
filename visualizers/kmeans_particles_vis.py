from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib

# Headless rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from solvers.euler_solver import EulerSolver
from solvers.rk4_solver import RK4Solver
from solvers.base_solver import BaseSolver
from visualizers.common import get_device, build_model_from_ckpt, ensure_dir_for
from omegaconf import OmegaConf
from helpers import build_dataloader_from_config


def animate_particles(
    trajectory: list[torch.Tensor],
    bounds: tuple[float, float],
    out_path: str,
    fps: int = 20,
    marker_size: int = 10,
    times: list[float] | None = None,
) -> None:
    minv, maxv = bounds
    frames = [t.detach().cpu().numpy() for t in trajectory]
    num_frames = len(frames)
    if times is None or len(times) != num_frames:
        times_list: list[float] = np.linspace(0.0, 1.0, num_frames).tolist()
    else:
        times_list = times

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)
    ax.set_aspect("equal")
    t0 = float(times_list[0]) if times_list else 0.0
    ax.set_title(f"Particle flow animation t={t0:.02f}")
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=marker_size, c="tab:blue")

    def update(frame_idx: int):
        pts = frames[frame_idx]
        scat.set_offsets(pts)
        ax.set_title(f"Particle flow animation t={float(times_list[frame_idx]):.02f}")
        return (scat,)

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / max(1, fps), blit=True
    )

    ensure_dir_for(out_path)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="outputs/ckpts/last.pt", help="Path to checkpoint"
    )
    parser.add_argument(
        "--out", type=str, default="outputs/vis/particles.gif", help="Output gif path"
    )
    parser.add_argument("--num", type=int, default=200, help="Number of particles")
    parser.add_argument("--bounds", type=float, nargs=2, default=[-4.5, 4.5])
    parser.add_argument(
        "--steps", type=int, default=50, help="Euler steps from t=0 to t=1"
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", type=int, default=10, help="Marker size")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/train.yaml",
        help="Hydra config path to build model and dataloader",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["euler", "rk4"],
        default="euler",
        help="Which ODE solver to use for integration",
    )
    args = parser.parse_args()

    # Device selection
    device = get_device()

    model = build_model_from_ckpt(args.ckpt, device, cfg_path=args.cfg)
    solver: BaseSolver
    if args.solver == "euler":
        solver = EulerSolver(model=model, num_steps=int(args.steps))
    else:
        solver = RK4Solver(model=model, num_steps=int(args.steps))

    # Build dataloader from config to draw a batch of positions
    cfg = OmegaConf.load(args.cfg)
    loader = build_dataloader_from_config(cfg.dataloaders.train, device)
    batch = next(iter(loader))
    init = batch["input"].to(device=device, dtype=torch.float32)
    if init.shape[0] > int(args.num):
        init = init[: int(args.num)]

    result = solver.solve(init)
    animate_particles(
        result.trajectory,
        (float(args.bounds[0]), float(args.bounds[1])),
        args.out,
        fps=int(args.fps),
        marker_size=int(args.size),
        times=result.times,
    )
    print(f"Saved particle animation to {args.out}")


if __name__ == "__main__":
    main()
