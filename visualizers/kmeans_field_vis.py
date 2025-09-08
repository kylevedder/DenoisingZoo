from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib

# Headless rendering (same as particles_anim)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from visualizers.common import (
    get_device,
    ensure_dir_for,
    build_model_from_ckpt,
    compute_flow_field,
    flow_to_rgb,
    add_direction_wheel_inset,
    mp4_to_gif,
    save_animation,
)


def animate_flow(
    model: torch.nn.Module,
    device: torch.device,
    t_values: list[float],
    grid_min: float,
    grid_max: float,
    n: int,
    out_gif: str,
    fps: int = 20,
) -> None:
    # First frame to initialize canvas
    X, Y, U, V, mag = compute_flow_field(
        model=model,
        device=device,
        grid_min=grid_min,
        grid_max=grid_max,
        num_points_per_dim=n,
        time_t=t_values[0],
    )
    rgb0 = flow_to_rgb(U, V, mag)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(
        rgb0,
        origin="lower",
        extent=(float(X.min()), float(X.max()), float(Y.min()), float(Y.max())),
        interpolation="bicubic",
    )
    ax.set_title(
        f"Learned flow field t={t_values[0]:.02f} (hue=direction, value=magnitude)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(False)
    add_direction_wheel_inset(ax)

    def update(idx: int):
        t = t_values[idx]
        X, Y, U, V, mag = compute_flow_field(
            model=model,
            device=device,
            grid_min=grid_min,
            grid_max=grid_max,
            num_points_per_dim=n,
            time_t=t,
        )
        rgb = flow_to_rgb(U, V, mag)
        im.set_data(rgb)
        ax.set_title(f"Learned flow field t={t:.02f} (hue=direction, value=magnitude)")
        return (im,)

    anim = FuncAnimation(
        fig, update, frames=len(t_values), interval=1000 / max(1, fps), blit=True
    )
    ensure_dir_for(out_gif)
    save_animation(anim, out_gif, fps=fps)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="outputs/ckpts/last.pt", help="Path to checkpoint"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/train.yaml",
        help="Hydra config path to build the model",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/vis/kmeans_flow.gif",
        help="Output GIF path",
    )
    parser.add_argument("--min", type=float, default=-4.5)
    parser.add_argument("--max", type=float, default=4.5)
    parser.add_argument("--n", type=int, default=500, help="Grid resolution per axis")
    parser.add_argument("--fps", type=int, default=20)
    # Time controls: sequence [t_start, t_end] with step
    parser.add_argument("--t_start", type=float, default=0.0)
    parser.add_argument("--t_end", type=float, default=1.0)
    parser.add_argument("--t_step", type=float, default=0.05)
    args = parser.parse_args()

    device = get_device()
    model = build_model_from_ckpt(args.ckpt, device, cfg_path=args.cfg)

    # Build list of t values (inclusive of end)
    t_values: list[float] = []
    t = float(args.t_start)
    while t <= args.t_end + 1e-9:
        t_values.append(round(t, 4))
        t += float(args.t_step)

    animate_flow(
        model=model,
        device=device,
        t_values=t_values,
        grid_min=float(args.min),
        grid_max=float(args.max),
        n=int(args.n),
        out_gif=str(args.out),
        fps=int(args.fps),
    )
    # Optional: if user asked for GIF but we generated MP4 for smoother colors
    # allow user to specify a .gif path directly
    if str(args.out).lower().endswith(".gif"):
        mp4_path = str(Path(args.out).with_suffix(".mp4"))
        # Render MP4 to avoid banding, then convert to high-quality GIF via ffmpeg
        animate_flow(
            model=model,
            device=device,
            t_values=t_values,
            grid_min=float(args.min),
            grid_max=float(args.max),
            n=int(args.n),
            out_gif=mp4_path,
            fps=int(args.fps),
        )
        mp4_to_gif(mp4_path, str(args.out), fps=int(args.fps))
        print(f"Saved flow animation GIF to {args.out}")
    else:
        print(f"Saved flow animation GIF to {args.out}")


if __name__ == "__main__":
    main()
