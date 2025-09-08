from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

# Headless rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from omegaconf import OmegaConf

from visualizers.common import (
    get_device,
    ensure_dir_for,
    build_model_from_ckpt,
    mp4_to_gif,
    tensor_to_display_image,
    save_animation,
)
from helpers import build_dataloader_from_config
from solvers.euler_solver import EulerSolver
from solvers.rk4_solver import RK4Solver
from solvers.base_solver import BaseSolver
from torch.utils.data import DataLoader
from itertools import islice


# image conversion handled by visualizers.common.tensor_to_display_image


def animate_denoising(
    frames: list[torch.Tensor],
    out_path: str,
    fps: int = 20,
    save_frames_dir: str | None = None,
) -> None:
    """Create an animation from a list of (C,H,W) tensors showing denoising progress."""
    # Prepare display frames
    disp_frames = [tensor_to_display_image(f) for f in frames]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.set_axis_off()

    init_img = disp_frames[0]
    im = ax.imshow(
        init_img, cmap="gray" if init_img.ndim == 2 else None, interpolation="bicubic"
    )
    ax.set_title("CelebA denoising t=0.00")

    # Generate evenly spaced times in [0,1]
    ts = np.linspace(0.0, 1.0, num=len(disp_frames)).tolist()

    def update(i: int):
        frame = disp_frames[i]
        im.set_data(frame)
        ax.set_title(f"CelebA denoising t={float(ts[i]):.02f}")
        if save_frames_dir is not None:
            # Save current frame as PNG
            ensure_dir_for(str(Path(save_frames_dir) / "frame.png"))
            frame_path = Path(save_frames_dir) / f"frame_{i:04d}.png"
            plt.imsave(frame_path, frame, cmap="gray" if frame.ndim == 2 else None)
        return (im,)

    anim = FuncAnimation(
        fig, update, frames=len(disp_frames), interval=1000 / max(1, fps), blit=True
    )

    ensure_dir_for(out_path)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)


def get_random_batch_source(
    loader: DataLoader, seed: int | None, device: torch.device
) -> torch.Tensor:
    dataset = loader.dataset
    rng = np.random.default_rng(seed)
    nth_batch = rng.integers(0, len(dataset))
    print(f"Using batch {nth_batch} of {len(dataset)}")
    batch = dataset[nth_batch]
    return batch["raw_source"].unsqueeze(0).to(device=device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="CelebA denoising visualizer")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/ckpts/smallcnn/last.pt",
        help="Path to checkpoint (defaults to CNN arch path)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/train.yaml",
        help="Hydra config path to build the model and dataloader",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/vis/celeba_denoise.gif",
        help="Output animation path (.gif or .mp4)",
    )
    parser.add_argument("--steps", type=int, default=50, help="Integration steps")
    parser.add_argument("--fps", type=int, default=20, help="Animation FPS")
    parser.add_argument(
        "--idx", type=int, default=0, help="Sample index from the loader"
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["euler", "rk4"],
        default="euler",
        help="ODE solver to integrate from t=0 to t=1",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the random number generator"
    )
    args = parser.parse_args()

    device = get_device()

    # Load model from checkpoint (uses resolved config inside the ckpt)
    model = build_model_from_ckpt(args.ckpt, device, cfg_path=args.cfg)

    # Build the train dataloader to sample an initial noisy image x ("raw_source")
    ckpt_blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt_blob, dict) and "config" in ckpt_blob):
        raise RuntimeError(
            "Checkpoint is missing a resolved 'config'. Re-train to create a compatible checkpoint."
        )
    cfg_ckpt = OmegaConf.create(ckpt_blob["config"])  # type: ignore[arg-type]
    loader = build_dataloader_from_config(cfg_ckpt.dataloaders.train, device)

    # Pull one batch and select a single example
    x0 = get_random_batch_source(loader, args.seed, device)

    # Choose solver
    solver: BaseSolver
    if args.solver == "euler":
        solver = EulerSolver(model=model, num_steps=int(args.steps))
    else:
        solver = RK4Solver(model=model, num_steps=int(args.steps))

    # Integrate from t=0 noisy to t=1 denoised
    result = solver.solve(x0)

    # Show the trajectory of the single sample
    frames = [f[0] for f in result.trajectory]  # (C,H,W) per frame
    # Optional per-frame saving directory alongside output
    frames_dir: str | None = None
    out_path = Path(args.out)
    if out_path.suffix.lower() in {".gif", ".mp4"}:
        frames_dir = str(out_path.with_suffix("").as_posix() + "_frames")

    animate_denoising(frames, args.out, fps=int(args.fps), save_frames_dir=frames_dir)

    # If GIF requested, optionally render via MP4 then convert for higher quality
    if str(args.out).lower().endswith(".gif"):
        mp4_path = str(Path(args.out).with_suffix(".mp4"))
        animate_denoising(
            frames, mp4_path, fps=int(args.fps), save_frames_dir=frames_dir
        )
        mp4_to_gif(mp4_path, str(args.out), fps=int(args.fps))
        print(f"Saved CelebA denoising GIF to {args.out}")
    else:
        print(f"Saved CelebA denoising animation to {args.out}")


if __name__ == "__main__":
    main()
