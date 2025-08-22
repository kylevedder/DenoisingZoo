from __future__ import annotations

import math
from pathlib import Path

import torch
import matplotlib
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
import shutil
import subprocess


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir_for(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    cfg_path: str = "configs/train.yaml",
) -> torch.nn.Module:
    """Instantiate model from Hydra config and load weights from checkpoint."""
    from helpers import load_checkpoint

    cfg = OmegaConf.load(cfg_path)
    model: torch.nn.Module = instantiate(cfg.model)
    model.to(device)
    load_checkpoint(
        ckpt_path, model=model, optimizer=None, scaler=None, map_location=device
    )
    model.eval()
    return model


@torch.no_grad()
def compute_flow_field(
    model: torch.nn.Module,
    device: torch.device,
    grid_min: float,
    grid_max: float,
    num_points_per_dim: int,
    time_t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(grid_min, grid_max, num_points_per_dim, device=device)
    ys = torch.linspace(grid_min, grid_max, num_points_per_dim, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    t = torch.full((pos.shape[0], 1), float(time_t), device=device)

    vec = model(pos, t)
    U = vec[:, 0].reshape(num_points_per_dim, num_points_per_dim)
    V = vec[:, 1].reshape(num_points_per_dim, num_points_per_dim)
    mag = torch.sqrt(U * U + V * V)
    return (
        X.detach().cpu(),
        Y.detach().cpu(),
        U.detach().cpu(),
        V.detach().cpu(),
        mag.detach().cpu(),
    )


def flow_to_rgb(U: torch.Tensor, V: torch.Tensor, mag: torch.Tensor) -> np.ndarray:
    """Encode direction (hue) and magnitude (value) into an HSV-derived RGB image."""
    angle = torch.atan2(V, U)
    angle01 = (angle + math.pi) / (2 * math.pi)
    mag_norm = mag / (mag.max() + 1e-12)
    hsv = torch.stack([angle01, torch.ones_like(angle01), mag_norm], dim=-1).numpy()
    return matplotlib.colors.hsv_to_rgb(hsv)


def add_direction_wheel_inset(
    ax, size_pct: str = "22%", loc: str = "upper right"
) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # lazy import

    Nw = 256
    g = torch.linspace(-1.0, 1.0, Nw)
    WX, WY = torch.meshgrid(g, g, indexing="xy")
    R = torch.sqrt(WX * WX + WY * WY)
    ANG = torch.atan2(WY, WX)
    H = (ANG + math.pi) / (2 * math.pi)
    S = torch.ones_like(H)
    Vv = torch.ones_like(H)
    HSV = torch.stack([H, S, Vv], dim=-1).numpy()
    RGB = matplotlib.colors.hsv_to_rgb(HSV)
    alpha = (R <= 1.0).to(torch.float32).numpy()
    RGBA = np.dstack([RGB, alpha])

    wheel_ax = inset_axes(ax, width=size_pct, height=size_pct, loc=loc, borderpad=0.8)
    wheel_ax.imshow(RGBA, origin="lower", extent=(-1, 1, -1, 1))
    wheel_ax.set_title("Direction", fontsize=8)
    wheel_ax.set_aspect("equal")
    wheel_ax.set_xticks([])
    wheel_ax.set_yticks([])
    for spine in wheel_ax.spines.values():
        spine.set_visible(False)


def mp4_to_gif(mp4_path: str, gif_path: str, fps: int = 20) -> None:
    """Convert an MP4 file to GIF using ffmpeg if available.

    This yields higher quality (palette + dithering) than naïve per-frame GIFs.
    Requires ffmpeg installed on the system.
    """
    ensure_dir_for(gif_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; please install ffmpeg to enable MP4→GIF conversion"
        )

    palette_path = str(Path(gif_path).with_suffix(".palette.png"))
    # Generate palette optimized for the clip
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            mp4_path,
            "-vf",
            f"fps={fps},scale=iw:ih:flags=lanczos,palettegen=stats_mode=full",
            palette_path,
        ],
        check=True,
    )
    # Use palette with good dithering
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            mp4_path,
            "-i",
            palette_path,
            "-lavfi",
            f"fps={fps},scale=iw:ih:flags=lanczos,paletteuse=dither=sierra2_4a",
            gif_path,
        ],
        check=True,
    )
    # Cleanup palette
    try:
        Path(palette_path).unlink(missing_ok=True)
    except Exception:
        pass
