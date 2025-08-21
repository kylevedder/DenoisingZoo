from __future__ import annotations

import os
import math
import argparse
import numpy as np

import torch
import matplotlib


# Use a non-interactive backend for headless saves
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from helpers import load_checkpoint
from models.mlp import MLP


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    # Infer feature dim from training config; for this project it's 2
    model = MLP(feature_dim=2, hidden_dim=256)
    model.to(device)
    # Load weights
    load_checkpoint(
        ckpt_path, model=model, optimizer=None, scaler=None, map_location=device
    )
    model.eval()
    return model


@torch.no_grad()
def compute_flow_field(
    model: torch.nn.Module,
    device: torch.device,
    grid_min: float = -4.0,
    grid_max: float = 4.0,
    num_points_per_dim: int = 200,
    time_t: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(grid_min, grid_max, num_points_per_dim, device=device)
    ys = torch.linspace(grid_min, grid_max, num_points_per_dim, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    t = torch.full((pos.shape[0], 1), float(time_t), device=device)

    # The learned vector field predicts the velocity (target - source)
    vec = model(pos, t)
    U = vec[:, 0].reshape(num_points_per_dim, num_points_per_dim)
    V = vec[:, 1].reshape(num_points_per_dim, num_points_per_dim)
    mag = torch.sqrt(U * U + V * V)
    return (
        X.detach().cpu(),
        Y.detach().cpu(),
        torch.stack([U, V, mag], dim=0).detach().cpu(),
    )


def plot_flow(
    X: torch.Tensor,
    Y: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    mag: torch.Tensor,
    out_path: str,
    time_t: float,
) -> None:
    # Color encodes direction (angle), intensity encodes magnitude
    angle = torch.atan2(V, U)
    # Normalize angle to [0, 1] for HSV
    angle01 = (angle + math.pi) / (2 * math.pi)
    # Normalize magnitude for brightness
    mag_norm = mag / (mag.max() + 1e-12)

    # Build HSV image: H=angle, S=1, V=mag_norm
    hsv = torch.stack([angle01, torch.ones_like(angle01), mag_norm], dim=-1).numpy()
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(
        rgb,
        origin="lower",
        extent=[float(X.min()), float(X.max()), float(Y.min()), float(Y.max())],
    )
    ax.set_title(f"Learned flow field t={time_t:.02f} (hue=direction, value=magnitude)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(False)

    # Overlay a circular hue wheel to indicate direction mapping
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

    wheel_ax = inset_axes(
        ax, width="22%", height="22%", loc="upper right", borderpad=0.8
    )
    wheel_ax.imshow(RGBA, origin="lower", extent=[-1, 1, -1, 1])
    wheel_ax.set_title("Direction", fontsize=8)
    wheel_ax.set_aspect("equal")
    wheel_ax.set_xticks([])
    wheel_ax.set_yticks([])
    for spine in wheel_ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _try_read_target_params_from_ckpt(
    ckpt_path: str,
) -> tuple[np.ndarray, float] | None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", None)
        if cfg is not None:
            centroids = np.array(cfg["dataset"]["centroids"])  # type: ignore[index]
            target_std = float(cfg["dataset"]["target_std"])  # type: ignore[index]
            return centroids, target_std
    except Exception:
        pass
    return None


def _fallback_read_target_params_from_yaml() -> tuple[np.ndarray, float]:
    cfg = OmegaConf.load("configs/train.yaml")
    centroids = np.array(cfg.dataset.centroids)
    target_std = float(cfg.dataset.target_std)
    return centroids, target_std


def sample_target_distribution(
    centroids: np.ndarray,
    target_std: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    k = centroids.shape[0]
    comp = rng.integers(0, k, size=(num_samples,))
    means = centroids[comp]
    samples = means + rng.normal(0.0, target_std, size=means.shape)
    return samples


def render_truth_density(
    centroids: np.ndarray,
    target_std: float,
    bounds: tuple[float, float, float, float],
    out_path: str,
    num_samples: int = 50000,
) -> None:
    xmin, xmax, ymin, ymax = bounds
    samples = sample_target_distribution(centroids, target_std, num_samples=num_samples)
    x = samples[:, 0]
    y = samples[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.hist2d(x, y, bins=200, range=[[xmin, xmax], [ymin, ymax]], cmap="Greys")
    ax.set_title("True target distribution (density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="outputs/ckpts/last.pt", help="Path to checkpoint"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Deprecated: base output path; if provided, saves <base>_flow.png and <base>_truth.png",
    )
    parser.add_argument(
        "--out_flow",
        type=str,
        default="outputs/vis/kmeans_flow.png",
        help="Output path for flow visualization",
    )
    parser.add_argument(
        "--out_truth",
        type=str,
        default="outputs/vis/kmeans_truth.png",
        help="Output path for true target density",
    )
    parser.add_argument("--min", type=float, default=-4.5)
    parser.add_argument("--max", type=float, default=4.5)
    parser.add_argument("--n", type=int, default=500, help="Points per dimension")
    # Time controls: render a sequence [t_start, t_end] with step
    parser.add_argument(
        "--t_start", type=float, default=0.0, help="Sequence start t (inclusive)"
    )
    parser.add_argument(
        "--t_end", type=float, default=1.0, help="Sequence end t (inclusive)"
    )
    parser.add_argument(
        "--t_step", type=float, default=0.05, help="Sequence step for t"
    )
    args = parser.parse_args()

    # Select device: prefer MPS, else CPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model_from_ckpt(args.ckpt, device)

    # Determine output paths
    if args.out is not None:
        base, ext = os.path.splitext(args.out)
        out_flow_base = f"{base}_flow{ext or '.png'}"
        out_truth = f"{base}_truth{ext or '.png'}"
    else:
        out_flow_base = args.out_flow
        out_truth = args.out_truth

    params = _try_read_target_params_from_ckpt(args.ckpt)
    if params is None:
        params = _fallback_read_target_params_from_yaml()
    centroids, target_std = params

    # Helper to add suffix before extension
    def add_suffix(path: str, suffix: str) -> str:
        b, e = os.path.splitext(path)
        if not e:
            e = ".png"
        return f"{b}_{suffix}{e}"

    # Build list of t values (always a sequence)
    vals = []
    t = float(args.t_start)
    while t <= args.t_end + 1e-9:
        vals.append(round(t, 4))
        t += float(args.t_step)
    t_values = vals

    # Precompute bounds from the grid (static across t)
    xmin, xmax = float(args.min), float(args.max)
    ymin, ymax = float(args.min), float(args.max)
    bounds = (xmin, xmax, ymin, ymax)

    # Render flow for each t; render truth once
    truth_written = False
    for tval in t_values:
        X, Y, stacked = compute_flow_field(
            model=model,
            device=device,
            grid_min=args.min,
            grid_max=args.max,
            num_points_per_dim=args.n,
            time_t=tval,
        )
        U, V, mag = stacked[0], stacked[1], stacked[2]
        flow_path = add_suffix(out_flow_base, f"t{tval:0.2f}")
        plot_flow(X, Y, U, V, mag, flow_path, tval)
        print(f"Saved flow visualization to {flow_path}")

        if not truth_written:
            render_truth_density(centroids, target_std, bounds, out_truth)
            print(f"Saved true target density to {out_truth}")
            truth_written = True


if __name__ == "__main__":
    main()
