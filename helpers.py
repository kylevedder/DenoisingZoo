from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dataloaders.base_dataloaders import DictDatasetAdapter, BaseDataset
from metal_fallbacks.pdist import pdist_compat


@dataclass
class PrecisionSettings:
    autocast_dtype: torch.dtype | None
    use_scaler: bool
    device_type: str


def build_device(device_str: str) -> torch.device:
    ds = device_str.lower()
    if ds == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "mps":
        try:
            mps_ok = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
        except Exception:
            mps_ok = False
        return torch.device("mps" if mps_ok else "cpu")
    return torch.device(ds)


def build_precision_settings(precision: str, device: torch.device) -> PrecisionSettings:
    p = precision.lower()
    if p in {"fp32", "float32"}:
        return PrecisionSettings(
            None, False, "cuda" if device.type == "cuda" else "cpu"
        )
    if p in {"bf16", "bfloat16"}:
        return PrecisionSettings(
            torch.bfloat16, False, "cuda" if device.type == "cuda" else "cpu"
        )
    if p in {"fp16", "float16", "half"}:
        return PrecisionSettings(torch.float16, device.type == "cuda", "cuda")
    raise ValueError(f"Unknown precision: {precision}")


def build_scaler(settings: PrecisionSettings) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(enabled=settings.use_scaler)


def build_dataset(cfg: DictConfig) -> BaseDataset:
    dataset: BaseDataset = instantiate(cfg.dataset)
    return dataset


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------


def broadcast_batch_scalar_to(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a batch-wise scalar time tensor `t` to match `like`'s shape.

    Assumes first dimension is batch. Supports `t` shaped (B,) or (B, 1, ..., 1).
    Returns a view suitable for elementwise ops with `like` (no materialized expand).
    """
    if t.shape[0] != like.shape[0]:
        raise ValueError(
            f"Batch size mismatch between t ({t.shape[0]}) and like ({like.shape[0]})"
        )
    # Reduce t to (B,) then reshape to (B, 1, ..., 1)
    t_batch = t.reshape(t.shape[0], -1)[:, :1]
    target_shape = (t.shape[0], *([1] * (like.ndim - 1)))
    t_view = t_batch.reshape(*target_shape)
    return t_view.to(dtype=like.dtype, device=like.device)


def reshape_to_samples_2d(x: torch.Tensor) -> torch.Tensor:
    """Return x as (N, D) by flattening all non-batch dims.

    Assumes first dimension is batch. If x is already (N, D) it is returned
    as-is. For (N, C, H, W, ...) returns (N, C*H*W*...).
    """
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def build_dataset_from_config(node: DictConfig) -> BaseDataset:
    dataset: BaseDataset = instantiate(node)
    return dataset


def build_dataloader(
    dataset: BaseDataset, device: torch.device, cfg: DictConfig
) -> DataLoader:
    adapter = DictDatasetAdapter(dataset)
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.get("batch_size", 256),
        "shuffle": cfg.get("shuffle", True),
        "num_workers": cfg.get("num_workers", 0),
        "drop_last": cfg.get("drop_last", False),
    }
    return DataLoader(adapter, **loader_kwargs)


def build_dataloader_from_config(node: DictConfig, device: torch.device) -> DataLoader:
    """Build a DataLoader from a structured config node containing a dataset.

    Expected structure:
      node:
        dataset: <hydra target>
        batch_size: int
        shuffle: bool
        num_workers: int
        pin_memory: bool
        drop_last: bool
    """
    dataset: BaseDataset = instantiate(node.dataset)
    adapter = DictDatasetAdapter(dataset)
    loader_kwargs: dict[str, Any] = {
        "batch_size": node.get("batch_size", 256),
        "shuffle": node.get("shuffle", True),
        "num_workers": node.get("num_workers", 0),
        "pin_memory": node.get("pin_memory", device.type == "cuda"),
        "drop_last": node.get("drop_last", False),
    }
    return DataLoader(adapter, **loader_kwargs)


def build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = instantiate(cfg.model)
    model.to(device)
    return model


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters()
    )
    return optimizer


def build_criterion(
    cfg: DictConfig,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    criterion = instantiate(cfg.loss)
    return criterion


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    cfg: DictConfig,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        # Store a fully resolved, plain config for robust loading outside Hydra
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: torch.device | str | None = None,
) -> int:
    # Load full checkpoint dictionary; this is safe for trusted, local checkpoints
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Model weights
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]

    # Optimizer state (optional)
    if (
        optimizer is not None
        and "optimizer" in checkpoint
        and checkpoint["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]

    # Grad scaler state (optional)
    if (
        scaler is not None
        and "scaler" in checkpoint
        and checkpoint["scaler"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler"])  # type: ignore[arg-type]

    # Return the epoch this checkpoint corresponds to
    return int(checkpoint.get("epoch", 0))


# -----------------------------------------------------------------------------
# Higher-level checkpoint orchestration helpers
# -----------------------------------------------------------------------------


def get_checkpoint_path(cfg: DictConfig) -> str:
    """Build and ensure the checkpoint path based on config.

    Returns the full path `<ckpt_dir>/<ckpt_name>` and creates the directory if
    needed.
    """
    ckpt_dir: str = str(cfg.get("ckpt_dir", "outputs/ckpts"))
    ckpt_name: str = str(cfg.get("ckpt_name", "last.pt"))
    os.makedirs(ckpt_dir, exist_ok=True)
    return f"{ckpt_dir}/{ckpt_name}"


def resume_if_requested(
    cfg: DictConfig,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    """Resume training if `cfg.resume` is true.

    Returns the starting epoch (1 if not resumed).
    """
    start_epoch = 1
    if bool(cfg.get("resume", False)):
        try:
            last_epoch = load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                map_location=device,
            )
            start_epoch = int(last_epoch) + 1
            print(f"Resumed from {ckpt_path} at epoch {last_epoch}")
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}; starting fresh")
    return start_epoch


def save_if_needed(
    cfg: DictConfig,
    ckpt_path: str,
    epoch: int,
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    """Save a checkpoint based on `save_every` policy and last-epoch condition."""
    save_every: int = int(cfg.get("save_every", 1))
    if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        # Also archive by run name/info + epoch to keep all checkpoints
        try:
            run_name = str(cfg.get("run_name", "run"))
        except Exception:
            run_name = "run"
        archive_dir = os.path.join(os.path.dirname(ckpt_path), "archive")
        os.makedirs(archive_dir, exist_ok=True)
        archive_name = f"{run_name}_epoch_{epoch:04d}.pt"
        archive_path = os.path.join(archive_dir, archive_name)
        save_checkpoint(
            path=archive_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg,
        )
        print(f"Archived checkpoint to {archive_path}")


# -----------------------------------------------------------------------------
# Distributional evaluation helpers (U-statistics)
# -----------------------------------------------------------------------------


def compute_energy_distance_u_statistic(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    p: float = 2.0,
) -> float:
    """Unbiased U-statistic estimator of the squared energy distance.

    For distributions X and Y with samples x~X (n samples) and y~Y (m samples):
        ED^2 = 2 E||x - y|| - E||x - x'|| - E||y - y'||

    This function computes the unbiased U-statistic version using pairwise
    distances. If n<2 or m<2, the corresponding intra-set term is treated as 0.

    Args:
        x_samples: Tensor of shape (n, d)
        y_samples: Tensor of shape (m, d)
        p: Norm degree for distances (passed to torch.cdist / torch.pdist). For
           Euclidean distance use p=2.0.

    Returns:
        Scalar float: estimated ED^2 (non-negative, 0 iff identical distributions in the limit).
    """
    if x_samples.ndim != 2 or y_samples.ndim != 2:
        raise ValueError(
            "x_samples and y_samples must be rank-2 tensors (N, D) and (M, D)"
        )

    # Cross term
    xy = torch.cdist(x_samples, y_samples, p=p)
    mean_xy = (
        xy.mean() if xy.numel() > 0 else torch.tensor(0.0, device=x_samples.device)
    )

    # Intra terms (use pairwise without diagonal via pdist)
    if x_samples.shape[0] > 1:
        xx = pdist_compat(x_samples, p=p)
        mean_xx = xx.mean()
    else:
        mean_xx = torch.tensor(0.0, device=x_samples.device)

    if y_samples.shape[0] > 1:
        yy = pdist_compat(y_samples, p=p)
        mean_yy = yy.mean()
    else:
        mean_yy = torch.tensor(0.0, device=y_samples.device)

    ed2 = 2.0 * mean_xy - mean_xx - mean_yy
    # Numerical safety: clamp at 0
    return float(torch.clamp(ed2, min=0.0).item())


def evaluate_epoch_energy_distance(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    solver: Any,
    p: float = 2.0,
) -> float:
    """Evaluate distributional error via energy distance U-statistic.

    Reconstructs source x and target y per batch using the dataset's provided
    fields where:
      x_t = (1 - t) * x + t * y
      v   = y - x  (provided as `target` in batches)
      => x = x_t - t * v,  y = x + v = x_t + (1 - t) * v

    Then integrates from x (t=0) to obtain model samples and computes the
    energy distance U-statistic between predicted samples and reconstructed y.
    """
    model.eval()
    with torch.no_grad():
        ed_sum = 0.0
        num = 0
        for batch in tqdm(eval_loader, desc="Evaluating energy distance"):
            x_t: torch.Tensor = batch["input"].to(device)
            t: torch.Tensor = batch["t"].to(device)
            v: torch.Tensor = batch["target"].to(device)

            t_b = broadcast_batch_scalar_to(t, x_t)

            x0 = x_t - t_b * v
            y_true = x0 + v  # = x_t + (1 - t) * v

            result = solver.solve(x0)
            y_pred = result.final_state

            # Flatten to (N, D) for distance computation
            y_pred_2d = reshape_to_samples_2d(y_pred)
            y_true_2d = reshape_to_samples_2d(y_true)

            ed2 = compute_energy_distance_u_statistic(y_pred_2d, y_true_2d, p=p)
            ed_sum += ed2
            num += 1
        return ed_sum / max(1, num)
