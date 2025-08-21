from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from hydra.utils import instantiate

from dataloaders.base_dataloaders import DictDatasetAdapter, BaseDataset


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
        "config": cfg,
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
