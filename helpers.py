from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
