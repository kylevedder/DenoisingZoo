from __future__ import annotations

from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from helpers import (
    PrecisionSettings,
    build_device,
    build_precision_settings,
    build_scaler,
    build_dataset,
    build_dataloader,
    build_model,
    build_optimizer,
    build_criterion,
)


Batch = dict[str, torch.Tensor]


def compute_loss(
    model: torch.nn.Module,
    batch: Batch,
    device: torch.device,
    settings: PrecisionSettings,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x = batch["input"].to(device, non_blocking=True)
    t = batch["t"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)

    with torch.autocast(
        device_type=settings.device_type,
        dtype=settings.autocast_dtype,
        enabled=settings.autocast_dtype is not None,
    ):
        pred = model(x, t)
        loss = criterion(pred, y)
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    settings: PrecisionSettings,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch, device, settings, criterion)

        if settings.use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item()
        num_batches += 1

    return running_loss / max(1, num_batches)


def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = build_device(cfg.get("device", "cuda"))
    settings = build_precision_settings(cfg.get("precision", "fp32"), device)
    scaler = build_scaler(settings)

    dataset = build_dataset(cfg)
    loader = build_dataloader(dataset, device, cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)

    epochs: int = int(cfg.get("epochs", 1))
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            settings=settings,
            scaler=scaler,
        )
        print(f"epoch {epoch:04d} | loss {avg_loss:.6f}")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
