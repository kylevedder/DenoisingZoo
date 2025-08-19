from __future__ import annotations

from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
import tqdm

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
    save_checkpoint,
    load_checkpoint,
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
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    bar = tqdm.tqdm(loader)
    for batch in bar:
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
        bar.set_description(f"loss: {running_loss / max(1, num_batches):.6f}")

    return running_loss / max(1, num_batches)


def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Runtime environment diagnostics
    print(f"torch version: {torch.__version__}")
    has_mps = bool(getattr(torch.backends, "mps", None))
    mps_available = bool(has_mps and torch.backends.mps.is_available())
    print(f"mps backend present: {has_mps}")
    print(f"mps available: {mps_available}")

    device = build_device(cfg.get("device", "cuda"))
    settings = build_precision_settings(cfg.get("precision", "fp32"), device)
    print(f"selected device: {device}")
    if device.type == "mps":
        sample = torch.ones(1, device=device)
        print(f"sample tensor device: {sample.device}")
    scaler = build_scaler(settings)

    dataset = build_dataset(cfg)
    loader = build_dataloader(dataset, device, cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)

    # Resume from checkpoint if requested
    start_epoch = 1
    ckpt_dir: str = str(cfg.get("ckpt_dir", "outputs/ckpts"))
    ckpt_name: str = str(cfg.get("ckpt_name", "last.pt"))
    ckpt_path = f"{ckpt_dir}/{ckpt_name}"
    if bool(cfg.get("resume", False)) and torch.cuda.is_available() is not None:
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

    epochs: int = int(cfg.get("epochs", 1))
    for epoch in range(start_epoch, epochs + 1):
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

        # Save checkpoint at the configured interval
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


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
