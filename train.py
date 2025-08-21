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
    build_dataset_from_config,
    build_dataloader_from_config,
    build_dataloader,
    build_model,
    build_optimizer,
    build_criterion,
    get_checkpoint_path,
    resume_if_requested,
    save_if_needed,
    evaluate_epoch_mse,
    load_checkpoint,
)
from hydra.utils import instantiate


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

    # Build train/eval dataloaders from structured config
    train_loader = build_dataloader_from_config(cfg.dataloaders.train, device)
    eval_loader = build_dataloader_from_config(cfg.dataloaders.eval, device)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)
    # Build solver
    solver = instantiate(cfg.solver, model=model)

    # Eval-only mode: if eval_checkpoint is provided, evaluate that path
    if (ckpt_override := cfg.get("eval_checkpoint", None)) is not None:
        ckpt_path = str(ckpt_override)
        try:
            _ = load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=None,
                scaler=None,
                map_location=device,
            )
            print(f"Loaded checkpoint from {ckpt_path}")
        except FileNotFoundError:
            print(f"Checkpoint not found: {ckpt_path}")
        eval_mse = evaluate_epoch_mse(
            model=model,
            eval_loader=eval_loader,
            device=device,
            solver=solver,
        )
        print(f"eval-only mse {eval_mse:.6f}")
        return

    # Resume from checkpoint if requested
    ckpt_path = get_checkpoint_path(cfg)
    start_epoch = resume_if_requested(
        cfg=cfg,
        ckpt_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    epochs: int = int(cfg.get("epochs", 1))
    for epoch in range(start_epoch, epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            settings=settings,
            scaler=scaler,
        )
        print(f"epoch {epoch:04d} | loss {avg_loss:.6f}")

        # Eval
        eval_mse = evaluate_epoch_mse(
            model=model,
            eval_loader=eval_loader,
            device=device,
            solver=solver,
        )
        print(f"eval mse {eval_mse:.6f}")

        # Save checkpoint based on policy
        save_if_needed(
            cfg=cfg,
            ckpt_path=ckpt_path,
            epoch=epoch,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
