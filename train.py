from __future__ import annotations

import time
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
import tqdm  # type: ignore[import-not-found]

import trackio

from helpers import (
    PrecisionSettings,
    build_device,
    build_precision_settings,
    build_scaler,
    build_dataloader_from_config,
    build_model,
    build_optimizer,
    build_criterion,
    get_checkpoint_path,
    resume_if_requested,
    save_if_needed,
    evaluate_epoch_energy_distance,
    load_checkpoint,
    has_mps_backend,
    is_mps_available,
)
from losses.meanflow_loss import MeanFlowLoss
from hydra.utils import instantiate
from gpu_monitor import GPUMetricsMonitor


Batch = dict[str, torch.Tensor]


def init_trackio(cfg: DictConfig) -> bool:
    """Initialize trackio experiment tracking if enabled.

    Returns True if trackio was initialized, False otherwise.
    """
    trackio_cfg = cfg.get("trackio", {})
    if not trackio_cfg.get("enabled", False):
        return False

    # Convert OmegaConf to plain dict for trackio config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    project = trackio_cfg.get("project", "denoising-zoo")
    run_name = cfg.get("run_name")

    trackio.init(project=project, name=run_name, config=config_dict)
    print(f"Trackio: project={project}, run={run_name}")
    return True


def log_trackio(
    metrics: dict,
    enabled: bool,
    device: torch.device,
    gpu_monitor: GPUMetricsMonitor | None = None,
    step: int | None = None,
) -> None:
    """Log metrics to trackio if enabled.

    If a gpu_monitor is provided and running, uses its rolling averages for
    GPU metrics. Otherwise, only logs MPS memory metrics.

    Args:
        step: Optional explicit step number. If None, trackio auto-increments.
    """
    if enabled:
        if device.type == "mps":
            if gpu_monitor is not None and gpu_monitor.is_running():
                metrics.update(gpu_monitor.get_metrics())
            else:
                # Fallback: just log MPS memory without GPU utilization
                metrics["mps/allocated_mb"] = (
                    torch.mps.current_allocated_memory() / 1024**2
                )
                metrics["mps/driver_mb"] = (
                    torch.mps.driver_allocated_memory() / 1024**2
                )
        trackio.log(metrics, step=step)


def finish_trackio(enabled: bool) -> None:
    """Finish trackio run if enabled."""
    if enabled:
        trackio.finish()
        print("Trackio run finished")


def is_meanflow_loss(criterion: object) -> bool:
    """Check if criterion is a MeanFlow loss that requires special handling."""
    return isinstance(criterion, MeanFlowLoss)


def compute_loss(
    model: torch.nn.Module,
    batch: Batch,
    device: torch.device,
    settings: PrecisionSettings,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | MeanFlowLoss,
) -> torch.Tensor:
    with torch.autocast(
        device_type=settings.device_type,
        dtype=settings.autocast_dtype,
        enabled=settings.autocast_dtype is not None,
    ):
        if is_meanflow_loss(criterion):
            # MeanFlow loss handles model forward internally
            loss = criterion(batch, device)
        else:
            # Standard loss: model(unified_input) -> pred, loss(pred, target)
            unified = batch["unified_input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = model(unified)
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
    gradient_accumulation_steps: int = 1,
    log_fn: Callable[[dict, int], None] | None = None,
    log_every: int = 10,
    global_step: int = 0,
    epoch: int = 1,
) -> tuple[float, int]:
    """Train for one epoch with optional gradient accumulation.

    Args:
        gradient_accumulation_steps: Number of mini-batches to accumulate
            before performing an optimizer step. Effective batch size =
            loader.batch_size * gradient_accumulation_steps.
        log_fn: Optional callback(metrics, step) to log metrics every log_every batches.
        log_every: Log train metrics every N batches.
        global_step: Starting global step counter (incremented each batch).
        epoch: Current epoch number (for logging).

    Returns:
        Tuple of (average_loss, updated_global_step).
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    bar = tqdm.tqdm(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(bar):
        # Scale loss by accumulation steps to normalize gradient magnitude
        loss = compute_loss(model, batch, device, settings, criterion)
        loss = loss / gradient_accumulation_steps

        if settings.use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Track unscaled loss for reporting
        batch_loss = loss.detach().item() * gradient_accumulation_steps
        running_loss += batch_loss
        num_batches += 1
        global_step += 1

        # Optimizer step after accumulating gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            if settings.use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(1, num_batches)
        bar.set_description(f"loss: {avg_loss:.6f}")

        # Log train metrics every K batches
        if log_fn is not None and global_step % log_every == 0:
            log_fn({"epoch": epoch, "batch": step + 1, "train/loss": avg_loss}, global_step)

    # Handle remaining gradients if dataset size not divisible by accum steps
    if (step + 1) % gradient_accumulation_steps != 0:
        if settings.use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(1, num_batches), global_step


def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Runtime environment diagnostics
    print(f"torch version: {torch.__version__}")
    print(f"mps backend present: {has_mps_backend()}")
    print(f"mps available: {is_mps_available()}")

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

    # Optionally compile model for faster training
    if cfg.get("compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled")

    # Derive architecture-specific checkpoint directory: outputs/ckpts/<arch>
    # Use unwrap_compiled to get the underlying model class name for compiled models
    from model_contracts import unwrap_compiled
    arch_name = unwrap_compiled(model).__class__.__name__.lower()
    base_ckpt_dir = str(cfg.get("ckpt_dir", "outputs/ckpts"))
    cfg.ckpt_dir = f"{base_ckpt_dir}/{arch_name}"
    # keep ckpt_name as-is (defaults to last.pt)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)

    # If using MeanFlow loss, inject the model
    if is_meanflow_loss(criterion):
        criterion.set_model(model)
        print("Using MeanFlow loss")
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
        ed = evaluate_epoch_energy_distance(
            model=model, eval_loader=eval_loader, device=device, solver=solver
        )
        print(f"eval-only energy_distance {ed:.6f}")
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

    # Initialize experiment tracking
    trackio_enabled = init_trackio(cfg)

    # Start async GPU metrics monitor for MPS devices
    gpu_monitor: GPUMetricsMonitor | None = None
    if device.type == "mps":
        gpu_monitor = GPUMetricsMonitor()  # streams from macmon, 500ms interval, 5s window
        gpu_monitor.start()
        print("GPU metrics monitor started (streaming from macmon)")

    epochs: int = int(cfg.get("epochs", 1))
    eval_every: int = int(cfg.get("eval_every", 1))  # how often to run eval (in epochs)
    gradient_accumulation_steps: int = int(cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps > 1:
        effective_batch = train_loader.batch_size * gradient_accumulation_steps
        print(f"Gradient accumulation: {gradient_accumulation_steps} steps, effective batch size: {effective_batch}")

    # Trackio logging config
    trackio_cfg = cfg.get("trackio", {})
    log_every: int = int(trackio_cfg.get("log_every", 10))

    # Create logging callback for per-batch logging
    def log_fn(metrics: dict, step: int) -> None:
        log_trackio(metrics, trackio_enabled, device, gpu_monitor, step=step)

    train_start_time = time.time()
    global_step = 0
    for epoch in range(start_epoch, epochs + 1):
        avg_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            settings=settings,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_fn=log_fn if trackio_enabled else None,
            log_every=log_every,
            global_step=global_step,
            epoch=epoch,
        )
        print(f"epoch {epoch:04d} | loss {avg_loss:.6f}")

        # Eval (distributional) based on eval_every policy
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == epochs):
            ed = evaluate_epoch_energy_distance(
                model=model, eval_loader=eval_loader, device=device, solver=solver
            )
            print(f"eval energy_distance {ed:.6f}")
            # Log eval metrics separately (only at end of epoch)
            elapsed_seconds = time.time() - train_start_time
            log_trackio(
                {"epoch": epoch, "eval/energy_distance": ed, "train/elapsed_seconds": elapsed_seconds},
                trackio_enabled,
                device,
                gpu_monitor,
                step=global_step,
            )

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

    # Stop GPU monitor
    if gpu_monitor is not None:
        gpu_monitor.stop()
        print("GPU metrics monitor stopped")

    # Finish experiment tracking
    finish_trackio(trackio_enabled)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
