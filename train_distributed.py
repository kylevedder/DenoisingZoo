"""Distributed training script for multi-GPU support.

This script extends train.py with DDP support. Use with torchrun:
    torchrun --nproc_per_node=4 train_distributed.py run_name=exp distributed.enabled=true
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import tqdm

import trackio

from helpers import (
    PrecisionSettings,
    build_precision_settings,
    build_scaler,
    build_optimizer,
    build_criterion,
    get_checkpoint_path,
    has_mps_backend,
    is_mps_available,
    # Distributed utilities
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    seed_everything,
    unwrap_model,
    build_dataloader_from_config_distributed,
    build_model_distributed,
    save_checkpoint_distributed,
    load_checkpoint_distributed,
    reduce_sum,
    evaluate_epoch_energy_distance,
)
from losses.meanflow_loss import MeanFlowLoss
from hydra.utils import instantiate
from gpu_monitor import GPUMetricsMonitor
from model_contracts import unwrap_compiled


Batch = dict[str, torch.Tensor]


def init_trackio(cfg: DictConfig) -> bool:
    """Initialize trackio experiment tracking if enabled (rank 0 only)."""
    if not is_main_process():
        return False

    trackio_cfg = cfg.get("trackio", {})
    if not trackio_cfg.get("enabled", False):
        return False

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
    """Log metrics to trackio if enabled (rank 0 only)."""
    if not is_main_process():
        return

    if enabled:
        if device.type == "mps":
            if gpu_monitor is not None and gpu_monitor.is_running():
                metrics.update(gpu_monitor.get_metrics())
            else:
                metrics["mps/allocated_mb"] = torch.mps.current_allocated_memory() / 1024**2
                metrics["mps/driver_mb"] = torch.mps.driver_allocated_memory() / 1024**2
        elif device.type == "cuda":
            # Log CUDA memory for distributed training
            metrics["cuda/allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
            metrics["cuda/reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        trackio.log(metrics, step=step)


def finish_trackio(enabled: bool) -> None:
    """Finish trackio run if enabled (rank 0 only)."""
    if not is_main_process():
        return
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
            loss = criterion(batch, device)
        else:
            unified = batch["unified_input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = model(unified)
            loss = criterion(pred, y)
    return loss


def train_one_epoch_distributed(
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
    distributed: bool = False,
) -> tuple[float, int]:
    """Train for one epoch with DDP-aware gradient accumulation.

    Uses model.no_sync() context to skip gradient synchronization during
    accumulation steps, only syncing on the final micro-batch or epoch end.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    total_steps = len(loader)

    # Only show progress bar on rank 0
    if is_main_process():
        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
    else:
        bar = loader

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(bar):
        batch_size = batch["unified_input"].shape[0]
        is_last_step = (step + 1) == total_steps
        is_sync_step = ((step + 1) % gradient_accumulation_steps == 0) or is_last_step

        # Use no_sync() to skip gradient sync for non-sync steps (DDP only)
        if distributed and isinstance(model, DDP):
            ctx = model.no_sync() if not is_sync_step else contextlib.nullcontext()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            loss = compute_loss(model, batch, device, settings, criterion)
            loss = loss / gradient_accumulation_steps

            if settings.use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Track loss (unscaled)
        batch_loss = loss.detach().item() * gradient_accumulation_steps
        running_loss += batch_loss * batch_size
        total_samples += batch_size
        global_step += 1

        # Optimizer step on sync steps
        if is_sync_step:
            if settings.use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar on rank 0 (bar is tqdm when is_main_process)
        if is_main_process():
            avg_loss = running_loss / max(1, total_samples)
            bar.set_description(f"Epoch {epoch} | loss: {avg_loss:.6f}")

        # Log train metrics
        if log_fn is not None and global_step % log_every == 0:
            avg_loss = running_loss / max(1, total_samples)
            log_fn({"epoch": epoch, "batch": step + 1, "train/loss": avg_loss}, global_step)

    # Synchronize loss across ranks for accurate reporting
    if distributed:
        loss_tensor = torch.tensor(running_loss, device=device)
        samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float32)
        loss_tensor = reduce_sum(loss_tensor)
        samples_tensor = reduce_sum(samples_tensor)
        avg_loss = (loss_tensor / samples_tensor).item()
    else:
        avg_loss = running_loss / max(1, total_samples)

    return avg_loss, global_step


def train(cfg: DictConfig) -> None:
    # Setup distributed training (must be first)
    local_rank, global_rank, world_size = setup_distributed(cfg)
    distributed = world_size > 1

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))
        print(f"torch version: {torch.__version__}")
        print(f"Distributed: {distributed}, world_size: {world_size}, rank: {global_rank}")

    # Device setup
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device_str = cfg.get("device", "cuda")
        if device_str == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "mps":
            device = torch.device("mps" if is_mps_available() else "cpu")
        else:
            device = torch.device(device_str)

    if is_main_process():
        print(f"selected device: {device}")

    settings = build_precision_settings(cfg.get("precision", "fp32"), device)
    scaler = build_scaler(settings)

    # CUDA optimizations
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if is_main_process():
            print("CUDA optimizations enabled: TF32 matmul, cuDNN benchmark")

    # Seed with rank offset for distributed
    seed = cfg.get("seed", 42)
    worker_init_fn = seed_everything(seed, global_rank)

    # Check FSDP + MeanFlow incompatibility
    distributed_cfg = cfg.get("distributed", {})
    strategy = distributed_cfg.get("strategy", "ddp") if distributed else None
    if strategy == "fsdp":
        loss_target = cfg.loss.get("_target_", "")
        if "MeanFlowLoss" in loss_target:
            raise ValueError("FSDP is not compatible with MeanFlow loss (JVP). Use DDP instead.")

    # Build dataloaders: train uses DistributedSampler, eval runs on rank 0 only
    train_loader, train_sampler = build_dataloader_from_config_distributed(
        cfg.dataloaders.train, device, distributed, is_train=True, worker_init_fn=worker_init_fn
    )
    # Eval loader: no distributed sampler (rank 0 processes full dataset for accurate metrics)
    eval_loader, _ = build_dataloader_from_config_distributed(
        cfg.dataloaders.eval, device, distributed=False, is_train=False, worker_init_fn=worker_init_fn
    )

    # Build model with DDP wrapping
    model = build_model_distributed(cfg, device, local_rank if distributed else None, strategy)

    # Get architecture name for checkpoint path (unwrap if needed)
    arch_name = unwrap_compiled(unwrap_model(model)).__class__.__name__.lower()
    base_ckpt_dir = str(cfg.get("ckpt_dir", "outputs/ckpts"))
    cfg.ckpt_dir = f"{base_ckpt_dir}/{arch_name}"

    # Build optimizer on wrapped model
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)

    # Set model for MeanFlow loss
    if is_meanflow_loss(criterion):
        criterion.set_model(model)
        if is_main_process():
            print("Using MeanFlow loss")

    # Build solver (uses unwrapped model for inference)
    solver = instantiate(cfg.solver, model=model)

    # Resume from checkpoint
    ckpt_path = get_checkpoint_path(cfg)
    start_epoch = 1
    if cfg.get("resume", False):
        try:
            start_epoch = load_checkpoint_distributed(
                ckpt_path, model, optimizer, scaler, device
            ) + 1
            if is_main_process():
                print(f"Resumed from {ckpt_path} at epoch {start_epoch - 1}")
        except FileNotFoundError:
            if is_main_process():
                print(f"No checkpoint found at {ckpt_path}; starting fresh")

    # Initialize tracking (rank 0 only)
    trackio_enabled = init_trackio(cfg)

    # GPU monitor (rank 0 only)
    gpu_monitor: GPUMetricsMonitor | None = None
    if device.type == "mps" and is_main_process():
        gpu_monitor = GPUMetricsMonitor()
        gpu_monitor.start()
        print("GPU metrics monitor started")

    epochs = int(cfg.get("epochs", 1))
    eval_every = int(cfg.get("eval_every", 1))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))

    if gradient_accumulation_steps > 1 and is_main_process():
        effective_batch = train_loader.batch_size * gradient_accumulation_steps * world_size
        print(f"Gradient accumulation: {gradient_accumulation_steps} steps, "
              f"effective global batch size: {effective_batch}")

    trackio_cfg = cfg.get("trackio", {})
    log_every = int(trackio_cfg.get("log_every", 10))

    def log_fn(metrics: dict, step: int) -> None:
        log_trackio(metrics, trackio_enabled, device, gpu_monitor, step=step)

    train_start_time = time.time()
    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for DistributedSampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_loss, global_step = train_one_epoch_distributed(
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
            distributed=distributed,
        )

        if is_main_process():
            print(f"epoch {epoch:04d} | loss {avg_loss:.6f}")

        # Evaluation (rank 0 only to avoid sharding bias)
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == epochs):
            if is_main_process():
                model.eval()
                ed = evaluate_epoch_energy_distance(
                    model=model, eval_loader=eval_loader, device=device, solver=solver
                )
                print(f"eval energy_distance {ed:.6f}")
                elapsed_seconds = time.time() - train_start_time
                log_trackio(
                    {"epoch": epoch, "eval/energy_distance": ed, "train/elapsed_seconds": elapsed_seconds},
                    trackio_enabled, device, gpu_monitor, step=global_step
                )
            # Synchronize all ranks before continuing
            if distributed:
                dist.barrier()

        # Save checkpoint (rank 0 only)
        save_every = int(cfg.get("save_every", 1))
        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
            save_checkpoint_distributed(ckpt_path, model, optimizer, scaler, epoch, cfg)
            if is_main_process():
                print(f"Saved checkpoint to {ckpt_path}")
                # Archive
                run_name = str(cfg.get("run_name", "run"))
                archive_dir = Path(ckpt_path).parent / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"{run_name}_epoch_{epoch:04d}.pt"
                save_checkpoint_distributed(str(archive_path), model, optimizer, scaler, epoch, cfg)
                print(f"Archived checkpoint to {archive_path}")

    # Cleanup
    if gpu_monitor is not None:
        gpu_monitor.stop()
        print("GPU metrics monitor stopped")

    finish_trackio(trackio_enabled)
    cleanup_distributed()


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
