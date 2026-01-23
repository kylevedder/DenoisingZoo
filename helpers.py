from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dataloaders.base_dataloaders import DictDatasetAdapter, BaseDataset
from metal_fallbacks.pdist import pdist_compat


@dataclass
class PrecisionSettings:
    autocast_dtype: torch.dtype | None
    use_scaler: bool
    device_type: str


def has_mps_backend() -> bool:
    return bool(getattr(torch.backends, "mps", None))


def is_mps_available() -> bool:
    if not has_mps_backend():
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError) as exc:
        print(f"Warning: MPS availability check failed: {exc}")
        return False


def build_device(device_str: str) -> torch.device:
    ds = device_str.lower()
    if ds == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "mps":
        return torch.device("mps" if is_mps_available() else "cpu")
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
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    cfg: DictConfig,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        # Store a fully resolved, plain config for robust loading outside Hydra
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
    }
    torch.save(state, path_obj)


def load_checkpoint(
    path: str | Path,
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
    ckpt_dir = Path(str(cfg.get("ckpt_dir", "outputs/ckpts")))
    ckpt_name = str(cfg.get("ckpt_name", "last.pt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return str(ckpt_dir / ckpt_name)


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
        run_name = str(cfg.get("run_name", "run"))
        ckpt_dir = Path(ckpt_path).parent
        archive_dir = ckpt_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{run_name}_epoch_{epoch:04d}.pt"
        archive_path = archive_dir / archive_name
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


# -----------------------------------------------------------------------------
# Distributed training utilities
# -----------------------------------------------------------------------------


def setup_distributed(cfg: DictConfig) -> tuple[int, int, int]:
    """Initialize distributed training.

    Args:
        cfg: Hydra config with distributed.enabled flag

    Returns:
        (local_rank, global_rank, world_size)
    """
    # Guard against double initialization
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return local_rank, dist.get_rank(), dist.get_world_size()

    distributed_enabled = cfg.get("distributed", {}).get("enabled", False)

    if not dist.is_available():
        if distributed_enabled:
            print("WARNING: distributed.enabled=true but torch.distributed is not available")
        return 0, 0, 1

    # Check if launched via torchrun
    if "LOCAL_RANK" not in os.environ:
        if distributed_enabled:
            print("WARNING: distributed.enabled=true but not launched via torchrun. "
                  "Running single-process. Use: torchrun --nproc_per_node=N train_distributed.py")
        return 0, 0, 1

    # Safety check: error if torchrun was used but distributed.enabled=false
    if not distributed_enabled:
        raise RuntimeError(
            "Launched via torchrun but distributed.enabled=false. "
            "Set distributed.enabled=true or run without torchrun."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set device before init (only for CUDA)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return local_rank, global_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def seed_everything(seed: int, rank: int = 0) -> Callable[[int], None]:
    """Seed RNG with rank offset for distributed training.

    Args:
        seed: Base seed from config
        rank: Global rank (0 for single-GPU)

    Returns:
        worker_init_fn for DataLoader workers
    """
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = effective_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively unwrap DDP/compiled model to get base module."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):  # DDP
            unwrapped = unwrapped.module
        elif hasattr(unwrapped, "_orig_mod"):  # torch.compile
            unwrapped = unwrapped._orig_mod
        else:
            break
    return unwrapped


def build_dataloader_from_config_distributed(
    node: DictConfig,
    device: torch.device,
    distributed: bool = False,
    is_train: bool = True,
    worker_init_fn: Callable[[int], None] | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a DataLoader with optional DistributedSampler support.

    Args:
        node: Config node with dataset and loader settings
        device: Target device
        distributed: Whether to use DistributedSampler
        is_train: True for training loader (affects drop_last, shuffle)
        worker_init_fn: Optional worker init function for seeding

    Returns:
        (DataLoader, sampler or None)
    """
    dataset: BaseDataset = instantiate(node.dataset)
    adapter = DictDatasetAdapter(dataset)

    sampler = None
    shuffle = node.get("shuffle", True) if is_train else False

    if distributed:
        sampler = DistributedSampler(
            adapter,
            shuffle=shuffle,
            drop_last=True if is_train else False,
        )
        shuffle = False  # Sampler handles shuffling

    loader = DataLoader(
        adapter,
        batch_size=node.get("batch_size", 256),
        shuffle=shuffle,
        num_workers=node.get("num_workers", 0),
        pin_memory=node.get("pin_memory", device.type == "cuda"),
        drop_last=True if (distributed and is_train) else node.get("drop_last", False),
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    return loader, sampler


def build_model_distributed(
    cfg: DictConfig,
    device: torch.device,
    local_rank: int | None = None,
    distributed_strategy: str | None = None,
) -> torch.nn.Module:
    """Build model with optional DDP wrapping.

    Args:
        cfg: Config with model and compile settings
        device: Target device
        local_rank: Local GPU rank (required for distributed)
        distributed_strategy: "ddp" or None

    Returns:
        Model (possibly DDP-wrapped)
    """
    model: torch.nn.Module = instantiate(cfg.model)
    model.to(device)

    compile_before = cfg.get("compile", False) and not cfg.get("distributed", {}).get("compile_after_ddp", False)
    compile_after = cfg.get("compile", False) and cfg.get("distributed", {}).get("compile_after_ddp", False)

    # Optional torch.compile BEFORE DDP (default)
    if compile_before:
        compile_mode = cfg.get("compile_mode", "default")
        model = torch.compile(model, mode=compile_mode)

    # Wrap with DDP
    if distributed_strategy == "ddp" and local_rank is not None:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.get("distributed", {}).get("find_unused_parameters", False),
            static_graph=False,
        )

    # Optional torch.compile AFTER DDP (experimental)
    if compile_after:
        compile_mode = cfg.get("compile_mode", "default")
        model = torch.compile(model, mode=compile_mode)

    return model


def save_checkpoint_distributed(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    cfg: DictConfig,
) -> None:
    """Save checkpoint (rank-0 only in distributed mode)."""
    if not is_main_process():
        return  # Only rank 0 saves

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),  # No .module. prefix
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
    }
    torch.save(state, path_obj)


def load_checkpoint_distributed(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: torch.device | str | None = None,
) -> int:
    """Load checkpoint into (possibly wrapped) model."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Load into unwrapped model
    unwrap_model(model).load_state_dict(checkpoint["model"])

    if (
        optimizer is not None
        and "optimizer" in checkpoint
        and checkpoint["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    if (
        scaler is not None
        and "scaler" in checkpoint
        and checkpoint["scaler"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler"])

    return int(checkpoint.get("epoch", 0))


# -----------------------------------------------------------------------------
# Distributed metric synchronization
# -----------------------------------------------------------------------------


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce sum across ranks."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def all_gather_variable_size(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors of potentially different sizes from all ranks.

    Handles the case where ranks have different numbers of samples
    (e.g., when drop_last=False for evaluation).
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    all_sizes = [torch.zeros(1, device=tensor.device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)

    # Pad tensor to max_size
    if tensor.shape[0] < max_size:
        padding = torch.zeros(max_size - tensor.shape[0], *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)

    # Gather padded tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Trim to actual sizes and concatenate
    result = []
    for i, size in enumerate(all_sizes):
        result.append(gathered[i][:size.item()])

    return torch.cat(result, dim=0)
