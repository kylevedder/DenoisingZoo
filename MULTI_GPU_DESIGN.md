# Multi-GPU Training Design Document

This document describes the design for adding multi-GPU training support to DenoisingZoo on a single machine (2-8 GPUs).

## Executive Summary

Add support for training on multiple GPUs within a single node using PyTorch's DistributedDataParallel (DDP) as the primary strategy, with FullyShardedDataParallel (FSDP) as an option for memory-constrained scenarios. The implementation uses `torchrun` as the launch mechanism and requires modifications to the training loop, data loading, checkpointing, and metric synchronization.

## Current State

The codebase is currently single-GPU only:
- `train.py`: Training loop with single device placement
- `helpers.py`: Model/dataloader builders target one device
- `launcher.py`: Launches training via subprocess or Modal
- Checkpoints save single model state_dict
- Metrics computed locally without synchronization

## Design Decisions

### 1. DDP vs FSDP Strategy

**Recommendation: Start with DDP, add FSDP as opt-in for large models**

| Aspect | DDP | FSDP |
|--------|-----|------|
| Memory | Full model replicated per GPU | Sharded params/grads/optimizer |
| Complexity | Simple wrapper | More complex lifecycle |
| Speed | Faster (less communication) | Slower (all-gather/reduce-scatter) |
| Use Case | Model fits in GPU memory | Memory-constrained |

**Memory Analysis for DiT-XL (675M params):**
- FP32: 2.7GB weights, 5.4GB optimizer (AdamW), ~8GB activations
- BF16: ~10GB total, fits comfortably on A100-40GB with DDP
- FSDP only needed if using very large batch sizes or larger models

**Config Surface:**
```yaml
# configs/train.yaml
distributed:
  enabled: false           # Enable multi-GPU
  strategy: ddp            # ddp | fsdp
  fsdp_sharding: full      # full | grad_op | no (FSDP only)
  find_unused_parameters: false
  compile_after_ddp: false # Experimental: compile after wrapping
```

### 2. MeanFlow Loss + JVP Compatibility

The MeanFlow loss uses `torch.func.jvp` for forward-mode automatic differentiation. Analysis of `losses/meanflow_loss.py`:

**JVP Usage Pattern (lines 168-172):**
```python
v_t_mf, jvp_mf = torch.func.jvp(
    model_fn_mf,
    (z_t_mf, t_mf),
    (v_t_mf_tangent, tangent_t),
)
```

**Compatibility Assessment:**
- JVP operates during forward pass
- DDP gradient synchronization happens during backward pass
- These are independent operations → **JVP is compatible with DDP**
- The model is used for ALL samples (line 243), so all parameters participate in backward
- **`find_unused_parameters=False`** is sufficient (default)

**Required Change:** The `set_model()` call receives the DDP-wrapped model, and JVP will work on it directly:
```python
# train.py
criterion.set_model(ddp_model)  # DDP wrapper passed to loss
```

**FSDP Compatibility Warning:**
> **IMPORTANT:** `torch.func.jvp` compatibility with FSDP is NOT validated. FSDP's parameter sharding may interfere with forward-mode AD. **Prohibit FSDP when using MeanFlow loss** until explicitly tested:
> ```python
> if is_meanflow_loss(criterion) and distributed_strategy == "fsdp":
>     raise ValueError("FSDP is not compatible with MeanFlow loss (JVP). Use DDP instead.")
> ```

### 3. torch.compile Integration

Current code optionally compiles the model (train.py:238-242). With DDP:

**Order of Operations (Configurable):**

The optimal order depends on PyTorch version and model. Both approaches have tradeoffs:

```python
# Option A: compile BEFORE DDP (default, recommended for most cases)
model = build_model(cfg, device)
if cfg.get("compile", False) and not cfg.distributed.get("compile_after_ddp", False):
    model = torch.compile(model, mode=cfg.get("compile_mode", "default"))
ddp_model = DDP(model, device_ids=[local_rank], static_graph=False)

# Option B: compile AFTER DDP (experimental, may enable DDP-specific optimizations)
if cfg.get("compile", False) and cfg.distributed.get("compile_after_ddp", False):
    ddp_model = torch.compile(ddp_model, mode=cfg.get("compile_mode", "default"))
```

**Key Points:**
- Default: compile BEFORE DDP wrapping (more stable)
- Use `static_graph=False` (default) for compiled models
- If issues arise, try `compile_after_ddp=true` or compile the `train_step` function instead
- Test both orders for your specific model/PyTorch version

### 4. Gradient Accumulation with DDP

The codebase already supports gradient accumulation (train.py:161-199). With DDP, we must use `no_sync()` to avoid redundant all-reduce operations.

**Critical: Handle the final partial step** when epoch length isn't divisible by accumulation steps:

```python
def train_one_epoch(...):
    total_steps = len(loader)

    for step, batch in enumerate(bar):
        is_last_step = (step + 1) == total_steps
        is_sync_step = ((step + 1) % gradient_accumulation_steps == 0) or is_last_step

        # Skip all-reduce for accumulation steps (but always sync on last step)
        ctx = ddp_model.no_sync() if not is_sync_step else contextlib.nullcontext()
        with ctx:
            loss = compute_loss(model, batch, device, settings, criterion)
            loss = loss / gradient_accumulation_steps

            if settings.use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_sync_step:
            if settings.use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

### 5. Launch Mechanism: torchrun

**Recommendation: Use `torchrun`** (modern standard, supports elastic training)

**Launcher Changes (`launcher.py`):**
```python
def run_local_distributed(cfg_overrides: list[str], num_gpus: int):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--standalone",
        "train.py",
        *cfg_overrides,
    ]
    subprocess.run(cmd, check=True)
```

**Usage:**
```bash
# Direct torchrun
torchrun --nproc_per_node=4 train.py run_name=experiment dataloaders=cifar10

# Via launcher (proposed)
python launcher.py --gpus 4 run_name=experiment dataloaders=cifar10
```

**Environment Variables (set by torchrun):**
- `LOCAL_RANK`: GPU index on this node (0-7)
- `RANK`: Global rank across all nodes
- `WORLD_SIZE`: Total number of processes

## Implementation Details

### 6. Process Group Initialization

New function in `helpers.py`. **Gated by config** to prevent accidental multi-process runs:

```python
import os
import torch.distributed as dist

def setup_distributed(cfg: DictConfig) -> tuple[int, int, int]:
    """Initialize distributed training.

    Args:
        cfg: Hydra config with distributed.enabled flag

    Returns:
        (local_rank, global_rank, world_size)
    """
    # Check if distributed is enabled in config
    distributed_enabled = cfg.get("distributed", {}).get("enabled", False)

    if not dist.is_available():
        return 0, 0, 1

    # Check if launched via torchrun
    if "LOCAL_RANK" not in os.environ:
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

    # Set device before init
    torch.cuda.set_device(local_rank)

    # Initialize NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return local_rank, global_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
```

### 7. Per-Rank RNG Seeding

Ensure each rank gets different random augmentations/dropout:

```python
def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed RNG with rank offset for distributed training.

    Args:
        seed: Base seed from config
        rank: Global rank (0 for single-GPU)
    """
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

    # For reproducibility in data loading workers
    def worker_init_fn(worker_id: int):
        worker_seed = effective_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn
```

### 8. Model Wrapping

Modify `build_model()` in `helpers.py`:

```python
def build_model(
    cfg: DictConfig,
    device: torch.device,
    local_rank: int | None = None,
    distributed_strategy: str | None = None,
) -> nn.Module:
    model: nn.Module = instantiate(cfg.model)
    model.to(device)

    compile_before = cfg.get("compile", False) and not cfg.get("distributed", {}).get("compile_after_ddp", False)
    compile_after = cfg.get("compile", False) and cfg.get("distributed", {}).get("compile_after_ddp", False)

    # Optional torch.compile BEFORE DDP (default)
    if compile_before:
        model = torch.compile(model, mode=cfg.get("compile_mode", "default"))

    # Wrap with DDP or FSDP
    if distributed_strategy == "ddp" and local_rank is not None:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.get("distributed", {}).get("find_unused_parameters", False),
            static_graph=False,
        )
    elif distributed_strategy == "fsdp" and local_rank is not None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no": ShardingStrategy.NO_SHARD,
        }
        sharding = strategy_map.get(cfg.get("distributed", {}).get("fsdp_sharding", "full"))
        model = FSDP(model, sharding_strategy=sharding, device_id=local_rank)

    # Optional torch.compile AFTER DDP (experimental)
    if compile_after:
        model = torch.compile(model, mode=cfg.get("compile_mode", "default"))

    return model
```

### 9. DataLoader with DistributedSampler

Modify `build_dataloader_from_config()` in `helpers.py`:

```python
from torch.utils.data.distributed import DistributedSampler

def build_dataloader_from_config(
    node: DictConfig,
    device: torch.device,
    distributed: bool = False,
    is_train: bool = True,
    worker_init_fn: Callable | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset: BaseDataset = instantiate(node.dataset)
    adapter = DictDatasetAdapter(dataset)

    sampler = None
    shuffle = node.get("shuffle", True) if is_train else False

    if distributed:
        sampler = DistributedSampler(
            adapter,
            shuffle=shuffle,
            drop_last=True if is_train else False,  # Train: drop for even batches; Eval: keep all
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
```

**Training Loop Update:**
```python
for epoch in range(start_epoch, epochs + 1):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)  # Required for proper shuffling
    # Note: eval_sampler also needs set_epoch if shuffle=True (but eval should use shuffle=False)

    avg_loss, global_step = train_one_epoch(...)
```

**Tradeoffs for `drop_last`:**
| Setting | Train | Eval |
|---------|-------|------|
| `drop_last=True` | Even batch sizes, no special handling | May drop samples, affects metrics |
| `drop_last=False` | Uneven batches, need padded gather | All samples included, accurate metrics |

**Recommendation:** `drop_last=True` for train (simplicity), `drop_last=False` for eval with padded gather.

### 10. Checkpointing

Save only from rank 0, recursively unwrap model to handle DDP+compile:

```python
def unwrap_model(model: nn.Module) -> nn.Module:
    """Recursively unwrap DDP/FSDP/compiled model to get base module."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):  # DDP, FSDP
            unwrapped = unwrapped.module
        elif hasattr(unwrapped, "_orig_mod"):  # torch.compile
            unwrapped = unwrapped._orig_mod
        else:
            break
    return unwrapped


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    cfg: DictConfig,
) -> None:
    if not is_main_process():
        return  # Only rank 0 saves

    state = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),  # No .module. prefix
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load into unwrapped model
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt.get("epoch", 0)
```

**FSDP Checkpointing (if using FSDP):**
```python
def save_fsdp_checkpoint(path: Path, model, optimizer=None):
    from torch.distributed.fsdp import (
        StateDictType, FullStateDictConfig, FullOptimStateDictConfig
    )

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg, optim_cfg):
        state = {"model": model.state_dict()}
        if optimizer is not None:
            state["optimizer"] = FSDP.optim_state_dict(model, optimizer)

        # Barrier to ensure all ranks finish before rank 0 saves
        dist.barrier()

        if is_main_process():
            torch.save(state, path)
```

### 11. Metric Synchronization

Synchronize loss and evaluation metrics across ranks. **Fixed: reduce both sum AND count for correct averaging:**

```python
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
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
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
```

**Training Loss Sync (CORRECTED):**
```python
# In train_one_epoch
# Track sum of losses and count of samples
loss_sum = torch.tensor(total_loss, device=device)
sample_count = torch.tensor(total_samples, device=device, dtype=torch.float32)

if dist.is_initialized():
    # Reduce BOTH sum and count, then divide
    loss_sum = reduce_sum(loss_sum)
    sample_count = reduce_sum(sample_count)

avg_loss = (loss_sum / sample_count).item()
```

**Energy Distance Eval:**
```python
def evaluate_epoch_energy_distance(model, eval_loader, device, solver):
    predictions = []
    targets = []

    for batch in eval_loader:
        # ... compute predictions ...
        predictions.append(y_pred)
        targets.append(y_true)

    preds = torch.cat(predictions, dim=0)
    targs = torch.cat(targets, dim=0)

    # Gather from all ranks (handles variable sizes)
    if dist.is_initialized():
        preds = all_gather_variable_size(preds)
        targs = all_gather_variable_size(targs)

    # Only rank 0 computes and logs
    if is_main_process():
        ed = compute_energy_distance_u_statistic(preds, targs)
        return ed
    return None
```

### 12. Trackio Logging

Only log from rank 0:

```python
def log_trackio(metrics, enabled, device, monitor, step):
    if not enabled or not is_main_process():
        return
    # ... existing logging code ...
```

### 13. Batch Size Semantics and LR Scaling

**Important:** With distributed training, `batch_size` in config is **per-GPU**:

```
Effective Global Batch Size = batch_size × world_size × gradient_accumulation_steps
```

**Example:**
- `batch_size: 64`, 4 GPUs, `gradient_accumulation_steps: 2`
- Effective batch size: 64 × 4 × 2 = 512

**Learning Rate Scaling:**
When scaling to more GPUs, consider linear LR scaling:
```python
# In train.py
effective_batch_size = cfg.batch_size * world_size * cfg.gradient_accumulation_steps
base_batch_size = 256  # Your reference batch size
lr_scale = effective_batch_size / base_batch_size

# Apply scaling (optional, may need warmup)
for param_group in optimizer.param_groups:
    param_group['lr'] *= lr_scale
```

**Recommendation:** Start without LR scaling; add if loss diverges or converges too slowly.

### 14. Training Script Changes

Main changes to `train.py`:

```python
@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # 1. Setup distributed (before any CUDA ops) - gated by config
    local_rank, global_rank, world_size = setup_distributed(cfg)
    distributed = world_size > 1

    # 2. Device setup
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.device)

    # 3. Seed with rank offset
    worker_init_fn = seed_everything(cfg.get("seed", 42), global_rank)

    # 4. Build model with distributed wrapper
    strategy = cfg.get("distributed", {}).get("strategy", "ddp") if distributed else None

    # Check FSDP + MeanFlow incompatibility
    if strategy == "fsdp" and cfg.loss.get("_target_", "").endswith("MeanFlowLoss"):
        raise ValueError("FSDP is not compatible with MeanFlow loss (JVP). Use DDP instead.")

    model = build_model(cfg, device, local_rank if distributed else None, strategy)

    # 5. Build dataloaders with samplers
    train_loader, train_sampler = build_dataloader_from_config(
        cfg.dataloaders.train, device, distributed, is_train=True, worker_init_fn=worker_init_fn
    )
    eval_loader, _ = build_dataloader_from_config(
        cfg.dataloaders.eval, device, distributed, is_train=False, worker_init_fn=worker_init_fn
    )

    # 6. Set model for MeanFlow loss
    if is_meanflow_loss(criterion):
        criterion.set_model(model)  # Pass DDP-wrapped model

    # ... rest of training loop with:
    # - train_sampler.set_epoch(epoch) before each epoch
    # - model.no_sync() context for gradient accumulation (with tail flush)
    # - Metric synchronization (reduce sum AND count)
    # - Rank-0-only checkpointing and logging

    # Cleanup
    cleanup_distributed()
```

## File Changes Summary

| File | Changes |
|------|---------|
| `train.py` | Add distributed setup, sampler.set_epoch(), no_sync() with tail flush, metric sync |
| `helpers.py` | Add setup_distributed(), seed_everything(), modify build_model(), build_dataloader_from_config() |
| `launcher.py` | Add torchrun launch path with `--gpus` argument |
| `configs/train.yaml` | Add `distributed` config group |
| `losses/meanflow_loss.py` | No changes needed (JVP compatible with DDP; FSDP blocked) |

## Configuration

New config options:

```yaml
# configs/train.yaml
distributed:
  enabled: false              # Enable multi-GPU (must be true when using torchrun)
  strategy: ddp               # ddp | fsdp
  fsdp_sharding: full         # full | grad_op | no (FSDP only)
  find_unused_parameters: false
  compile_after_ddp: false    # Experimental: compile after DDP wrapping

seed: 42  # Base seed (offset by rank for distributed)
```

## Usage Examples

```bash
# Single GPU (unchanged)
python launcher.py run_name=exp1 dataloaders=cifar10 model=unet

# Multi-GPU via launcher
python launcher.py --gpus 4 run_name=exp1 dataloaders=cifar10 model=unet

# Multi-GPU via torchrun directly
torchrun --nproc_per_node=4 train.py run_name=exp1 dataloaders=cifar10 model=unet distributed.enabled=true

# FSDP for large models (NOT compatible with MeanFlow)
torchrun --nproc_per_node=8 train.py run_name=dit_xl distributed.enabled=true distributed.strategy=fsdp model=dit_xl loss=mse
```

**Effective Batch Size Examples:**
```bash
# 4 GPUs, batch_size=64, grad_accum=1 → effective batch = 256
torchrun --nproc_per_node=4 train.py ... dataloaders.train.batch_size=64

# 4 GPUs, batch_size=32, grad_accum=4 → effective batch = 512
torchrun --nproc_per_node=4 train.py ... dataloaders.train.batch_size=32 gradient_accumulation_steps=4
```

## Testing Plan

1. **Correctness Tests:**
   - Compare 1-GPU vs 2-GPU training loss curves (should match with same effective batch size)
   - Verify checkpoint loading/saving across ranks
   - Test MeanFlow loss gradient computation with DDP
   - Verify per-rank seeding produces different augmentations

2. **Scaling Tests:**
   - Measure throughput (samples/sec) with 1, 2, 4, 8 GPUs
   - Verify near-linear scaling (expect ~85-95% efficiency)

3. **Edge Cases:**
   - Resume from checkpoint with different GPU count
   - Gradient accumulation + DDP + mixed precision (verify tail flush)
   - torch.compile + DDP (both orderings)
   - Variable batch sizes in eval gather

4. **Error Handling:**
   - Verify FSDP + MeanFlow raises clear error
   - Verify torchrun without distributed.enabled=true raises clear error

## Future Extensions

- **Multi-node training**: Add `--nnodes` and `--node_rank` support to launcher
- **Gradient checkpointing**: Reduce activation memory for larger batch sizes
- **ZeRO optimizer**: Memory-efficient optimizer via DeepSpeed or FSDP
- **FSDP + JVP**: Validate and enable once PyTorch adds forward-mode AD support for FSDP

---

*Document prepared with technical input from Codex (OpenAI gpt-5.2-codex). All issues from initial review have been addressed.*
