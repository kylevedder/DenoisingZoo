"""
Modal application for remote training on NVIDIA GPUs.
"""

import os
import sys
import subprocess
from pathlib import Path
import modal


# Enable Modal client output early so image build logs are streamed locally
modal.enable_output()

app = modal.App("denoisingzoo-training")

# Define what to exclude when copying the project
EXCLUDE_PATTERNS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "outputs",
    "data",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.egg-info",
}


def should_ignore(path: Path) -> bool:
    """Filter function for add_local_dir to exclude unwanted files."""
    parts = path.parts
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            # Glob pattern for extensions
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in parts or path.name == pattern:
            return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "hydra-core>=1.3,<2.0",
        "omegaconf>=2.3,<3.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "gdown>=5.2",
        "trackio>=0.2",
        "clean-fid>=0.1",
    )
    .run_commands(
        # Install CUDA-enabled torch/vision
        "pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6,<2.7' 'torchvision>=0.21,<0.22'"
    )
    .run_commands("mkdir -p /root/.cache/torch/hub/checkpoints")
    # Copy entire project, excluding venv/git/etc
    .add_local_dir(
        ".",
        "/root/app",
        copy=True,
        ignore=should_ignore,
    )
)


# Shared volume for datasets and trackio logs
training_volume = modal.Volume.from_name("training-data", create_if_missing=True)

# Minimal image for utility functions (no torch, just Python stdlib)
util_image = modal.Image.debian_slim(python_version="3.12")


def _setup_modal_environment(app_dir: Path) -> None:
    """Common setup for Modal training environment."""
    os.chdir(app_dir)

    # Create data directories
    Path("/data").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # Set trackio storage to persistent volume
    trackio_dir = Path("/data/trackio")
    trackio_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(trackio_dir)
    print(f"[modal] Trackio storage: {trackio_dir}")

    # Extract CelebA if needed
    try:
        data_dir = Path("/data")
        archive = data_dir / "celeba.tar.gz"
        extracted = data_dir / "celeba"
        if archive.exists() and not extracted.exists():
            print("[modal] Found /data/celeba.tar.gz in volume, extracting...")
            subprocess.run(
                ["tar", "-xzf", str(archive), "-C", str(data_dir)],
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[modal] Warning: dataset extraction skipped: {exc}")

    # Setup checkpoint symlink
    import shutil
    ckpt_volume_dir = Path("/data/ckpts")
    ckpt_volume_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_local_dir = outputs_dir / "ckpts"
    if ckpt_local_dir.is_symlink():
        if ckpt_local_dir.resolve() != ckpt_volume_dir.resolve():
            ckpt_local_dir.unlink()
    elif ckpt_local_dir.exists():
        if ckpt_local_dir.is_dir():
            shutil.rmtree(ckpt_local_dir)
        else:
            ckpt_local_dir.unlink()
    if not ckpt_local_dir.exists() and not ckpt_local_dir.is_symlink():
        ckpt_local_dir.symlink_to(ckpt_volume_dir)
    print(f"[modal] Checkpoints: {ckpt_local_dir} -> {ckpt_volume_dir}")


def _prepare_training_args(extra_args: list[str]) -> list[str]:
    """Prepare training arguments with Modal-specific defaults."""
    args = list(extra_args)

    def _has_prefix(prefix: str) -> bool:
        return any(str(a).startswith(prefix) for a in args)

    using_cifar = any("dataloaders=cifar10" in str(a) for a in args)

    if not using_cifar:
        if not _has_prefix("dataloaders.train.dataset.root="):
            args.append("dataloaders.train.dataset.root=/data/celeba")
        if not _has_prefix("dataloaders.eval.dataset.root="):
            args.append("dataloaders.eval.dataset.root=/data/celeba")

    if not any(a.startswith("resume=") for a in args):
        args.append("resume=true")
        print("[modal] Auto-enabled resume=true for retry resilience")

    return args


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=43200,  # 12 hours max per attempt
    volumes={"/data": training_volume},
    retries=modal.Retries(
        max_retries=5,           # Up to 5 retries (60 hours total worst case)
        initial_delay=30.0,      # 30 second delay before retry
        backoff_coefficient=1.0, # Fixed delay (not exponential)
    ),
)
def train_on_modal(extra_args: list[str]) -> None:
    """Run single-GPU training on Modal with NVIDIA GPU."""
    app_dir = Path("/root/app")
    _setup_modal_environment(app_dir)

    # Verify GPU availability
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    args = _prepare_training_args(extra_args)
    cmd = ["python", "train.py", "device=cuda", *args]
    print("Running command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    training_volume.commit()
    print("[modal] Final volume commit - training complete")


@app.function(
    image=image,
    gpu="A100-40GB:2",  # Request 2 A100 GPUs
    timeout=43200,  # 12 hours max per attempt
    volumes={"/data": training_volume},
    retries=modal.Retries(
        max_retries=5,
        initial_delay=30.0,
        backoff_coefficient=1.0,
    ),
)
def train_on_modal_multigpu(extra_args: list[str], num_gpus: int = 2) -> None:
    """Run multi-GPU training on Modal using DDP with torchrun.

    Args:
        extra_args: Hydra config overrides
        num_gpus: Number of GPUs to use (must match function GPU allocation)
    """
    app_dir = Path("/root/app")
    _setup_modal_environment(app_dir)

    # Verify GPU availability
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        if device_count < num_gpus:
            raise RuntimeError(f"Requested {num_gpus} GPUs but only {device_count} available")

    args = _prepare_training_args(extra_args)

    # Enable distributed training
    if not any(a.startswith("distributed.enabled=") for a in args):
        args.append("distributed.enabled=true")

    # Use torchrun for multi-GPU training
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--standalone",
        "train_distributed.py",
        "device=cuda",
        *args,
    ]
    print("Running command:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    training_volume.commit()
    print("[modal] Final volume commit - multi-GPU training complete")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours for FID (50k samples takes a while)
    volumes={"/data": training_volume},
)
def compute_fid_on_modal(
    checkpoint_path: str,
    num_samples: int = 50000,
    use_solver: bool = False,
    solver_steps: int = 50,
    batch_size: int = 256,
) -> dict:
    """Compute FID score on Modal A100 GPU.

    Args:
        checkpoint_path: Relative path in /data/ckpts (e.g., "unet/archive/cifar10_ratio0_20ep_epoch_0020.pt")
        num_samples: Number of samples to generate (default 50000 for CIFAR-10 FID)
        use_solver: If True, use ODE solver; else use MeanFlow 1-step
        solver_steps: Number of solver steps (only if use_solver=True)
        batch_size: Batch size for generation

    Returns:
        Dict with FID score and metadata
    """
    import tempfile
    from pathlib import Path

    import torch
    from cleanfid import fid
    from torchvision.utils import save_image
    from tqdm import tqdm

    app_dir = Path("/root/app")
    os.chdir(app_dir)

    # Import model and helpers
    sys.path.insert(0, str(app_dir))
    from models.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input
    from solvers.euler_solver import EulerSolver

    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name()})")

    # Load checkpoint
    ckpt_full_path = Path("/data/ckpts") / checkpoint_path
    if not ckpt_full_path.exists():
        return {"error": f"Checkpoint not found: {ckpt_full_path}"}

    print(f"Loading checkpoint: {ckpt_full_path}")
    checkpoint = torch.load(ckpt_full_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Create model
    model = UNet(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 3),
        base_channels=model_config.get("base_channels", 128),
        time_channels=model_config.get("time_channels", 2),
        channel_mult=model_config.get("channel_mult", [1, 2, 2, 2]),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        attention_resolutions=model_config.get("attention_resolutions", [16]),
        dropout=model_config.get("dropout", 0.1),
        num_heads=model_config.get("num_heads", 4),
        input_resolution=model_config.get("input_resolution", 32),
        use_separate_time_embeds=model_config.get("use_separate_time_embeds", True),
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Model loaded from epoch {config.get('epoch', 'unknown')}")

    # Sample shape
    sample_shape = (
        model_config.get("in_channels", 3),
        model_config.get("input_resolution", 32),
        model_config.get("input_resolution", 32),
    )
    print(f"Sample shape: {sample_shape}")

    # Create solver if needed
    solver = None
    if use_solver:
        solver = EulerSolver(model, num_steps=solver_steps)

    # Generate samples to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        sample_idx = 0

        method = "solver" if use_solver else "meanflow_1step"
        print(f"Generating {num_samples} samples using {method}...")

        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc="Generating")
            while sample_idx < num_samples:
                current_batch = min(batch_size, num_samples - sample_idx)
                noise = torch.randn(current_batch, *sample_shape, device=device)

                if use_solver:
                    # Solver-based sampling
                    result = solver.solve(noise)
                    samples = result.final_state
                else:
                    # MeanFlow 1-step: x1 = x0 + v(x0, r=0, t=1)
                    B = noise.shape[0]
                    r_tensor = torch.zeros(B, 1, device=device, dtype=noise.dtype)
                    t_tensor = torch.ones(B, 1, device=device, dtype=noise.dtype)
                    time_input = make_time_input(t_tensor, r=r_tensor)
                    unified = make_unified_flow_matching_input(noise, time_input)
                    v = model(unified)
                    samples = noise + v

                # Clamp and save
                samples = samples.clamp(-1, 1)
                for i in range(current_batch):
                    img = (samples[i] + 1) / 2  # [-1, 1] -> [0, 1]
                    save_image(img, output_dir / f"{sample_idx:05d}.png")
                    sample_idx += 1
                pbar.update(current_batch)
            pbar.close()

        # Compute FID
        print("Computing FID against CIFAR-10 train set...")
        fid_score = fid.compute_fid(
            fdir1=str(output_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="train",
            mode="clean",
            device=device,
            verbose=True,
        )

    print(f"\nFID Score ({method}): {fid_score:.2f}")

    return {
        "fid": fid_score,
        "checkpoint": checkpoint_path,
        "num_samples": num_samples,
        "method": method,
        "solver_steps": solver_steps if use_solver else None,
        "epoch": config.get("epoch", "unknown"),
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,  # 1 hour max for benchmarks
    volumes={"/data": training_volume},
)
def run_benchmark(script_name: str, script_args: str = "") -> str:
    """Run a benchmark script on Modal with NVIDIA GPU.

    Args:
        script_name: Name of the script in scripts/ directory (e.g., "benchmark_meanflow_loss.py")
        script_args: Additional arguments to pass to the script (e.g., "--dtype bfloat16")

    Returns:
        Output from the benchmark script
    """
    app_dir = Path("/root/app")
    os.chdir(app_dir)

    # Create outputs directory for results
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Verify GPU availability
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name()}")

    script_path = app_dir / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Set PYTHONPATH so imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_dir)

    cmd = ["python", str(script_path)]
    if script_args:
        cmd.extend(script_args.split())
    print(f"Running benchmark: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    output = result.stdout
    if result.stderr:
        output += "\n\nSTDERR:\n" + result.stderr

    print(output)

    # Copy any output files to volume for persistence
    outputs_dir = Path("outputs")
    volume_outputs = Path("/data/benchmark_outputs")
    volume_outputs.mkdir(parents=True, exist_ok=True)

    for json_file in outputs_dir.glob("benchmark_*.json"):
        import shutil
        dest = volume_outputs / json_file.name
        shutil.copy(json_file, dest)
        print(f"Saved results to volume: {dest}")

    training_volume.commit()

    return output


@app.function(
    image=util_image,
    volumes={"/data": training_volume},
)
def list_trackio_runs() -> list[str]:
    """List trackio runs stored in the Modal volume."""
    trackio_dir = Path("/data/trackio")
    if not trackio_dir.exists():
        return []

    # Find all .db files (each is a project)
    db_files = list(trackio_dir.glob("*.db"))
    runs = []
    for db_file in db_files:
        project = db_file.stem
        # Query the database for run names
        import sqlite3

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Check all tables for run names
        for table in ["configs", "metrics", "system_metrics"]:
            try:
                cursor.execute(f"SELECT DISTINCT run_name FROM {table}")
                for row in cursor.fetchall():
                    run_info = f"{project}/{row[0]}"
                    if run_info not in runs:
                        runs.append(run_info)
            except sqlite3.OperationalError:
                pass
        conn.close()
    return sorted(runs)


@app.function(
    image=util_image,
    volumes={"/data": training_volume},
)
def get_trackio_db(project: str = "denoising-zoo") -> bytes | None:
    """Download trackio database for a project from Modal volume."""
    db_path = Path(f"/data/trackio/{project}.db")
    if not db_path.exists():
        return None
    return db_path.read_bytes()


@app.function(
    image=util_image,
    volumes={"/data": training_volume},
)
def list_checkpoints() -> list[dict]:
    """List checkpoints stored in the Modal volume.

    Returns list of dicts with keys: path, size_mb, modified
    """
    ckpt_dir = Path("/data/ckpts")
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for pt_file in ckpt_dir.rglob("*.pt"):
        stat = pt_file.stat()
        checkpoints.append({
            "path": str(pt_file.relative_to(ckpt_dir)),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
        })
    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return checkpoints


def _safe_ckpt_path(base: Path, user_path: str) -> Path | None:
    """Safely resolve a user-provided path within a base directory.

    Returns the resolved path if it's within the base directory, None otherwise.
    Prevents path traversal attacks (e.g., "../../../etc/passwd").
    """
    # Reject absolute paths
    if Path(user_path).is_absolute():
        return None
    # Resolve the full path
    full_path = (base / user_path).resolve()
    # Ensure it's within the base directory
    try:
        full_path.relative_to(base.resolve())
        return full_path
    except ValueError:
        return None


@app.function(
    image=util_image,
    volumes={"/data": training_volume},
)
def get_checkpoint(path: str) -> bytes | None:
    """Download a checkpoint file from Modal volume.

    Args:
        path: Relative path within /data/ckpts (e.g., "unet/last.pt")

    Returns:
        Checkpoint bytes or None if not found.
        Returns None for invalid paths (absolute or traversal attempts).

    Note:
        Large checkpoints (500MB+) may hit Modal's response size limits.
        For very large files, consider using Modal volumes directly.
    """
    base = Path("/data/ckpts")
    ckpt_path = _safe_ckpt_path(base, path)
    if ckpt_path is None:
        return None
    if not ckpt_path.exists() or not ckpt_path.is_file():
        return None
    return ckpt_path.read_bytes()


def download_checkpoint_from_modal(remote_path: str, local_path: str | None = None) -> Path | None:
    """Download a checkpoint from Modal volume to local storage.

    Args:
        remote_path: Relative path in Modal volume (e.g., "unet/last.pt")
        local_path: Local destination path. If None, uses outputs/ckpts/<remote_path>

    Returns:
        Path to downloaded checkpoint or None if not found

    Note:
        Large checkpoints (500MB+) may hit Modal's response size limits.
        For very large files, consider mounting the Modal volume locally.
    """
    # Validate remote path (reject absolute or traversal paths)
    if Path(remote_path).is_absolute() or ".." in Path(remote_path).parts:
        print(f"Invalid remote path: {remote_path}")
        return None

    if local_path is None:
        local_dest = Path("outputs/ckpts") / remote_path
    else:
        local_dest = Path(local_path)

    local_dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint '{remote_path}' from Modal...")
    with app.run():
        ckpt_bytes = get_checkpoint.remote(remote_path)
        if ckpt_bytes is None:
            print(f"Checkpoint not found or invalid path: {remote_path}")
            return None

        local_dest.write_bytes(ckpt_bytes)
        size_mb = len(ckpt_bytes) / (1024 * 1024)
        print(f"Downloaded {size_mb:.1f} MB to {local_dest}")
        return local_dest


def run_modal_training(extra_args: list[str]) -> None:
    """Helper function to run single-GPU Modal training from launcher.py."""
    print("Starting Modal training job (single GPU)...")
    with app.run():
        train_on_modal.remote(extra_args)
    print("Modal training job completed!")


def run_modal_training_multigpu(extra_args: list[str], num_gpus: int = 2) -> None:
    """Helper function to run multi-GPU Modal training from launcher.py."""
    print(f"Starting Modal training job ({num_gpus} GPUs)...")
    with app.run():
        train_on_modal_multigpu.remote(extra_args, num_gpus)
    print("Modal multi-GPU training job completed!")


def sync_trackio_from_modal(project: str = "denoising-zoo") -> None:
    """Download trackio database from Modal volume to local storage."""
    local_trackio_dir = Path.home() / ".cache" / "huggingface" / "trackio"
    local_trackio_dir.mkdir(parents=True, exist_ok=True)
    local_db_path = local_trackio_dir / f"{project}.db"

    print(f"Syncing trackio data for project '{project}' from Modal...")
    with app.run():
        # First list what's available
        runs = list_trackio_runs.remote()
        if runs:
            print(f"Found {len(runs)} run(s) in Modal volume:")
            for run in runs:
                print(f"  - {run}")
        else:
            print("No runs found in Modal volume")
            return

        # Download the database
        db_bytes = get_trackio_db.remote(project)
        if db_bytes is None:
            print(f"No trackio database found for project '{project}'")
            return

        # Merge with local database if it exists
        if local_db_path.exists():
            print(f"Merging with existing local database at {local_db_path}")
            _merge_trackio_dbs(local_db_path, db_bytes)
        else:
            print(f"Writing new database to {local_db_path}")
            local_db_path.write_bytes(db_bytes)

    print(f"Trackio data synced to {local_db_path}")
    print(f"View with: trackio show --project {project}")


def _merge_trackio_dbs(local_path: Path, remote_bytes: bytes) -> None:
    """Merge remote trackio database into local one."""
    import sqlite3
    import tempfile

    # Write remote DB to temp file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        f.write(remote_bytes)
        remote_path = f.name

    try:
        local_conn = sqlite3.connect(local_path)
        local_conn.execute(f"ATTACH DATABASE '{remote_path}' AS remote")

        # Merge each table
        for table in ["configs", "metrics", "system_metrics"]:
            try:
                # Get columns from remote table
                cursor = local_conn.execute(f"PRAGMA remote.table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                if not columns:
                    continue

                cols_str = ", ".join(columns)
                # Insert or replace from remote
                local_conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({cols_str}) "
                    f"SELECT {cols_str} FROM remote.{table}"
                )
            except sqlite3.OperationalError as e:
                print(f"  Warning: could not merge table {table}: {e}")

        local_conn.commit()
        local_conn.close()
    finally:
        Path(remote_path).unlink()


def run_fid_evaluation(
    checkpoint_path: str,
    num_samples: int = 50000,
    use_solver: bool = False,
    solver_steps: int = 50,
) -> dict:
    """Run FID evaluation on Modal and return results."""
    print(f"Computing FID for checkpoint: {checkpoint_path}")
    print(f"  num_samples: {num_samples}")
    print(f"  method: {'solver' if use_solver else 'meanflow_1step'}")
    if use_solver:
        print(f"  solver_steps: {solver_steps}")

    with app.run():
        result = compute_fid_on_modal.remote(
            checkpoint_path=checkpoint_path,
            num_samples=num_samples,
            use_solver=use_solver,
            solver_steps=solver_steps,
        )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nResults:")
        print(f"  FID: {result['fid']:.2f}")
        print(f"  Checkpoint: {result['checkpoint']}")
        print(f"  Epoch: {result['epoch']}")
        print(f"  Method: {result['method']}")

    return result


def _print_usage():
    print("Usage: python scripts/modal_app.py <command> [args]")
    print()
    print("Commands:")
    print("  list              List trackio runs in Modal volume")
    print("  sync [project]    Sync trackio data to local (default: denoising-zoo)")
    print("  ckpts             List checkpoints in Modal volume")
    print("  download <path>   Download checkpoint from Modal (e.g., unet/last.pt)")
    print("  multigpu <args>   Run multi-GPU (2x A100) training on Modal")
    print("  benchmark <script> Run a benchmark script on Modal A100")
    print("  fid <ckpt> [--samples N] [--solver] [--steps N]  Compute FID on Modal")
    print("  <hydra args>      Run single-GPU training on Modal with given arguments")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        # Sync trackio data: python scripts/modal_app.py sync [project]
        project = sys.argv[2] if len(sys.argv) > 2 else "denoising-zoo"
        sync_trackio_from_modal(project)
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        # List trackio runs: python scripts/modal_app.py list
        with app.run():
            runs = list_trackio_runs.remote()
            if runs:
                print("Trackio runs in Modal volume:")
                for run in runs:
                    print(f"  - {run}")
            else:
                print("No runs found in Modal volume")
    elif len(sys.argv) > 1 and sys.argv[1] == "ckpts":
        # List checkpoints: python scripts/modal_app.py ckpts
        from datetime import datetime
        with app.run():
            ckpts = list_checkpoints.remote()
            if ckpts:
                print("Checkpoints in Modal volume:")
                for ckpt in ckpts:
                    mtime = datetime.fromtimestamp(ckpt["modified"]).strftime("%Y-%m-%d %H:%M")
                    print(f"  {ckpt['path']:50} {ckpt['size_mb']:>8.1f} MB  {mtime}")
            else:
                print("No checkpoints found in Modal volume")
    elif len(sys.argv) > 1 and sys.argv[1] == "download":
        # Download checkpoint: python scripts/modal_app.py download <path> [local_path]
        if len(sys.argv) < 3:
            print("Usage: python scripts/modal_app.py download <path> [local_path]")
            print("  path: Relative path in Modal volume (e.g., unet/last.pt)")
            sys.exit(1)
        remote_path = sys.argv[2]
        local_path = sys.argv[3] if len(sys.argv) > 3 else None
        download_checkpoint_from_modal(remote_path, local_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "multigpu":
        # Multi-GPU training: python scripts/modal_app.py multigpu <hydra args>
        if len(sys.argv) < 3:
            print("Usage: python scripts/modal_app.py multigpu <hydra args>")
            print("  Runs training on 2x A100 GPUs using DDP")
            sys.exit(1)
        extra_args = sys.argv[2:]
        run_modal_training_multigpu(extra_args, num_gpus=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark script: python scripts/modal_app.py benchmark <script_name> [script_args...]
        if len(sys.argv) < 3:
            print("Usage: python scripts/modal_app.py benchmark <script_name> [script_args...]")
            print("  script_name: Name of script in scripts/ (e.g., benchmark_meanflow_loss.py)")
            print("  script_args: Additional args to pass to script (e.g., --dtype bfloat16)")
            sys.exit(1)
        script_name = sys.argv[2]
        script_args = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        print(f"Running benchmark script: {script_name} {script_args}")
        with app.run():
            output = run_benchmark.remote(script_name, script_args)
            print("\n" + "="*60)
            print("BENCHMARK COMPLETE")
            print("="*60)
    elif len(sys.argv) > 1 and sys.argv[1] == "fid":
        # FID evaluation: python scripts/modal_app.py fid <checkpoint> [--samples N] [--solver] [--steps N]
        if len(sys.argv) < 3:
            print("Usage: python scripts/modal_app.py fid <checkpoint> [--samples N] [--solver] [--steps N]")
            print("  checkpoint: Path in Modal volume (e.g., unet/archive/cifar10_ratio0_20ep_epoch_0020.pt)")
            print("  --samples N: Number of samples (default: 50000)")
            print("  --solver: Use ODE solver instead of MeanFlow 1-step")
            print("  --steps N: Solver steps (default: 50, only with --solver)")
            sys.exit(1)
        checkpoint = sys.argv[2]
        num_samples = 50000
        use_solver = False
        solver_steps = 50
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--samples" and i + 1 < len(sys.argv):
                num_samples = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--solver":
                use_solver = True
                i += 1
            elif sys.argv[i] == "--steps" and i + 1 < len(sys.argv):
                solver_steps = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        run_fid_evaluation(checkpoint, num_samples, use_solver, solver_steps)
    elif len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        _print_usage()
    elif len(sys.argv) == 1:
        _print_usage()
    else:
        # Default: run training with given args
        extra_args = sys.argv[1:]
        run_modal_training(extra_args)


@app.local_entrypoint()
def main(*extra_args: str) -> None:
    """CLI entrypoint so `modal run scripts/modal_app.py -- ...` streams logs."""
    train_on_modal.remote(list(extra_args))
