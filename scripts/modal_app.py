"""
Modal application for remote training on NVIDIA GPUs.
"""

import os
import sys
import subprocess
import threading
import time
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


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,  # 4 hours for longer training runs
    volumes={"/data": training_volume},
)
def train_on_modal(extra_args: list[str]) -> None:
    """Run training on Modal with NVIDIA GPU."""
    app_dir = Path("/root/app")
    os.chdir(app_dir)

    # Create data directories
    Path("/data").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # Set trackio storage to persistent volume so logs survive container shutdown
    trackio_dir = Path("/data/trackio")
    trackio_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(trackio_dir)
    print(f"[modal] Trackio storage: {trackio_dir}")

    # If a pre-uploaded archive exists in the persisted volume, extract it
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

    # Verify GPU availability
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Persist checkpoints to volume via symlink
    import shutil
    ckpt_volume_dir = Path("/data/ckpts")
    ckpt_volume_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_local_dir = outputs_dir / "ckpts"
    # Handle existing path: remove if it's not a correct symlink
    if ckpt_local_dir.is_symlink():
        # Check if symlink points to the right place
        if ckpt_local_dir.resolve() != ckpt_volume_dir.resolve():
            ckpt_local_dir.unlink()
    elif ckpt_local_dir.exists():
        # It's a file or directory, remove it
        if ckpt_local_dir.is_dir():
            shutil.rmtree(ckpt_local_dir)
        else:
            ckpt_local_dir.unlink()
    # Create symlink if it doesn't exist
    if not ckpt_local_dir.exists() and not ckpt_local_dir.is_symlink():
        ckpt_local_dir.symlink_to(ckpt_volume_dir)
    print(f"[modal] Checkpoints: {ckpt_local_dir} -> {ckpt_volume_dir}")

    args = list(extra_args)

    def _has_prefix(prefix: str) -> bool:
        return any(str(a).startswith(prefix) for a in args)

    # Check if using CIFAR-10 (auto-downloads) vs CelebA (needs volume)
    using_cifar = any("dataloaders=cifar10" in str(a) for a in args)

    if not using_cifar:
        # CelebA: use persisted volume paths
        if not _has_prefix("dataloaders.train.dataset.root="):
            args.append("dataloaders.train.dataset.root=/data/celeba")
        if not _has_prefix("dataloaders.eval.dataset.root="):
            args.append("dataloaders.eval.dataset.root=/data/celeba")

    cmd = ["python", "train.py", "device=cuda", *args]
    print("Running command:", " ".join(cmd))

    # Background thread to commit volume periodically (every 5 min)
    # This allows syncing trackio data locally during long runs
    stop_commit_thread = threading.Event()

    def periodic_commit():
        commit_interval = 300  # 5 minutes
        while not stop_commit_thread.wait(commit_interval):
            try:
                training_volume.commit()
                print("[modal] Periodic volume commit completed")
            except Exception as e:
                print(f"[modal] Periodic commit failed: {e}")

    commit_thread = threading.Thread(target=periodic_commit, daemon=True)
    commit_thread.start()
    print("[modal] Started periodic volume commit (every 5 min)")

    try:
        subprocess.run(cmd, check=True)
    finally:
        stop_commit_thread.set()
        commit_thread.join(timeout=5)

    # Final commit to persist trackio logs and any checkpoints
    training_volume.commit()
    print("[modal] Final volume commit - trackio logs persisted")


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
    """Helper function to run Modal training from launcher.py."""
    print("Starting Modal training job...")
    with app.run():
        train_on_modal.remote(extra_args)
    print("Modal training job completed!")


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


def _print_usage():
    print("Usage: python scripts/modal_app.py <command> [args]")
    print()
    print("Commands:")
    print("  list              List trackio runs in Modal volume")
    print("  sync [project]    Sync trackio data to local (default: denoising-zoo)")
    print("  ckpts             List checkpoints in Modal volume")
    print("  download <path>   Download checkpoint from Modal (e.g., unet/last.pt)")
    print("  <hydra args>      Run training on Modal with given arguments")


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
