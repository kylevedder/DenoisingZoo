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

    Path("outputs/ckpts").mkdir(parents=True, exist_ok=True)

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
    subprocess.run(cmd, check=True)

    # Commit volume to persist trackio logs and any checkpoints
    training_volume.commit()
    print("[modal] Volume committed - trackio logs persisted")


@app.function(
    image=image,
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
    image=image,
    volumes={"/data": training_volume},
)
def get_trackio_db(project: str = "denoising-zoo") -> bytes | None:
    """Download trackio database for a project from Modal volume."""
    db_path = Path(f"/data/trackio/{project}.db")
    if not db_path.exists():
        return None
    return db_path.read_bytes()


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        # Sync trackio data: python scripts/modal_app.py sync [project]
        project = sys.argv[2] if len(sys.argv) > 2 else "denoising-zoo"
        sync_trackio_from_modal(project)
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        # List runs: python scripts/modal_app.py list
        with app.run():
            runs = list_trackio_runs.remote()
            if runs:
                print("Trackio runs in Modal volume:")
                for run in runs:
                    print(f"  - {run}")
            else:
                print("No runs found in Modal volume")
    else:
        # Default: run training
        extra_args = sys.argv[1:] if len(sys.argv) > 1 else []
        run_modal_training(extra_args)


@app.local_entrypoint()
def main(*extra_args: str) -> None:
    """CLI entrypoint so `modal run scripts/modal_app.py -- ...` streams logs."""
    train_on_modal.remote(list(extra_args))
