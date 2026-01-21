#!/usr/bin/env python3
"""Delete trackio runs matching a regex pattern."""

import argparse
import re
import sqlite3
from pathlib import Path


def get_db_path(project: str) -> Path:
    return Path.home() / ".cache/huggingface/trackio" / f"{project}.db"


def list_runs(project: str) -> list[tuple[str, str]]:
    """Return list of (run_name, created_at) tuples from all tables."""
    db_path = get_db_path(project)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    # Get runs from configs table (with created_at)
    cursor = conn.execute("SELECT run_name, created_at FROM configs")
    runs_with_time = {name: created for name, created in cursor.fetchall()}

    # Get any runs that only exist in metrics table
    cursor = conn.execute("SELECT DISTINCT run_name FROM metrics")
    for (name,) in cursor.fetchall():
        if name not in runs_with_time:
            runs_with_time[name] = "unknown"

    # Get any runs that only exist in system_metrics table
    cursor = conn.execute("SELECT DISTINCT run_name FROM system_metrics")
    for (name,) in cursor.fetchall():
        if name not in runs_with_time:
            runs_with_time[name] = "unknown"

    conn.close()
    # Sort by created_at, with "unknown" at the end
    return sorted(runs_with_time.items(), key=lambda x: (x[1] == "unknown", x[1]))


def delete_runs(project: str, run_names: list[str]) -> int:
    """Delete specified runs. Returns count of deleted runs."""
    db_path = get_db_path(project)
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(db_path)
    for run_name in run_names:
        conn.execute("DELETE FROM configs WHERE run_name = ?", (run_name,))
        conn.execute("DELETE FROM metrics WHERE run_name = ?", (run_name,))
        conn.execute("DELETE FROM system_metrics WHERE run_name = ?", (run_name,))
    conn.commit()
    conn.close()
    return len(run_names)


def main():
    parser = argparse.ArgumentParser(description="Delete trackio runs matching a regex pattern")
    parser.add_argument("pattern", nargs="?", help="Regex pattern to match run names")
    parser.add_argument("--project", default="denoising-zoo", help="Project name (default: denoising-zoo)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--list", action="store_true", dest="list_only", help="Just list all runs")
    args = parser.parse_args()

    if not args.list_only and not args.pattern:
        parser.error("pattern is required unless using --list")

    runs = list_runs(args.project)

    if not runs:
        print(f"No runs found in project '{args.project}'")
        return

    if args.list_only:
        print(f"Runs in '{args.project}':")
        for run_name, created_at in runs:
            print(f"  {run_name:30} {created_at}")
        return

    # Find matching runs
    pattern = re.compile(args.pattern)
    matching = [(name, created) for name, created in runs if pattern.search(name)]

    if not matching:
        print(f"No runs matching pattern '{args.pattern}'")
        return

    print(f"Runs matching '{args.pattern}':")
    for run_name, created_at in matching:
        print(f"  {run_name:30} {created_at}")

    if args.dry_run:
        print(f"\nDry run: would delete {len(matching)} run(s)")
        return

    # Confirm deletion
    response = input(f"\nDelete {len(matching)} run(s)? [y/N] ")
    if response.lower() != 'y':
        print("Aborted")
        return

    deleted = delete_runs(args.project, [name for name, _ in matching])
    print(f"Deleted {deleted} run(s)")


if __name__ == "__main__":
    main()
