"""Experiment 3.1: Compare MeanFlow 1-step vs solver-based sampling.

Verifies that MeanFlow 1-step sampling achieves comparable energy distance
to solver-based sampling on the deterministic dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.base_dataloaders import (
    DictDatasetAdapter,
    make_time_input,
    make_unified_flow_matching_input,
)
from dataloaders.deterministic_dataset import DeterministicFlowDataset
from helpers import compute_energy_distance_u_statistic, reshape_to_samples_2d
from models.unet import UNet
from solvers.euler_solver import EulerSolver
from solvers.rk4_solver import RK4Solver


def evaluate_meanflow_1step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate MeanFlow 1-step sampling energy distance.

    For each sample in the dataset:
    - Use raw_source (x0) as the starting point
    - Generate: x1 = x0 + (1 - 0) * model(x0, r=0, t=1)
    - Compare to raw_target (y_true)
    """
    model.eval()
    ed_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MeanFlow 1-step eval"):
            x0 = batch["raw_source"].to(device)
            y_true = batch["raw_target"].to(device)

            B = x0.shape[0]

            # Build time input for r=0, t=1
            r_tensor = torch.zeros(B, 1, device=device, dtype=x0.dtype)
            t_tensor = torch.ones(B, 1, device=device, dtype=x0.dtype)
            time_input = make_time_input(t_tensor, r=r_tensor)

            # Build unified input
            unified = make_unified_flow_matching_input(x0, time_input)

            # MeanFlow 1-step: x1 = x0 + (t - r) * v = x0 + 1 * v
            v = model(unified)
            y_pred = x0 + v

            # Compute energy distance
            y_pred_2d = reshape_to_samples_2d(y_pred)
            y_true_2d = reshape_to_samples_2d(y_true)

            ed2 = compute_energy_distance_u_statistic(y_pred_2d, y_true_2d)
            ed_sum += ed2
            num_batches += 1

    return ed_sum / max(1, num_batches)


def evaluate_solver(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    solver_type: str = "euler",
    num_steps: int = 10,
) -> float:
    """Evaluate solver-based sampling energy distance.

    For each sample in the dataset:
    - Use raw_source (x0) as the starting point
    - Integrate ODE from t=0 to t=1
    - Compare to raw_target (y_true)
    """
    model.eval()

    if solver_type == "euler":
        solver = EulerSolver(model, num_steps=num_steps)
    elif solver_type == "rk4":
        solver = RK4Solver(model, num_steps=num_steps)
    else:
        raise ValueError(f"Unknown solver: {solver_type}")

    ed_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Solver ({solver_type}, {num_steps} steps)"):
            x0 = batch["raw_source"].to(device)
            y_true = batch["raw_target"].to(device)

            # Integrate ODE from x0
            result = solver.solve(x0)
            y_pred = result.final_state

            # Compute energy distance
            y_pred_2d = reshape_to_samples_2d(y_pred)
            y_true_2d = reshape_to_samples_2d(y_true)

            ed2 = compute_energy_distance_u_statistic(y_pred_2d, y_true_2d)
            ed_sum += ed2
            num_batches += 1

    return ed_sum / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Compare MeanFlow 1-step vs solver sampling")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ckpts/unet/archive/meanflow_ratio025_epoch_0200.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--solver-steps", type=int, default=10, help="Number of solver steps")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract config from checkpoint to get model params
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Create model with same config as training
    model = UNet(
        in_channels=model_config.get("in_channels", 3),
        base_channels=model_config.get("base_channels", 128),
        channel_mult=model_config.get("channel_mult", [1, 2, 2, 2]),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        attention_resolutions=model_config.get("attention_resolutions", [16]),
        dropout=model_config.get("dropout", 0.1),
        num_heads=model_config.get("num_heads", 4),
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create dataset (same config as training)
    dataset = DeterministicFlowDataset(
        num_classes=4,
        samples_per_class=10,
        image_size=32,
        num_channels=3,
        seed=42,
    )

    dataloader = DataLoader(
        DictDatasetAdapter(dataset),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"\nDataset: {len(dataset)} samples ({dataset.num_classes} classes, {dataset.samples_per_class} per class)")
    print(f"Batch size: {args.batch_size}")

    # Evaluate MeanFlow 1-step
    print("\n" + "=" * 60)
    print("Evaluating MeanFlow 1-step (r=0, t=1)")
    print("=" * 60)
    ed_meanflow = evaluate_meanflow_1step(model, dataloader, device)
    print(f"MeanFlow 1-step Energy Distance: {ed_meanflow:.6f}")

    # Evaluate solvers with different step counts
    print("\n" + "=" * 60)
    print("Evaluating Solver-based sampling")
    print("=" * 60)

    results = {"meanflow_1step": ed_meanflow}

    for solver_type in ["euler", "rk4"]:
        for steps in [10, 50, 100]:
            key = f"{solver_type}_{steps}"
            ed = evaluate_solver(model, dataloader, device, solver_type, steps)
            results[key] = ed
            print(f"{solver_type.upper()} ({steps} steps) Energy Distance: {ed:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Energy Distance':>15}")
    print("-" * 42)
    for method, ed in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:<25} {ed:>15.6f}")

    # Compare MeanFlow to best solver
    best_solver = min(
        [(k, v) for k, v in results.items() if k != "meanflow_1step"],
        key=lambda x: x[1],
    )
    print(f"\nMeanFlow 1-step vs best solver ({best_solver[0]}):")
    print(f"  MeanFlow: {ed_meanflow:.6f}")
    print(f"  Solver:   {best_solver[1]:.6f}")
    print(f"  Ratio:    {ed_meanflow / best_solver[1]:.2f}x")


if __name__ == "__main__":
    main()
