"""Profile JVP computation to identify hot ops.

Identifies which operations dominate JVP time to guide further optimization.

Usage:
    python scripts/profile_jvp.py
    modal run scripts/modal_app.py::run_benchmark --script-name profile_jvp.py
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity


def create_unet_model(device: torch.device) -> nn.Module:
    """Create UNet model for profiling."""
    from models.unet.unet import UNet

    model = UNet(
        in_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Profile JVP computation")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    from dataloaders.base_dataloaders import make_unified_flow_matching_input

    model = create_unet_model(device)
    batch_size = args.batch_size
    img_size = 32

    # Create test data
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randn(batch_size, 3, img_size, img_size, device=device)
    t = torch.rand(batch_size, 1, device=device).clamp(0.01, 0.99)
    r = torch.rand(batch_size, 1, device=device).clamp(0.01, 0.99)
    r, t = torch.minimum(r, t), torch.maximum(r, t)

    t_broad = t.view(-1, 1, 1, 1)
    z_t = (1 - t_broad) * x + t_broad * y
    v_true = y - x

    time_input = torch.cat([r, t], dim=1)
    unified = make_unified_flow_matching_input(z_t, time_input)

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        _ = model(unified)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Profile baseline forward
    print("\n" + "=" * 70)
    print(" Profiling Baseline Forward Pass")
    print("=" * 70)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("baseline_forward"):
            _ = model(unified)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print("\nBaseline Forward - Top 20 ops by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))

    # Profile JVP
    print("\n" + "=" * 70)
    print(" Profiling torch.func.jvp")
    print("=" * 70)

    def u_func(z_in, r_in, t_in):
        time_in = torch.cat([r_in, t_in], dim=1)
        unified_in = make_unified_flow_matching_input(z_in, time_in)
        return model(unified_in)

    tangent_z = v_true.float()
    tangent_r = torch.zeros_like(r).float()
    tangent_t = torch.ones_like(t).float()

    # Warmup JVP
    for _ in range(3):
        with torch.amp.autocast(device.type, enabled=False):
            _, _ = torch.func.jvp(
                u_func,
                (z_t.float(), r.float(), t.float()),
                (tangent_z, tangent_r, tangent_t),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("jvp_forward"):
            with torch.amp.autocast(device.type, enabled=False):
                _, _ = torch.func.jvp(
                    u_func,
                    (z_t.float(), r.float(), t.float()),
                    (tangent_z, tangent_r, tangent_t),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()

    print("\nJVP - Top 20 ops by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))

    # Compare specific ops
    print("\n" + "=" * 70)
    print(" Op Comparison: Baseline vs JVP")
    print("=" * 70)

    # Get baseline times
    baseline_times = {}
    with profile(activities=activities, record_shapes=True) as prof_baseline:
        _ = model(unified)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for item in prof_baseline.key_averages():
        key = item.key
        time_ms = item.cuda_time_total / 1000 if device.type == "cuda" else item.cpu_time_total / 1000
        if time_ms > 0.1:  # Only significant ops
            baseline_times[key] = time_ms

    # Get JVP times
    jvp_times = {}
    with profile(activities=activities, record_shapes=True) as prof_jvp:
        with torch.amp.autocast(device.type, enabled=False):
            _, _ = torch.func.jvp(
                u_func,
                (z_t.float(), r.float(), t.float()),
                (tangent_z, tangent_r, tangent_t),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    for item in prof_jvp.key_averages():
        key = item.key
        time_ms = item.cuda_time_total / 1000 if device.type == "cuda" else item.cpu_time_total / 1000
        if time_ms > 0.1:
            jvp_times[key] = time_ms

    # Find ops with biggest JVP overhead
    print("\nOps with biggest JVP overhead:")
    overhead = []
    for key in jvp_times:
        jvp_t = jvp_times[key]
        base_t = baseline_times.get(key, 0)
        if jvp_t > 0.5:  # At least 0.5ms
            overhead.append((key, base_t, jvp_t, jvp_t - base_t, jvp_t / max(base_t, 0.001)))

    overhead.sort(key=lambda x: -x[3])  # Sort by absolute overhead
    print(f"{'Op':<40} {'Baseline':<12} {'JVP':<12} {'Overhead':<12} {'Ratio':<10}")
    print("-" * 86)
    for key, base_t, jvp_t, diff, ratio in overhead[:15]:
        print(f"{key[:40]:<40} {base_t:>10.2f}ms {jvp_t:>10.2f}ms {diff:>10.2f}ms {ratio:>8.1f}x")


if __name__ == "__main__":
    main()
