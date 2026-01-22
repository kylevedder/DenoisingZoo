"""Benchmark micro-batching strategies for JVP.

Tests whether smaller micro-batches with gradient accumulation
are more efficient than larger batches due to non-linear JVP scaling.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input
from models.unet.unet import UNet


def create_model(device, dtype):
    """Create CIFAR-10 UNet model."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        time_channels=2,
    ).to(device, dtype)
    model.eval()
    return model


def benchmark_microbatch(model, total_samples: int, micro_batch_size: int, ratio: float = 0.75):
    """Benchmark JVP with micro-batching."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    num_microbatches = total_samples // micro_batch_size
    n_jvp_per_mb = int(micro_batch_size * ratio)

    if n_jvp_per_mb == 0:
        return None

    def model_fn(z_in, t_in):
        time_input = make_time_input(t_in)
        unified = make_unified_flow_matching_input(z_in, time_input)
        return model(unified)

    # Generate all data
    all_z = torch.randn(total_samples, 3, 32, 32, device=device, dtype=dtype)
    all_t = torch.rand(total_samples, 1, device=device, dtype=dtype) * 0.8 + 0.1

    # Warmup
    z_mb = all_z[:micro_batch_size]
    t_mb = all_t[:micro_batch_size]
    for _ in range(3):
        z_jvp = z_mb[:n_jvp_per_mb]
        t_jvp = t_mb[:n_jvp_per_mb]
        with torch.no_grad():
            v = model_fn(z_jvp, t_jvp)
        tangent_t = torch.ones_like(t_jvp)
        _, _ = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.detach(), tangent_t))
        torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()

        for mb_idx in range(num_microbatches):
            mb_start = mb_idx * micro_batch_size
            mb_end = mb_start + micro_batch_size
            z_mb = all_z[mb_start:mb_end]
            t_mb = all_t[mb_start:mb_end]

            # JVP on subset
            z_jvp = z_mb[:n_jvp_per_mb]
            t_jvp = t_mb[:n_jvp_per_mb]

            with torch.no_grad():
                v = model_fn(z_jvp, t_jvp)
            tangent_t = torch.ones_like(t_jvp)
            _, jvp = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.detach(), tangent_t))

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_time = sum(times) / len(times) * 1000
    std_time = (sum((x - mean_time / 1000) ** 2 for x in times) / len(times)) ** 0.5 * 1000
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "total_samples": total_samples,
        "micro_batch_size": micro_batch_size,
        "num_microbatches": num_microbatches,
        "n_jvp_per_mb": n_jvp_per_mb,
        "total_jvp_samples": n_jvp_per_mb * num_microbatches,
        "mean_ms": mean_time,
        "std_ms": std_time,
        "mem_gb": mem_gb,
    }


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Dtype:  {dtype}")

    results = {"device": torch.cuda.get_device_name(), "dtype": str(dtype), "benchmarks": []}

    model = create_model(device, dtype)

    total_samples = 128
    ratio = 0.75

    print(f"\nTotal samples: {total_samples}, ratio: {ratio}")
    print(f"Total JVP samples: {int(total_samples * ratio)}")
    print()

    print(f"{'Micro-batch':<12} {'N batches':<10} {'JVP/batch':<10} {'Total ms':<12} {'Mem GB':<10}")
    print("-" * 54)

    for mb_size in [16, 32, 48, 64, 96, 128]:
        if total_samples % mb_size != 0:
            continue

        torch.cuda.empty_cache()

        try:
            result = benchmark_microbatch(model, total_samples, mb_size, ratio)
            if result:
                results["benchmarks"].append(result)
                print(
                    f"{mb_size:<12} {result['num_microbatches']:<10} {result['n_jvp_per_mb']:<10} "
                    f"{result['mean_ms']:<12.2f} {result['mem_gb']:<10.2f}"
                )
        except torch.cuda.OutOfMemoryError:
            print(f"{mb_size:<12} OOM")
            results["benchmarks"].append({"micro_batch_size": mb_size, "error": "OOM"})

    # Find optimal configuration
    valid_results = [r for r in results["benchmarks"] if "error" not in r]
    if valid_results:
        optimal = min(valid_results, key=lambda x: x["mean_ms"])
        print()
        print(f"Optimal micro-batch size: {optimal['micro_batch_size']}")
        print(f"  Time: {optimal['mean_ms']:.2f}ms")
        print(f"  Memory: {optimal['mem_gb']:.2f}GB")

        # Compare to largest batch
        largest = max(valid_results, key=lambda x: x["micro_batch_size"])
        if largest["micro_batch_size"] != optimal["micro_batch_size"]:
            speedup = largest["mean_ms"] / optimal["mean_ms"]
            print(f"  Speedup vs batch={largest['micro_batch_size']}: {speedup:.2f}x")

    # Save results
    output_path = Path("outputs/benchmark_microbatch.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
