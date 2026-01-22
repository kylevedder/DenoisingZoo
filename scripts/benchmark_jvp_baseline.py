"""Benchmark JVP performance across batch sizes and ratios."""

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


def benchmark_jvp(model, batch_size, n_jvp_samples, device, dtype, num_warmup=5, num_iter=15):
    """Benchmark JVP for given configuration."""
    z = torch.randn(n_jvp_samples, 3, 32, 32, device=device, dtype=dtype)
    t = torch.rand(n_jvp_samples, 1, device=device, dtype=dtype) * 0.8 + 0.1

    def model_fn(z_in, t_in):
        time_input = make_time_input(t_in)
        unified = make_unified_flow_matching_input(z_in, time_input)
        return model(unified)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            v = model_fn(z, t)
        tangent_t = torch.ones_like(t)
        _, _ = torch.func.jvp(model_fn, (z, t), (v.detach(), tangent_t))
        torch.cuda.synchronize()

    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(num_iter):
        with torch.no_grad():
            v = model_fn(z, t)
        torch.cuda.synchronize()
        start = time.perf_counter()
        tangent_t = torch.ones_like(t)
        _, jvp = torch.func.jvp(model_fn, (z, t), (v.detach(), tangent_t))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mem_allocated = torch.cuda.max_memory_allocated() / 1e9

    mean_time = sum(times) / len(times)
    std_time = (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5

    return {
        "mean_ms": mean_time * 1000,
        "std_ms": std_time * 1000,
        "mem_gb": mem_allocated,
    }


def benchmark_forward(model, batch_size, device, dtype, num_warmup=5, num_iter=15):
    """Benchmark forward pass for reference."""
    z = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
    t = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.8 + 0.1

    def model_fn(z_in, t_in):
        time_input = make_time_input(t_in)
        unified = make_unified_flow_matching_input(z_in, time_input)
        return model(unified)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model_fn(z, t)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_fn(z, t)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Dtype:  {dtype}")

    results = {"device": torch.cuda.get_device_name(), "dtype": str(dtype), "benchmarks": []}

    baseline_per_sample = None

    for batch_size in [32, 64, 96, 128]:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        model = create_model(device, dtype)

        # Forward pass baseline
        try:
            fwd_ms = benchmark_forward(model, batch_size, device, dtype)
            print(f"Forward pass (full batch): {fwd_ms:.2f}ms")
        except torch.cuda.OutOfMemoryError:
            print("Forward pass: OOM")
            continue

        # JVP at different ratios
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            n_jvp = int(batch_size * ratio)
            if n_jvp == 0:
                continue

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                result = benchmark_jvp(model, batch_size, n_jvp, device, dtype)

                # Compute overhead vs linear expectation
                if ratio == 0.25:
                    baseline_per_sample = result["mean_ms"] / n_jvp
                    overhead = 1.0
                elif baseline_per_sample is not None:
                    expected_ms = baseline_per_sample * n_jvp
                    overhead = result["mean_ms"] / expected_ms
                else:
                    overhead = 0.0

                print(
                    f"  ratio={ratio:.2f} ({n_jvp:3d} samples): {result['mean_ms']:>8.2f}ms ± {result['std_ms']:.2f}ms, "
                    f"mem={result['mem_gb']:.2f}GB, overhead={overhead:.2f}x"
                )

                results["benchmarks"].append(
                    {
                        "batch_size": batch_size,
                        "ratio": ratio,
                        "n_jvp_samples": n_jvp,
                        "mean_ms": result["mean_ms"],
                        "std_ms": result["std_ms"],
                        "mem_gb": result["mem_gb"],
                        "forward_ms": fwd_ms,
                        "overhead_vs_linear": overhead,
                    }
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  ratio={ratio:.2f} ({n_jvp:3d} samples): OOM")
                results["benchmarks"].append(
                    {
                        "batch_size": batch_size,
                        "ratio": ratio,
                        "n_jvp_samples": n_jvp,
                        "error": "OOM",
                    }
                )

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = Path("outputs/benchmark_jvp_baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
