"""Benchmark split JVP: separate spatial and time derivatives.

Tests whether computing ∂v/∂z and ∂v/∂t separately is faster than combined.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input
from models.unet.unet import UNet


def benchmark_jvp_methods(batch_size: int = 64):
    device = torch.device("cuda")
    dtype = torch.bfloat16

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

    z = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
    t = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.8 + 0.1

    def model_fn(z_in, t_in):
        time_input = make_time_input(t_in)
        unified = make_unified_flow_matching_input(z_in, time_input)
        return model(unified)

    with torch.no_grad():
        v_tangent = model_fn(z, t)

    # Method 1: Combined JVP (current implementation)
    def combined_jvp():
        tangent_t = torch.ones_like(t)
        _, jvp = torch.func.jvp(model_fn, (z, t), (v_tangent.detach(), tangent_t))
        return jvp

    # Method 2: Split JVP (spatial + time separately)
    def split_jvp():
        # Spatial JVP: z varies, t fixed
        _, jvp_z = torch.func.jvp(
            lambda z_in: model_fn(z_in, t),
            (z,),
            (v_tangent.detach(),),
        )
        # Time JVP: t varies, z fixed
        tangent_t = torch.ones_like(t)
        _, jvp_t = torch.func.jvp(
            lambda t_in: model_fn(z, t_in),
            (t,),
            (tangent_t,),
        )
        return jvp_z + jvp_t

    # Warmup
    print(f"Batch size: {batch_size}")
    print("Warming up...")
    for _ in range(5):
        combined_jvp()
        split_jvp()
        torch.cuda.synchronize()

    # Verify correctness
    jvp_c = combined_jvp()
    jvp_s = split_jvp()
    max_error = (jvp_c - jvp_s).abs().max().item()
    mean_error = (jvp_c - jvp_s).abs().mean().item()
    print(f"Max error between methods: {max_error:.2e}")
    print(f"Mean error between methods: {mean_error:.2e}")

    if max_error > 1e-3:
        print("WARNING: Methods produce different results!")
        return None

    # Benchmark combined
    print("Benchmarking combined JVP...")
    times_combined = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter()
        combined_jvp()
        torch.cuda.synchronize()
        times_combined.append(time.perf_counter() - start)

    # Benchmark split
    print("Benchmarking split JVP...")
    times_split = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter()
        split_jvp()
        torch.cuda.synchronize()
        times_split.append(time.perf_counter() - start)

    mean_combined = sum(times_combined) / len(times_combined) * 1000
    mean_split = sum(times_split) / len(times_split) * 1000

    print(f"\nResults:")
    print(f"  Combined JVP: {mean_combined:.2f}ms")
    print(f"  Split JVP:    {mean_split:.2f}ms")
    print(f"  Speedup:      {mean_combined/mean_split:.2f}x")

    return {
        "batch_size": batch_size,
        "combined_ms": mean_combined,
        "split_ms": mean_split,
        "speedup": mean_combined / mean_split,
        "max_error": max_error,
    }


if __name__ == "__main__":
    results = []
    for bs in [32, 48, 64]:
        print(f"\n{'='*50}")
        try:
            result = benchmark_jvp_methods(bs)
            if result:
                results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {bs}: OOM")

    if results:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"{'Batch':<8} {'Combined':<12} {'Split':<12} {'Speedup':<10}")
        print("-" * 42)
        for r in results:
            print(f"{r['batch_size']:<8} {r['combined_ms']:<12.2f} {r['split_ms']:<12.2f} {r['speedup']:<10.2f}")
