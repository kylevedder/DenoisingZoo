#!/usr/bin/env python3
"""
Robust benchmark runner for Modal A100.

Usage:
    python scripts/run_benchmarks_modal.py [benchmark_name]

If no benchmark_name is provided, runs all benchmarks.
"""

import json
import sys
import time
from pathlib import Path

import modal

# Modal app for benchmarks
app = modal.App("denoisingzoo-benchmarks")

# Exclude patterns for copying
EXCLUDE_PATTERNS = {
    ".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "outputs", "data", "*.pyc", "*.pyo", ".DS_Store", "*.egg-info",
}


def should_ignore(path: Path) -> bool:
    parts = path.parts
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in parts or path.name == pattern:
            return True
    return False


# Image with PyTorch + CUDA
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "tqdm")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6,<2.7' 'torchvision>=0.21,<0.22'"
    )
    .add_local_dir(".", "/root/app", copy=True, ignore=should_ignore)
)

# Volume for results
results_volume = modal.Volume.from_name("benchmark-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,  # 30 min max
    volumes={"/results": results_volume},
)
def run_jvp_baseline_benchmark() -> dict:
    """Run JVP baseline benchmark and return results."""
    import os
    import time as time_module
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    # Keep model in fp32, use autocast for bf16 - this is compatible with torch.func.jvp
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    from models.unet.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

    results = {"device": torch.cuda.get_device_name(), "dtype": "bf16 (autocast)", "benchmarks": []}
    baseline_per_sample = None

    for batch_size in [32, 64, 96, 128]:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()

        # Create model in fp32, use autocast
        model = UNet(
            in_channels=3, out_channels=3, base_channels=128,
            num_res_blocks=2, attention_resolutions=[16, 8],
            channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
        ).to(device)  # fp32
        model.eval()

        def model_fn(z_in, t_in):
            time_input = make_time_input(t_in)
            unified = make_unified_flow_matching_input(z_in, time_input)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(unified)

        # Benchmark forward pass
        try:
            z = torch.randn(batch_size, 3, 32, 32, device=device)  # fp32
            t = torch.rand(batch_size, 1, device=device) * 0.8 + 0.1  # fp32

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model_fn(z, t)
                torch.cuda.synchronize()

            # Benchmark
            fwd_times = []
            for _ in range(15):
                torch.cuda.synchronize()
                start = time_module.perf_counter()
                with torch.no_grad():
                    _ = model_fn(z, t)
                torch.cuda.synchronize()
                fwd_times.append(time_module.perf_counter() - start)

            fwd_ms = sum(fwd_times) / len(fwd_times) * 1000
            print(f"Forward pass (full batch): {fwd_ms:.2f}ms")
        except torch.cuda.OutOfMemoryError:
            print("Forward pass: OOM")
            del model
            continue
        except Exception as e:
            print(f"Forward pass error: {e}")
            del model
            continue

        for ratio in [0.25, 0.5, 0.75, 1.0]:
            n_jvp = int(batch_size * ratio)
            if n_jvp == 0:
                continue

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create tensors for JVP (fp32)
                z_jvp = torch.randn(n_jvp, 3, 32, 32, device=device)
                t_jvp = torch.rand(n_jvp, 1, device=device) * 0.8 + 0.1

                # Warmup JVP
                for _ in range(5):
                    with torch.no_grad():
                        v = model_fn(z_jvp, t_jvp)
                    tangent_t = torch.ones_like(t_jvp)
                    # Convert v to fp32 if needed for JVP tangent
                    _, _ = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.float().detach(), tangent_t))
                    torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()

                # Benchmark JVP
                jvp_times = []
                for _ in range(15):
                    with torch.no_grad():
                        v = model_fn(z_jvp, t_jvp)
                    torch.cuda.synchronize()
                    start = time_module.perf_counter()
                    tangent_t = torch.ones_like(t_jvp)
                    _, jvp = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.float().detach(), tangent_t))
                    torch.cuda.synchronize()
                    jvp_times.append(time_module.perf_counter() - start)

                mean_ms = sum(jvp_times) / len(jvp_times) * 1000
                std_ms = (sum((x*1000 - mean_ms)**2 for x in jvp_times) / len(jvp_times)) ** 0.5
                mem_gb = torch.cuda.max_memory_allocated() / 1e9

                if ratio == 0.25:
                    baseline_per_sample = mean_ms / n_jvp
                    overhead = 1.0
                elif baseline_per_sample:
                    expected_ms = baseline_per_sample * n_jvp
                    overhead = mean_ms / expected_ms
                else:
                    overhead = 0.0

                print(f"  ratio={ratio:.2f} ({n_jvp:3d} samples): {mean_ms:>8.2f}ms ± {std_ms:.2f}ms, "
                      f"mem={mem_gb:.2f}GB, overhead={overhead:.2f}x")

                results["benchmarks"].append({
                    "batch_size": batch_size,
                    "ratio": ratio,
                    "n_jvp_samples": n_jvp,
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "mem_gb": mem_gb,
                    "forward_ms": fwd_ms,
                    "overhead_vs_linear": overhead,
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  ratio={ratio:.2f} ({n_jvp:3d} samples): OOM")
                results["benchmarks"].append({"batch_size": batch_size, "ratio": ratio, "error": "OOM"})
            except Exception as e:
                print(f"  ratio={ratio:.2f}: Error - {e}")
                import traceback
                traceback.print_exc()
                results["benchmarks"].append({"batch_size": batch_size, "ratio": ratio, "error": str(e)})

        del model
        torch.cuda.empty_cache()

    # Save to volume
    volume_path = Path("/results/benchmark_jvp_baseline.json")
    volume_path.write_text(json.dumps(results, indent=2))
    results_volume.commit()
    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/results": results_volume},
)
def run_split_jvp_benchmark() -> dict:
    """Run split JVP benchmark and return results."""
    import os
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch
    import time as time_module
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    from models.unet.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

    results = {"device": torch.cuda.get_device_name(), "dtype": "bf16 (autocast)", "benchmarks": []}

    for batch_size in [32, 48, 64]:
        print(f"\n{'='*50}")
        print(f"Batch size: {batch_size}")

        try:
            torch.cuda.empty_cache()

            # Keep model in fp32, use autocast
            model = UNet(
                in_channels=3, out_channels=3, base_channels=128,
                num_res_blocks=2, attention_resolutions=[16, 8],
                channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
            ).to(device)
            model.eval()

            z = torch.randn(batch_size, 3, 32, 32, device=device)  # fp32
            t = torch.rand(batch_size, 1, device=device) * 0.8 + 0.1

            def model_fn(z_in, t_in):
                time_input = make_time_input(t_in)
                unified = make_unified_flow_matching_input(z_in, time_input)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(unified)

            with torch.no_grad():
                v_tangent = model_fn(z, t)

            # Combined JVP
            def combined_jvp():
                tangent_t = torch.ones_like(t)
                _, jvp = torch.func.jvp(model_fn, (z, t), (v_tangent.float().detach(), tangent_t))
                return jvp

            # Split JVP
            def split_jvp():
                _, jvp_z = torch.func.jvp(lambda z_in: model_fn(z_in, t), (z,), (v_tangent.float().detach(),))
                tangent_t = torch.ones_like(t)
                _, jvp_t = torch.func.jvp(lambda t_in: model_fn(z, t_in), (t,), (tangent_t,))
                return jvp_z + jvp_t

            # Warmup
            for _ in range(5):
                combined_jvp()
                split_jvp()
                torch.cuda.synchronize()

            # Verify correctness
            jvp_c = combined_jvp()
            jvp_s = split_jvp()
            max_error = (jvp_c - jvp_s).abs().max().item()
            print(f"Max error between methods: {max_error:.2e}")

            # Benchmark combined
            times_combined = []
            for _ in range(15):
                torch.cuda.synchronize()
                start = time_module.perf_counter()
                combined_jvp()
                torch.cuda.synchronize()
                times_combined.append(time_module.perf_counter() - start)

            # Benchmark split
            times_split = []
            for _ in range(15):
                torch.cuda.synchronize()
                start = time_module.perf_counter()
                split_jvp()
                torch.cuda.synchronize()
                times_split.append(time_module.perf_counter() - start)

            mean_combined = sum(times_combined) / len(times_combined) * 1000
            mean_split = sum(times_split) / len(times_split) * 1000

            print(f"Combined JVP: {mean_combined:.2f}ms")
            print(f"Split JVP:    {mean_split:.2f}ms")
            print(f"Speedup:      {mean_combined/mean_split:.2f}x")

            results["benchmarks"].append({
                "batch_size": batch_size,
                "combined_ms": mean_combined,
                "split_ms": mean_split,
                "speedup": mean_combined / mean_split,
                "max_error": max_error,
            })

            del model
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size}: OOM")
            results["benchmarks"].append({"batch_size": batch_size, "error": "OOM"})
        except Exception as e:
            print(f"Batch size {batch_size}: Error - {e}")
            results["benchmarks"].append({"batch_size": batch_size, "error": str(e)})

    # Save to volume
    volume_path = Path("/results/benchmark_split_jvp.json")
    volume_path.write_text(json.dumps(results, indent=2))
    results_volume.commit()

    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/results": results_volume},
)
def run_microbatch_benchmark() -> dict:
    """Run micro-batching benchmark and return results."""
    import os
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch
    import time as time_module
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    from models.unet.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

    results = {"device": torch.cuda.get_device_name(), "dtype": "bf16 (autocast)", "benchmarks": []}

    # Create model in fp32, use autocast for bf16
    model = UNet(
        in_channels=3, out_channels=3, base_channels=128,
        num_res_blocks=2, attention_resolutions=[16, 8],
        channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
    ).to(device)  # fp32
    model.eval()

    def model_fn(z_in, t_in):
        time_input = make_time_input(t_in)
        unified = make_unified_flow_matching_input(z_in, time_input)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(unified)

    total_samples = 128
    ratio = 0.75

    print(f"Total samples: {total_samples}, ratio: {ratio}")
    print(f"Total JVP samples: {int(total_samples * ratio)}")
    print()
    print(f"{'Micro-batch':<12} {'N batches':<10} {'JVP/batch':<10} {'Total ms':<12} {'Mem GB':<10}")
    print("-" * 54)

    for mb_size in [16, 32, 48, 64, 96, 128]:
        if total_samples % mb_size != 0:
            continue

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            num_microbatches = total_samples // mb_size
            n_jvp_per_mb = int(mb_size * ratio)

            if n_jvp_per_mb == 0:
                continue

            # Generate data in fp32
            all_z = torch.randn(total_samples, 3, 32, 32, device=device)
            all_t = torch.rand(total_samples, 1, device=device) * 0.8 + 0.1

            # Warmup
            z_mb = all_z[:mb_size]
            t_mb = all_t[:mb_size]
            for _ in range(3):
                z_jvp = z_mb[:n_jvp_per_mb]
                t_jvp = t_mb[:n_jvp_per_mb]
                with torch.no_grad():
                    v = model_fn(z_jvp, t_jvp)
                tangent_t = torch.ones_like(t_jvp)
                _, _ = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.float().detach(), tangent_t))
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()

            # Benchmark
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time_module.perf_counter()

                for mb_idx in range(num_microbatches):
                    mb_start = mb_idx * mb_size
                    mb_end = mb_start + mb_size
                    z_mb = all_z[mb_start:mb_end]
                    t_mb = all_t[mb_start:mb_end]

                    z_jvp = z_mb[:n_jvp_per_mb]
                    t_jvp = t_mb[:n_jvp_per_mb]

                    with torch.no_grad():
                        v = model_fn(z_jvp, t_jvp)
                    tangent_t = torch.ones_like(t_jvp)
                    _, jvp = torch.func.jvp(model_fn, (z_jvp, t_jvp), (v.float().detach(), tangent_t))

                torch.cuda.synchronize()
                times.append(time_module.perf_counter() - start)

            mean_time = sum(times) / len(times) * 1000
            mem_gb = torch.cuda.max_memory_allocated() / 1e9

            print(f"{mb_size:<12} {num_microbatches:<10} {n_jvp_per_mb:<10} {mean_time:<12.2f} {mem_gb:<10.2f}")

            results["benchmarks"].append({
                "micro_batch_size": mb_size,
                "num_microbatches": num_microbatches,
                "n_jvp_per_mb": n_jvp_per_mb,
                "total_jvp_samples": n_jvp_per_mb * num_microbatches,
                "mean_ms": mean_time,
                "mem_gb": mem_gb,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"{mb_size:<12} OOM")
            results["benchmarks"].append({"micro_batch_size": mb_size, "error": "OOM"})
        except Exception as e:
            print(f"{mb_size:<12} Error: {e}")
            results["benchmarks"].append({"micro_batch_size": mb_size, "error": str(e)})

    del model
    torch.cuda.empty_cache()

    # Save to volume
    volume_path = Path("/results/benchmark_microbatch.json")
    volume_path.write_text(json.dumps(results, indent=2))
    results_volume.commit()

    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,  # 1 hour for max-autotune
    volumes={"/results": results_volume},
)
def run_compile_jvp_benchmark() -> dict:
    """Benchmark torch.compile + JVP with different compile modes."""
    import os
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch
    import time as time_module
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    from models.unet.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

    results = {"device": torch.cuda.get_device_name(), "dtype": "bf16 (autocast)", "benchmarks": []}

    batch_size = 32
    n_jvp = 24  # ratio=0.75

    z = torch.randn(n_jvp, 3, 32, 32, device=device)
    t = torch.rand(n_jvp, 1, device=device) * 0.8 + 0.1

    def benchmark_mode(model, mode_name, compile_mode=None):
        """Benchmark a specific compile mode."""
        print()
        print("=" * 50)
        print(f"Testing: {mode_name}")
        print("=" * 50)

        if compile_mode is not None:
            print(f"Compiling with mode={compile_mode}...")
            compiled_model = torch.compile(model, mode=compile_mode)
        else:
            compiled_model = model

        def model_fn(z_in, t_in):
            time_input = make_time_input(t_in)
            unified = make_unified_flow_matching_input(z_in, time_input)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return compiled_model(unified)

        # Extended warmup for max-autotune (per Codex recommendation)
        compile_start = time_module.perf_counter()
        first_forward_ms = None
        first_jvp_ms = None

        print("Forward warmup (20 iterations for max-autotune)...")
        for i in range(20):
            with torch.no_grad():
                v = model_fn(z, t)
            torch.cuda.synchronize()
            if i == 0:
                first_forward_ms = (time_module.perf_counter() - compile_start) * 1000
                print(f"  First forward: {first_forward_ms:.0f}ms")

        print("JVP warmup (15 iterations)...")
        jvp_start = time_module.perf_counter()
        for i in range(15):
            with torch.no_grad():
                v = model_fn(z, t)
            tangent_t = torch.ones_like(t)
            _, _ = torch.func.jvp(model_fn, (z, t), (v.float().detach(), tangent_t))
            torch.cuda.synchronize()
            if i == 0:
                first_jvp_ms = (time_module.perf_counter() - jvp_start) * 1000
                print(f"  First JVP: {first_jvp_ms:.0f}ms")

        # Benchmark after warmup
        times = []
        for _ in range(15):
            with torch.no_grad():
                v = model_fn(z, t)
            torch.cuda.synchronize()
            start = time_module.perf_counter()
            tangent_t = torch.ones_like(t)
            _, jvp = torch.func.jvp(model_fn, (z, t), (v.float().detach(), tangent_t))
            torch.cuda.synchronize()
            times.append(time_module.perf_counter() - start)

        mean_ms = sum(times) / len(times) * 1000
        print(f"  JVP after warmup: {mean_ms:.2f}ms")

        return {
            "mode": mode_name,
            "compile_mode": compile_mode,
            "batch_size": batch_size,
            "n_jvp": n_jvp,
            "mean_ms": mean_ms,
            "first_forward_ms": first_forward_ms,
            "first_jvp_ms": first_jvp_ms,
        }

    # Test uncompiled
    model = UNet(
        in_channels=3, out_channels=3, base_channels=128,
        num_res_blocks=2, attention_resolutions=[16, 8],
        channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
    ).to(device)
    model.train()  # Use train mode for realistic benchmark

    uncompiled_result = benchmark_mode(model, "uncompiled", compile_mode=None)
    results["benchmarks"].append(uncompiled_result)
    uncompiled_ms = uncompiled_result["mean_ms"]

    # Test different compile modes
    for compile_mode in ["default", "reduce-overhead", "max-autotune"]:
        try:
            torch.cuda.empty_cache()
            torch._dynamo.reset()

            # Fresh model for each compile mode
            model = UNet(
                in_channels=3, out_channels=3, base_channels=128,
                num_res_blocks=2, attention_resolutions=[16, 8],
                channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
            ).to(device)
            model.train()  # Use train mode for realistic benchmark

            result = benchmark_mode(model, f"compiled-{compile_mode}", compile_mode=compile_mode)
            result["speedup_vs_uncompiled"] = uncompiled_ms / result["mean_ms"]
            print(f"  Speedup vs uncompiled: {result['speedup_vs_uncompiled']:.2f}x")
            results["benchmarks"].append(result)

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            results["benchmarks"].append({
                "mode": f"compiled-{compile_mode}",
                "error": str(e),
            })

    # Print summary
    print()
    print("=" * 60)
    print("COMPILE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<25} {'JVP (ms)':<12} {'Speedup':<10} {'Compile (s)':<12}")
    print("-" * 60)
    for b in results["benchmarks"]:
        if "error" in b:
            print(f"{b['mode']:<25} ERROR: {b['error'][:30]}")
        else:
            compile_time = b.get("first_forward_ms", 0) / 1000
            speedup = b.get("speedup_vs_uncompiled", 1.0)
            print(f"{b['mode']:<25} {b['mean_ms']:<12.2f} {speedup:<10.2f}x {compile_time:<12.1f}")

    # Save to volume
    volume_path = Path("/results/benchmark_compile_jvp.json")
    volume_path.write_text(json.dumps(results, indent=2))
    results_volume.commit()

    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_full_training_step_benchmark() -> dict:
    """Benchmark full training step (forward + JVP + backward) with compile modes.

    This is more realistic than JVP-only benchmarks because it includes:
    - Full forward pass
    - JVP computation
    - Loss computation
    - Backward pass with gradients
    """
    import os
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch
    import time as time_module
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    from models.unet.unet import UNet
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

    results = {"device": torch.cuda.get_device_name(), "dtype": "bf16 (autocast)", "benchmarks": []}

    batch_size = 64
    ratio = 0.75
    n_jvp = int(batch_size * ratio)

    def run_training_step(model, optimizer, z, t, n_jvp):
        """Simulate one MeanFlow training step."""
        optimizer.zero_grad()

        def model_fn(z_in, t_in):
            time_input = make_time_input(t_in)
            unified = make_unified_flow_matching_input(z_in, time_input)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(unified)

        # Forward pass
        with torch.no_grad():
            v = model_fn(z, t)

        # JVP for MeanFlow samples
        z_jvp = z[:n_jvp]
        t_jvp = t[:n_jvp]
        v_jvp = v[:n_jvp]
        tangent_t = torch.ones_like(t_jvp)

        # This needs gradients for backward
        z_jvp_grad = z_jvp.clone().requires_grad_(True)
        t_jvp_grad = t_jvp.clone().requires_grad_(True)

        def model_fn_grad(z_in, t_in):
            time_input = make_time_input(t_in)
            unified = make_unified_flow_matching_input(z_in, time_input)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(unified)

        # JVP computation
        _, jvp_output = torch.func.jvp(
            model_fn_grad, (z_jvp_grad, t_jvp_grad), (v_jvp.float().detach(), tangent_t)
        )

        # Simple loss (MSE between JVP output and target)
        target = torch.zeros_like(jvp_output)
        loss = torch.nn.functional.mse_loss(jvp_output.float(), target)

        # Backward
        loss.backward()
        optimizer.step()

        return loss.item()

    def benchmark_training_mode(compile_mode=None):
        """Benchmark training with specific compile mode."""
        mode_name = f"compiled-{compile_mode}" if compile_mode else "uncompiled"
        print()
        print("=" * 50)
        print(f"Testing: {mode_name} (full training step)")
        print("=" * 50)

        torch.cuda.empty_cache()
        if compile_mode:
            torch._dynamo.reset()

        # Create model and optimizer
        model = UNet(
            in_channels=3, out_channels=3, base_channels=128,
            num_res_blocks=2, attention_resolutions=[16, 8],
            channel_mult=[1, 2, 2, 2], num_heads=4, time_channels=2,
        ).to(device)

        if compile_mode:
            print(f"Compiling with mode={compile_mode}...")
            model = torch.compile(model, mode=compile_mode)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Generate data
        z = torch.randn(batch_size, 3, 32, 32, device=device)
        t = torch.rand(batch_size, 1, device=device) * 0.8 + 0.1

        # Extended warmup for max-autotune (20 iterations per Codex recommendation)
        print("Warmup (20 iterations)...")
        compile_start = time_module.perf_counter()
        first_step_ms = None
        for i in range(20):
            loss = run_training_step(model, optimizer, z, t, n_jvp)
            torch.cuda.synchronize()
            if i == 0:
                first_step_ms = (time_module.perf_counter() - compile_start) * 1000
                print(f"  First step: {first_step_ms:.0f}ms (includes compilation)")

        # Benchmark
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time_module.perf_counter()
            loss = run_training_step(model, optimizer, z, t, n_jvp)
            torch.cuda.synchronize()
            times.append(time_module.perf_counter() - start)

        mean_ms = sum(times) / len(times) * 1000
        print(f"  Training step after warmup: {mean_ms:.2f}ms")

        return {
            "mode": mode_name,
            "compile_mode": compile_mode,
            "batch_size": batch_size,
            "n_jvp": n_jvp,
            "mean_ms": mean_ms,
            "first_step_ms": first_step_ms,
        }

    # Test uncompiled
    uncompiled_result = benchmark_training_mode(compile_mode=None)
    results["benchmarks"].append(uncompiled_result)
    uncompiled_ms = uncompiled_result["mean_ms"]

    # Test compile modes
    for compile_mode in ["default", "reduce-overhead", "max-autotune"]:
        try:
            result = benchmark_training_mode(compile_mode=compile_mode)
            result["speedup_vs_uncompiled"] = uncompiled_ms / result["mean_ms"]
            print(f"  Speedup vs uncompiled: {result['speedup_vs_uncompiled']:.2f}x")
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            results["benchmarks"].append({
                "mode": f"compiled-{compile_mode}",
                "error": str(e),
            })

    # Print summary
    print()
    print("=" * 60)
    print("FULL TRAINING STEP BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<25} {'Step (ms)':<12} {'Speedup':<10} {'Compile (s)':<12}")
    print("-" * 60)
    for b in results["benchmarks"]:
        if "error" in b:
            print(f"{b['mode']:<25} ERROR: {b['error'][:30]}")
        else:
            compile_time = b.get("first_step_ms", 0) / 1000
            speedup = b.get("speedup_vs_uncompiled", 1.0)
            print(f"{b['mode']:<25} {b['mean_ms']:<12.2f} {speedup:<10.2f}x {compile_time:<12.1f}")

    # Save
    volume_path = Path("/results/benchmark_full_training.json")
    volume_path.write_text(json.dumps(results, indent=2))
    results_volume.commit()

    return results


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/results": results_volume},
)
def get_benchmark_results() -> dict:
    """Get all benchmark results from volume."""
    results = {}
    results_dir = Path("/results")

    for json_file in results_dir.glob("*.json"):
        try:
            results[json_file.stem] = json.loads(json_file.read_text())
        except Exception as e:
            results[json_file.stem] = {"error": str(e)}

    return results


def run_all_benchmarks():
    """Run all benchmarks in parallel and collect results."""
    print("Starting all benchmarks in parallel on Modal A100...")
    print()

    # Run benchmarks in parallel
    baseline_handle = run_jvp_baseline_benchmark.spawn()
    split_handle = run_split_jvp_benchmark.spawn()
    microbatch_handle = run_microbatch_benchmark.spawn()

    print("Benchmarks spawned. Waiting for results...")
    print()

    # Collect results
    results = {}

    print("="*60)
    print("BASELINE BENCHMARK")
    print("="*60)
    try:
        results["baseline"] = baseline_handle.get()
        print("Baseline benchmark completed successfully")
    except Exception as e:
        print(f"Baseline benchmark failed: {e}")
        results["baseline"] = {"error": str(e)}

    print()
    print("="*60)
    print("SPLIT JVP BENCHMARK")
    print("="*60)
    try:
        results["split_jvp"] = split_handle.get()
        print("Split JVP benchmark completed successfully")
    except Exception as e:
        print(f"Split JVP benchmark failed: {e}")
        results["split_jvp"] = {"error": str(e)}

    print()
    print("="*60)
    print("MICROBATCH BENCHMARK")
    print("="*60)
    try:
        results["microbatch"] = microbatch_handle.get()
        print("Microbatch benchmark completed successfully")
    except Exception as e:
        print(f"Microbatch benchmark failed: {e}")
        results["microbatch"] = {"error": str(e)}

    return results


def run_single_benchmark(name: str):
    """Run a single benchmark by name."""
    benchmarks = {
        "baseline": run_jvp_baseline_benchmark,
        "split": run_split_jvp_benchmark,
        "microbatch": run_microbatch_benchmark,
        "compile": run_compile_jvp_benchmark,
        "training": run_full_training_step_benchmark,
    }

    if name not in benchmarks:
        print(f"Unknown benchmark: {name}")
        print(f"Available: {list(benchmarks.keys())}")
        return None

    print(f"Running {name} benchmark on Modal A100...")
    return benchmarks[name].remote()


def save_results_locally(results: dict, output_dir: Path = Path("outputs")):
    """Save results to local files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save combined results
    combined_path = output_dir / f"jvp_benchmarks_{timestamp}.json"
    combined_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {combined_path}")

    # Also save individual results
    for name, data in results.items():
        if "error" not in data:
            individual_path = output_dir / f"benchmark_{name}.json"
            individual_path.write_text(json.dumps(data, indent=2))


def print_summary(results: dict):
    """Print a summary of benchmark results."""
    print()
    print("="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    # Baseline summary
    if "baseline" in results and "benchmarks" in results["baseline"]:
        print("\n--- JVP Baseline (time vs ratio) ---")
        print(f"{'Batch':<8} {'Ratio':<8} {'Time (ms)':<12} {'Memory (GB)':<12} {'Overhead':<10}")
        print("-"*50)
        for b in results["baseline"]["benchmarks"]:
            if "error" in b:
                print(f"{b.get('batch_size', '?'):<8} {b.get('ratio', '?'):<8} {b['error']}")
            else:
                print(f"{b['batch_size']:<8} {b['ratio']:<8.2f} {b['mean_ms']:<12.2f} {b['mem_gb']:<12.2f} {b.get('overhead_vs_linear', 1.0):<10.2f}")

    # Split JVP summary
    if "split_jvp" in results and "benchmarks" in results["split_jvp"]:
        print("\n--- Split JVP (combined vs separate ∂v/∂z + ∂v/∂t) ---")
        print(f"{'Batch':<8} {'Combined (ms)':<15} {'Split (ms)':<12} {'Speedup':<10}")
        print("-"*45)
        for b in results["split_jvp"]["benchmarks"]:
            if "error" in b:
                print(f"{b.get('batch_size', '?'):<8} {b['error']}")
            else:
                print(f"{b['batch_size']:<8} {b['combined_ms']:<15.2f} {b['split_ms']:<12.2f} {b['speedup']:<10.2f}x")

    # Microbatch summary
    if "microbatch" in results and "benchmarks" in results["microbatch"]:
        print("\n--- Micro-batching (total 128 samples, ratio=0.75) ---")
        print(f"{'Micro-batch':<12} {'N batches':<10} {'Time (ms)':<12} {'Memory (GB)':<12}")
        print("-"*46)
        for b in results["microbatch"]["benchmarks"]:
            if "error" in b:
                print(f"{b.get('micro_batch_size', '?'):<12} {b['error']}")
            else:
                print(f"{b['micro_batch_size']:<12} {b['num_microbatches']:<10} {b['mean_ms']:<12.2f} {b['mem_gb']:<12.2f}")


@app.local_entrypoint()
def main(benchmark: str = "all"):
    """Run benchmarks on Modal."""
    if benchmark == "all":
        results = run_all_benchmarks()
    else:
        results = {benchmark: run_single_benchmark(benchmark)}

    save_results_locally(results)
    print_summary(results)


if __name__ == "__main__":
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "all"

    with app.run():
        if benchmark == "all":
            results = run_all_benchmarks()
        else:
            results = {benchmark: run_single_benchmark(benchmark)}

        save_results_locally(results)
        print_summary(results)
