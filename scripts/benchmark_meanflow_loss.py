"""Microbenchmark for MeanFlow loss computation.

Measures the overhead of JVP computation vs standard flow matching loss.
According to MeanFlow paper Appendix B.4, JVP should add ~16% wall clock time.

This benchmark directly uses the MeanFlowLoss class to ensure we measure
the actual implementation, not a copy.

Usage:
    python scripts/benchmark_meanflow_loss.py
    python scripts/benchmark_meanflow_loss.py --device cuda --batch-size 64
    python scripts/benchmark_meanflow_loss.py --model unet
    python scripts/benchmark_meanflow_loss.py --compile  # Test with torch.compile (model only)
    python scripts/benchmark_meanflow_loss.py --compile-loss --compile-mode reduce-overhead
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from losses.meanflow_loss import MeanFlowLoss


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    n_runs: int
    first_ms: float | None = None

    def __str__(self) -> str:
        first = f" (first {self.first_ms:.0f} ms)" if self.first_ms is not None else ""
        return f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms{first}"


def sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_fn(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    n_runs: int = 100,
    name: str = "unnamed",
    measure_first: bool = False,
) -> BenchmarkResult:
    """Benchmark a function with proper warmup and synchronization."""
    first_ms = None
    if measure_first:
        sync_device(device)
        start = time.perf_counter()
        fn()
        sync_device(device)
        end = time.perf_counter()
        first_ms = (end - start) * 1000

    # Warmup
    for _ in range(warmup):
        fn()
        sync_device(device)

    # Benchmark
    times_ms: list[float] = []
    for _ in range(n_runs):
        sync_device(device)
        start = time.perf_counter()
        fn()
        sync_device(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    times_t = torch.tensor(times_ms)
    return BenchmarkResult(
        name=name,
        mean_ms=times_t.mean().item(),
        std_ms=times_t.std().item(),
        n_runs=n_runs,
        first_ms=first_ms,
    )


def create_unet_model(
    device: torch.device, compile_model: bool = False, compile_mode: str = "default"
) -> nn.Module:
    """Create UNet model for benchmarking.

    Model is always created in fp32 - use autocast for mixed precision.
    """
    from models.unet.unet import UNet

    model = UNet(
        in_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
    )
    model = model.to(device)  # Keep in fp32, use autocast for mixed precision
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    return model


def create_batch(
    batch_size: int,
    img_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching MeanFlowLoss expected format."""
    x = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
    y = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
    return {
        "raw_source": x,  # Noise
        "raw_target": y,  # Data
    }


def run_unet_benchmarks(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    n_runs: int,
    warmup: int,
    compile_model: bool = False,
    compile_loss: bool = False,
    compile_mode: str = "default",
) -> list[BenchmarkResult]:
    """Run benchmarks for UNet model using actual MeanFlowLoss class.

    Benchmark breakdown:
    1. Forward only (no_grad) - baseline forward pass cost
    2. Standard FM (fwd+bwd) - typical flow matching training step (MSE loss)
    3. MeanFlowLoss 0% ratio - MeanFlowLoss with ratio=0 (equivalent to standard FM)
    4. MeanFlowLoss 25% ratio - matches paper's training setup
    5. MeanFlowLoss 100% ratio - all samples use MeanFlow (worst case)
    """
    results: list[BenchmarkResult] = []

    # Model stays in fp32; use autocast for mixed precision
    model = create_unet_model(device, compile_model, compile_mode)
    img_size = 32

    # Determine if we need autocast (for bf16/fp16)
    use_autocast = dtype != torch.float32
    autocast_dtype = dtype if use_autocast else None

    # Create batch in target dtype
    batch = create_batch(batch_size, img_size, device, dtype)
    x = batch["raw_source"]
    y = batch["raw_target"]

    # Pre-compute unified input for forward-only benchmark
    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input
    t = torch.rand(batch_size, 1, device=device, dtype=dtype).clamp(0.01, 0.99)
    t_broad = t.view(-1, 1, 1, 1)
    z_t = (1 - t_broad) * x + t_broad * y
    v_true = y - x
    time_input = make_time_input(t)
    unified_input = make_unified_flow_matching_input(z_t, time_input)

    # --- Benchmark 1: Forward only (no_grad) ---
    def forward_only():
        with torch.no_grad():
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                return model(unified_input)

    results.append(benchmark_fn(
        forward_only, device, warmup=warmup, n_runs=n_runs,
        name="1. Forward only (no_grad)"
    ))

    # --- Benchmark 2: Standard FM loss (MSE, fwd+bwd) ---
    def standard_fm_loss():
        model.zero_grad()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            pred = model(unified_input)
            loss = F.mse_loss(pred, v_true)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        standard_fm_loss, device, warmup=warmup, n_runs=n_runs,
        name="2. Standard FM MSE (fwd+bwd)"
    ))

    # --- Benchmark 3: MeanFlowLoss with ratio=0 (no JVP, should match standard FM) ---
    loss_fn_0pct = MeanFlowLoss(model=model, meanflow_ratio=0.0)

    def meanflow_loss_0pct():
        model.zero_grad()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = loss_fn_0pct(batch, device=device)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        meanflow_loss_0pct, device, warmup=warmup, n_runs=n_runs,
        name="3. MeanFlowLoss 0% ratio (no JVP)"
    ))

    # --- Benchmark 4: MeanFlowLoss with ratio=0.25 (paper's default) ---
    loss_fn_25pct = MeanFlowLoss(model=model, meanflow_ratio=0.25)

    def meanflow_loss_25pct():
        model.zero_grad()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = loss_fn_25pct(batch, device=device)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        meanflow_loss_25pct, device, warmup=warmup, n_runs=n_runs,
        name="4. MeanFlowLoss 25% ratio (paper default)"
    ))

    # --- Benchmark 5: MeanFlowLoss with ratio=1.0 (all samples use JVP) ---
    loss_fn_100pct = MeanFlowLoss(model=model, meanflow_ratio=1.0)

    def meanflow_loss_100pct():
        model.zero_grad()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = loss_fn_100pct(batch, device=device)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        meanflow_loss_100pct, device, warmup=warmup, n_runs=n_runs,
        name="5. MeanFlowLoss 100% ratio (all JVP)"
    ))

    # --- Benchmark 5b: MeanFlowLoss 100% with full_batch_jvp mode ---
    loss_fn_fullbatch = MeanFlowLoss(model=model, meanflow_ratio=1.0, full_batch_jvp=True)

    def meanflow_loss_fullbatch():
        model.zero_grad()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss = loss_fn_fullbatch(batch, device=device)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        meanflow_loss_fullbatch, device, warmup=warmup, n_runs=n_runs,
        name="5b. MeanFlowLoss 100% (full_batch_jvp)"
    ))

    # --- Benchmark 5c: MeanFlowLoss 100% with hybrid CUDA graph mode (CUDA only) ---
    if device.type == "cuda":
        loss_fn_cuda_graph = MeanFlowLoss(
            model=model, meanflow_ratio=1.0, full_batch_jvp=True, use_cuda_graph=True
        )

        def meanflow_loss_cuda_graph():
            model.zero_grad()
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss = loss_fn_cuda_graph(batch, device=device)
            loss.backward()
            return loss

        results.append(benchmark_fn(
            meanflow_loss_cuda_graph, device, warmup=warmup, n_runs=n_runs,
            name="5c. MeanFlowLoss 100% (hybrid CUDA graph)"
        ))

    # --- Benchmark 5d: MeanFlowLoss 100% compiled loss (wrap entire loss forward) ---
    if compile_loss:
        torch._dynamo.reset()
        loss_fn_compiled = MeanFlowLoss(
            model=model, meanflow_ratio=1.0, full_batch_jvp=True
        )

        def loss_forward(x_in: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
            batch_in = {"raw_source": x_in, "raw_target": y_in}
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                return loss_fn_compiled(batch_in, device=device)

        compiled_loss_forward = torch.compile(loss_forward, mode=compile_mode)

        def meanflow_loss_compiled():
            model.zero_grad()
            loss = compiled_loss_forward(x, y)
            loss.backward()
            return loss

        results.append(benchmark_fn(
            meanflow_loss_compiled, device, warmup=warmup, n_runs=n_runs,
            name=f"5d. MeanFlowLoss 100% (compiled loss, mode={compile_mode})",
            measure_first=True,
        ))

    # --- Benchmark 6: Isolate JVP cost via _compute_target ---
    # Call the internal method directly to measure JVP overhead
    loss_fn_for_jvp = MeanFlowLoss(model=model, meanflow_ratio=1.0)

    # Pre-compute inputs for _compute_target
    t_jvp, r_jvp = loss_fn_for_jvp._sample_two_timesteps(batch_size, device, dtype)
    t_broad_jvp = t_jvp.view(-1, 1, 1, 1)
    z_t_jvp = (1 - t_broad_jvp) * x + t_broad_jvp * y
    use_meanflow = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

    def jvp_compute_target_only():
        return loss_fn_for_jvp._compute_target(z_t_jvp, t_jvp, r_jvp, v_true, use_meanflow)

    results.append(benchmark_fn(
        jvp_compute_target_only, device, warmup=warmup, n_runs=n_runs,
        name="6. _compute_target (JVP only)"
    ))

    # --- Benchmark 7: CUDA graph captured JVP (CUDA only) ---
    if device.type == "cuda":
        # Create static buffers for CUDA graph capture (all float32 for JVP)
        static_z = z_t_jvp.float().clone()
        static_r = r_jvp.float().clone()
        static_t = t_jvp.float().clone()
        static_tang_z = v_true.float().clone()
        static_tang_r = torch.zeros_like(r_jvp).float()
        static_tang_t = torch.ones_like(t_jvp).float()

        # Pure JVP function (no conditionals, no data-dependent branches)
        def pure_jvp_func(z_in, r_in, t_in):
            time_input = torch.cat([r_in, t_in], dim=1)
            unified = make_unified_flow_matching_input(z_in, time_input)
            return model(unified)

        # Warmup for CUDA graph capture
        for _ in range(3):
            torch.func.jvp(
                pure_jvp_func,
                (static_z, static_r, static_t),
                (static_tang_z, static_tang_r, static_tang_t),
            )
            torch.cuda.synchronize()

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                static_primal, static_tangent = torch.func.jvp(
                    pure_jvp_func,
                    (static_z, static_r, static_t),
                    (static_tang_z, static_tang_r, static_tang_t),
                )

            def jvp_cuda_graph():
                # Copy new data into static buffers
                static_z.copy_(z_t_jvp.float())
                static_r.copy_(r_jvp.float())
                static_t.copy_(t_jvp.float())
                static_tang_z.copy_(v_true.float())
                # Replay the graph
                g.replay()
                return static_primal, static_tangent

            results.append(benchmark_fn(
                jvp_cuda_graph, device, warmup=warmup, n_runs=n_runs,
                name="7. Pure JVP (CUDA graph)"
            ))
        except RuntimeError as e:
            print(f"\n  CUDA graph capture failed: {e}")
            print("  Skipping CUDA graph benchmark.")

        # --- Benchmark 8: Hybrid approach - graphed JVP (detached) + forward (fwd+bwd) ---
        # Theory: CUDA graphs can't capture backward, but we can:
        # 1. Use graphed JVP to compute target (detached)
        # 2. Run separate forward pass for prediction (with gradients)
        # Total: ~52ms (graphed JVP) + ~45ms (fwd+bwd) = ~97ms vs 145ms current
        try:
            # Static buffers for target computation
            static_delta_t = (static_t - static_r).view(-1, 1, 1, 1)

            def hybrid_meanflow_loss():
                model.zero_grad()
                with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                    # Step 1: Use graphed JVP for target (detached, no gradients)
                    static_z.copy_(z_t_jvp.float())
                    static_r.copy_(r_jvp.float())
                    static_t.copy_(t_jvp.float())
                    static_tang_z.copy_(v_true.float())
                    g.replay()
                    # Compute target: u_tgt = v - delta_t * dudt (all detached)
                    u_tgt = (v_true.float() - static_delta_t * static_tangent).to(dtype).detach()

                    # Step 2: Separate forward pass for prediction (with gradients)
                    time_pred = torch.cat([r_jvp, t_jvp], dim=1)
                    unified = make_unified_flow_matching_input(z_t_jvp, time_pred)
                    u_pred = model(unified)

                    # Step 3: Compute loss
                    loss = F.mse_loss(u_pred, u_tgt)
                loss.backward()
                return loss

            results.append(benchmark_fn(
                hybrid_meanflow_loss, device, warmup=warmup, n_runs=n_runs,
                name="8. Hybrid: graphed JVP (target) + forward (pred)"
            ))
        except RuntimeError as e:
            print(f"\n  Hybrid approach failed: {e}")

    return results


def print_results(results: list[BenchmarkResult], title: str) -> None:
    """Print benchmark results with analysis."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    for r in results:
        print(f"  {r}")

    # Compute analysis
    if len(results) >= 5:
        forward = results[0].mean_ms
        standard_fm = results[1].mean_ms
        mf_0pct = results[2].mean_ms
        mf_25pct = results[3].mean_ms
        mf_100pct = results[4].mean_ms

        backward_est = standard_fm - forward

        print(f"\n  Cost Breakdown:")
        print(f"    Forward pass:        {forward:.3f} ms")
        print(f"    Backward pass (est): {backward_est:.3f} ms")

        print(f"\n  MeanFlowLoss Overhead vs Standard FM:")
        overhead_0 = (mf_0pct - standard_fm) / standard_fm * 100
        overhead_25 = (mf_25pct - standard_fm) / standard_fm * 100
        overhead_100 = (mf_100pct - standard_fm) / standard_fm * 100
        print(f"    0% ratio (no JVP):   {overhead_0:+.1f}% overhead")
        print(f"    25% ratio (paper):   {overhead_25:+.1f}% overhead (paper claims ~16%)")
        print(f"    100% ratio (all):    {overhead_100:+.1f}% overhead")

        jvp_result = next((r for r in results if r.name.startswith("6. _compute_target")), None)
        if jvp_result is not None:
            jvp_only = jvp_result.mean_ms
            print(f"\n  JVP Cost Analysis:")
            print(f"    _compute_target (JVP): {jvp_only:.3f} ms")
            print(f"    JVP as % of forward:   {100*jvp_only/forward:.1f}%")
            print(f"    JVP as % of standard:  {100*jvp_only/standard_fm:.1f}%")


def get_default_device() -> str:
    """Auto-detect the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Benchmark MeanFlow loss computation")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on model")
    parser.add_argument("--compile-loss", action="store_true", help="Use torch.compile on loss forward")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--log-cudagraphs", action="store_true", help="Enable cudagraph logging")
    args = parser.parse_args()

    device_str = args.device if args.device else get_default_device()
    device = torch.device(device_str)

    # Enable TF32 for CUDA (significant speedup for FP32 ops on Ampere+)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Benchmark Configuration:")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  N runs: {args.n_runs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  torch.compile: {args.compile}")
    print(f"  compile_loss: {args.compile_loss}")
    print(f"  compile_mode: {args.compile_mode}")
    if device.type == "cuda":
        print(f"  TF32 enabled: True")

    if args.log_cudagraphs:
        import logging
        torch._logging.set_logs(inductor=logging.INFO, cudagraphs=True)

    results = run_unet_benchmarks(
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        n_runs=args.n_runs,
        warmup=args.warmup,
        compile_model=args.compile,
        compile_loss=args.compile_loss,
        compile_mode=args.compile_mode,
    )
    print_results(results, "UNet Benchmark (CIFAR-10 scale, ~51M params)")


if __name__ == "__main__":
    main()
