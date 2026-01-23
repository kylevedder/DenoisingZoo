"""Benchmark torch.autograd.forward_ad vs torch.func.jvp.

The hypothesis: forward_ad allows computing both u_pred (with gradients) and dudt
(tangent) in a single pass, eliminating the separate forward pass needed in the
hybrid CUDA graph approach.

Usage:
    python scripts/benchmark_forward_ad.py
    modal run scripts/modal_app.py::run_benchmark --script-name benchmark_forward_ad.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
import torch.func


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    n_runs: int

    def __str__(self) -> str:
        return f"{self.name}: {self.mean_ms:.3f} +/- {self.std_ms:.3f} ms"


def sync_device(device: torch.device) -> None:
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
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
        sync_device(device)

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
    )


def create_unet_model(device: torch.device) -> nn.Module:
    """Create UNet model for benchmarking."""
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
    parser = argparse.ArgumentParser(description="Benchmark forward_ad vs torch.func.jvp")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
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

    from dataloaders.base_dataloaders import make_time_input, make_unified_flow_matching_input

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

    # Pre-compute unified input
    time_input = torch.cat([r, t], dim=1)
    unified = make_unified_flow_matching_input(z_t, time_input)

    results: list[BenchmarkResult] = []

    # --- Benchmark 1: Baseline forward + backward ---
    def baseline_fwd_bwd():
        model.zero_grad()
        u_pred = model(unified)
        loss = ((u_pred - v_true) ** 2).mean()
        loss.backward()
        return loss

    results.append(benchmark_fn(
        baseline_fwd_bwd, device, args.warmup, args.n_runs,
        name="1. Baseline forward + backward"
    ))

    # --- Benchmark 2: torch.func.jvp (current approach) ---
    def torch_func_jvp():
        def u_func(z_in, r_in, t_in):
            time_in = torch.cat([r_in, t_in], dim=1)
            unified_in = make_unified_flow_matching_input(z_in, time_in)
            return model(unified_in)

        tangent_z = v_true.float()
        tangent_r = torch.zeros_like(r).float()
        tangent_t = torch.ones_like(t).float()

        with torch.amp.autocast(device.type, enabled=False):
            u_primal, dudt = torch.func.jvp(
                u_func,
                (z_t.float(), r.float(), t.float()),
                (tangent_z, tangent_r, tangent_t),
            )
        return u_primal, dudt

    results.append(benchmark_fn(
        torch_func_jvp, device, args.warmup, args.n_runs,
        name="2. torch.func.jvp (no backward)"
    ))

    # --- Benchmark 3: torch.autograd.forward_ad single pass ---
    # Note: forward_ad works by running the forward pass with dual numbers
    # The primal output tracks gradients (can backprop), tangent output is dudt
    def forward_ad_single_pass():
        model.zero_grad()

        with fwAD.dual_level():
            # Attach tangents: dz/d_param = v, dr/d_param = 0, dt/d_param = 1
            z_dual = fwAD.make_dual(z_t, v_true)
            r_dual = fwAD.make_dual(r, torch.zeros_like(r))
            t_dual = fwAD.make_dual(t, torch.ones_like(t))

            # Build unified input from dual tensors
            time_dual = torch.cat([r_dual, t_dual], dim=1)
            time_dual_expanded = time_dual[:, :, None, None].expand(
                -1, -1, z_dual.shape[2], z_dual.shape[3]
            )
            unified_dual = torch.cat([z_dual, time_dual_expanded], dim=1)

            # Forward pass - primal has autograd graph attached
            u_dual = model(unified_dual)

            # Unpack primal and tangent
            u_primal, u_tangent = fwAD.unpack_dual(u_dual)
            dudt = u_tangent if u_tangent is not None else torch.zeros_like(u_primal)

        # Compute target (outside dual_level)
        delta_t = (t - r).view(-1, 1, 1, 1)
        u_tgt = v_true - delta_t * dudt.detach()

        # Compute loss and backward (on primal with gradients)
        loss = ((u_primal - u_tgt.detach()) ** 2).mean()
        loss.backward()
        return loss

    results.append(benchmark_fn(
        forward_ad_single_pass, device, args.warmup, args.n_runs,
        name="3. forward_ad single pass (fwd+bwd)"
    ))

    # --- Benchmark 3b: forward_ad forward only (no backward) ---
    def forward_ad_no_bwd():
        with fwAD.dual_level():
            z_dual = fwAD.make_dual(z_t, v_true)
            r_dual = fwAD.make_dual(r, torch.zeros_like(r))
            t_dual = fwAD.make_dual(t, torch.ones_like(t))
            time_dual = torch.cat([r_dual, t_dual], dim=1)
            time_dual_expanded = time_dual[:, :, None, None].expand(-1, -1, z_dual.shape[2], z_dual.shape[3])
            unified_dual = torch.cat([z_dual, time_dual_expanded], dim=1)
            u_dual = model(unified_dual)
            u_primal, u_tangent = fwAD.unpack_dual(u_dual)
        return u_primal, u_tangent

    results.append(benchmark_fn(
        forward_ad_no_bwd, device, args.warmup, args.n_runs,
        name="3b. forward_ad (forward only, no backward)"
    ))

    # --- Benchmark 4: Hybrid approach (graphed JVP + forward) ---
    # This is our current best: separate JVP (detached) + forward (with grad)
    def hybrid_approach():
        model.zero_grad()

        # JVP for target (detached)
        def u_func(z_in, r_in, t_in):
            time_in = torch.cat([r_in, t_in], dim=1)
            unified_in = make_unified_flow_matching_input(z_in, time_in)
            return model(unified_in)

        with torch.amp.autocast(device.type, enabled=False):
            _, dudt = torch.func.jvp(
                u_func,
                (z_t.float(), r.float(), t.float()),
                (v_true.float(), torch.zeros_like(r).float(), torch.ones_like(t).float()),
            )

        delta_t = (t - r).view(-1, 1, 1, 1)
        u_tgt = (v_true - delta_t * dudt).detach()

        # Forward for prediction (with gradients)
        u_pred = model(unified)
        loss = ((u_pred - u_tgt) ** 2).mean()
        loss.backward()
        return loss

    results.append(benchmark_fn(
        hybrid_approach, device, args.warmup, args.n_runs,
        name="4. Hybrid: JVP (target) + forward (pred)"
    ))

    # --- Benchmark 5: forward_ad with CUDA graph ---
    if device.type == "cuda":
        # Static buffers for CUDA graph
        static_z = z_t.clone()
        static_r = r.clone()
        static_t = t.clone()
        static_v = v_true.clone()

        def fwd_ad_compute():
            with fwAD.dual_level():
                z_dual = fwAD.make_dual(static_z, static_v)
                r_dual = fwAD.make_dual(static_r, torch.zeros_like(static_r))
                t_dual = fwAD.make_dual(static_t, torch.ones_like(static_t))

                time_dual = torch.cat([r_dual, t_dual], dim=1)
                time_dual_expanded = time_dual[:, :, None, None].expand(
                    -1, -1, static_z.shape[2], static_z.shape[3]
                )
                unified_dual = torch.cat([z_dual, time_dual_expanded], dim=1)

                u_dual = model(unified_dual)
                u_primal, u_tangent = fwAD.unpack_dual(u_dual)
            return u_primal, u_tangent

        # Warmup
        for _ in range(3):
            fwd_ad_compute()
            torch.cuda.synchronize()

        # Try to capture CUDA graph
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_primal, static_tangent = fwd_ad_compute()

            def forward_ad_cuda_graph():
                static_z.copy_(z_t)
                static_r.copy_(r)
                static_t.copy_(t)
                static_v.copy_(v_true)
                g.replay()
                return static_primal, static_tangent

            results.append(benchmark_fn(
                forward_ad_cuda_graph, device, args.warmup, args.n_runs,
                name="5. forward_ad (CUDA graph, no backward)"
            ))
        except RuntimeError as e:
            print(f"\nCUDA graph capture failed for forward_ad: {e}")

    # --- Benchmark 6: forward_ad (graphed) + backward (separate) ---
    # Key insight: forward_ad primal has gradients attached, so we can backward through it
    if device.type == "cuda" and 'g' in dir():
        backward_success = [False]  # Use list to track in closure

        def forward_ad_hybrid():
            model.zero_grad()
            # Use graphed forward_ad to get primal + tangent
            static_z.copy_(z_t)
            static_r.copy_(r)
            static_t.copy_(t)
            static_v.copy_(v_true)
            g.replay()
            # static_primal has gradients, static_tangent is dudt
            delta_t = (static_t - static_r).view(-1, 1, 1, 1)
            u_tgt = (static_v - delta_t * static_tangent).detach()
            # Note: static_primal may not have gradients attached after graph replay
            # Need to check if this works
            loss = ((static_primal - u_tgt) ** 2).mean()
            if loss.requires_grad:
                loss.backward()
                backward_success[0] = True
            return loss

        try:
            results.append(benchmark_fn(
                forward_ad_hybrid, device, args.warmup, args.n_runs,
                name="6. forward_ad (graphed fwd) + backward"
            ))
            if backward_success[0]:
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                print(f"    Backward succeeded! Grad norm: {grad_norm:.4f}")
            else:
                print(f"    WARNING: Backward did NOT run (loss.requires_grad=False)")
        except Exception as e:
            print(f"\nforward_ad hybrid failed: {e}")

    # --- Benchmark 7: Fast JVP GroupNorm approximation ---
    # Patch model with FastJVPGroupNorm and benchmark
    from models.unet.fast_jvp_norm import patch_model_with_fast_jvp_norm
    import copy

    model_fast = create_unet_model(device)
    patch_model_with_fast_jvp_norm(model_fast)

    def torch_func_jvp_fast():
        def u_func(z_in, r_in, t_in):
            time_in = torch.cat([r_in, t_in], dim=1)
            unified_in = make_unified_flow_matching_input(z_in, time_in)
            return model_fast(unified_in)

        tangent_z = v_true.float()
        tangent_r = torch.zeros_like(r).float()
        tangent_t = torch.ones_like(t).float()

        with torch.amp.autocast(device.type, enabled=False):
            u_primal, dudt = torch.func.jvp(
                u_func,
                (z_t.float(), r.float(), t.float()),
                (tangent_z, tangent_r, tangent_t),
            )
        return u_primal, dudt

    results.append(benchmark_fn(
        torch_func_jvp_fast, device, args.warmup, args.n_runs,
        name="7. torch.func.jvp (FastJVPGroupNorm)"
    ))

    # --- Benchmark 8: Full MeanFlow loss with FastJVPGroupNorm ---
    from losses.meanflow_loss import MeanFlowLoss
    loss_fn_fast = MeanFlowLoss(model=model_fast, meanflow_ratio=1.0, full_batch_jvp=True)
    batch_dict = {"raw_source": x, "raw_target": y}

    def meanflow_loss_fast():
        model_fast.zero_grad()
        loss = loss_fn_fast(batch_dict, device=device)
        loss.backward()
        return loss

    results.append(benchmark_fn(
        meanflow_loss_fast, device, args.warmup, args.n_runs,
        name="8. MeanFlowLoss 100% (FastJVPGroupNorm)"
    ))

    # Print results
    print("\n" + "=" * 70)
    print(" Forward-AD vs torch.func.jvp Benchmark")
    print("=" * 70)
    for r in results:
        print(f"  {r}")

    # Analysis
    baseline = results[0].mean_ms
    torch_jvp = results[1].mean_ms
    fwd_ad = results[2].mean_ms
    hybrid = results[3].mean_ms

    print(f"\nAnalysis:")
    print(f"  Baseline (fwd+bwd):       {baseline:.3f} ms")
    print(f"  torch.func.jvp (no bwd):  {torch_jvp:.3f} ms ({100*torch_jvp/baseline:.1f}% of baseline)")
    print(f"  forward_ad (fwd+bwd):     {fwd_ad:.3f} ms ({100*fwd_ad/baseline:.1f}% of baseline)")
    print(f"  Hybrid (JVP + fwd):       {hybrid:.3f} ms ({100*hybrid/baseline:.1f}% of baseline)")
    print(f"\nConclusion:")
    if fwd_ad < hybrid:
        print(f"  forward_ad is {100*(hybrid - fwd_ad)/hybrid:.1f}% FASTER than hybrid!")
        print(f"  Potential overhead reduction: {100*(fwd_ad/baseline - 1):.1f}%")
    else:
        print(f"  Hybrid is {100*(fwd_ad - hybrid)/fwd_ad:.1f}% faster than forward_ad")


if __name__ == "__main__":
    main()
