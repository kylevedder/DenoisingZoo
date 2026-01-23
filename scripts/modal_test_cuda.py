"""
Modal script to run precision tests on CUDA.
Usage: modal run scripts/modal_test_cuda.py
"""

import modal

app = modal.App("denoisingzoo-test-cuda")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install("pytest>=8.0")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6,<2.7'"
    )
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cuda_tests():
    """Run precision tests on CUDA GPU."""
    import subprocess
    import sys
    import os

    # Create test files in container
    helpers_code = '''
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PrecisionSettings:
    autocast_dtype: torch.dtype | None
    use_scaler: bool
    device_type: str


def build_precision_settings(precision: str, device: torch.device) -> PrecisionSettings:
    """Build precision settings for mixed-precision training."""
    p = precision.lower()

    supported_devices = ("cuda", "mps", "cpu")
    if device.type not in supported_devices:
        raise ValueError(
            f"Unsupported device type: {device.type}. "
            f"Supported devices: {', '.join(supported_devices)}"
        )

    device_type = device.type

    if p in {"fp32", "float32"}:
        return PrecisionSettings(None, False, device_type)

    if p in {"bf16", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "bf16 precision requires Ampere or newer GPU. "
                "Your GPU does not support bf16. Use fp16 or fp32 instead."
            )
        return PrecisionSettings(torch.bfloat16, False, device_type)

    if p in {"fp16", "float16", "half"}:
        if device.type != "cuda":
            raise ValueError(
                f"fp16 precision requires CUDA device, got {device.type}. "
                "Use bf16 for mixed-precision on MPS/CPU."
            )
        return PrecisionSettings(torch.float16, True, "cuda")

    raise ValueError(f"Unknown precision: {precision}. Choose from: fp32, bf16, fp16")
'''

    test_code = '''
import pytest
import torch
from helpers import build_precision_settings, PrecisionSettings


class TestCudaPrecision:
    """Tests that require CUDA hardware."""

    def test_cuda_available(self) -> None:
        """Verify CUDA is available."""
        assert torch.cuda.is_available(), "CUDA should be available"
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    def test_fp32_cuda(self) -> None:
        """FP32 on CUDA should have no autocast."""
        device = torch.device("cuda")
        settings = build_precision_settings("fp32", device)

        assert settings.autocast_dtype is None
        assert settings.use_scaler is False
        assert settings.device_type == "cuda"

    def test_fp16_cuda(self) -> None:
        """FP16 on CUDA should use fp16 autocast with scaler."""
        device = torch.device("cuda")
        settings = build_precision_settings("fp16", device)

        assert settings.autocast_dtype == torch.float16
        assert settings.use_scaler is True
        assert settings.device_type == "cuda"

    def test_bf16_cuda_if_supported(self) -> None:
        """BF16 on CUDA should work if hardware supports it."""
        device = torch.device("cuda")

        if torch.cuda.is_bf16_supported():
            settings = build_precision_settings("bf16", device)
            assert settings.autocast_dtype == torch.bfloat16
            assert settings.use_scaler is False
            assert settings.device_type == "cuda"
        else:
            with pytest.raises(ValueError, match="bf16 precision requires Ampere"):
                build_precision_settings("bf16", device)

    def test_bf16_autocast_cuda(self) -> None:
        """Test bf16 autocast on CUDA produces finite outputs."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip("GPU does not support bf16")

        device = torch.device("cuda")
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        ).to(device)
        x = torch.randn(4, 10, device=device)

        settings = build_precision_settings("bf16", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)

        assert torch.isfinite(output).all()
        print(f"BF16 autocast output dtype: {output.dtype}")

    def test_fp16_autocast_cuda(self) -> None:
        """Test fp16 autocast on CUDA produces finite outputs."""
        device = torch.device("cuda")
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        ).to(device)
        x = torch.randn(4, 10, device=device)

        settings = build_precision_settings("fp16", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)

        assert torch.isfinite(output).all()
        print(f"FP16 autocast output dtype: {output.dtype}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    # Write files
    os.makedirs("/tmp/test_precision", exist_ok=True)
    with open("/tmp/test_precision/helpers.py", "w") as f:
        f.write(helpers_code)
    with open("/tmp/test_precision/test_cuda_precision.py", "w") as f:
        f.write(test_code)

    # Run tests
    os.chdir("/tmp/test_precision")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_cuda_precision.py", "-v"],
        capture_output=True,
        text=True
    )

    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.returncode == 0


@app.local_entrypoint()
def main():
    """Run CUDA precision tests on Modal."""
    print("Running CUDA precision tests on Modal T4 GPU...")
    success = run_cuda_tests.remote()
    if success:
        print("\n✓ All CUDA tests passed!")
    else:
        print("\n✗ Some CUDA tests failed!")
        raise SystemExit(1)
