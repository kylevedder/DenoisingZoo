"""Tests for precision settings and bf16 mixed-precision support.

Tests the build_precision_settings function and verifies bf16 autocast
works correctly across different devices.
"""

from __future__ import annotations

import pytest
import torch

from helpers import build_precision_settings, PrecisionSettings


class TestBuildPrecisionSettings:
    """Tests for build_precision_settings function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp32_cuda(self) -> None:
        """FP32 on CUDA should have no autocast."""
        device = torch.device("cuda")
        settings = build_precision_settings("fp32", device)

        assert settings.autocast_dtype is None
        assert settings.use_scaler is False
        assert settings.device_type == "cuda"

    def test_fp32_mps(self) -> None:
        """FP32 on MPS should have no autocast with mps device_type."""
        device = torch.device("mps")
        settings = build_precision_settings("fp32", device)

        assert settings.autocast_dtype is None
        assert settings.use_scaler is False
        assert settings.device_type == "mps"

    def test_fp32_cpu(self) -> None:
        """FP32 on CPU should have no autocast."""
        device = torch.device("cpu")
        settings = build_precision_settings("fp32", device)

        assert settings.autocast_dtype is None
        assert settings.use_scaler is False
        assert settings.device_type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="CUDA with bf16 support not available"
    )
    def test_bf16_cuda(self) -> None:
        """BF16 on CUDA should use bf16 autocast without scaler."""
        device = torch.device("cuda")
        settings = build_precision_settings("bf16", device)

        assert settings.autocast_dtype == torch.bfloat16
        assert settings.use_scaler is False
        assert settings.device_type == "cuda"

    def test_bf16_mps(self) -> None:
        """BF16 on MPS should use bf16 autocast with mps device_type."""
        device = torch.device("mps")
        settings = build_precision_settings("bf16", device)

        assert settings.autocast_dtype == torch.bfloat16
        assert settings.use_scaler is False
        assert settings.device_type == "mps"

    def test_bf16_cpu(self) -> None:
        """BF16 on CPU should use bf16 autocast (slow but works)."""
        device = torch.device("cpu")
        settings = build_precision_settings("bf16", device)

        assert settings.autocast_dtype == torch.bfloat16
        assert settings.use_scaler is False
        assert settings.device_type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_cuda(self) -> None:
        """FP16 on CUDA should use fp16 autocast with scaler."""
        device = torch.device("cuda")
        settings = build_precision_settings("fp16", device)

        assert settings.autocast_dtype == torch.float16
        assert settings.use_scaler is True
        assert settings.device_type == "cuda"

    def test_fp16_mps_raises(self) -> None:
        """FP16 on MPS should raise ValueError."""
        device = torch.device("mps")
        with pytest.raises(ValueError, match="fp16 precision requires CUDA"):
            build_precision_settings("fp16", device)

    def test_fp16_cpu_raises(self) -> None:
        """FP16 on CPU should raise ValueError."""
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="fp16 precision requires CUDA"):
            build_precision_settings("fp16", device)

    def test_unknown_precision_raises(self) -> None:
        """Unknown precision should raise ValueError."""
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="Unknown precision"):
            build_precision_settings("fp8", device)

    @pytest.mark.parametrize("alias", ["float16", "half"])
    def test_fp16_aliases_raise_on_cpu(self, alias: str) -> None:
        """FP16 aliases should also raise on non-CUDA."""
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="fp16 precision requires CUDA"):
            build_precision_settings(alias, device)

    @pytest.mark.parametrize("alias", ["float16", "half"])
    def test_fp16_aliases_raise_on_mps(self, alias: str) -> None:
        """FP16 aliases should also raise on MPS."""
        device = torch.device("mps")
        with pytest.raises(ValueError, match="fp16 precision requires CUDA"):
            build_precision_settings(alias, device)

    def test_unsupported_device_raises(self) -> None:
        """Unsupported device types should raise ValueError."""
        device = torch.device("meta")
        with pytest.raises(ValueError, match="Unsupported device type"):
            build_precision_settings("fp32", device)

    @pytest.mark.parametrize("alias,expected", [
        ("fp32", None),
        ("float32", None),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ])
    def test_precision_aliases(self, alias: str, expected: torch.dtype | None) -> None:
        """Test that precision aliases work correctly."""
        device = torch.device("cpu")
        settings = build_precision_settings(alias, device)
        assert settings.autocast_dtype == expected


class TestBf16Autocast:
    """Tests for bf16 autocast functionality."""

    @pytest.fixture
    def simple_model(self) -> torch.nn.Module:
        """Create a simple model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        )

    def test_bf16_autocast_cpu(self, simple_model: torch.nn.Module) -> None:
        """Test bf16 autocast on CPU produces finite outputs."""
        device = torch.device("cpu")
        model = simple_model.to(device)
        x = torch.randn(4, 10, device=device)

        settings = build_precision_settings("bf16", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)

        assert torch.isfinite(output).all()
        # Output dtype depends on backend - CPU bf16 autocast may produce bf16 output
        assert output.dtype in (torch.float32, torch.bfloat16)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_bf16_autocast_mps(self, simple_model: torch.nn.Module) -> None:
        """Test bf16 autocast on MPS produces finite outputs."""
        device = torch.device("mps")
        model = simple_model.to(device)
        x = torch.randn(4, 10, device=device)

        settings = build_precision_settings("bf16", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)

        assert torch.isfinite(output).all()

    def test_fp32_no_autocast(self, simple_model: torch.nn.Module) -> None:
        """Test fp32 runs without autocast."""
        device = torch.device("cpu")
        model = simple_model.to(device)
        x = torch.randn(4, 10, device=device)

        settings = build_precision_settings("fp32", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)

        assert torch.isfinite(output).all()
        assert output.dtype == torch.float32


class TestBf16Gradients:
    """Tests for gradient computation under bf16 autocast."""

    def test_bf16_gradients_finite(self) -> None:
        """Test that gradients computed under bf16 autocast are finite."""
        device = torch.device("cpu")
        model = torch.nn.Linear(10, 10).to(device)
        x = torch.randn(4, 10, device=device, requires_grad=False)
        target = torch.randn(4, 10, device=device)

        settings = build_precision_settings("bf16", device)

        with torch.autocast(
            device_type=settings.device_type,
            dtype=settings.autocast_dtype,
            enabled=settings.autocast_dtype is not None,
        ):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)

        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_bf16_loss_close_to_fp32(self) -> None:
        """Test that bf16 loss is close to fp32 loss."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = torch.nn.Linear(10, 10).to(device)
        x = torch.randn(4, 10, device=device)
        target = torch.randn(4, 10, device=device)

        # FP32 loss
        fp32_settings = build_precision_settings("fp32", device)
        with torch.autocast(
            device_type=fp32_settings.device_type,
            dtype=fp32_settings.autocast_dtype,
            enabled=fp32_settings.autocast_dtype is not None,
        ):
            fp32_output = model(x)
            fp32_loss = torch.nn.functional.mse_loss(fp32_output, target)

        # BF16 loss
        bf16_settings = build_precision_settings("bf16", device)
        with torch.autocast(
            device_type=bf16_settings.device_type,
            dtype=bf16_settings.autocast_dtype,
            enabled=bf16_settings.autocast_dtype is not None,
        ):
            bf16_output = model(x)
            bf16_loss = torch.nn.functional.mse_loss(bf16_output, target)

        # BF16 loss should be close to FP32 (within 1%)
        assert torch.isclose(bf16_loss, fp32_loss, rtol=0.01, atol=0.01)
