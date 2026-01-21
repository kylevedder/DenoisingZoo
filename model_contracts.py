"""Model contracts and base classes."""

from __future__ import annotations

from torch import nn

from constants import TIME_CHANNELS_REQUIRED


class TimeChannelModule(nn.Module):
    def __init__(self, time_channels: int = TIME_CHANNELS_REQUIRED) -> None:
        super().__init__()
        if time_channels != TIME_CHANNELS_REQUIRED:
            raise ValueError(f"time_channels must be {TIME_CHANNELS_REQUIRED}")
        self.time_channels = time_channels


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Unwrap a torch.compile'd model to get the underlying module."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def is_time_channel_module(model: nn.Module) -> bool:
    """Check if model is a TimeChannelModule (handles compiled models)."""
    return isinstance(unwrap_compiled(model), TimeChannelModule)
