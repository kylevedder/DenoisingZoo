"""Tests for model contract helpers."""

import pytest

import helpers
from model_contracts import TimeChannelModule


def test_time_channel_module_sets_default():
    class Model(TimeChannelModule):
        def __init__(self):
            super().__init__()

    model = Model()
    assert model.time_channels == 2


def test_time_channel_module_mismatch_raises():
    class Model(TimeChannelModule):
        def __init__(self):
            super().__init__(time_channels=1)

    with pytest.raises(ValueError, match="time_channels"):
        Model()


def test_has_mps_backend_returns_bool():
    assert isinstance(helpers.has_mps_backend(), bool)


def test_is_mps_available_returns_bool():
    assert isinstance(helpers.is_mps_available(), bool)


def test_is_mps_available_false_when_backend_disabled(monkeypatch):
    monkeypatch.setattr(helpers, "has_mps_backend", lambda: False)
    assert helpers.is_mps_available() is False
