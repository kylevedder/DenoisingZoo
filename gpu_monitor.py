"""Async GPU metrics monitoring with rolling averages.

Runs a background thread that continuously reads from macmon's streaming output,
storing readings in a circular buffer. Provides rolling averages of recent
measurements to capture actual GPU utilization during training.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch


@dataclass
class GPUReading:
    """Single GPU metrics reading."""

    timestamp: float
    freq_mhz: float | None = None
    util_pct: float | None = None
    power_w: float | None = None
    temp_c: float | None = None


@dataclass
class GPUMetricsMonitor:
    """Background monitor for GPU metrics with rolling averages.

    Runs macmon in continuous streaming mode and reads from its stdout.
    Maintains a circular buffer of recent readings for rolling averages.

    Args:
        interval_ms: Macmon sampling interval in milliseconds (default 500)
        window_seconds: How many seconds of history to keep (default 5)
    """

    interval_ms: int = 500  # macmon sampling interval
    window_seconds: float = 5.0  # rolling window size

    _readings: Deque[GPUReading] = field(default_factory=deque, init=False)
    _last_reading: GPUReading | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _process: subprocess.Popen | None = field(default=None, init=False)
    _warned_exception_types: set[type[BaseException]] = field(
        default_factory=set, init=False
    )
    _warn_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread and macmon process."""
        self._stop_event.set()

        # Terminate macmon process
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired as exc:
                self._warn_once(exc, "macmon terminate timed out")
                try:
                    self._process.kill()
                except OSError as kill_exc:
                    self._warn_once(kill_exc, "macmon kill failed")
            except OSError as exc:
                self._warn_once(exc, "macmon terminate failed")
            self._process = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _stream_loop(self) -> None:
        """Background loop that reads from macmon's streaming output."""
        try:
            # Start macmon in continuous mode
            self._process = subprocess.Popen(
                ["macmon", "pipe", "-s", "0", "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # Line buffered
            )
        except FileNotFoundError:
            # macmon not installed
            return

        try:
            while not self._stop_event.is_set() and self._process.poll() is None:
                stdout = self._process.stdout
                if stdout is None:
                    self._warn_once(
                        RuntimeError("macmon stdout pipe is None"),
                        "macmon stream failed",
                    )
                    break
                line = stdout.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.strip())
                    reading = GPUReading(timestamp=time.monotonic())

                    if "gpu_usage" in data:
                        reading.freq_mhz = data["gpu_usage"][0]
                        reading.util_pct = data["gpu_usage"][1] * 100
                    if "gpu_power" in data:
                        reading.power_w = data["gpu_power"]
                    if "temp" in data and "gpu_temp_avg" in data["temp"]:
                        reading.temp_c = data["temp"]["gpu_temp_avg"]

                    with self._lock:
                        self._readings.append(reading)
                        self._last_reading = reading
                        self._prune_old_readings()

                except json.JSONDecodeError:
                    continue

        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self._warn_once(exc, "macmon stream failed")
        finally:
            if self._process is not None:
                try:
                    self._process.terminate()
                except OSError as exc:
                    self._warn_once(exc, "macmon terminate failed")

    def _prune_old_readings(self) -> None:
        """Remove readings older than window_seconds. Must hold lock."""
        cutoff = time.monotonic() - self.window_seconds
        while self._readings and self._readings[0].timestamp < cutoff:
            self._readings.popleft()

    def _warn_once(self, exc: BaseException, context: str) -> None:
        exc_type = type(exc)
        with self._warn_lock:
            if exc_type in self._warned_exception_types:
                return
            self._warned_exception_types.add(exc_type)
        print(f"[gpu_monitor] {context}: {exc}")

    def get_metrics(self) -> dict:
        """Get current MPS memory and rolling average GPU metrics.

        Returns dict with keys:
            mps/allocated_mb: Current MPS allocated memory
            mps/driver_mb: Current MPS driver memory
            gpu/freq_mhz: Rolling average GPU frequency
            gpu/util_pct: Rolling average GPU utilization
            gpu/power_w: Rolling average GPU power
            gpu/temp_c: Rolling average GPU temperature
            gpu/samples: Number of samples in the rolling window
        """
        metrics: dict = {
            "mps/allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
            "mps/driver_mb": torch.mps.driver_allocated_memory() / 1024**2,
        }

        with self._lock:
            self._prune_old_readings()
            n = len(self._readings)

            if n == 0:
                # Use last known reading as fallback
                if self._last_reading is not None:
                    r = self._last_reading
                    metrics["gpu/samples"] = 0  # indicates stale data
                    if r.freq_mhz is not None:
                        metrics["gpu/freq_mhz"] = r.freq_mhz
                    if r.util_pct is not None:
                        metrics["gpu/util_pct"] = r.util_pct
                    if r.power_w is not None:
                        metrics["gpu/power_w"] = r.power_w
                    if r.temp_c is not None:
                        metrics["gpu/temp_c"] = r.temp_c
                return metrics

            metrics["gpu/samples"] = n

            # Compute rolling averages
            freq_sum, freq_count = 0.0, 0
            util_sum, util_count = 0.0, 0
            power_sum, power_count = 0.0, 0
            temp_sum, temp_count = 0.0, 0

            for r in self._readings:
                if r.freq_mhz is not None:
                    freq_sum += r.freq_mhz
                    freq_count += 1
                if r.util_pct is not None:
                    util_sum += r.util_pct
                    util_count += 1
                if r.power_w is not None:
                    power_sum += r.power_w
                    power_count += 1
                if r.temp_c is not None:
                    temp_sum += r.temp_c
                    temp_count += 1

            if freq_count > 0:
                metrics["gpu/freq_mhz"] = freq_sum / freq_count
            if util_count > 0:
                metrics["gpu/util_pct"] = util_sum / util_count
            if power_count > 0:
                metrics["gpu/power_w"] = power_sum / power_count
            if temp_count > 0:
                metrics["gpu/temp_c"] = temp_sum / temp_count

        return metrics

    def is_running(self) -> bool:
        """Check if the monitor thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "GPUMetricsMonitor":
        """Context manager entry - starts monitoring."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - stops monitoring."""
        self.stop()
