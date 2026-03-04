import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class MetricsSnapshot:
    timestamp: float
    fps: float
    latency_ms: float
    captured_frames: int
    displayed_frames: int
    dropped_frames: int
    inference_latency_ms: float
    detection_count: int
    cpu_percent: float
    memory_percent: float
    temperature_c: Optional[float]
    cpu_freq_mhz: Optional[int]


def _read_temperature_c() -> Optional[float]:
    """Pi 5: max over all thermal zones."""
    import glob
    temps = []
    for zone in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(zone) as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def _read_cpu_freq_mhz() -> Optional[int]:
    """CPU frequency in MHz (shows throttling)."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) // 1000
    except (OSError, ValueError):
        return None


class RuntimeMetrics:
    def __init__(self, include_system_metrics: bool = True) -> None:
        self._lock = threading.Lock()
        self._captured_frames = 0
        self._displayed_frames = 0
        self._dropped_frames = 0
        self._last_display_time = None
        self._fps = 0.0
        self._latency_ms = 0.0
        self._inference_latency_ms = 0.0
        self._detection_count = 0
        self._include_system = include_system_metrics
        self._psutil = None
        if include_system_metrics:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                self._psutil = None

    def on_frame_captured(self) -> None:
        with self._lock:
            self._captured_frames += 1

    def on_frame_dropped(self) -> None:
        with self._lock:
            self._dropped_frames += 1

    def on_frame_displayed(self, latency_ms: float) -> None:
        now = time.time()
        with self._lock:
            self._displayed_frames += 1
            if self._last_display_time is not None:
                dt = now - self._last_display_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    alpha = 0.2
                    self._fps = alpha * instant_fps + (1 - alpha) * self._fps
            self._last_display_time = now
            self._latency_ms = latency_ms

    def on_detection_completed(self, inference_ms: float, count: int) -> None:
        with self._lock:
            self._inference_latency_ms = inference_ms
            self._detection_count = count

    def get_snapshot(self) -> Dict[str, float]:
        cpu = 0.0
        mem = 0.0
        if self._psutil:
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                mem = self._psutil.virtual_memory().percent
            except Exception:
                pass
        temp = _read_temperature_c() if self._include_system else None
        freq = _read_cpu_freq_mhz() if self._include_system else None
        with self._lock:
            snap = MetricsSnapshot(
                timestamp=time.time(),
                fps=self._fps,
                latency_ms=self._latency_ms,
                captured_frames=self._captured_frames,
                displayed_frames=self._displayed_frames,
                dropped_frames=self._dropped_frames,
                inference_latency_ms=self._inference_latency_ms,
                detection_count=self._detection_count,
                cpu_percent=cpu,
                memory_percent=mem,
                temperature_c=temp,
                cpu_freq_mhz=freq,
            )
        return asdict(snap)

