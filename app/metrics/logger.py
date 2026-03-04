import csv
import os
from typing import Dict


class CSVLogger:
    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        d = os.path.dirname(self._csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0:
            with open(self._csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "mode",
                        "fps",
                        "latency_ms",
                        "captured_frames",
                        "displayed_frames",
                        "dropped_frames",
                        "inference_latency_ms",
                        "detection_count",
                        "cpu_percent",
                        "memory_percent",
                        "temperature_c",
                        "cpu_freq_mhz",
                    ]
                )

    def log(self, metrics_snapshot: Dict[str, float], mode_name: str) -> None:
        t = metrics_snapshot.get("temperature_c")
        f = metrics_snapshot.get("cpu_freq_mhz")
        row = [
            metrics_snapshot.get("timestamp", 0.0),
            mode_name,
            metrics_snapshot.get("fps", 0.0),
            metrics_snapshot.get("latency_ms", 0.0),
            metrics_snapshot.get("captured_frames", 0),
            metrics_snapshot.get("displayed_frames", 0),
            metrics_snapshot.get("dropped_frames", 0),
            metrics_snapshot.get("inference_latency_ms", 0.0),
            metrics_snapshot.get("detection_count", 0),
            metrics_snapshot.get("cpu_percent", 0.0),
            metrics_snapshot.get("memory_percent", 0.0),
            t if t is not None else 0.0,
            f if f is not None else "",
        ]
        with open(self._csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

