"""Shared state between pipeline and web server: latest frame (MJPEG) and metrics (WebSocket)."""
import threading
from typing import Any, Dict, Optional

import numpy as np


class ServerState:
    """Thread-safe shared state for /video and /ws."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame_bgr: Optional[np.ndarray] = None
        self._latest_metrics: Dict[str, Any] = {}
        self._frame_event = threading.Event()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_frame_bgr = frame_bgr.copy() if frame_bgr is not None else None
        self._frame_event.set()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame_bgr is None:
                return None
            return self._latest_frame_bgr.copy()

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._lock:
            self._latest_metrics = dict(metrics)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_metrics)

    def wait_frame(self, timeout: float = 1.0) -> bool:
        return self._frame_event.wait(timeout=timeout)
