"""Picamera2 lifecycle and IMX500 model hot-swap for NPU inference."""
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from app.metrics.runtime_metrics import RuntimeMetrics

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages Picamera2 lifecycle including IMX500 model hot-swap."""

    def __init__(
        self,
        config: Dict[str, Any],
        frame_queue: "queue.Queue",
        metrics: RuntimeMetrics,
    ) -> None:
        self._config = config
        self._frame_queue = frame_queue
        self._metrics = metrics
        self._picam2: Any = None
        self._imx500: Any = None
        self._current_imx500_model: Optional[str] = None
        self._lock = threading.Lock()

    def start(self, imx500_model_path: Optional[str] = None) -> None:
        """Start camera at maximum ISP resolution with optional IMX500 model."""
        Picamera2 = __import__("picamera2").Picamera2
        if imx500_model_path and os.path.exists(imx500_model_path):
            try:
                from picamera2.devices.imx500 import IMX500
                self._imx500 = IMX500(imx500_model_path)
                self._current_imx500_model = imx500_model_path
            except Exception as e:
                logger.debug("IMX500 init failed: %s", e)
                self._imx500 = None
                self._current_imx500_model = None
        else:
            self._imx500 = None
            self._current_imx500_model = None

        self._picam2 = Picamera2()

        cam_conf = self._config["camera"]
        main_size = tuple(cam_conf.get("resolution", [3840, 2160]))
        fps = cam_conf.get("framerate", 15)
        fd = int(1e6 / fps)
        sensor_size = tuple(cam_conf.get("sensor_resolution", [4056, 3040]))
        camera_cfg = self._picam2.create_video_configuration(
            main={"size": main_size, "format": "RGB888"},
            raw={"size": sensor_size},
            controls={
                "FrameDurationLimits": (fd, fd),
                "NoiseReductionMode": 2,
            },
            buffer_count=2,
        )
        self._picam2.configure(camera_cfg)
        self._picam2.start()
        time.sleep(1.0)

    def restart_with_model(self, imx500_model_path: Optional[str] = None) -> None:
        """Hot-swap IMX500 model by restarting camera. No-op if model unchanged."""
        if imx500_model_path == self._current_imx500_model:
            return
        self.stop()
        time.sleep(0.3)
        self.start(imx500_model_path)
        time.sleep(0.5)  # allow camera to produce first frames before capture_loop starts

    def get_imx500(self) -> Any:
        """Return IMX500 device wrapper if loaded; else None."""
        return self._imx500

    def get_picam2(self) -> Any:
        """Return Picamera2 instance for runtime controls (AE/AWB/AF)."""
        return self._picam2

    def stop(self) -> None:
        """Stop and close camera; clear IMX500 state."""
        with self._lock:
            try:
                if self._picam2 is not None:
                    self._picam2.stop()
                    self._picam2.close()
                    self._picam2 = None
                self._imx500 = None
                self._current_imx500_model = None
            except Exception:
                pass

    def capture_loop(self, running: threading.Event) -> None:
        """Producer loop: push (frame, metadata, timestamp_ns) into frame_queue."""
        while running.is_set():
            try:
                if self._picam2 is None:
                    time.sleep(0.05)
                    continue
                request = self._picam2.capture_request()
                frame = request.make_array("main")
                metadata = request.get_metadata()
                request.release()
                if not isinstance(frame, np.ndarray):
                    continue
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                ts = time.time_ns()
                try:
                    self._frame_queue.put_nowait((frame, metadata, ts))
                    self._metrics.on_frame_captured()
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_queue.put_nowait((frame, metadata, ts))
                    self._metrics.on_frame_dropped()
            except Exception as e:
                if running.is_set():
                    logger.warning("capture_loop error: %s", e)
                break
