"""MiDaS v2.1 small depth estimation. Returns colormap overlay and inference time."""
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

MIDAS_URL = "https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/midas_v21_small_256.onnx"


def _draw_colorbar(frame: np.ndarray) -> np.ndarray:
    """Draw FAR→NEAR colorbar on right edge of frame."""
    h, w = frame.shape[:2]
    bar_w, bar_h = 20, h // 3
    x_start = w - bar_w - 10
    y_start = h // 3
    for i in range(bar_h):
        t = i / bar_h
        val = np.array([[[int(t * 255)]]], dtype=np.uint8)
        color = cv2.applyColorMap(val, cv2.COLORMAP_MAGMA)[0][0].tolist()
        cv2.rectangle(
            frame,
            (x_start, y_start + i),
            (x_start + bar_w, y_start + i + 1),
            color,
            -1,
        )
    cv2.putText(
        frame, "FAR", (x_start - 2, y_start - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
    )
    cv2.putText(
        frame, "NEAR", (x_start - 5, y_start + bar_h + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
    )
    return frame


class DepthEstimator:
    """MiDaS v2.1 small (256). estimate() -> (visualization_frame_bgr, inference_ms)."""

    def __init__(self, model_path: str = "models/midas_v21_small_256.onnx") -> None:
        self._path = Path(model_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session = None
        self._input_name: Optional[str] = None
        self._input_size = (256, 256)

    def _ensure_model(self) -> bool:
        if not self._path.is_file():
            try:
                import urllib.request
                self._path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MIDAS_URL, str(self._path))
            except Exception:
                return False
        return True

    def _load_session(self) -> bool:
        if self._session is not None:
            return True
        if not self._ensure_model():
            return False
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self._path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            return True
        except Exception:
            return False

    def estimate(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run depth; input RGB888 (Picamera2). Return (frame with MAGMA overlay, inference_ms)."""
        if not self._load_session():
            return frame_rgb.copy(), 0.0
        h, w = frame_rgb.shape[:2]
        t0 = time.perf_counter()
        # MiDaS expects RGB; frame is already RGB888 from Picamera2
        blob = cv2.resize(frame_rgb, self._input_size)
        blob = blob.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        blob = (blob - mean) / std
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        out = self._session.run(None, {self._input_name: blob})[0]
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        depth = out.squeeze()
        depth = cv2.resize(depth, (w, h))
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        colormap_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame_rgb, 0.5, colormap_rgb, 0.5, 0)
        overlay = _draw_colorbar(overlay)
        cx, cy = w // 2, h // 2
        crop_size = min(w, h) // 4
        y1, y2 = max(0, cy - crop_size), min(h, cy + crop_size)
        x1, x2 = max(0, cx - crop_size), min(w, cx + crop_size)
        raw = float(np.median(depth[y1:y2, x1:x2]))
        d_min, d_max = depth.min(), depth.max()
        center_rel = (raw - d_min) / (d_max - d_min + 1e-6)
        cv2.putText(
            overlay,
            f"depth: {center_rel:.2f} (rel)",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return overlay, inference_ms

    def estimate_raw(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (depth_map same HxW as input, inference_ms). Input RGB888."""
        if not self._load_session():
            return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32), 0.0
        h, w = frame_rgb.shape[:2]
        t0 = time.perf_counter()
        blob = cv2.resize(frame_rgb, self._input_size)
        blob = blob.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        blob = (blob - mean) / std
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        out = self._session.run(None, {self._input_name: blob})[0]
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        depth = out.squeeze()
        depth = cv2.resize(depth, (w, h))
        return depth, inference_ms
