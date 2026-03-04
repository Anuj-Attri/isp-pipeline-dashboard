from typing import Dict

import cv2
import numpy as np


def draw_metrics_overlay(
    frame: np.ndarray,
    metrics_snapshot: Dict[str, float],
    mode_name: str,
) -> np.ndarray:
    """Draw metrics text on frame (RGB888)."""
    overlay = frame.copy()

    fps = metrics_snapshot.get("fps", 0.0)
    latency_ms = metrics_snapshot.get("latency_ms", 0.0)
    dropped = metrics_snapshot.get("dropped_frames", 0)

    lines = [
        f"Mode: {mode_name}",
        f"FPS: {fps:.1f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Dropped: {dropped}",
    ]
    inference_ms = metrics_snapshot.get("inference_latency_ms", 0.0)
    det_count = metrics_snapshot.get("detection_count", 0)
    if inference_ms > 0 or det_count > 0:
        lines.append(f"Inference: {inference_ms:.1f} ms")
        lines.append(f"Detections: {det_count}")

    x, y = 10, 20
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    for i, text in enumerate(lines):
        yy = y + i * dy
        cv2.putText(
            overlay,
            text,
            (x + 1, yy + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (x, yy),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return overlay

