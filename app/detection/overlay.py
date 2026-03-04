"""Draw detection boxes and labels with per-class colors."""
import colorsys
from typing import List

import cv2
import numpy as np

from app.detection.base import Detection


def _generate_colors(n: int) -> List[tuple]:
    """Generate n perceptually distinct colors using HSV space."""
    colors = []
    for i in range(n):
        h = i / n
        s, v = 0.85, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


CLASS_COLORS = _generate_colors(80)


def get_class_color(class_id: int) -> tuple:
    """Return a consistent BGR color for the given class index."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def draw_detection_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels with per-class colors and filled label background."""
    for det in detections:
        c = get_class_color(det.class_id)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), c, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame
