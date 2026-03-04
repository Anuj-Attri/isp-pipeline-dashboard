"""Create detection backend from config. IMX500 detector is created in main on mode switch."""
from typing import Any, Dict, Optional

from app.detection.base import DetectionBackend
from app.detection.yolo_detector import YoloDetector


def create_detector(config: Dict[str, Any]) -> Optional[DetectionBackend]:
    """Return YOLO detector for CPU segmentation/detection fallback; None if disabled."""
    det = config.get("detection") or {}
    backend = (det.get("backend") or "yolo").strip().lower()
    if backend in ("yolo", "auto"):
        return YoloDetector(det)
    return None
