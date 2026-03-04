"""Denoising: fast (Gaussian) or quality (fastNlMeansDenoisingColored)."""
from typing import Any, Dict

import cv2
import numpy as np


def apply_denoising(frame: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Apply denoising. Config: denoising_mode ('fast'/'quality'), denoising_strength (float). Frame in RGB888."""
    mode = (config.get("denoising_mode") or "fast").strip().lower()
    strength = float(config.get("denoising_strength", 1.0))
    strength = max(0.1, min(2.0, strength))

    if mode == "quality":
        h = 3.0 * strength
        h_for_color = 3.0 * strength
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=h,
            hForColorComponents=h_for_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    k = max(3, int(strength * 2) * 2 + 1)
    return cv2.GaussianBlur(frame, (k, k), strength)
