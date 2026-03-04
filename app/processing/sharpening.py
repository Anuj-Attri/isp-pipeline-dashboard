from typing import Any, Dict

import cv2
import numpy as np


def apply_sharpening(frame: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Simple sharpening filter using an unsharp mask-style kernel. Frame in RGB888."""
    strength = float(config.get("sharpening_strength", 1.0))
    strength = max(0.0, min(strength, 2.0))

    kernel = (
        np.array(
            [
                [0, -1, 0],
                [-1, 5 + strength, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
    )

    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened

