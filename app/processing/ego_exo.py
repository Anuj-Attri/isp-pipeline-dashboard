"""Ego→Exo: Inverse Perspective Mapping (IPM) for top-down / bird's-eye view (ADAS-style)."""
from typing import Optional

import cv2
import numpy as np


class EgoExoProjector:
    """IPM: map ego view (bottom trapezoid = road/floor) to top-down rectangle."""

    def __init__(self, frame_w: int = 1280, frame_h: int = 720) -> None:
        # IPM source points — bottom trapezoid of ego view (road/floor region)
        src = np.float32([
            [frame_w * 0.1, frame_h],         # bottom-left
            [frame_w * 0.9, frame_h],         # bottom-right
            [frame_w * 0.6, frame_h * 0.5],   # top-right
            [frame_w * 0.4, frame_h * 0.5],   # top-left
        ])
        # IPM destination points — rectangle (top-down view)
        dst = np.float32([
            [frame_w * 0.1, frame_h],
            [frame_w * 0.9, frame_h],
            [frame_w * 0.9, 0],
            [frame_w * 0.1, 0],
        ])
        self._H = cv2.getPerspectiveTransform(src, dst)
        self._frame_w = frame_w
        self._frame_h = frame_h

    def project(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply IPM warp for top-down view. depth_map unused but kept for API compatibility."""
        h, w = frame.shape[:2]
        if (w, h) != (self._frame_w, self._frame_h):
            H = self._get_homography(w, h)
        else:
            H = self._H
        warped = cv2.warpPerspective(
            frame,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(20, 20, 20),
        )
        return warped

    def _get_homography(self, w: int, h: int) -> np.ndarray:
        src = np.float32([
            [w * 0.1, h],
            [w * 0.9, h],
            [w * 0.6, h * 0.5],
            [w * 0.4, h * 0.5],
        ])
        dst = np.float32([
            [w * 0.1, h],
            [w * 0.9, h],
            [w * 0.9, 0],
            [w * 0.1, 0],
        ])
        return cv2.getPerspectiveTransform(src, dst)
