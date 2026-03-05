"""IMX500 native pose estimation using HigherHRNet — 17 COCO keypoints on NPU."""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMX500_POSE_MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SKELETON_COLORS = {
    "face": (255, 200, 0),
    "shoulders": (0, 200, 255),
    "arms": (0, 255, 128),
    "torso": (255, 100, 200),
    "legs": (100, 150, 255),
}

SKELETON_COLOR_MAP = [
    SKELETON_COLORS["face"], SKELETON_COLORS["face"],
    SKELETON_COLORS["face"], SKELETON_COLORS["face"],
    SKELETON_COLORS["shoulders"], SKELETON_COLORS["arms"],
    SKELETON_COLORS["arms"], SKELETON_COLORS["arms"], SKELETON_COLORS["arms"],
    SKELETON_COLORS["torso"], SKELETON_COLORS["torso"], SKELETON_COLORS["torso"],
    SKELETON_COLORS["legs"], SKELETON_COLORS["legs"],
    SKELETON_COLORS["legs"], SKELETON_COLORS["legs"],
]


@dataclass
class Pose:
    """Single person pose with 17 COCO keypoints."""
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]


class PoseEstimator:
    """Reads pose estimation results from IMX500 HigherHRNet metadata."""

    def __init__(self, imx500: Any, picam2: Any, conf_threshold: float = 0.3) -> None:
        self._imx500 = imx500
        self._picam2 = picam2
        self._conf = conf_threshold

    def estimate(
        self, frame: np.ndarray, metadata: Optional[Dict] = None
    ) -> Tuple[List[Pose], float]:
        """Extract poses from IMX500 metadata passed from capture loop."""
        if self._imx500 is None or metadata is None:
            return [], 0.0
        t0 = time.perf_counter()
        try:
            outputs = self._imx500.get_outputs(metadata, add_batch=True)
            if outputs is None:
                return [], (time.perf_counter() - t0) * 1000.0
            poses = self._parse_hrnet_outputs(outputs, frame.shape)
            return poses, (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.debug("Pose estimation error: %s", e)
            return [], 0.0

    def _parse_hrnet_outputs(self, outputs: Any, frame_shape: tuple) -> List[Pose]:
        """Parse HigherHRNet outputs into Pose objects."""
        h, w = frame_shape[:2]
        poses: List[Pose] = []
        try:
            kps_raw = np.array(outputs[0][0])
            scores_raw = np.array(outputs[1][0])
            for i in range(len(kps_raw)):
                kps = kps_raw[i]
                scores = scores_raw[i]
                if np.mean(scores) < self._conf * 0.5:
                    continue
                kps_px = kps.copy()
                kps_px[:, 0] *= w
                kps_px[:, 1] *= h
                visible = kps_px[scores > self._conf]
                if len(visible) < 3:
                    continue
                x1 = int(visible[:, 0].min())
                y1 = int(visible[:, 1].min())
                x2 = int(visible[:, 0].max())
                y2 = int(visible[:, 1].max())
                poses.append(Pose(
                    keypoints=kps_px.astype(np.int32),
                    scores=scores,
                    bbox=(x1, y1, x2, y2),
                ))
        except Exception as e:
            logger.debug("HRNet parse error: %s", e)
        return poses

    def draw(self, frame: np.ndarray, poses: List[Pose]) -> np.ndarray:
        """Draw skeleton and keypoints on frame."""
        out = frame.copy()
        for pose in poses:
            kps = pose.keypoints
            scores = pose.scores
            for idx, (i, j) in enumerate(SKELETON):
                if scores[i] > self._conf and scores[j] > self._conf:
                    pt1 = tuple(int(x) for x in kps[i])
                    pt2 = tuple(int(x) for x in kps[j])
                    color = SKELETON_COLOR_MAP[idx]
                    cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)
            for k in range(len(kps)):
                if scores[k] > self._conf:
                    pt = tuple(int(x) for x in kps[k])
                    cv2.circle(out, pt, 4, (255, 255, 255), -1)
                    cv2.circle(out, pt, 4, (0, 0, 0), 1)
        return out
