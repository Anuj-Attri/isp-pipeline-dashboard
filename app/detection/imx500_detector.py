"""IMX500 native object detection — inference runs on Sony NPU, zero CPU cost."""
import logging
import time
from typing import Any, List, Tuple

import numpy as np

from app.detection.base import COCO_NAMES, Detection, DetectionBackend

logger = logging.getLogger(__name__)

IMX500_DET_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


class Imx500Detector(DetectionBackend):
    """Reads detection results from IMX500 frame metadata. Zero CPU inference cost."""

    def __init__(self, config: dict, imx500: Any, picam2: Any) -> None:
        self._imx500 = imx500
        self._picam2 = picam2
        self._conf = config.get("confidence_threshold", 0.4)

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Extract detections from IMX500 metadata embedded in the last frame."""
        if self._imx500 is None:
            return [], 0.0
        t0 = time.perf_counter()
        try:
            metadata = self._picam2.capture_metadata()
            outputs = self._imx500.get_outputs(metadata, add_batch=True)
            if outputs is None:
                return [], 0.0
            detections = self._parse_ssd_outputs(outputs, frame.shape)
            return detections, (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.debug("IMX500 detection error: %s", e)
            return [], 0.0

    def _parse_ssd_outputs(self, outputs: Any, frame_shape: tuple) -> List[Detection]:
        """Parse SSD MobileNetV2 FPN post-processed outputs into Detection list."""
        h, w = frame_shape[:2]
        detections: List[Detection] = []
        try:
            boxes = np.array(outputs[0][0])
            classes = np.array(outputs[1][0])
            scores = np.array(outputs[2][0])
            for i in range(len(scores)):
                if float(scores[i]) < self._conf:
                    continue
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                class_id = int(classes[i])
                name = COCO_NAMES[class_id] if 0 <= class_id < len(COCO_NAMES) else f"class_{class_id}"
                detections.append(
                    Detection(class_id, name, float(scores[i]), (x1, y1, x2, y2))
                )
        except Exception as e:
            logger.debug("SSD parse error: %s", e)
        return detections
