"""IMX500 native object detection — runs on Sony NPU, zero CPU inference cost."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.detection.base import COCO_NAMES, Detection, DetectionBackend

logger = logging.getLogger(__name__)

IMX500_DET_MODEL = (
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)


class Imx500Detector(DetectionBackend):
    """Reads SSD detection results from IMX500 NPU via frame metadata.

    Metadata is captured alongside each frame in the capture loop and passed
    directly to detect() — no additional camera calls, no thread contention.
    """

    def __init__(self, config: dict, imx500: Any, picam2: Any) -> None:
        self._imx500 = imx500
        self._picam2 = picam2
        self._conf = config.get("confidence_threshold", 0.5)

    def detect(
        self, frame: np.ndarray, metadata: Optional[Dict] = None
    ) -> Tuple[List[Detection], float]:
        """Extract detections from IMX500 metadata passed from capture loop."""
        if self._imx500 is None or metadata is None:
            return [], 0.0
        t0 = time.perf_counter()
        outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if outputs is None or len(outputs) < 3:
            return [], (time.perf_counter() - t0) * 1000.0
        detections = self._parse(outputs, frame.shape)
        return detections, (time.perf_counter() - t0) * 1000.0

    def _parse(self, outputs: Any, frame_shape: tuple) -> List[Detection]:
        """Decode SSD outputs to Detection list with NMS.

        Output tensors: [boxes(1,N,4), classes(1,N), scores(1,N), count(1,1)]
        - boxes: ymin, xmin, ymax, xmax normalized 0.0-1.0
        - scores: raw integers 0-100 — divide by 100 to normalize
        - NMS IoU threshold 0.35 to suppress overlapping boxes
        """
        h, w = frame_shape[:2]
        boxes = np.asarray(outputs[0][0])           # (N, 4)
        classes = np.asarray(outputs[1][0])        # (N,)
        scores = np.asarray(outputs[2][0]) / 100.0  # normalize 0-100 -> 0.0-1.0

        nms_boxes, nms_scores, nms_classes = [], [], []
        for i in range(len(scores)):
            if scores[i] < self._conf:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = min(w, int(xmax * w))
            y2 = min(h, int(ymax * h))
            if x2 <= x1 or y2 <= y1:
                continue
            nms_boxes.append([x1, y1, x2 - x1, y2 - y1])
            nms_scores.append(float(scores[i]))
            nms_classes.append(int(classes[i]))

        if not nms_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self._conf, 0.35)
        if len(indices) == 0:
            return []

        result = []
        for idx in np.asarray(indices).flatten():
            x, y, bw, bh = nms_boxes[idx]
            cid = nms_classes[idx]
            name = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else f"class_{cid}"
            result.append(
                Detection(cid, name, nms_scores[idx], (x, y, x + bw, y + bh))
            )
        return result
