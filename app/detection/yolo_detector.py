"""YOLO11n detection (ONNX Runtime) and segmentation (ultralytics). DetectionBackend interface."""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.detection.base import COCO_NAMES, Detection, DetectionBackend

SegResult = Tuple[np.ndarray, float]
IMGSZ = 320


def _ensure_onnx_detection(model_pt_path: str) -> Optional[str]:
    """Export to ONNX if missing. Return path to .onnx or None."""
    path = Path(model_pt_path)
    onnx_path = path.with_suffix(".onnx") if path.suffix == ".pt" else Path(str(path) + ".onnx")
    if onnx_path.is_file():
        return str(onnx_path)
    try:
        from ultralytics import YOLO
        load_name = str(path) if path.is_file() else "yolo11n.pt"
        model = YOLO(load_name)
        model.export(format="onnx", imgsz=IMGSZ, simplify=True)
        return str(onnx_path) if onnx_path.is_file() else None
    except Exception:
        return None


def _decode_yolo_onnx(output: np.ndarray, conf_thresh: float, inp_w: int, inp_h: int) -> List[Detection]:
    """Decode (1, 84, 8400) to list of Detection. 84 = 4 box + 80 classes."""
    # (1, 84, 8400) -> (8400, 84)
    out = output[0].T
    if out.shape[1] < 84:
        return []
    boxes = out[:, :4]
    scores = out[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    keep = max_scores >= conf_thresh
    if not np.any(keep):
        return []
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    max_scores = max_scores[keep]
    # cx, cy, w, h (normalized 0-1 in ultralytics) -> x1, y1, x2, y2 in pixel
    cx = boxes[:, 0] * inp_w
    cy = boxes[:, 1] * inp_h
    w = boxes[:, 2] * inp_w
    h = boxes[:, 3] * inp_h
    x1 = np.clip(cx - w / 2, 0, inp_w)
    y1 = np.clip(cy - h / 2, 0, inp_h)
    x2 = np.clip(cx + w / 2, 0, inp_w)
    y2 = np.clip(cy + h / 2, 0, inp_h)
    # NMSBoxes expects [x, y, w, h]
    boxes_xywh = np.column_stack([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        max_scores.tolist(),
        conf_thresh,
        0.45,
    )
    if len(indices) == 0:
        return []
    if hasattr(indices, '__len__') and not isinstance(indices, np.ndarray):
        indices = np.array(indices).flatten()
    detections = []
    for i in indices:
        idx = i if isinstance(i, int) else int(i)
        cls = int(class_ids[idx])
        if cls < 0 or cls >= len(COCO_NAMES):
            continue
        detections.append(
            Detection(
                class_id=cls,
                class_name=COCO_NAMES[cls],
                confidence=float(max_scores[idx]),
                bbox=(int(round(x1[idx])), int(round(y1[idx])), int(round(x2[idx])), int(round(y2[idx]))),
            )
        )
    return detections


class YoloDetector(DetectionBackend):
    """YOLO11n: detection via ONNX Runtime; segmentation via ultralytics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._conf = float(config.get("confidence_threshold", 0.4))
        model_det = config.get("model_detection", "models/yolo11n.pt")
        model_seg = config.get("model_segmentation", "models/yolo11n-seg.pt")
        self._model_det_path = Path(model_det)
        self._model_seg_path = Path(model_seg)
        self._session_det: Any = None
        self._input_name_det: Optional[str] = None
        self._model_seg: Any = None
        self._ensure_models()

    def _ensure_models(self) -> None:
        self._model_det_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_seg_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_det_onnx(self) -> bool:
        if self._session_det is not None:
            return True
        onnx_path = _ensure_onnx_detection(str(self._model_det_path))
        if not onnx_path or not os.path.isfile(onnx_path):
            return False
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session_det = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name_det = self._session_det.get_inputs()[0].name
            return True
        except Exception:
            return False

    def _load_seg(self) -> Any:
        if self._model_seg is not None:
            return self._model_seg
        try:
            from ultralytics import YOLO
            path = str(self._model_seg_path)
            load_name = path if os.path.isfile(path) else "yolo11n-seg.pt"
            self._model_seg = YOLO(load_name)
            onnx_path = path.replace(".pt", ".onnx") if path.endswith(".pt") else path + ".onnx"
            if not os.path.isfile(onnx_path):
                self._model_seg.export(format="onnx", imgsz=IMGSZ, simplify=True)
            return self._model_seg
        except Exception:
            return None

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Run detection via ONNX Runtime. Returns (detections, inference_ms)."""
        if not self._load_det_onnx():
            return [], 0.0
        h, w = frame.shape[:2]
        blob = cv2.resize(frame, (IMGSZ, IMGSZ))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        t0 = time.perf_counter()
        out = self._session_det.run(None, {self._input_name_det: blob})[0]
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        detections = _decode_yolo_onnx(out, self._conf, IMGSZ, IMGSZ)
        # Model output is in IMGSZ×IMGSZ; scale bboxes to input frame size (e.g. 320×240)
        if detections and (w != IMGSZ or h != IMGSZ):
            sx, sy = w / IMGSZ, h / IMGSZ
            detections = [
                Detection(
                    d.class_id,
                    d.class_name,
                    d.confidence,
                    (
                        int(round(d.bbox[0] * sx)),
                        int(round(d.bbox[1] * sy)),
                        int(round(d.bbox[2] * sx)),
                        int(round(d.bbox[3] * sy)),
                    ),
                )
                for d in detections
            ]
        return detections, inference_ms

    def segment(self, frame: np.ndarray) -> SegResult:
        """Run segmentation via ultralytics (masks). Returns (frame with masks, inference_ms)."""
        model = self._load_seg()
        if model is None:
            return frame.copy(), 0.0
        t0 = time.perf_counter()
        results = model.predict(frame, conf=self._conf, imgsz=IMGSZ, verbose=False)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        out = frame.copy()
        if not results:
            return out, inference_ms
        r = results[0]
        if r.masks is None:
            return out, inference_ms
        h, w = frame.shape[:2]
        for mask_data in r.masks.data:
            mask = mask_data.cpu().numpy()
            if mask.ndim == 2:
                mask = cv2.resize(mask, (w, h))
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                overlay = out.copy()
                overlay[mask > 0.5] = color
                cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        return out, inference_ms
