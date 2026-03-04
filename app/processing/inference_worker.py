"""InferenceWorker: run AI inference in a dedicated thread so the main loop never blocks."""
import queue
import threading
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from app.processing.pipeline_state import AIMode

AI_INFER_SIZE = (320, 240)


class InferenceWorker:
    """Single-thread inference: submit(mode, frame) non-blocking, get_result() returns latest (frame, ms, mode, count)."""

    def __init__(
        self,
        detector: Optional[Any],
        depth_estimator: Any,
        ego_exo_projector: Any,
    ) -> None:
        self._detector = detector
        self._depth_estimator = depth_estimator
        self._ego_exo_projector = ego_exo_projector
        self._input_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result: Optional[Tuple[np.ndarray, float, AIMode, int]] = None
        self._result_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, mode: AIMode, frame: np.ndarray) -> None:
        """Submit a frame for inference; drop if queue full (non-blocking)."""
        try:
            self._input_queue.put_nowait((mode, frame.copy()))
        except queue.Full:
            pass

    def get_result(self) -> Tuple[Optional[np.ndarray], float, Optional[AIMode], int]:
        """Return (latest_result_frame, inference_ms, mode, detection_count). Never blocks."""
        with self._result_lock:
            if self._result is None:
                return None, 0.0, None, 0
            out, ms, m, cnt = self._result
            return (out.copy() if out is not None else None, ms, m, cnt)

    def stop(self) -> None:
        self._running.clear()
        try:
            self._input_queue.put_nowait((None, None))
        except queue.Full:
            pass

    def _run(self) -> None:
        while self._running.is_set():
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None or item[0] is None:
                continue
            mode, frame = item
            if frame is None:
                continue
            out, ms, cnt = self._do_inference(mode, frame)
            with self._result_lock:
                self._result = (out, ms, mode, cnt)

    def _do_inference(self, mode: AIMode, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, AI_INFER_SIZE)

        if mode == AIMode.DETECTION:
            return frame.copy(), 0.0, 0

        if mode == AIMode.SEGMENTATION and self._detector is not None:
            from app.detection.yolo_detector import YoloDetector
            if isinstance(self._detector, YoloDetector):
                seg_small, inference_ms = self._detector.segment(small)
                out = cv2.resize(seg_small, (w, h))
                return out, inference_ms, 0
            return frame.copy(), 0.0, 0

        if mode == AIMode.DEPTH:
            vis_small, inference_ms = self._depth_estimator.estimate(small)
            out = cv2.resize(vis_small, (w, h))
            return out, inference_ms, 0

        if mode == AIMode.EGO_EXO:
            depth_small, inference_ms = self._depth_estimator.estimate_raw(small)
            depth_full = cv2.resize(depth_small, (w, h))
            projected = self._ego_exo_projector.project(frame, depth_full)
            half_w = w // 2
            ego_half = cv2.resize(frame, (half_w, h))
            exo_half = cv2.resize(projected, (half_w, h))
            out = np.hstack([ego_half, exo_half])
            return out, inference_ms, 0

        return frame.copy(), 0.0, 0
