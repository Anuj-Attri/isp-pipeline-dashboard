"""Main entrypoint: camera pipeline, ISP, IMX500/CPU AI dispatch, metrics, server."""
import argparse
import logging
import os
import queue
import signal
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from app.camera.camera_manager import CameraManager
from app.config.config_loader import load_config
from app.detection import create_detector, draw_detection_overlay
from app.detection.base import DetectionBackend
from app.detection.imx500_detector import Imx500Detector
from app.detection.pose_estimator import PoseEstimator
from app.metrics.logger import CSVLogger
from app.metrics.metrics_overlay import draw_metrics_overlay
from app.metrics.runtime_metrics import RuntimeMetrics
from app.processing.denoising import apply_denoising
from app.processing.sharpening import apply_sharpening
from app.processing.depth_estimator import DepthEstimator
from app.processing.ego_exo import EgoExoProjector
from app.processing.inference_worker import InferenceWorker
from app.processing.pipeline_state import (
    AIMode,
    CPU_MODES,
    IMX500_DETECTION_MODEL,
    IMX500_MODES,
    IMX500_POSE_MODEL,
    get_pipeline_state,
)

logger = logging.getLogger(__name__)
WATCHDOG_EMPTY_SEC = 5.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Camera Platform (Pi 5 / AI Camera)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display; server still serves /video and /ws",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    csv_path = config["app"]["csv_log_path"]
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    frame_queue_size = config["app"].get("frame_queue_size", 2)
    frame_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=frame_queue_size)
    include_system = config["app"].get("include_system_metrics", True)
    metrics = RuntimeMetrics(include_system_metrics=include_system)
    logger_csv = CSVLogger(csv_path)

    camera_manager = CameraManager(config, frame_queue, metrics)
    pipeline_state = get_pipeline_state()
    proc = config.get("processing") or {}
    pipeline_state.set_isp("sharpening", proc.get("sharpening_enabled", True))
    pipeline_state.set_isp("sharpening_strength", proc.get("sharpening_strength", 1.0))
    pipeline_state.set_isp("denoising", proc.get("denoising_enabled", False))
    pipeline_state.set_isp("denoising_strength", proc.get("denoising_strength", 1.0))
    pipeline_state.set_isp("denoising_mode", proc.get("denoising_mode", "fast"))
    pipeline_state.set_isp("demosaicing_quality_fast", True)
    pipeline_state.set_isp("ae_lock", False)
    pipeline_state.set_isp("awb_lock", False)
    pipeline_state.set_isp("af_enabled", False)
    pipeline_state.set_ai_mode("none")

    running = threading.Event()
    running.set()

    def on_stop(_signum=None, _frame=None) -> None:
        try:
            camera_manager.stop()
        except Exception:
            pass
        running.clear()

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    server_state: Optional[Any] = None
    server_config = config.get("server") or {}
    if server_config.get("enabled", False):
        from app.server.state import ServerState
        from app.server.app import run_server
        server_state = ServerState()
        host = server_config.get("host", "0.0.0.0")
        port = int(server_config.get("port", 8765))
        server_thread = threading.Thread(
            target=run_server,
            args=(server_state, host, port, pipeline_state, camera_manager),
            daemon=True,
        )
        server_thread.start()

    detector: Optional[DetectionBackend] = create_detector(config)
    depth_cfg = config.get("depth") or {}
    depth_estimator = DepthEstimator(
        depth_cfg.get("model_path", "models/midas_v21_small_256.onnx")
    )
    ego_exo_projector = EgoExoProjector()
    inference_worker = InferenceWorker(detector, depth_estimator, ego_exo_projector)

    capture_thread = threading.Thread(
        target=camera_manager.capture_loop,
        args=(running,),
        daemon=True,
    )

    camera_manager.start(None)
    capture_thread.start()

    headless = args.headless or config["app"].get("headless", False)
    window_name = config["app"].get("display_window_name", "Pi Camera Platform")
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_frame_time = time.time()
    processing_config = config.get("processing") or {}
    processing_config.setdefault("denoising_mode", "fast")
    processing_config.setdefault("denoising_strength", 1.0)
    processing_config.setdefault("sharpening_strength", 1.0)

    imx500_detector: Optional[Imx500Detector] = None
    pose_estimator: Optional[PoseEstimator] = None
    prev_ai_mode: Optional[AIMode] = None
    det_config = config.get("detection") or {}

    try:
        while running.is_set():
            try:
                frame, capture_timestamp_ns = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if time.time() - last_frame_time > WATCHDOG_EMPTY_SEC:
                    logger.warning("Frame queue empty for >%.1fs", WATCHDOG_EMPTY_SEC)
                continue

            last_frame_time = time.time()
            now_ns = time.time_ns()
            latency_ms = (now_ns - capture_timestamp_ns) / 1e6
            metrics.on_frame_displayed(latency_ms)

            snap = pipeline_state.get_snapshot()
            try:
                current_ai_mode = AIMode(snap.get("ai_mode", "none"))
            except ValueError:
                current_ai_mode = AIMode.NONE

            if current_ai_mode != prev_ai_mode:
                if current_ai_mode == AIMode.DETECTION:
                    camera_manager.restart_with_model(IMX500_DETECTION_MODEL)
                    imx500_detector = Imx500Detector(
                        det_config,
                        camera_manager.get_imx500(),
                        camera_manager.get_picam2(),
                    )
                    pose_estimator = None
                elif current_ai_mode == AIMode.POSE:
                    camera_manager.restart_with_model(IMX500_POSE_MODEL)
                    pose_estimator = PoseEstimator(
                        camera_manager.get_imx500(),
                        camera_manager.get_picam2(),
                        det_config.get("confidence_threshold", 0.4),
                    )
                    imx500_detector = None
                else:
                    camera_manager.restart_with_model(None)
                    imx500_detector = None
                    pose_estimator = None

                capture_thread = threading.Thread(
                    target=camera_manager.capture_loop,
                    args=(running,),
                    daemon=True,
                )
                capture_thread.start()

                prev_ai_mode = current_ai_mode

            proc_frame = {
                **processing_config,
                "sharpening_strength": snap.get("sharpening_strength", 1.0),
                "denoising_strength": snap.get("denoising_strength", 1.0),
                "denoising_mode": snap.get("denoising_mode", "fast"),
            }
            display_frame = frame.copy()

            if snap.get("sharpening", False):
                display_frame = apply_sharpening(display_frame, proc_frame)
            if snap.get("denoising", False):
                display_frame = apply_denoising(display_frame, proc_frame)

            mode_name = current_ai_mode.value

            if current_ai_mode == AIMode.DETECTION and imx500_detector is not None:
                detections, inf_ms = imx500_detector.detect(display_frame)
                display_frame = draw_detection_overlay(display_frame, detections)
                metrics.on_detection_completed(inf_ms, len(detections))
            elif current_ai_mode == AIMode.POSE and pose_estimator is not None:
                poses, inf_ms = pose_estimator.estimate(display_frame)
                display_frame = pose_estimator.draw(display_frame, poses)
                metrics.on_detection_completed(inf_ms, len(poses))
            elif current_ai_mode in CPU_MODES:
                inference_worker.submit(current_ai_mode, display_frame)
                result_frame, inf_ms, result_mode, det_count = inference_worker.get_result()
                if result_mode == current_ai_mode and result_frame is not None:
                    display_frame = result_frame
                    metrics.on_detection_completed(inf_ms, det_count)
                else:
                    metrics.on_detection_completed(0.0, 0)
            else:
                metrics.on_detection_completed(0.0, 0)

            snapshot = metrics.get_snapshot()
            snapshot["inference_backend"] = (
                "IMX500 NPU" if current_ai_mode in IMX500_MODES else "ONNX CPU"
            )
            display_frame = draw_metrics_overlay(
                display_frame,
                snapshot,
                mode_name=mode_name,
            )
            logger_csv.log(snapshot, mode_name)

            if server_state is not None:
                server_state.set_frame(display_frame)
                server_state.set_metrics(snapshot)

            if not headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running.clear()
                    break

    finally:
        running.clear()
        inference_worker.stop()
        capture_thread.join(timeout=2.0)
        camera_manager.stop()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
