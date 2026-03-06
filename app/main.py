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


def _normalize_processing_config(proc: dict) -> dict:
    """Support both nested (sharpening.enabled) and flat (sharpening_enabled) config."""
    sh = proc.get("sharpening")
    dn = proc.get("denoising")
    return {
        "sharpening_enabled": sh.get("enabled", True) if isinstance(sh, dict) else proc.get("sharpening_enabled", True),
        "sharpening_strength": (sh or {}).get("strength", 1.0) if isinstance(sh, dict) else proc.get("sharpening_strength", 1.0),
        "denoising_enabled": dn.get("enabled", False) if isinstance(dn, dict) else proc.get("denoising_enabled", False),
        "denoising_strength": (dn or {}).get("strength", 1.0) if isinstance(dn, dict) else proc.get("denoising_strength", 1.0),
        "denoising_mode": (dn or {}).get("mode", "fast") if isinstance(dn, dict) else proc.get("denoising_mode", "fast"),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Edge CV Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Run ISP pipeline on a single image file",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="Run ISP pipeline on a video file instead of camera",
    )
    parser.add_argument(
        "--ai-mode",
        type=str,
        default="none",
        help="Initial AI mode: none|detection|pose|segmentation|depth|ego_exo",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for image/video mode")
    return parser.parse_args()


def _run_image_mode(args: argparse.Namespace, config: dict, flat_proc: dict) -> None:
    """Run ISP + AI on a single image and save result."""
    img = cv2.imread(args.input_image)
    if img is None:
        logger.error("Could not read image: %s", args.input_image)
        return
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    display_frame = img.copy()
    proc_frame = dict(flat_proc)
    if flat_proc["sharpening_enabled"]:
        display_frame = apply_sharpening(display_frame, proc_frame)
    if flat_proc["denoising_enabled"]:
        display_frame = apply_denoising(display_frame, proc_frame)

    try:
        ai_mode = AIMode(args.ai_mode)
    except ValueError:
        ai_mode = AIMode.NONE

    det_config = config.get("detection") or {}
    depth_cfg = config.get("depth") or {}
    depth_estimator = DepthEstimator(depth_cfg.get("model_path", "models/midas_v21_small_256.onnx"))
    detector = create_detector(config)
    ego_exo_projector = EgoExoProjector()
    metadata: dict = {}

    if ai_mode == AIMode.DETECTION:
        if detector is not None:
            detections, _ = detector.detect(display_frame)
            display_frame = draw_detection_overlay(display_frame, detections)
    elif ai_mode == AIMode.SEGMENTATION and detector is not None:
        from app.detection.yolo_detector import YoloDetector
        if isinstance(detector, YoloDetector):
            display_frame, _ = detector.segment(display_frame)
    elif ai_mode == AIMode.DEPTH:
        display_frame, _ = depth_estimator.estimate(display_frame)
    elif ai_mode == AIMode.EGO_EXO:
        depth_map, _ = depth_estimator.estimate_raw(display_frame)
        display_frame = ego_exo_projector.project(display_frame, depth_map)
        h, w = display_frame.shape[:2]
        half = w // 2
        display_frame = np.hstack([
            cv2.resize(img, (half, h)),
            cv2.resize(display_frame, (half, h)),
        ])

    out_path = args.output or ("output_" + os.path.basename(args.input_image))
    out_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) if display_frame.shape[2] == 3 else display_frame
    cv2.imwrite(out_path, out_bgr)
    logger.info("Saved: %s", out_path)


def _video_loop(
    running: threading.Event,
    frame_queue: "queue.Queue[tuple]",
    video_path: str,
    metrics: RuntimeMetrics,
) -> None:
    """Producer: read video frames and put (frame, metadata, ts) into queue."""
    cap = cv2.VideoCapture(video_path)
    try:
        while running.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            elif frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = time.time_ns()
            try:
                frame_queue.put_nowait((frame, {}, ts))
                metrics.on_frame_captured()
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait((frame, {}, ts))
                metrics.on_frame_dropped()
    finally:
        cap.release()
        running.clear()


def _run_video_mode(
    args: argparse.Namespace,
    config: dict,
    frame_queue: "queue.Queue[tuple]",
    metrics: RuntimeMetrics,
    logger_csv: CSVLogger,
    flat_proc: dict,
) -> None:
    """Run pipeline with video file as frame source; serve MJPEG dashboard."""
    from app.server.state import ServerState
    from app.server.app import run_server

    server_config = config.get("server") or {}
    server_state = ServerState()
    host = server_config.get("host", "0.0.0.0")
    port = int(server_config.get("port", 8765))
    server_thread = threading.Thread(
        target=run_server,
        args=(server_state, host, port, get_pipeline_state(), None, config),
        daemon=True,
    )
    server_thread.start()

    detector = create_detector(config)
    depth_cfg = config.get("depth") or {}
    depth_estimator = DepthEstimator(depth_cfg.get("model_path", "models/midas_v21_small_256.onnx"))
    ego_exo_projector = EgoExoProjector()
    inference_worker = InferenceWorker(detector, depth_estimator, ego_exo_projector)
    det_config = config.get("detection") or {}

    running = threading.Event()
    running.set()
    video_thread = threading.Thread(
        target=_video_loop,
        args=(running, frame_queue, args.input_video, metrics),
        daemon=True,
    )
    video_thread.start()

    headless = True
    last_frame_time = time.time()
    processing_config = dict(flat_proc)
    imx500_detector = None
    pose_estimator = None
    current_ai_mode = AIMode(args.ai_mode) if args.ai_mode else AIMode.NONE
    prev_ai_mode = current_ai_mode

    try:
        while running.is_set():
            try:
                frame, metadata, capture_timestamp_ns = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if time.time() - last_frame_time > WATCHDOG_EMPTY_SEC:
                    break
                continue

            last_frame_time = time.time()
            metrics.on_frame_displayed((time.time_ns() - capture_timestamp_ns) / 1e6)
            snap = get_pipeline_state().get_snapshot()
            try:
                current_ai_mode = AIMode(snap.get("ai_mode", "none"))
            except ValueError:
                current_ai_mode = AIMode.NONE

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
                detections, inf_ms = imx500_detector.detect(display_frame, metadata)
                display_frame = draw_detection_overlay(display_frame, detections)
                metrics.on_detection_completed(inf_ms, len(detections))
            elif current_ai_mode == AIMode.POSE and pose_estimator is not None:
                poses, inf_ms = pose_estimator.estimate(display_frame, metadata)
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
            snapshot["inference_backend"] = "ONNX CPU" if current_ai_mode in CPU_MODES else "IMX500 NPU"
            display_frame = draw_metrics_overlay(display_frame, snapshot, mode_name=mode_name)
            logger_csv.log(snapshot, mode_name)
            server_state.set_frame(display_frame)
            server_state.set_metrics(snapshot)

    finally:
        running.clear()
        inference_worker.stop()
        video_thread.join(timeout=2.0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    app_cfg = config.get("app") or {}
    csv_path = app_cfg.get("csv_log_path", "./logs/runtime_metrics.csv")
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    proc = config.get("processing") or {}
    flat_proc = _normalize_processing_config(proc)
    pipeline_state = get_pipeline_state()
    pipeline_state.set_isp("sharpening", flat_proc["sharpening_enabled"])
    pipeline_state.set_isp("sharpening_strength", flat_proc["sharpening_strength"])
    pipeline_state.set_isp("denoising", flat_proc["denoising_enabled"])
    pipeline_state.set_isp("denoising_strength", flat_proc["denoising_strength"])
    pipeline_state.set_isp("denoising_mode", flat_proc["denoising_mode"])
    pipeline_state.set_isp("demosaicing_quality_fast", not proc.get("demosaicing_quality", False))
    pipeline_state.set_isp("ae_lock", False)
    pipeline_state.set_isp("awb_lock", False)
    pipeline_state.set_isp("af_enabled", False)
    pipeline_state.set_ai_mode(args.ai_mode)

    # --- Single image mode: load, run ISP + AI once, save, exit ---
    if args.input_image is not None:
        _run_image_mode(args, config, flat_proc)
        return

    # --- Video or live: need queue, metrics, optional server ---
    frame_queue_size = app_cfg.get("frame_queue_size", 2)
    frame_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=frame_queue_size)
    include_system = app_cfg.get("include_system_metrics", True)
    metrics = RuntimeMetrics(include_system_metrics=include_system)
    logger_csv = CSVLogger(csv_path)

    # --- Video mode: frame source is video file, no camera ---
    if args.input_video is not None:
        _run_video_mode(args, config, frame_queue, metrics, logger_csv, flat_proc)
        return

    # --- Live camera mode ---
    camera_manager = CameraManager(config, frame_queue, metrics)

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
            args=(server_state, host, port, pipeline_state, camera_manager, config),
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

    headless = args.headless or app_cfg.get("headless", False)
    window_name = app_cfg.get("display_window_name", "Pi Camera Platform")
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_frame_time = time.time()
    processing_config = dict(flat_proc)

    imx500_detector: Optional[Imx500Detector] = None
    pose_estimator: Optional[PoseEstimator] = None
    prev_ai_mode: Optional[AIMode] = None
    det_config = config.get("detection") or {}

    try:
        while running.is_set():
            try:
                frame, metadata, capture_timestamp_ns = frame_queue.get(timeout=1.0)
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
                detections, inf_ms = imx500_detector.detect(display_frame, metadata)
                display_frame = draw_detection_overlay(display_frame, detections)
                metrics.on_detection_completed(inf_ms, len(detections))
            elif current_ai_mode == AIMode.POSE and pose_estimator is not None:
                poses, inf_ms = pose_estimator.estimate(display_frame, metadata)
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
