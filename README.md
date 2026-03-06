# Edge CV Pipeline — Raspberry Pi 5 + Sony IMX500

A modular, real-time computer vision pipeline for the Raspberry Pi 5 using the
Sony IMX500 intelligent image sensor. Designed to demonstrate the full stack from
raw sensor data through ISP processing to on-device AI inference — with hardware
acceleration where available and graceful CPU fallback where not.

The web dashboard exists purely as a runtime inspection surface: it lets you
toggle ISP stages, switch inference modes, and observe the effect on the live
stream and metrics without restarting the pipeline.

## Why This Exists

Most edge CV demos either:
(a) run a pretrained model on static images and call it "edge AI", or
(b) offload everything to the cloud and stream JPEG frames

This pipeline runs the full inference stack on-device, uses the IMX500's onboard
ARM Cortex-M55 + Ethos-U55 NPU for zero-CPU-cost detection and pose estimation,
and keeps the ISP pipeline observable and configurable at runtime.

It is intended as a reference architecture for:
- Edge inference system design on ARM SoCs
- IMX500 NPU integration with picamera2/libcamera
- ISP algorithm evaluation on real sensor data
- Hardware-software codesign prototyping

## Hardware

| Component | Details |
|---|---|
| SoC | Raspberry Pi 5 (BCM2712, 4× Cortex-A76 @ 2.4GHz) |
| Sensor | Sony IMX500 — 12MP stacked CMOS + onboard NPU |
| NPU | ARM Cortex-M55 + Ethos-U55 (on IMX500 die) |
| ISP | Raspberry Pi PiSP (dedicated ISP hardware block) |
| OS | Raspberry Pi OS Bookworm 64-bit, libcamera v0.7+ |

The IMX500 is a stacked sensor — the NPU sits on the same die as the image
sensor. Neural network weights are uploaded to the chip at camera init time.
Inference results are embedded in frame metadata and read with zero CPU cycles.

## Pipeline Architecture

```
IMX500 Sensor
│
├── RAW frames (2028×1520 RGGB) ──▶ PiSP ISP Hardware
│                                        │
│                                        ▼
│                               RGB888 frames (1280×720)
│                                        │
│   NPU (on-sensor)                      ▼
│   ├── SSD MobileNetV2 FPN   ──▶  capture_loop() ──▶ frame_queue
│   └── HigherHRNet COCO            (producer thread)       │
│         │                                                  │
│         └── metadata ──────────────────────────────▶ main_loop()
│                                                      (consumer thread)
│                                                           │
│                                              ┌────────────┼────────────┐
│                                              │            │            │
│                                         ISP stages   AI dispatch   Metrics
│                                         (inline)     (inline or    (inline)
│                                                       async worker)
│                                                           │
│                                              ┌────────────▼──────────┐
│                                              │   AsyncInferenceWorker │
│                                              │   ONNX Runtime         │
│                                              │   ├── YOLO11n-seg      │
│                                              │   ├── MiDaS v2.1       │
│                                              │   └── MiDaS + IPM      │
│                                              └───────────────────────┘
│                                                           │
│                                              ┌────────────▼──────────┐
│                                              │   FastAPI Server       │
│                                              │   MJPEG + WebSocket    │
│                                              │   REST control API     │
│                                              └───────────────────────┘
```

## Inference Backends

### IMX500 NPU (zero CPU cost)
The NPU runs in parallel with image capture. Switching models requires a
camera restart (~2 seconds) since the network must be re-uploaded to the chip.

| Mode | Model | RPK |
|---|---|---|
| Object Detection | SSD MobileNetV2 FPN 320×320 | imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk |
| Pose Estimation | HigherHRNet COCO | imx500_network_higherhrnet_coco.rpk |

`_pp` suffix = post-processing (NMS) runs on NPU. CPU receives filtered boxes only.

### ONNX Runtime (async CPU worker)
CPU inference runs in a dedicated thread. The main loop never blocks —
it reads the latest result from a shared slot and continues at camera FPS.

| Mode | Model | Input | ~FPS |
|---|---|---|---|
| Segmentation | YOLO11n-seg | 320×320 | 15 |
| Depth Estimation | MiDaS v2.1 small | 256×256 | 10 |
| Ego→Exo Projection | MiDaS + IPM homography | 256×256 | 8 |

## ISP Pipeline

All ISP stages run inline on the main loop thread. Picamera2 runtime controls
(AE, AWB, AF) are applied directly to the libcamera control list.

| Stage | Method | Latency | Configurable |
|---|---|---|---|
| Sharpening | 3×3 unsharp mask kernel | <1ms | Strength 0–2 |
| Denoising (fast) | Gaussian blur | <1ms | Strength 0–2 |
| Denoising (quality) | Non-Local Means (NLM) | ~15ms | Strength 0–2 |
| Auto Exposure Lock | libcamera AeEnable control | 0ms | Toggle |
| Auto White Balance Lock | libcamera AwbEnable control | 0ms | Toggle |
| Auto Focus | IMX500 phase-detect PDAF | 0ms | Toggle |
| Demosaicing quality | libcamera NoiseReductionMode | 0ms | Toggle |

## Performance (Pi 5, 2.4GHz locked)

| Mode | FPS | Inference | CPU% |
|---|---|---|---|
| ISP only | 30 | — | ~15% |
| Detection (IMX500) | 30 | ~0ms CPU | ~20% |
| Pose (IMX500) | 30 | ~0ms CPU | ~20% |
| Segmentation (ONNX) | ~15 | ~65ms | ~85% |
| Depth (ONNX) | ~10 | ~100ms | ~90% |
| Ego→Exo (ONNX) | ~8 | ~130ms | ~95% |

## Quick Start

```bash
git clone https://github.com/Anuj-Attri/isp-pipeline-dashboard
cd isp-pipeline-dashboard
pip3 install --break-system-packages -r requirements.txt
```

### Live camera (4K ISP, 1080p stream)

```bash
./scripts/launch.sh
# Dashboard: https://<pi-ip>:8765
```

### Single image — run full ISP + AI pipeline, save result

```bash
./scripts/launch.sh --image /path/to/image.jpg --mode depth
```

### Video file — feed through pipeline, serve on dashboard

```bash
./scripts/launch.sh --video /path/to/video.mp4 --mode segmentation
```

### AI mode options

```bash
# none | detection | pose | segmentation | depth | ego_exo
./scripts/launch.sh --mode detection
```

Requires Raspberry Pi OS Bookworm with `python3-picamera2` and `imx500-all` installed:

```bash
sudo apt install python3-picamera2 python3-opencv python3-psutil imx500-all
```

## Project Structure

```
app/
  camera/           # Picamera2 + IMX500 lifecycle, capture loop, model hot-swap
  detection/        # IMX500 detector, YOLO ONNX detector, pose estimator, overlay
  processing/       # ISP stages, depth estimator, ego-exo projector,
                    # async inference worker, pipeline state
  metrics/          # Frame metrics, CSV telemetry, HUD overlay
  server/           # FastAPI MJPEG server, WebSocket metrics, REST control API
config/             # YAML pipeline configuration
models/             # ONNX weights — gitignored, auto-downloaded on first run
scripts/
  launch.sh         # Start pipeline + print dashboard URL
  install_service.sh # Register as systemd service
  perf_mode.sh      # Lock CPU to performance governor
```

## Configuration

```yaml
# config/default_config.yaml
camera:
  resolution: [1280, 720]
  framerate: 30

detection:
  backend: auto              # "auto": IMX500 first, YOLO fallback
  confidence_threshold: 0.4

server:
  host: "0.0.0.0"
  port: 8765
```

## Extending

**New ISP stage:** implement in `app/processing/`, add field to `PipelineState`,
call from the main loop ISP section, expose toggle + slider in the server API.

**New inference mode:** add to `AIMode` enum, implement backend class with
`.detect(frame)` or `.estimate(frame)` interface, register in `app/main.py`
mode dispatch. If using IMX500, add the `.rpk` path to `pipeline_state.py`
constants and handle camera restart in the mode-switch block.

**New metric:** add to `RuntimeMetrics.get_snapshot()` — propagates automatically
to the WebSocket stream, dashboard UI, and CSV telemetry.

## License

MIT — Anuj Attri, 2026
