# ISP Pipeline Dashboard

> Real-time edge imaging platform for Raspberry Pi 5 + Sony IMX500 AI Camera

![Dashboard Screenshot](docs/dashboard.png)

## What This Is

A full-stack camera pipeline that runs directly on a Raspberry Pi 5, demonstrating
real-time ISP algorithms and AI inference on the edge. The Sony IMX500 sensor runs
neural networks directly on the camera chip — detection and pose estimation run at
30 FPS with zero CPU inference cost.

## Features

**ISP Pipeline**
- Sharpening, Denoising (Fast/Quality), AE Lock, AWB Lock, Auto Focus
- All toggleable in real-time with visual confirmation in the stream
- Strength sliders for sharpening and denoising

**AI Modes — IMX500 NPU (zero CPU cost)**
- Object Detection — SSD MobileNetV2 FPN, 80 COCO classes, 30 FPS
- Pose Estimation — HigherHRNet, 17 COCO keypoints, 30 FPS

**AI Modes — CPU ONNX Runtime**
- Instance Segmentation — YOLO11n-seg, ~15 FPS
- Depth Estimation — MiDaS v2.1, relative depth colormap, ~10 FPS
- Ego→Exo Projection — depth-guided IPM top-down view, ~8 FPS

**Dashboard**
- Live MJPEG stream + WebSocket metrics at https://\<pi-ip\>:8765
- Real-time charts: FPS, Latency, Inference time
- ISP control panel with tooltips explaining each algorithm
- AI mode selector with per-mode color-coded video border

## Hardware

| Component | Requirement |
|-----------|-------------|
| Board | Raspberry Pi 5 (4GB+ recommended) |
| Camera | Raspberry Pi AI Camera (Sony IMX500) |
| Storage | 32GB+ MicroSD, Raspberry Pi OS Bookworm 64-bit |
| Network | Ethernet or WiFi (for dashboard access) |

## Quick Start

```bash
# On Raspberry Pi
git clone https://github.com/<you>/isp-pipeline-dashboard
cd isp-pipeline-dashboard
pip3 install --break-system-packages -r requirements.txt
./scripts/launch.sh
```

Open **https://\<pi-ip\>:8765** in your browser.
Accept the self-signed certificate warning (local network only).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                              │
│                                                                 │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ Sony IMX500  │    │  Pi Camera  │    │   Main Loop      │   │
│  │   Sensor     │───▶│  Stack      │───▶│   (30 FPS)       │   │
│  │              │    │ (libcamera/ │    │                  │   │
│  │ NPU runs:    │    │  picamera2) │    │  ISP Processing  │   │
│  │ • Detection  │    └─────────────┘    │  AI Dispatch     │   │
│  │ • Pose Est.  │                       │  Metrics         │   │
│  │              │    Metadata ──────────▶  CSV Logger      │   │
│  └──────────────┘    (inference result) └────────┬─────────┘   │
│                                                   │             │
│  ┌────────────────────────────────────────────────▼──────────┐  │
│  │                  ONNX Runtime (async thread)               │  │
│  │   Segmentation (YOLO11n-seg) │ Depth (MiDaS) │ Ego→Exo   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                   │             │
│  ┌────────────────────────────────────────────────▼──────────┐  │
│  │              FastAPI Server (daemon thread)                │  │
│  │   GET /          → Dashboard HTML                         │  │
│  │   GET /video     → MJPEG stream                           │  │
│  │   WS  /ws        → Metrics JSON @ 10Hz                    │  │
│  │   POST /api/*    → ISP toggles, AI mode switch            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Browser Client   │
                    │  https://<pi>:8765 │
                    └────────────────────┘
```

## ISP Algorithms

| Algorithm | Method | Latency | Notes |
|-----------|--------|---------|-------|
| Sharpening | 3×3 convolution kernel | <1ms | Adjustable strength 0–2 |
| Denoising Fast | Gaussian blur | <1ms | Low latency |
| Denoising Quality | Non-Local Means (NLM) | ~15ms | Edge-preserving |
| AE Lock | libcamera runtime control | 0ms | Fixes exposure time + gain |
| AWB Lock | libcamera runtime control | 0ms | Fixes color temperature matrix |
| Auto Focus | IMX500 phase-detect AF | 0ms | Locks focus at current distance |

## AI Modes

| Mode | Model | Backend | FPS | Notes |
|------|-------|---------|-----|-------|
| Detection | SSD MobileNetV2 FPN 320×320 | IMX500 NPU | 30 | Zero CPU cost |
| Pose Estimation | HigherHRNet COCO | IMX500 NPU | 30 | 17 keypoints, zero CPU cost |
| Segmentation | YOLO11n-seg | ONNX Runtime | ~15 | Per-instance masks |
| Depth Estimation | MiDaS v2.1 small 256 | ONNX Runtime | ~10 | Relative depth, MAGMA colormap |
| Ego→Exo | MiDaS + IPM homography | ONNX Runtime | ~8 | Top-down projection |

## Configuration

All settings in `config/default_config.yaml`:

```yaml
camera:
  resolution: [1280, 720]  # capture resolution
  framerate: 30

detection:
  backend: auto             # "auto" = IMX500 first, YOLO fallback
  confidence_threshold: 0.4

server:
  host: "0.0.0.0"
  port: 8765
```

## Project Structure

```
app/
  camera/           # Picamera2 + IMX500 lifecycle manager
  detection/        # IMX500 detector, YOLO ONNX detector, pose estimator
  processing/       # ISP algorithms, depth, ego-exo, pipeline state
  metrics/          # Runtime metrics, CSV logger, frame overlay
  server/           # FastAPI dashboard, WebSocket, REST API
config/             # YAML configuration
models/             # Downloaded weights — gitignored, auto-downloaded
scripts/
  launch.sh         # One-command startup
  install_service.sh # systemd service setup
  perf_mode.sh      # CPU performance governor
logs/               # CSV telemetry — gitignored
```

## Extending

**Add an ISP stage:** implement in `app/processing/`, add field to `PipelineState`,
call from `app/main.py`, add toggle + tooltip to dashboard.

**Add an AI mode:** add value to `AIMode` enum, implement backend in `app/detection/`
or `app/processing/`, handle in `app/main.py` mode dispatch, add radio + tooltip to dashboard.

**Add a metric:** add to `RuntimeMetrics.get_snapshot()` — appears automatically in
CSV log and WebSocket stream.

## License

MIT
