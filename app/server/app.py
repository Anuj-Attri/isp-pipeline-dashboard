"""FastAPI app: ISP + AI dashboard, MJPEG, WebSocket metrics, API for toggles. HTTPS on first run."""
import asyncio
import os
import subprocess
import time
from typing import Any, Iterator, Optional, Union

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.server.state import ServerState

_pipeline_state_ref: Optional[Any] = None
_camera_manager_ref: Optional[Any] = None
_config_ref: Optional[Any] = None


class IspToggleBody(BaseModel):
    key: str
    value: Union[bool, float, str]


class AiModeBody(BaseModel):
    mode: str


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ISP Pipeline Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { display: flex; flex-direction: column; height: 100vh; overflow: hidden; margin: 0; font-family: sans-serif; background: #1a1a1a; color: #eee; }
    .header { flex-shrink: 0; padding: 8px 12px; display: flex; align-items: center; gap: 16px; }
    .header h1 { margin: 0; font-size: 1.25rem; }
    .ws-status { font-size: 12px; color: #888; }
    #health { width: 12px; height: 12px; border-radius: 50%; }
    .main-content { display: flex; flex: 1; overflow: hidden; min-height: 0; }
    .stream-panel { flex: 1; position: sticky; top: 0; height: 100%; min-width: 0; background: #111; }
    .stream-panel img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .sidebar { width: 340px; overflow-y: auto; padding: 16px; background: #1e1e1e; flex-shrink: 0; }
    .sidebar h3 { margin: 0 0 8px 0; font-size: 14px; color: #aaa; }
    #status { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12px; font-family: monospace; }
    .info-icon { display: inline-block; width: 16px; height: 16px; line-height: 16px; background: #444; border-radius: 50%; text-align: center; font-size: 11px; cursor: help; margin-left: 6px; color: #aaa; position: relative; }
    .info-icon:hover::after { content: attr(data-tooltip); position: absolute; left: 20px; top: 0; background: #222; color: #ddd; padding: 8px 12px; border-radius: 6px; font-size: 11px; max-width: 260px; z-index: 100; white-space: pre-wrap; border: 1px solid #555; font-family: sans-serif; font-weight: normal; }
    .control-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .control-row label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }
    .control-row input[type="checkbox"] { width: 18px; height: 18px; }
    .control-row input[type="range"] { width: 80px; }
    .control-row select { font-size: 12px; padding: 2px 6px; }
    .ai-modes label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 12px; margin-bottom: 4px; }
    .ai-modes input[type="radio"] { width: 14px; height: 14px; }
    .chart-wrap { max-height: 150px; margin-bottom: 12px; }
    .chart-wrap canvas { max-height: 150px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>ISP Pipeline Dashboard</h1>
    <span id="health" style="background:#666"></span>
    <span class="ws-status">WebSocket: <span id="ws-state">connecting...</span></span>
  </div>
  <div class="main-content">
    <div class="stream-panel" id="stream-panel">
      <img id="video" src="/video" alt="MJPEG stream" />
    </div>
    <div class="sidebar">
      <h3>Status</h3>
      <div id="status">
        <span>FPS</span><span id="val-fps">-</span>
        <span>Latency (ms)</span><span id="val-latency">-</span>
        <span>Inference (ms)</span><span id="val-inference">-</span>
        <span>CPU %</span><span id="val-cpu">-</span>
        <span>Mem %</span><span id="val-mem">-</span>
        <span>Temp °C</span><span id="val-temp">-</span>
        <span>GPU %</span><span id="val-gpu">N/A</span>
        <span>CPU MHz</span><span id="val-cpu-mhz">-</span>
        <span>Dropped</span><span id="val-dropped">-</span>
        <span>Inference Backend</span><span id="val-inference-backend">-</span>
      </div>
      <h3 style="margin-top:12px">ISP Controls</h3>
      <div class="control-row"><label><input type="checkbox" id="isp-sharpening" /> Sharpening</label><span class="info-icon" data-tooltip="Applies a 3x3 convolution kernel to enhance edge contrast. Higher values increase perceived sharpness but may amplify noise.">(i)</span></div>
      <div class="control-row"><input type="range" id="sharpening-strength" min="0" max="20" value="10" step="1" /><span id="sharpening-val">1.0</span></div>
      <div class="control-row"><label><input type="checkbox" id="isp-denoising" /> Denoising</label><span class="info-icon" data-tooltip="Reduces sensor noise. Fast mode uses Gaussian blur (low latency). Quality mode uses Non-Local Means (NLM) — slower but preserves edges.">(i)</span></div>
      <div class="control-row"><input type="range" id="denoising-strength" min="0" max="20" value="10" step="1" /><span id="denoising-val">1.0</span> <select id="denoising-mode"><option value="fast">Fast</option><option value="quality">Quality</option></select></div>
      <div class="control-row"><label><input type="checkbox" id="isp-demosaicing-fast" /> Demosaicing fast</label><span class="info-icon" data-tooltip="Controls raw Bayer pattern interpolation quality. When enabled, uses high-quality bicubic interpolation via libcamera's ISP. Disable for faster but lower-quality nearest-neighbor demosaicing.">(i)</span></div>
      <div class="control-row"><label><input type="checkbox" id="isp-ae-lock" /> AE Lock</label><span class="info-icon" data-tooltip="Auto Exposure Lock. When enabled, fixes the current exposure time and gain. Useful for comparing scenes without brightness variation.">(i)</span></div>
      <div class="control-row"><label><input type="checkbox" id="isp-awb-lock" /> AWB Lock</label><span class="info-icon" data-tooltip="Auto White Balance Lock. Freezes the current color temperature correction matrix. Useful for consistent color across lighting changes.">(i)</span></div>
      <div class="control-row"><label><input type="checkbox" id="isp-af" /> Auto Focus</label><span class="info-icon" data-tooltip="Phase-detect autofocus using the IMX500 sensor. Toggle off to lock focus at current distance for stable inference frames.">(i)</span></div>
      <h3 style="margin-top:12px">AI Mode</h3>
      <div class="ai-modes">
        <label><input type="radio" name="ai" value="none" /> None <span class="info-icon" data-tooltip="Raw ISP output only. No AI inference running.">(i)</span></label>
        <label><input type="radio" name="ai" value="detection" /> Object Detection <span class="info-icon" data-tooltip="Object detection with IMX500 NPU acceleration. Draws bounding boxes with per-class colors and confidence scores. Detects 80 COCO classes. Runs at 30 FPS on Sony NPU — zero CPU inference cost. Automatically loads SSD MobileNetV2 FPN model onto IMX500 chip.">(i)</span></label>
        <label><input type="radio" name="ai" value="pose" /> Pose Estimation <span class="info-icon" data-tooltip="HigherHRNet pose estimation running on IMX500 NPU. Detects 17 COCO body keypoints per person: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. Draws color-coded skeleton overlay. Runs at 30 FPS — inference on Sony NPU, zero CPU cost. Switches IMX500 model automatically (~2s camera restart).">(i)</span></label>
        <label><input type="radio" name="ai" value="segmentation" /> Segmentation <span class="info-icon" data-tooltip="YOLOv11n-seg instance segmentation. Draws per-object pixel masks with semi-transparent color overlays. ~5-10 FPS on Pi 5.">(i)</span></label>
        <label><input type="radio" name="ai" value="depth" /> Depth Estimation <span class="info-icon" data-tooltip="MiDaS v2.1 monocular depth estimation. Colorizes the scene by estimated distance using the MAGMA colormap (purple=far, yellow=close). ~5-8 FPS on Pi 5.">(i)</span></label>
        <label><input type="radio" name="ai" value="ego_exo" /> Ego→Exo Projection <span class="info-icon" data-tooltip="Converts the egocentric (first-person) camera view into an estimated exocentric (top-down/bird's-eye) view using depth-guided inverse perspective mapping (IPM). Left: ego view. Right: projected exo view. ~3-5 FPS.">(i)</span></label>
      </div>
      <h3 style="margin-top:12px">FPS</h3>
      <div class="chart-wrap"><canvas id="chart-fps"></canvas></div>
      <h3>Latency (ms)</h3>
      <div class="chart-wrap"><canvas id="chart-latency"></canvas></div>
      <h3>Inference (ms)</h3>
      <div class="chart-wrap"><canvas id="chart-inference"></canvas></div>
    </div>
  </div>
  <script>
    const maxPoints = 60;
    const opts = { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } } };
    const fpsChart = new Chart(document.getElementById('chart-fps'), { type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#4ade80', fill: false }] }, options: { ...opts, scales: { y: { min: 0 }, x: { ticks: { maxTicksLimit: 5 } } } } });
    const latChart = new Chart(document.getElementById('chart-latency'), { type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#60a5fa', fill: false }] }, options: { ...opts, scales: { y: { min: 0 }, x: { ticks: { maxTicksLimit: 5 } } } } });
    const infChart = new Chart(document.getElementById('chart-inference'), { type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#a78bfa', fill: false }] }, options: { ...opts, scales: { y: { min: 0 }, x: { ticks: { maxTicksLimit: 5 } } } } });
    function addPoint(chart, label, value) { chart.data.labels.push(label); chart.data.datasets[0].data.push(value); if (chart.data.labels.length > maxPoints) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); } chart.update('none'); }
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');
    ws.onopen = () => { document.getElementById('ws-state').textContent = 'connected'; };
    ws.onclose = () => { document.getElementById('ws-state').textContent = 'disconnected'; };
    ws.onerror = () => { document.getElementById('ws-state').textContent = 'error'; };
    ws.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      const t = new Date().toLocaleTimeString();
      document.getElementById('val-fps').textContent = (d.fps != null ? d.fps.toFixed(1) : '-');
      document.getElementById('val-latency').textContent = (d.latency_ms != null ? d.latency_ms.toFixed(1) : '-');
      document.getElementById('val-inference').textContent = (d.inference_latency_ms != null ? d.inference_latency_ms.toFixed(1) : '-');
      document.getElementById('val-cpu').textContent = (d.cpu_percent != null ? d.cpu_percent.toFixed(1) : '-');
      document.getElementById('val-mem').textContent = (d.memory_percent != null ? d.memory_percent.toFixed(1) : '-');
      document.getElementById('val-temp').textContent = (d.temperature_c != null ? d.temperature_c.toFixed(1) : '-');
      document.getElementById('val-cpu-mhz').textContent = (d.cpu_freq_mhz != null ? d.cpu_freq_mhz : '-');
      document.getElementById('val-dropped').textContent = (d.dropped_frames != null ? d.dropped_frames : '-');
      const be = document.getElementById('val-inference-backend');
      if (be) be.textContent = (d.inference_backend != null ? d.inference_backend : '-');
      const fps = d.fps != null ? d.fps : 0;
      const h = document.getElementById('health');
      if (fps >= 20) h.style.background = '#22c55e';
      else if (fps >= 10) h.style.background = '#eab308';
      else h.style.background = '#ef4444';
      addPoint(fpsChart, t, d.fps ?? 0);
      addPoint(latChart, t, d.latency_ms ?? 0);
      addPoint(infChart, t, d.inference_latency_ms ?? 0);
    };
    const ispKeys = { 'isp-sharpening': 'sharpening', 'isp-denoising': 'denoising', 'isp-demosaicing-fast': 'demosaicing_quality_fast', 'isp-ae-lock': 'ae_lock', 'isp-awb-lock': 'awb_lock', 'isp-af': 'af_enabled' };
    ['isp-sharpening','isp-denoising','isp-demosaicing-fast','isp-ae-lock','isp-awb-lock','isp-af'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', function() { const key = ispKeys[id]; fetch('/api/isp_toggle', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key, value: this.checked }) }).catch(() => {}); });
    });
    document.getElementById('sharpening-strength').addEventListener('input', function() { const v = this.value / 10; document.getElementById('sharpening-val').textContent = v.toFixed(1); fetch('/api/isp_toggle', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: 'sharpening_strength', value: v }) }).catch(() => {}); });
    document.getElementById('denoising-strength').addEventListener('input', function() { const v = this.value / 10; document.getElementById('denoising-val').textContent = v.toFixed(1); fetch('/api/isp_toggle', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: 'denoising_strength', value: v }) }).catch(() => {}); });
    document.getElementById('denoising-mode').addEventListener('change', function() { fetch('/api/isp_toggle', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: 'denoising_mode', value: this.value }) }).catch(() => {}); });
    function setStreamBorder(mode) {
      const panel = document.getElementById('stream-panel');
      if (!panel) return;
      panel.style.border = (mode === 'pose') ? '3px solid rgb(255, 200, 0)' : 'none';
    }
    document.querySelectorAll('input[name="ai"]').forEach(el => {
      el.addEventListener('change', function() {
        fetch('/api/ai_mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: this.value }) }).catch(() => {});
        setStreamBorder(this.value);
      });
    });
    fetch('/api/pipeline_state').then(r => r.json()).then(s => {
      if (s.sharpening != null) document.getElementById('isp-sharpening').checked = s.sharpening;
      if (s.denoising != null) document.getElementById('isp-denoising').checked = s.denoising;
      if (s.demosaicing_quality_fast != null) document.getElementById('isp-demosaicing-fast').checked = s.demosaicing_quality_fast;
      if (s.ae_lock != null) document.getElementById('isp-ae-lock').checked = s.ae_lock;
      if (s.awb_lock != null) document.getElementById('isp-awb-lock').checked = s.awb_lock;
      if (s.af_enabled != null) document.getElementById('isp-af').checked = s.af_enabled;
      if (s.sharpening_strength != null) { document.getElementById('sharpening-strength').value = Math.round(s.sharpening_strength * 10); document.getElementById('sharpening-val').textContent = s.sharpening_strength.toFixed(1); }
      if (s.denoising_strength != null) { document.getElementById('denoising-strength').value = Math.round(s.denoising_strength * 10); document.getElementById('denoising-val').textContent = s.denoising_strength.toFixed(1); }
      if (s.denoising_mode != null) document.getElementById('denoising-mode').value = s.denoising_mode;
      if (s.ai_mode) { const r = document.querySelector('input[name="ai"][value="' + s.ai_mode + '"]'); if (r) { r.checked = true; setStreamBorder(s.ai_mode); } }
    }).catch(() => {});
  </script>
</body>
</html>
"""


def create_app(
    state: ServerState,
    pipeline_state: Optional[Any] = None,
    camera_manager: Optional[Any] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    global _pipeline_state_ref, _camera_manager_ref, _config_ref
    _pipeline_state_ref = pipeline_state
    _camera_manager_ref = camera_manager
    _config_ref = config

    app = FastAPI(title="ISP Pipeline Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/pipeline_state")
    def get_pipeline_state() -> dict:
        if _pipeline_state_ref is None:
            return {}
        return _pipeline_state_ref.get_snapshot()

    @app.post("/api/isp_toggle")
    def post_isp_toggle(body: IspToggleBody) -> dict:
        if _pipeline_state_ref is not None:
            _pipeline_state_ref.set_isp(body.key, body.value)
        if _camera_manager_ref is not None and body.key in ("ae_lock", "awb_lock", "af_enabled"):
            from app.processing.isp_controls import ISPController
            picam2 = _camera_manager_ref.get_picam2()
            val = bool(body.value) if isinstance(body.value, (bool, int, float)) else False
            if body.key == "ae_lock":
                ISPController.set_ae_lock(picam2, val)
            elif body.key == "awb_lock":
                ISPController.set_awb_lock(picam2, val)
            elif body.key == "af_enabled":
                ISPController.set_af_mode(picam2, val)
        return {"ok": True}

    @app.post("/api/ai_mode")
    def post_ai_mode(body: AiModeBody) -> dict:
        if _pipeline_state_ref is not None:
            _pipeline_state_ref.set_ai_mode(body.mode)
        return {"ok": True}

    @app.get("/video")
    def video_stream():
        app_config = (_config_ref or {}).get("app", {})
        mjpeg_res = tuple(app_config.get("mjpeg_resolution", [1920, 1080]))
        mjpeg_quality = app_config.get("mjpeg_quality", 85)

        def generate() -> Iterator[bytes]:
            boundary = "frame"
            while True:
                frame = state.get_frame()
                if frame is None:
                    time.sleep(0.033)
                    continue
                frame_resized = cv2.resize(frame, mjpeg_res, interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(
                    ".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, mjpeg_quality]
                )
                if buf is None:
                    continue
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                    + buf.tobytes() + b"\r\n"
                )
                time.sleep(0.001)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws")
    async def websocket_metrics(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                metrics = state.get_metrics()
                await websocket.send_json(metrics)
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    return app


def run_server(
    state: ServerState,
    host: str = "0.0.0.0",
    port: int = 8765,
    pipeline_state: Optional[Any] = None,
    camera_manager: Optional[Any] = None,
    config: Optional[Any] = None,
) -> None:
    """Run FastAPI with uvicorn. HTTPS via self-signed cert if certs/ not present."""
    import uvicorn
    # Cert paths relative to project root (working dir when launched as python -m app.main)
    try:
        from app.config.config_loader import load_config
        _cfg = load_config(None)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception:
        _root = os.getcwd()
    cert_path = os.path.join(_root, "certs", "cert.pem")
    key_path = os.path.join(_root, "certs", "key.pem")
    if not os.path.exists(cert_path):
        os.makedirs(os.path.dirname(cert_path), exist_ok=True)
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "365", "-nodes", "-subj", "/CN=picamera",
            ],
            check=False,
            capture_output=True,
        )
    app = create_app(state, pipeline_state, camera_manager, config)
    if os.path.exists(cert_path) and os.path.exists(key_path):
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            ssl_certfile=cert_path,
            ssl_keyfile=key_path,
        )
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
