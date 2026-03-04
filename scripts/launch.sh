#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🎥  ISP Pipeline Dashboard — Raspberry Pi AI Camera"
echo "===================================================="

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>/dev/null || true
echo "✓ CPU locked to performance mode (2.4GHz)"

PI_IP=$(hostname -I | awk '{print $1}')
echo "✓ IP: $PI_IP"
echo "✓ Dashboard: https://$PI_IP:8765"
echo ""

cd "$PROJECT_DIR"
python3 -m app.main --headless &
PIPELINE_PID=$!

echo "⏳ Starting pipeline (PID: $PIPELINE_PID)..."
for i in $(seq 1 30); do
  if curl -sk "https://$PI_IP:8765" > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo ""
echo "✅ Ready — open https://$PI_IP:8765 in your browser"
echo "   Press Ctrl+C to stop"
echo ""

if [ -n "$DISPLAY" ]; then
  chromium-browser --start-maximized "https://$PI_IP:8765" 2>/dev/null &
fi

wait $PIPELINE_PID
