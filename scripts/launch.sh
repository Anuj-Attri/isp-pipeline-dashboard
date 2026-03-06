#!/bin/bash
# launch.sh — Edge CV Pipeline launcher
# Usage:
#   ./scripts/launch.sh                          # live camera
#   ./scripts/launch.sh --image <path>           # single image
#   ./scripts/launch.sh --video <path>           # video file
#   ./scripts/launch.sh --mode <none|detection|pose|depth|segmentation|ego_exo>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODE="live"
INPUT_PATH=""
AI_MODE="none"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) MODE="image"; INPUT_PATH="$2"; shift 2 ;;
        --video) MODE="video"; INPUT_PATH="$2"; shift 2 ;;
        --mode)  AI_MODE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--image PATH | --video PATH] [--mode AI_MODE]"
            echo "AI modes: none, detection, pose, segmentation, depth, ego_exo"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Edge CV Pipeline — Raspberry Pi 5 + IMX500"
echo "================================================"

# Lock CPU to performance mode
if sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null; then
    echo "✓ CPU locked to performance mode (2.4GHz)"
else
    echo "Could not set CPU governor (run with sudo for best performance)"
fi

PI_IP=$(hostname -I | awk '{print $1}')
echo "✓ IP: $PI_IP"

cd "$PROJECT_DIR"

case "$MODE" in
    live)
        echo "✓ Mode: Live camera (4K ISP → 1080p stream)"
        echo "✓ AI: $AI_MODE"
        echo "✓ Dashboard: https://$PI_IP:8765"
        echo ""
        python3 -m app.main --headless --ai-mode "$AI_MODE" &
        PIPELINE_PID=$!
        echo "Starting pipeline (PID: $PIPELINE_PID)..."

        for i in $(seq 1 30); do
            if curl -sk "https://$PI_IP:8765" > /dev/null 2>&1; then
                echo "Ready — open https://$PI_IP:8765 in your browser"
                echo "Press Ctrl+C to stop"
                break
            fi
            sleep 1
        done

        if [ -n "$DISPLAY" ]; then
            chromium-browser --new-window "https://$PI_IP:8765" 2>/dev/null &
        fi

        wait $PIPELINE_PID
        ;;

    image)
        if [ -z "$INPUT_PATH" ] || [ ! -f "$INPUT_PATH" ]; then
            echo "Error: image file not found: $INPUT_PATH"
            exit 1
        fi
        echo "✓ Mode: Single image — $INPUT_PATH"
        echo "✓ AI: $AI_MODE"
        echo ""
        python3 -m app.main --headless --input-image "$INPUT_PATH" --ai-mode "$AI_MODE"
        echo "Output saved to: output_$(basename "$INPUT_PATH")"
        ;;

    video)
        if [ -z "$INPUT_PATH" ] || [ ! -f "$INPUT_PATH" ]; then
            echo "Error: video file not found: $INPUT_PATH"
            exit 1
        fi
        echo "✓ Mode: Video file — $INPUT_PATH"
        echo "✓ AI: $AI_MODE"
        echo "✓ Dashboard: https://$PI_IP:8765"
        echo ""
        python3 -m app.main --headless --input-video "$INPUT_PATH" --ai-mode "$AI_MODE" &
        PIPELINE_PID=$!
        echo "Starting pipeline (PID: $PIPELINE_PID)..."

        for i in $(seq 1 30); do
            if curl -sk "https://$PI_IP:8765" > /dev/null 2>&1; then
                echo "Ready — open https://$PI_IP:8765 in your browser"
                break
            fi
            sleep 1
        done

        wait $PIPELINE_PID
        ;;
esac
