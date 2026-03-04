#!/usr/bin/env bash
# Install systemd service for Pi Camera Pipeline (auto-start on boot).
# Usage: sudo ./scripts/install_service.sh
# Then: sudo systemctl enable picamera && sudo systemctl start picamera

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="picamera"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0"
  exit 1
fi

# Use the actual project path (e.g. /home/anuj/camera-pipeline or current user)
WORKING_DIR="${PROJECT_ROOT}"
PYTHON="python3"

cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=Pi Camera ISP Pipeline (headless dashboard)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${WORKING_DIR}
ExecStartPre=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
ExecStart=${PYTHON} -m app.main --headless
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Created ${SERVICE_FILE}"
systemctl daemon-reload
echo "Run: sudo systemctl enable ${SERVICE_NAME}  # enable on boot"
echo "Run: sudo systemctl start ${SERVICE_NAME}  # start now"
echo "Run: sudo systemctl status ${SERVICE_NAME} # check status"
