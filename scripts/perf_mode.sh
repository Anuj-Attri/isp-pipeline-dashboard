#!/usr/bin/env bash
# Set CPU governor to performance for maximum FPS (Pi 5).
# Run once before starting the pipeline, or rely on systemd ExecStartPre.
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
