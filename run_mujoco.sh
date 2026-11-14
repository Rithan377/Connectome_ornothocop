#!/usr/bin/env bash
# Quick launcher for Native MuJoCo Interactive Flight Simulator
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="glfw"

echo "=========================================================="
echo " Launching Native MuJoCo Interactive Connectome Simulator"
echo " Display: $DISPLAY"
echo "=========================================================="

"$SCRIPT_DIR/myenv/bin/python" "$SCRIPT_DIR/scripts/run_mujoco_interactive.py" "$@"
