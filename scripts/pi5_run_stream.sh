#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_DEVICE="${CAMERA_DEVICE:-}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-20}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
SERIAL_BAUD="${SERIAL_BAUD:-9600}"
DISABLE_SERIAL="${DISABLE_SERIAL:-0}"
HR_DELTA_ALERT="${HR_DELTA_ALERT:-8}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/pi5_camera_stream.py"
  --host "${HOST}"
  --port "${PORT}"
  --camera-index "${CAMERA_INDEX}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --fps "${FPS}"
  --jpeg-quality "${JPEG_QUALITY}"
  --serial-port "${SERIAL_PORT}"
  --serial-baud "${SERIAL_BAUD}"
  --hr-delta-alert "${HR_DELTA_ALERT}"
)

if [[ -n "${CAMERA_DEVICE}" ]]; then
  CMD+=(--camera-device "${CAMERA_DEVICE}")
fi
if [[ "${DISABLE_SERIAL}" == "1" ]]; then
  CMD+=(--disable-serial)
fi

exec "${CMD[@]}" \
  "$@"
