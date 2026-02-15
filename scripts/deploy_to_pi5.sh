#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pi-host-or-ip> [pi-user] [remote-dir]" >&2
  exit 1
fi

PI_HOST="$1"
PI_USER="${2:-pi}"
REMOTE_DIR="${3:-/home/${PI_USER}/makeUofT}"

echo "Syncing project to ${PI_USER}@${PI_HOST}:${REMOTE_DIR}"
rsync -avz \
  --exclude '.env' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'yolov8*.pt' \
  ./ "${PI_USER}@${PI_HOST}:${REMOTE_DIR}/"

echo "Running Pi install script remotely"
ssh "${PI_USER}@${PI_HOST}" "bash '${REMOTE_DIR}/scripts/pi5_install.sh' '${REMOTE_DIR}'"

echo "Deployment complete. Start stream on Pi with:"
echo "  ssh ${PI_USER}@${PI_HOST} 'bash ${REMOTE_DIR}/scripts/pi5_run_stream.sh'"
