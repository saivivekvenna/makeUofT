#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"

if [[ ! -f "${PROJECT_DIR}/requirements-pi.txt" ]]; then
  echo "requirements-pi.txt not found in ${PROJECT_DIR}" >&2
  exit 1
fi

python3 -m venv "${PROJECT_DIR}/.venv"
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements-pi.txt"

echo "Pi 5 dependencies installed in ${PROJECT_DIR}/.venv"
echo "Start stream with:"
echo "  ${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/pi5_camera_stream.py --host 0.0.0.0 --port 8080"
