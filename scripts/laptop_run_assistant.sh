#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pi-host-or-ip> [extra vision_assistant args...]" >&2
  exit 1
fi

PI_HOST="$1"
shift

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
PI_PORT="${PI_PORT:-8080}"
STREAM_URL="http://${PI_HOST}:${PI_PORT}/video"
TTS_VOICE="${TTS_VOICE:-Daniel}"
TTS_RATE="${TTS_RATE:-185}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/vision_assistant.py"
  --model "${LLM_MODEL}"
  --source-url "${STREAM_URL}"
  --show-raw-stream
  --gpt-push-url "http://${PI_HOST}:${PI_PORT}/gpt-log"
  --gpt-control-url "http://${PI_HOST}:${PI_PORT}/gpt-control"
  --sensor-url "http://${PI_HOST}:${PI_PORT}/sensor-state"
  --annotated-push-url "http://${PI_HOST}:${PI_PORT}/annotated-frame"
  --speak-output
  --interrupt-speech
  --push-to-talk
  --sensor-button-ptt
  --tts-voice "${TTS_VOICE}"
  --tts-rate "${TTS_RATE}"
)

if [[ -n "${AUDIO_OUTPUT_DEVICE:-}" ]]; then
  CMD+=(--audio-output-device "${AUDIO_OUTPUT_DEVICE}")
fi

CMD+=("$@")

exec "${CMD[@]}"
