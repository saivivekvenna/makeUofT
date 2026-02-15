#!/usr/bin/env python3
import argparse
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
from dotenv import load_dotenv
from openai import OpenAI

try:
    import numpy as np
except Exception:
    np = None

try:
    import sounddevice as sd
except Exception:
    sd = None

HAZARD_HINTS = {
    "car": "traffic danger nearby",
    "truck": "large moving vehicle risk",
    "bus": "heavy traffic flow",
    "motorcycle": "fast close-by movement",
    "bicycle": "moving object collision risk",
    "knife": "sharp object injury risk",
    "scissors": "sharp object injury risk",
    "fire hydrant": "street-side/roadway indicator",
    "stairs": "fall risk if moving quickly",
}

RESOURCE_HINTS = {
    "person": "possible source of help",
    "cell phone": "communication/emergency call tool",
    "bottle": "possible hydration source",
    "backpack": "portable supplies likely",
    "chair": "rest/support option",
    "bench": "rest/support option",
    "dining table": "stable work/supply surface",
    "book": "information source",
    "laptop": "information/communication tool",
    "tv": "information source",
    "clock": "time tracking",
}

MEMORY_PREFIX = "MEMORY_UPDATE_JSON:"


class ContextMemory:
    def __init__(self, seed_context: str, bootstrap_rounds: int):
        self.seed_context = seed_context
        self.bootstrap_rounds = max(1, bootstrap_rounds)
        self.analysis_count = 0
        self.environment = seed_context
        self.priority = "Maintain safety and situational awareness."
        self.risks = []
        self.resources = []
        self.confidence = 0.0

    def phase(self) -> str:
        if self.analysis_count < self.bootstrap_rounds:
            return "bootstrap"
        return "steady"

    def remaining_bootstrap(self) -> int:
        return max(0, self.bootstrap_rounds - self.analysis_count)

    def apply_update(self, update):
        if not isinstance(update, dict):
            return

        environment = update.get("environment")
        if isinstance(environment, str) and environment.strip():
            self.environment = environment.strip()[:220]

        priority = update.get("priority")
        if isinstance(priority, str) and priority.strip():
            self.priority = priority.strip()[:220]

        risks = update.get("risks")
        if isinstance(risks, list):
            self.risks = [str(x).strip()[:120] for x in risks if str(x).strip()][:5]

        resources = update.get("resources")
        if isinstance(resources, list):
            self.resources = [str(x).strip()[:120] for x in resources if str(x).strip()][:5]

        confidence = update.get("confidence")
        if isinstance(confidence, (int, float)):
            self.confidence = max(0.0, min(1.0, float(confidence)))

    def tick(self):
        self.analysis_count += 1

    def render(self) -> str:
        risks = ", ".join(self.risks) if self.risks else "none established"
        resources = ", ".join(self.resources) if self.resources else "none established"
        return (
            f"Environment memory: {self.environment}\n"
            f"Priority memory: {self.priority}\n"
            f"Known risks: {risks}\n"
            f"Known resources: {resources}\n"
            f"Memory confidence: {self.confidence:.2f}"
        )


def route_audio_output(device_name: str) -> bool:
    if not device_name:
        return True
    if platform.system() != "Darwin":
        print("Audio device routing is only supported on macOS.")
        return False
    switch_bin = shutil.which("SwitchAudioSource")
    if not switch_bin:
        print(
            "SwitchAudioSource is not installed. Install with: brew install switchaudio-osx "
            "or set output manually in macOS Sound settings."
        )
        return False
    try:
        subprocess.run([switch_bin, "-s", device_name, "-t", "output"], check=True)
        print(f"Audio output routed to: {device_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"Could not route audio output to: {device_name}")
        return False


def list_audio_output_devices():
    if platform.system() != "Darwin":
        return []
    switch_bin = shutil.which("SwitchAudioSource")
    if not switch_bin:
        return []
    try:
        proc = subprocess.run(
            [switch_bin, "-a", "-t", "output"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


class SpeechManager:
    def __init__(self, enabled: bool, voice: str, rate: int, interrupt: bool):
        self.enabled = enabled and platform.system() == "Darwin"
        self.voice = voice
        self.rate = max(120, min(340, int(rate)))
        self.interrupt = interrupt
        self._queue = queue.Queue()
        self._thread = None
        self._current_proc = None
        self._lock = threading.Lock()
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        elif enabled:
            print("Speech output requested, but this host is not macOS.")

    def speak(self, text: str):
        if not self.enabled:
            return
        spoken = " ".join((text or "").strip().split())
        if not spoken:
            return
        if self.interrupt:
            self._clear_pending()
            with self._lock:
                if self._current_proc and self._current_proc.poll() is None:
                    self._current_proc.terminate()
        self._queue.put(spoken[:500])

    def stop(self):
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()

    def interrupt_now(self):
        if not self.enabled:
            return
        self._clear_pending()
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()

    def _clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            cmd = ["say", "-v", self.voice, "-r", str(self.rate), text]
            try:
                proc = subprocess.Popen(cmd)
                with self._lock:
                    self._current_proc = proc
                proc.wait()
            except FileNotFoundError:
                print("macOS 'say' command not available.")
                return
            finally:
                with self._lock:
                    self._current_proc = None


def list_mic_devices():
    if sd is None:
        return []
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0:
            devices.append((idx, str(dev.get("name", f"Device {idx}"))))
    return devices


def parse_mic_device(value: str):
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return int(raw)
    return raw


class PushToTalkRecorder:
    def __init__(self, sample_rate: int, channels: int, mic_device: str = ""):
        self.sample_rate = max(8000, int(sample_rate))
        self.channels = max(1, int(channels))
        self.mic_device = mic_device if mic_device else None
        self.stream = None
        self.lock = threading.Lock()
        self.chunks = []
        self.started_at = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            return
        with self.lock:
            self.chunks.append(indata.copy())

    def start(self):
        if sd is None or np is None:
            raise RuntimeError(
                "Push-to-talk requires sounddevice and numpy. Run: pip install -r requirements.txt"
            )
        if self.stream is not None:
            return
        with self.lock:
            self.chunks = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
            device=self.mic_device,
        )
        self.stream.start()
        self.started_at = time.time()

    def stop_to_wav(self):
        if self.stream is None:
            return None, 0.0
        self.stream.stop()
        self.stream.close()
        self.stream = None
        with self.lock:
            chunks = list(self.chunks)
            self.chunks = []
        if not chunks:
            return None, 0.0
        audio = np.concatenate(chunks, axis=0)
        if audio.size == 0:
            return None, 0.0

        duration_s = float(audio.shape[0]) / float(max(1, self.sample_rate))
        out = tempfile.NamedTemporaryFile(prefix="ptt_", suffix=".wav", delete=False)
        out_path = out.name
        out.close()
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return out_path, duration_s


def transcribe_with_openai(client: OpenAI, wav_path: str, model: str, language: str = "") -> str:
    if not wav_path:
        return ""
    with open(wav_path, "rb") as f:
        kwargs = {"model": model, "file": f}
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
    text = getattr(result, "text", "") or ""
    return str(text).strip()


class YoloHelper:
    def __init__(self, model_name: str):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.names = self.model.names

    def detect(self, frame, min_conf: float):
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        detections = []
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < min_conf:
                    continue
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, w * h))
                detections.append(
                    {
                        "name": self.names.get(cls_id, str(cls_id)),
                        "conf": conf,
                        "area": area_ratio,
                        "cx": ((x1 + x2) * 0.5) / float(max(1, w)),
                        "cy": ((y1 + y2) * 0.5) / float(max(1, h)),
                    }
                )
        return detections, results[0].plot()


class RollingSceneState:
    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self.frames = deque()

    def add(self, timestamp: float, detections):
        self.frames.append((timestamp, detections))
        self._purge(timestamp)

    def _purge(self, now: float):
        cutoff = now - self.window_seconds
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()

    def summarize(self, top_k: int):
        total_frames = len(self.frames)
        if total_frames == 0:
            return []

        per_class = {}
        for _, detections in self.frames:
            frame_seen = set()
            for det in detections:
                name = det["name"]
                stats = per_class.setdefault(
                    name,
                    {
                        "frames_seen": 0,
                        "instances": 0,
                        "conf_sum": 0.0,
                        "conf_max": 0.0,
                        "area_sum": 0.0,
                        "cx_sum": 0.0,
                        "cy_sum": 0.0,
                    },
                )
                stats["instances"] += 1
                stats["conf_sum"] += det["conf"]
                stats["conf_max"] = max(stats["conf_max"], det["conf"])
                stats["area_sum"] += det["area"]
                stats["cx_sum"] += det["cx"]
                stats["cy_sum"] += det["cy"]
                if name not in frame_seen:
                    frame_seen.add(name)
                    stats["frames_seen"] += 1

        summary = []
        for name, stats in per_class.items():
            presence = stats["frames_seen"] / float(total_frames)
            avg_conf = stats["conf_sum"] / float(max(1, stats["instances"]))
            avg_area = stats["area_sum"] / float(max(1, stats["instances"]))
            avg_instances = stats["instances"] / float(max(1, stats["frames_seen"]))
            avg_x = stats["cx_sum"] / float(max(1, stats["instances"]))
            avg_y = stats["cy_sum"] / float(max(1, stats["instances"]))
            score = (presence * 0.6) + (avg_conf * 0.3) + (min(1.0, avg_area * 4.0) * 0.1)
            summary.append(
                {
                    "name": name,
                    "presence": presence,
                    "avg_conf": avg_conf,
                    "max_conf": stats["conf_max"],
                    "avg_area": avg_area,
                    "avg_instances": avg_instances,
                    "position": bucket_position(avg_x, avg_y),
                    "score": score,
                }
            )
        summary.sort(key=lambda x: x["score"], reverse=True)
        return summary[:top_k]


def bucket_position(x: float, y: float) -> str:
    if x < 0.33:
        horizontal = "left"
    elif x > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    if y < 0.33:
        vertical = "upper"
    elif y > 0.66:
        vertical = "lower"
    else:
        vertical = "middle"
    return f"{horizontal}-{vertical}"


def format_tags(tag_summary) -> str:
    if not tag_summary:
        return "No stable objects detected in the current window."
    lines = []
    for tag in tag_summary:
        lines.append(
            f"- {tag['name']} | presence {tag['presence']:.0%} | avg_conf {tag['avg_conf']:.2f} | "
            f"size {tag['avg_area']:.1%} | position {tag['position']} | density {tag['avg_instances']:.1f}/frame"
        )
    return "\n".join(lines)


def derive_survival_cues(tag_summary) -> str:
    if not tag_summary:
        return "No cue objects detected yet."

    hazards = []
    resources = []
    for tag in tag_summary:
        name = tag["name"]
        if name in HAZARD_HINTS:
            hazards.append(f"- {name}: {HAZARD_HINTS[name]}")
        if name in RESOURCE_HINTS:
            resources.append(f"- {name}: {RESOURCE_HINTS[name]}")

    if not hazards:
        hazards_text = "No obvious hazard-tag matches."
    else:
        hazards_text = "\n".join(hazards)

    if not resources:
        resources_text = "No obvious resource-tag matches."
    else:
        resources_text = "\n".join(resources)

    return f"Hazard cues:\n{hazards_text}\nResource cues:\n{resources_text}"


def build_system_prompt(
    user_context: str,
    window_seconds: float,
    memory_context: str,
    phase: str,
    remaining_bootstrap: int,
) -> str:
    phase_line = (
        f"Phase: bootstrap ({remaining_bootstrap} bootstrap analyses remaining after this one)."
        if phase == "bootstrap"
        else "Phase: steady-state memory usage."
    )
    return (
        "You are a real-time survival assistant speaking directly to the user.\n"
        "Address the user as 'Twin'.\n"
        "Use a natural, supportive Gen Z vibe: short, conversational, practical, and not cringe.\n"
        "No roleplay, no emojis, no excessive slang.\n"
        "Goal: keep Twin safe using environment understanding.\n"
        "You receive YOLO-derived object tags aggregated across time, not raw frames.\n"
        f"Window size: last {window_seconds:.1f} seconds.\n"
        f"User context: {user_context}\n"
        f"{phase_line}\n"
        "Use the existing context memory unless current evidence strongly contradicts it.\n"
        "Context memory provided by system:\n"
        f"{memory_context}\n"
        "If a user voice request is present, answer that request directly first, then ground it in environment context.\n"
        "Infer the likely environment and prioritize immediate survival-oriented advice.\n"
        "Prefer de-escalation, distance from hazards, stable shelter/exit options, hydration, and communication.\n"
        "Be explicit about uncertainty and avoid overclaiming.\n"
        "Output style rules:\n"
        "- Start with 'Twin,'.\n"
        "- 2-3 short sentences total in natural speech.\n"
        "- Directly answer Twin's spoken question/request first if given.\n"
        "- Include the top risk and best next action within those 2-3 sentences.\n"
        "- Do not use section headers or bullet points.\n"
        "Final line requirement: output one single-line JSON memory update using this exact prefix:\n"
        'MEMORY_UPDATE_JSON: {"environment":"...","priority":"...","risks":["..."],"resources":["..."],"confidence":0.0}'
    )


def extract_memory_update(analysis_text: str):
    for line in reversed(analysis_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(MEMORY_PREFIX):
            raw_json = stripped[len(MEMORY_PREFIX) :].strip()
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                return None
    return None


def strip_memory_line(analysis_text: str) -> str:
    cleaned = []
    for line in analysis_text.splitlines():
        if line.strip().startswith(MEMORY_PREFIX):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def clamp_to_max_sentences(text: str, max_sentences: int = 3) -> str:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return ""
    limit = max(1, int(max_sentences))
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", normalized) if s.strip()]
    if not sentences:
        return normalized
    return " ".join(sentences[:limit]).strip()


def analyze_with_gpt(
    client: OpenAI,
    model: str,
    system_prompt: str,
    tags_text: str,
    survival_cues: str,
    sensor_text: str,
    max_tokens: int,
    user_voice_prompt: str = "",
) -> str:
    voice_block = ""
    if user_voice_prompt.strip():
        voice_block = (
            "User voice request (answer this directly first, then use scene context):\n"
            f"{user_voice_prompt.strip()}\n\n"
        )
    user_prompt = (
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"{voice_block}"
        "Aggregated YOLO tags:\n"
        f"{tags_text}\n\n"
        "External sensor state:\n"
        f"{sensor_text}\n\n"
        "Derived cues:\n"
        f"{survival_cues}\n"
    )
    response = client.responses.create(
        model=model,
        max_output_tokens=max_tokens,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    return response.output_text.strip()


def push_gpt_update(push_url: str, text: str, phase: str, memory_conf: float):
    if not push_url:
        return

    safe_text = (text or "").strip()
    if not safe_text:
        safe_text = "No visible analysis text returned (memory update only)."

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "memory_conf": float(memory_conf),
        "text": safe_text,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        push_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(req, timeout=2.0):
            return
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        print(f"Warning: failed to push GPT update to {push_url}: {exc}")


def push_annotated_frame(push_url: str, frame, jpeg_quality: int):
    if not push_url:
        return

    quality = max(30, min(95, int(jpeg_quality)))
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return

    req = urlrequest.Request(
        push_url,
        data=enc.tobytes(),
        method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urlrequest.urlopen(req, timeout=1.0):
            return
    except (urlerror.URLError, TimeoutError, ValueError):
        return


def fetch_gpt_paused(control_url: str, fallback: bool) -> bool:
    if not control_url:
        return fallback

    req = urlrequest.Request(control_url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=1.2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return fallback

    return bool(payload.get("paused", False))


def fetch_sensor_state(sensor_url: str, fallback: dict):
    if not sensor_url:
        return fallback
    req = urlrequest.Request(sensor_url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=1.2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return fallback

    sensor = payload.get("sensor") if isinstance(payload, dict) else None
    if isinstance(sensor, dict):
        return sensor
    if isinstance(payload, dict):
        return payload
    return fallback


def format_sensor_text(sensor_state: dict, include_movement: bool) -> str:
    if not isinstance(sensor_state, dict) or not sensor_state:
        return "No sensor data available."
    heart_rate = sensor_state.get("heart_rate")
    speak_button = int(sensor_state.get("speak_button") or 0)
    updated_at = sensor_state.get("updated_at") or "unknown"
    connected = bool(sensor_state.get("connected", True))
    base = (
        f"connected={connected}, updated_at={updated_at}, heart_rate={heart_rate}, "
        f"speak_button={speak_button}"
    )
    if not include_movement:
        return base
    left = int(sensor_state.get("left") or 0)
    right = int(sensor_state.get("right") or 0)
    behind = int(sensor_state.get("behind") or 0)
    return f"{base}, left={left}, right={right}, behind={behind}"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + GPT survival assistant")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--list-cameras", action="store_true", help="List usable camera indexes, then exit")
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List macOS audio output devices (requires SwitchAudioSource), then exit",
    )
    parser.add_argument("--list-mics", action="store_true", help="List available microphone input devices")
    parser.add_argument(
        "--source-url",
        default="",
        help="Remote stream URL (e.g. http://pi5.local:8080/video or rtsp://...)",
    )
    parser.add_argument(
        "--phone-url",
        default="",
        help="Deprecated alias for --source-url",
    )
    parser.add_argument("--interval", type=float, default=6.0, help="Seconds between GPT analyses")
    parser.add_argument(
        "--analysis-window",
        type=float,
        default=4.0,
        help="Seconds of YOLO history to aggregate into robust tags",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for survival reasoning")
    parser.add_argument("--max-tokens", type=int, default=260, help="Max output tokens per analysis")
    parser.add_argument("--min-conf", type=float, default=0.35, help="Min YOLO confidence to keep")
    parser.add_argument("--top-k", type=int, default=10, help="Max number of aggregated tags sent to GPT")
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=2,
        help="Number of early analyses used to establish persistent context memory",
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="Ultralytics YOLO model name/path")
    parser.add_argument(
        "--context",
        default=os.getenv("VISION_CONTEXT", "General desktop/room environment"),
        help="Current user/environment context for system prompt",
    )
    parser.add_argument(
        "--show-raw-stream",
        action="store_true",
        help="Display a second window with the raw camera stream (useful with Pi source URL)",
    )
    parser.add_argument(
        "--gpt-push-url",
        default="",
        help="Optional endpoint to receive GPT text updates (e.g. http://<PI_IP>:8080/gpt-log)",
    )
    parser.add_argument(
        "--gpt-control-url",
        default="",
        help="Optional GPT pause-control endpoint (e.g. http://<PI_IP>:8080/gpt-control)",
    )
    parser.add_argument(
        "--sensor-url",
        default="",
        help="Optional sensor endpoint (e.g. http://<PI_IP>:8080/sensor-state)",
    )
    parser.add_argument(
        "--sensor-poll-interval",
        type=float,
        default=0.05,
        help="Polling interval for sensor endpoint in seconds",
    )
    parser.add_argument(
        "--sensor-button-ptt",
        action="store_true",
        help="Use sensor speak_button field for push-to-talk trigger instead of keyboard space",
    )
    parser.add_argument(
        "--heart-rate-alert-delta",
        type=int,
        default=8,
        help="Speak an alert if heart rate changes by at least this delta",
    )
    parser.add_argument(
        "--include-movement-in-gpt",
        action="store_true",
        help="Include left/right/behind sensor bits in GPT context (off by default)",
    )
    parser.add_argument(
        "--speak-movement-alerts",
        action="store_true",
        help="Speak left/right/behind movement alerts (off by default; still logs)",
    )
    parser.add_argument(
        "--annotated-push-url",
        default="",
        help="Optional endpoint to receive YOLO-annotated JPEG frames (e.g. http://<PI_IP>:8080/annotated-frame)",
    )
    parser.add_argument(
        "--annotated-push-fps",
        type=float,
        default=6.0,
        help="Max FPS for pushing annotated frames to remote dashboard",
    )
    parser.add_argument(
        "--annotated-jpeg-quality",
        type=int,
        default=72,
        help="JPEG quality (30-95) for pushed annotated frames",
    )
    parser.add_argument("--speak-output", action="store_true", help="Speak each GPT response on this Mac")
    parser.add_argument(
        "--tts-voice",
        default="Daniel",
        help="macOS voice name used by `say` (e.g. Daniel, Samantha, Alex)",
    )
    parser.add_argument("--tts-rate", type=int, default=185, help="Speech speed words/min for macOS `say`")
    parser.add_argument(
        "--interrupt-speech",
        action="store_true",
        help="Interrupt current speech when a newer GPT response arrives",
    )
    parser.add_argument(
        "--audio-output-device",
        default="",
        help="Optional macOS audio output device name (Bluetooth speaker/headphones) for speech",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Enable microphone push-to-talk recording flow",
    )
    parser.add_argument(
        "--mic-device",
        default="",
        help="Optional microphone device name/index for push-to-talk (default is system input)",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate for push-to-talk recording",
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Microphone channel count for push-to-talk recording",
    )
    parser.add_argument(
        "--transcribe-model",
        default="gpt-4o-mini-transcribe",
        help="OpenAI transcription model for push-to-talk",
    )
    parser.add_argument(
        "--transcribe-language",
        default="en",
        help="Language hint for transcription (e.g. en, es). Empty for auto.",
    )
    parser.add_argument(
        "--hold-release-timeout",
        type=float,
        default=0.35,
        help="Seconds without SPACE events before ending push-to-talk capture",
    )
    return parser.parse_args()


def open_camera(index: int):
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 10):
    found = []
    for idx in range(max_index + 1):
        cap = open_camera(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(idx)
        cap.release()
    return found


def open_source(args):
    source_url = args.source_url or args.phone_url
    if source_url:
        cap = cv2.VideoCapture(source_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, f"url:{source_url}"

    cap = open_camera(args.camera_index)
    return cap, f"camera-index:{args.camera_index}"


def main():
    load_dotenv()
    args = parse_args()

    if args.list_audio_devices:
        devices = list_audio_output_devices()
        if devices:
            print("Available macOS output audio devices:")
            for dev in devices:
                print(f"- {dev}")
        else:
            print("No audio devices listed. Install SwitchAudioSource: brew install switchaudio-osx")
        return
    if args.list_mics:
        mics = list_mic_devices()
        if mics:
            print("Available microphone input devices:")
            for idx, name in mics:
                print(f"- {idx}: {name}")
        else:
            print("No microphone devices found or sounddevice is not installed.")
        return

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("Missing OPENAI_API_KEY. Set it in your shell or a .env file.")
        sys.exit(1)

    if args.list_cameras:
        cams = list_cameras()
        if cams:
            print("Usable camera indexes:", ", ".join(str(c) for c in cams))
        else:
            print("No usable cameras found in indexes 0-10.")
        return

    if args.audio_output_device:
        route_audio_output(args.audio_output_device)
    speech = SpeechManager(
        enabled=args.speak_output,
        voice=args.tts_voice,
        rate=args.tts_rate,
        interrupt=args.interrupt_speech,
    )
    ptt_enabled = bool(args.push_to_talk or args.sensor_button_ptt)
    ptt_recorder = None
    if ptt_enabled:
        try:
            ptt_recorder = PushToTalkRecorder(
                sample_rate=args.mic_sample_rate,
                channels=args.mic_channels,
                mic_device=parse_mic_device(args.mic_device),
            )
        except Exception as exc:
            print(f"Push-to-talk disabled: {exc}")
            ptt_enabled = False

    client = OpenAI(api_key=api_key)
    memory = ContextMemory(seed_context=args.context, bootstrap_rounds=args.bootstrap_rounds)

    try:
        yolo = YoloHelper(args.yolo_model)
    except Exception as exc:
        print(f"Failed to load YOLO model ({args.yolo_model}): {exc}")
        sys.exit(1)

    cap, source_desc = open_source(args)
    if cap is None or not cap.isOpened():
        print(f"Could not open camera source ({source_desc}).")
        sys.exit(1)

    if ptt_enabled and args.sensor_button_ptt:
        print(
            "Live camera started. Use Arduino speak_button to talk (press=record, release=send), "
            "press 'f' to force analysis, 'q' to quit."
        )
        print("Auto GPT interval analysis is disabled while sensor-button push-to-talk is enabled.")
        print("Voice workflow is serialized: one request is fully processed before the next starts.")
    elif ptt_enabled:
        print(
            "Live camera started. Hold SPACE to talk, release SPACE to send voice prompt, "
            "press 'f' to force analysis, 'q' to quit."
        )
        print("Auto GPT interval analysis is disabled while push-to-talk is enabled.")
        print("Voice workflow is serialized: one request is fully processed before the next starts.")
    else:
        print(f"Live camera started from {source_desc}. Press 'q' to quit, 'space' to force analysis.")
    print("Video window and GPT analysis output are rendered on this machine.")
    if args.speak_output:
        print(f"Speech output enabled (voice={args.tts_voice}, rate={args.tts_rate}).")
    scene_state = RollingSceneState(window_seconds=args.analysis_window)
    last_analysis = 0.0
    last_text = "Waiting for first analysis..."
    last_annotated_push = 0.0
    last_control_poll = 0.0
    remote_gpt_paused = False
    ptt_recording = False
    last_space_signal = 0.0
    ptt_processing = False
    ptt_space_latched = False
    last_sensor_poll = 0.0
    sensor_state = {}
    prev_sensor_state = {}
    prev_button_state = 0

    control_url = args.gpt_control_url
    sensor_url = args.sensor_url
    annotated_push_url = args.annotated_push_url
    if args.gpt_push_url and not control_url and args.gpt_push_url.endswith("/gpt-log"):
        control_url = args.gpt_push_url[: -len("/gpt-log")] + "/gpt-control"
    if args.gpt_push_url and not sensor_url and args.gpt_push_url.endswith("/gpt-log"):
        sensor_url = args.gpt_push_url[: -len("/gpt-log")] + "/sensor-state"
    if args.gpt_push_url and not annotated_push_url and args.gpt_push_url.endswith("/gpt-log"):
        annotated_push_url = args.gpt_push_url[: -len("/gpt-log")] + "/annotated-frame"

    if control_url:
        print(f"Remote GPT control enabled: {control_url}")
    if sensor_url:
        print(f"Remote sensor feed enabled: {sensor_url}")
    if annotated_push_url:
        print(f"Annotated frame push enabled: {annotated_push_url}")

    def run_analysis(user_voice_prompt: str = ""):
        current_phase = memory.phase()
        tags = scene_state.summarize(top_k=args.top_k)
        tags_text = format_tags(tags)
        survival_cues = derive_survival_cues(tags)
        sensor_text = format_sensor_text(
            sensor_state=sensor_state,
            include_movement=args.include_movement_in_gpt,
        )
        system_prompt = build_system_prompt(
            user_context=args.context,
            window_seconds=args.analysis_window,
            memory_context=memory.render(),
            phase=current_phase,
            remaining_bootstrap=memory.remaining_bootstrap(),
        )
        analysis = analyze_with_gpt(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            tags_text=tags_text,
            survival_cues=survival_cues,
            sensor_text=sensor_text,
            max_tokens=args.max_tokens,
            user_voice_prompt=user_voice_prompt,
        )
        memory_update = extract_memory_update(analysis)
        memory.apply_update(memory_update)
        memory.tick()
        visible_analysis = strip_memory_line(analysis)
        if not visible_analysis:
            visible_analysis = "No visible analysis text returned (memory update only)."
        visible_analysis = clamp_to_max_sentences(visible_analysis, max_sentences=3)
        stamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"\n[{stamp}] GPT Context Analysis (phase={current_phase}, memory_conf={memory.confidence:.2f})\n"
            f"{visible_analysis}\n"
        )
        speech.speak(visible_analysis)
        push_gpt_update(
            push_url=args.gpt_push_url,
            text=visible_analysis,
            phase=current_phase,
            memory_conf=memory.confidence,
        )
        return visible_analysis

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Remote streams can drop momentarily; attempt reconnect before failing hard.
                print("Frame read failed; attempting reconnect...")
                cap.release()
                time.sleep(1.0)
                cap, source_desc = open_source(args)
                if cap is None or not cap.isOpened():
                    print(f"Reconnect failed for source ({source_desc}).")
                    break
                ok, frame = cap.read()
                if not ok:
                    print("Reconnect succeeded but still no frames available.")
                    break

            detections, annotated = yolo.detect(frame, min_conf=args.min_conf)
            scene_state.add(time.time(), detections)
            display_frame = annotated

            now = time.time()
            if control_url and now - last_control_poll >= 1.0:
                new_paused = fetch_gpt_paused(control_url, fallback=remote_gpt_paused)
                if new_paused != remote_gpt_paused:
                    state = "paused" if new_paused else "resumed"
                    print(f"Remote GPT has been {state} from dashboard.")
                remote_gpt_paused = new_paused
                last_control_poll = now

            if sensor_url and (now - last_sensor_poll) >= max(0.05, args.sensor_poll_interval):
                new_sensor = fetch_sensor_state(sensor_url, fallback=sensor_state)
                if isinstance(new_sensor, dict):
                    prev_sensor_state = dict(sensor_state)
                    sensor_state = dict(new_sensor)

                    prev_hr = prev_sensor_state.get("heart_rate")
                    curr_hr = sensor_state.get("heart_rate")
                    if (
                        isinstance(prev_hr, (int, float))
                        and isinstance(curr_hr, (int, float))
                        and abs(int(curr_hr) - int(prev_hr)) >= max(1, args.heart_rate_alert_delta)
                    ):
                        alert = f"Twin, heart rate changed from {int(prev_hr)} to {int(curr_hr)}."
                        print(alert)
                        speech.speak(alert)

                    for direction_key, direction_name in (
                        ("left", "left"),
                        ("right", "right"),
                        ("behind", "behind"),
                    ):
                        prev_val = int(prev_sensor_state.get(direction_key) or 0)
                        curr_val = int(sensor_state.get(direction_key) or 0)
                        if prev_val == 0 and curr_val == 1:
                            alert = f"[Sensor] movement detected on {direction_name}"
                            print(alert)
                            if args.speak_movement_alerts:
                                speech.speak(f"Twin, movement detected on your {direction_name}.")

                last_sensor_poll = now

            push_interval = 1.0 / float(max(1.0, args.annotated_push_fps))
            if annotated_push_url and (now - last_annotated_push) >= push_interval:
                push_annotated_frame(
                    push_url=annotated_push_url,
                    frame=display_frame,
                    jpeg_quality=args.annotated_jpeg_quality,
                )
                last_annotated_push = now

            current_button_state = 0
            if args.sensor_button_ptt:
                current_button_state = 1 if int(sensor_state.get("speak_button") or 0) > 0 else 0
                if current_button_state == 1 and prev_button_state == 0:
                    print("[SensorPTT] button pressed (speak_button=1)")
                elif current_button_state == 0 and prev_button_state == 1:
                    print("[SensorPTT] button released (speak_button=0)")
                if (
                    current_button_state == 1
                    and prev_button_state == 0
                    and ptt_enabled
                    and (not ptt_processing)
                    and (not ptt_recording)
                ):
                    try:
                        speech.interrupt_now()
                        ptt_recorder.start()
                        ptt_recording = True
                        print("[SensorPTT] recording started")
                        last_text = "Listening from button... release button to send your request."
                    except Exception as exc:
                        ptt_recording = False
                        last_text = f"Mic capture failed: {exc}"
                        print(last_text)
                        speech.speak(last_text)

            should_finalize_ptt = False
            if ptt_enabled and ptt_recording and (not ptt_processing):
                if args.sensor_button_ptt:
                    should_finalize_ptt = current_button_state == 0
                else:
                    should_finalize_ptt = (now - last_space_signal) >= args.hold_release_timeout

            if (
                ptt_enabled
                and should_finalize_ptt
                and (not ptt_processing)
            ):
                wav_path = None
                ptt_processing = True
                if args.sensor_button_ptt:
                    print("[SensorPTT] recording stopped; processing voice request")
                try:
                    wav_path, duration_s = ptt_recorder.stop_to_wav()
                    ptt_recording = False
                    if not wav_path or duration_s < 0.2:
                        last_text = "Twin, I barely caught audio. Hold SPACE and speak a bit longer."
                        speech.speak(last_text)
                    else:
                        transcript = transcribe_with_openai(
                            client=client,
                            wav_path=wav_path,
                            model=args.transcribe_model,
                            language=args.transcribe_language,
                        )
                        if not transcript:
                            last_text = "Twin, I couldn't understand that audio. Try again."
                            speech.speak(last_text)
                        elif remote_gpt_paused:
                            last_text = f"Twin said: {transcript}. GPT is paused, resume to process it."
                            print(f"[SensorPTT] transcript: {transcript}")
                            speech.speak(last_text)
                        else:
                            print(f"[SensorPTT] transcript: {transcript}")
                            print("[SensorPTT] sending transcript to GPT")
                            last_text = run_analysis(user_voice_prompt=transcript)
                            last_analysis = time.time()
                            print("[SensorPTT] GPT response complete")
                except Exception as exc:
                    ptt_recording = False
                    last_text = f"Voice capture/transcription failed: {exc}"
                    print(last_text)
                    speech.speak(last_text)
                finally:
                    ptt_processing = False
                    if wav_path and os.path.exists(wav_path):
                        os.remove(wav_path)
            if args.sensor_button_ptt:
                prev_button_state = current_button_state

            if (not ptt_enabled) and (now - last_analysis >= args.interval):
                if remote_gpt_paused:
                    last_text = "Twin, GPT is paused from the dashboard right now."
                    speech.speak(last_text)
                else:
                    try:
                        last_text = run_analysis()
                    except Exception as exc:
                        last_text = f"GPT call failed: {exc}"
                        print(last_text)
                        speech.speak(last_text)
                last_analysis = now

            preview_text = last_text.splitlines()[0][:100] if last_text else ""
            cv2.putText(
                display_frame,
                preview_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                (
                    f"detections:{len(detections)} window:{args.analysis_window:.1f}s "
                    f"phase:{memory.phase()} mem:{memory.confidence:.2f} "
                    f"gpt:{'paused' if remote_gpt_paused else 'live'}"
                ),
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if sensor_state:
                sensor_line = (
                    f"HR:{sensor_state.get('heart_rate')} "
                    f"L:{int(sensor_state.get('left') or 0)} "
                    f"R:{int(sensor_state.get('right') or 0)} "
                    f"B:{int(sensor_state.get('behind') or 0)} "
                    f"BTN:{int(sensor_state.get('speak_button') or 0)}"
                )
                cv2.putText(
                    display_frame,
                    sensor_line,
                    (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if ptt_recording:
                cv2.putText(
                    display_frame,
                    (
                        "LISTENING... press and hold your hardware button"
                        if args.sensor_button_ptt
                        else "LISTENING... hold SPACE and talk"
                    ),
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif ptt_processing:
                cv2.putText(
                    display_frame,
                    "PROCESSING VOICE REQUEST...",
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Vision Assistant (YOLO + GPT Context)", display_frame)
            if args.show_raw_stream:
                cv2.imshow("Raw Camera Stream", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ptt_enabled and (not args.sensor_button_ptt):
                if key == 32:
                    last_space_signal = time.time()
                    if not ptt_space_latched:
                        ptt_space_latched = True
                        if ptt_processing:
                            last_text = "Twin, still processing your last voice prompt. One sec."
                        elif not ptt_recording:
                            try:
                                speech.interrupt_now()
                                ptt_recorder.start()
                                ptt_recording = True
                                last_text = "Listening... release SPACE to send your request."
                            except Exception as exc:
                                ptt_recording = False
                                last_text = f"Mic capture failed: {exc}"
                                print(last_text)
                                speech.speak(last_text)
                else:
                    ptt_space_latched = False

            if key == ord("f") or (not ptt_enabled and key == 32):
                if ptt_processing:
                    last_text = "Twin, still processing your last voice prompt. One sec."
                    continue
                if remote_gpt_paused:
                    last_text = "Twin, GPT is paused from dashboard. Resume first."
                    speech.speak(last_text)
                else:
                    try:
                        last_text = run_analysis()
                        last_analysis = time.time()
                    except Exception as exc:
                        last_text = f"GPT call failed: {exc}"
                        print(last_text)
                        speech.speak(last_text)
    finally:
        if ptt_enabled and ptt_recording and ptt_recorder is not None:
            try:
                ptt_recorder.stop_to_wav()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()


if __name__ == "__main__":
    main()
