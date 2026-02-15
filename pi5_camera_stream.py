#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http import server
from socketserver import ThreadingMixIn

import math

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pi5 Camera + GPT Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1e293b 0%, var(--bg) 45%, #020617 100%);
      color: var(--text);
    }
    .wrap {
      display: grid;
      grid-template-columns: minmax(360px, 2fr) minmax(220px, 1fr) 180px;
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .panel {
      background: #0b1220;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      color: var(--muted);
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--accent);
      font-weight: 600;
    }
    .controls {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
    }
    .btn {
      border: 1px solid #1e40af;
      background: #1d4ed8;
      color: #e5edff;
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 12px;
      cursor: pointer;
    }
    .btn.paused {
      border-color: #92400e;
      background: #b45309;
      color: #fff7ed;
    }
    .video-box {
      padding: 10px;
      background: #020617;
    }
    img.stream {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      border: 1px solid var(--line);
    }
    .feed {
      padding: 10px;
      height: calc(100vh - 85px);
      overflow: auto;
    }
    .entry {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #0b1220;
      margin-bottom: 10px;
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .text {
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 13px;
    }
    .empty {
      color: var(--muted);
      font-size: 13px;
      border: 1px dashed var(--line);
      border-radius: 10px;
      padding: 12px;
    }
    .compass-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 14px 8px;
      gap: 8px;
    }
    .compass-box canvas {
      border-radius: 50%;
      border: 1px solid var(--line);
    }
    .compass-label {
      font-size: 22px;
      font-weight: 700;
      color: var(--accent);
    }
    .compass-sub {
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
      .feed { height: auto; max-height: 45vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <div class="head">
        <span>YOLO Stream</span>
        <span class="status"><span class="dot"></span>LIVE</span>
      </div>
      <div class="video-box">
        <img class="stream" src="/annotated-video" alt="YOLO camera stream" />
      </div>
    </section>
    <section class="panel">
      <div class="head">
        <span>GPT Output</span>
        <div class="controls">
          <span id="gptMode" class="meta">GPT live</span>
          <button id="gptToggle" class="btn" type="button">Pause GPT</button>
        </div>
      </div>
      <div id="feed" class="feed">
        <div class="empty">No GPT output received yet. Start the laptop assistant with `--gpt-push-url`.</div>
      </div>
    </section>
    <section class="panel">
      <div class="head">
        <span>Compass</span>
        <span class="status"><span class="dot"></span>TRACKING</span>
      </div>
      <div class="compass-box">
        <canvas id="compassCanvas" width="140" height="140"></canvas>
        <span id="headingValue" class="compass-label">0.0\u00b0</span>
        <span id="servoValue" class="compass-sub">servo 90\u00b0</span>
      </div>
    </section>
  </div>
  <script>
    const feedEl = document.getElementById("feed");
    const gptModeEl = document.getElementById("gptMode");
    const gptToggleEl = document.getElementById("gptToggle");
    let gptPaused = false;

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function render(entries) {
      if (!entries.length) {
        feedEl.innerHTML = '<div class="empty">No GPT output received yet. Start the laptop assistant with `--gpt-push-url`.</div>';
        return;
      }

      const html = entries
        .slice()
        .reverse()
        .map((entry) => {
          const phase = entry.phase ? ` | phase=${entry.phase}` : "";
          const mem = Number.isFinite(entry.memory_conf) ? ` | mem=${entry.memory_conf.toFixed(2)}` : "";
          const meta = `${entry.time || "unknown"}${phase}${mem}`;
          return `
            <div class="entry">
              <div class="meta">${escapeHtml(meta)}</div>
              <div class="text">${escapeHtml(entry.text || "")}</div>
            </div>
          `;
        })
        .join("");
      feedEl.innerHTML = html;
    }

    function setPausedState(paused) {
      gptPaused = !!paused;
      if (gptPaused) {
        gptModeEl.textContent = "GPT paused";
        gptToggleEl.textContent = "Resume GPT";
        gptToggleEl.classList.add("paused");
      } else {
        gptModeEl.textContent = "GPT live";
        gptToggleEl.textContent = "Pause GPT";
        gptToggleEl.classList.remove("paused");
      }
    }

    async function pollControl() {
      const resp = await fetch("/gpt-control", { cache: "no-store" });
      const data = await resp.json();
      setPausedState(!!data.paused);
    }

    async function poll() {
      try {
        const [feedResp, controlResp] = await Promise.all([
          fetch("/gpt-feed", { cache: "no-store" }),
          fetch("/gpt-control", { cache: "no-store" }),
        ]);
        const feedData = await feedResp.json();
        const controlData = await controlResp.json();
        render(feedData.entries || []);
        setPausedState(!!controlData.paused);
      } catch (err) {
        gptModeEl.textContent = "Feed offline";
      }
    }

    gptToggleEl.addEventListener("click", async () => {
      try {
        const resp = await fetch("/gpt-control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paused: !gptPaused }),
        });
        const data = await resp.json();
        setPausedState(!!data.paused);
      } catch (err) {
        gptModeEl.textContent = "Control failed";
      }
    });

    pollControl();
    poll();
    setInterval(poll, 1500);

    /* --- Compass widget --- */
    const compassCanvas = document.getElementById("compassCanvas");
    const headingEl = document.getElementById("headingValue");
    const servoEl = document.getElementById("servoValue");
    function drawCompass(angleDeg) {
      const ctx = compassCanvas.getContext("2d");
      const W = compassCanvas.width, H = compassCanvas.height;
      const cx = W / 2, cy = H / 2, r = Math.min(cx, cy) - 6;
      ctx.clearRect(0, 0, W, H);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.strokeStyle = "#334155"; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, r - 6, 0, 2 * Math.PI);
      ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.stroke();
      ["N","E","S","W"].forEach((l, i) => {
        const a = (i * 90 - 90) * Math.PI / 180;
        ctx.fillStyle = l==="N" ? "#22c55e" : "#94a3b8";
        ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(l, cx + (r - 18) * Math.cos(a), cy + (r - 18) * Math.sin(a));
      });
      const rad = (-angleDeg + 90) * Math.PI / 180;
      ctx.beginPath();
      ctx.moveTo(cx + r * 0.6 * Math.cos(rad), cy - r * 0.6 * Math.sin(rad));
      ctx.lineTo(cx + r * 0.15 * Math.cos(rad + 2.5), cy - r * 0.15 * Math.sin(rad + 2.5));
      ctx.lineTo(cx + r * 0.15 * Math.cos(rad - 2.5), cy - r * 0.15 * Math.sin(rad - 2.5));
      ctx.closePath(); ctx.fillStyle = "#ef4444"; ctx.fill();
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.35 * Math.cos(rad), cy + r * 0.35 * Math.sin(rad));
      ctx.lineTo(cx + r * 0.15 * Math.cos(rad + 2.5), cy - r * 0.15 * Math.sin(rad + 2.5));
      ctx.lineTo(cx + r * 0.15 * Math.cos(rad - 2.5), cy - r * 0.15 * Math.sin(rad - 2.5));
      ctx.closePath(); ctx.fillStyle = "#94a3b8"; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fillStyle = "#e5e7eb"; ctx.fill();
    }
    drawCompass(0);
    async function pollCompass() {
      try {
        const resp = await fetch("/compass-state", { cache: "no-store" });
        const data = await resp.json();
        const heading = data.heading_deg != null ? data.heading_deg : 0;
        const servo = data.servo_angle != null ? data.servo_angle : 90;
        drawCompass(heading);
        headingEl.textContent = heading.toFixed(1) + "\u00b0";
        servoEl.textContent = "servo " + servo + "\u00b0";
      } catch(e) {}
    }
    pollCompass();
    setInterval(pollCompass, 300);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Rotation tracker – sparse optical flow + affine estimation
# ---------------------------------------------------------------------------


class RotationTracker:
    """Detects cumulative camera yaw rotation using sparse optical flow.

    Uses goodFeaturesToTrack + calcOpticalFlowPyrLK to find point
    correspondences between consecutive frames, then extracts the
    rotation angle via estimateAffinePartial2D (similarity transform
    with RANSAC, robust to moving objects).

    The cumulative heading starts at 0 degrees and grows/shrinks as
    the camera pans left/right.  The servo angle is derived by mapping
    the heading into the 0-180 range.
    """

    _DOWNSCALE_WIDTH = 320  # process at low res for speed

    def __init__(
        self,
        enabled: bool = True,
        smoothing: float = 0.4,
        min_features: int = 40,
        max_features: int = 200,
    ):
        self.enabled = enabled
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.min_features = max(10, int(min_features))
        self.max_features = max(self.min_features, int(max_features))
        self.lock = threading.Lock()

        # Internal state
        self._prev_gray = None
        self._prev_pts = None
        self._heading_deg = 0.0        # cumulative raw heading
        self._smooth_heading = 0.0     # EMA-smoothed heading
        self._servo_angle = 90         # mapped servo position
        self._frame_count = 0

        # Shi-Tomasi corner detection params
        self._feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.05,
            minDistance=12,
            blockSize=7,
        )
        # Lucas-Kanade optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def update(self, frame):
        """Feed a new BGR frame; returns current (heading_deg, servo_angle)."""
        if not self.enabled:
            return self._heading_deg, self._servo_angle

        # Downscale for speed
        h, w = frame.shape[:2]
        scale = self._DOWNSCALE_WIDTH / float(max(1, w))
        if scale < 0.95:
            small = cv2.resize(frame, (self._DOWNSCALE_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
            scale = 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < self.min_features:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feature_params)
            self._frame_count += 1
            return self._heading_deg, self._servo_angle

        # Compute sparse optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feature_params)
            return self._heading_deg, self._servo_angle

        # Keep only successfully tracked points
        mask = status.flatten() == 1
        if mask.sum() < 6:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feature_params)
            return self._heading_deg, self._servo_angle

        prev_good = self._prev_pts[mask]
        next_good = next_pts[mask]

        # Estimate similarity transform (rotation + translation + uniform scale)
        mat, inliers = cv2.estimateAffinePartial2D(
            prev_good, next_good, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if mat is not None:
            # mat is 2x3: [[cos*s, -sin*s, tx], [sin*s, cos*s, ty]]
            angle_rad = math.atan2(mat[1, 0], mat[0, 0])
            angle_deg = math.degrees(angle_rad)

            # Reject implausible single-frame rotations (> 15 deg likely a glitch)
            if abs(angle_deg) < 15.0:
                self._heading_deg += angle_deg
                # Normalize to -180..180
                self._heading_deg = (self._heading_deg + 180) % 360 - 180

        # Exponential moving average smoothing
        self._smooth_heading += self.smoothing * (self._heading_deg - self._smooth_heading)

        # Map heading to servo: heading 0 -> servo 90 (center/backward),
        # heading -90 -> servo 180, heading +90 -> servo 0
        servo = 90 - self._smooth_heading
        servo = max(0, min(180, int(round(servo))))

        with self.lock:
            self._servo_angle = servo

        # Refresh features periodically or when count drops
        self._frame_count += 1
        if self._frame_count % 5 == 0 or mask.sum() < self.min_features:
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        else:
            self._prev_pts = next_good.reshape(-1, 1, 2)
        self._prev_gray = gray

        return self._smooth_heading, servo

    def get_state(self):
        """Thread-safe snapshot of heading and servo angle."""
        with self.lock:
            return {
                "heading_deg": round(self._smooth_heading, 1),
                "servo_angle": self._servo_angle,
            }

    def reset(self):
        """Reset heading to zero (re-center compass)."""
        with self.lock:
            self._heading_deg = 0.0
            self._smooth_heading = 0.0
            self._servo_angle = 90
        self._prev_gray = None
        self._prev_pts = None


class CameraSource:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        device_path: str = "",
        rotation_tracker: RotationTracker | None = None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.jpeg_quality = max(20, min(95, jpeg_quality))
        self.device_path = device_path.strip()
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.thread = None
        self.cap = None
        self.active_source = None
        self.rotation_tracker = rotation_tracker

    def start(self):
        self.cap = self._open_camera()
        if not self.cap.isOpened():
            attempted = [self.device_path] if self.device_path else []
            attempted.extend([self.index, "/dev/video0", "/dev/video1"])
            raise RuntimeError(f"Could not open camera. Tried sources: {attempted}")
        self._apply_capture_settings(self.cap)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    def _read_loop(self):
        sleep_s = 1.0 / float(self.fps)
        consecutive_failures = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    self._reopen_camera()
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            # Run rotation tracking on the raw frame
            if self.rotation_tracker is not None:
                self.rotation_tracker.update(frame)

            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_jpeg = enc.tobytes()
            time.sleep(sleep_s)

    def _open_camera(self):
        candidates = []
        if self.device_path:
            candidates.append(self.device_path)
        candidates.append(self.index)
        if not self.device_path:
            candidates.extend(["/dev/video0", "/dev/video1"])

        for source in candidates:
            normalized = self._normalize_source(source)
            cap = cv2.VideoCapture(normalized, cv2.CAP_V4L2)
            if cap.isOpened():
                self.active_source = source
                return cap
            cap.release()

            cap = cv2.VideoCapture(normalized)
            if cap.isOpened():
                self.active_source = source
                return cap
            cap.release()

        return cv2.VideoCapture(self.index)

    @staticmethod
    def _normalize_source(source):
        # Some OpenCV builds on Pi fail with CAP_V4L2 when given "/dev/videoN" string;
        # convert to integer index where possible.
        if isinstance(source, str) and source.startswith("/dev/video"):
            tail = source[len("/dev/video") :]
            if tail.isdigit():
                return int(tail)
        return source

    def _reopen_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = self._open_camera()
        self._apply_capture_settings(self.cap)

    def _apply_capture_settings(self, cap):
        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg


class SerialSensorSource:
    def __init__(self, port: str, baud: int, enabled: bool, hr_delta_alert: int, log_raw: bool):
        self.port = port
        self.baud = max(1200, int(baud))
        self.enabled = bool(enabled)
        self.hr_delta_alert = max(1, int(hr_delta_alert))
        self.log_raw = bool(log_raw)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.connected = False
        self._ser = None  # shared serial handle for read + write
        self._ser_lock = threading.Lock()  # guards write access
        self._last_servo_angle = -1  # dedup: only write on change
        self.latest = {
            "heart_rate": None,
            "left": 0,
            "right": 0,
            "behind": 0,
            "speak_button": 0,
            "updated_at": None,
            "raw": "",
        }
        self.events = deque(maxlen=60)

    def start(self):
        if not self.enabled:
            return
        if serial is None:
            print("Serial disabled: pyserial is not installed on Pi.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def get_state(self):
        with self.lock:
            state = dict(self.latest)
            state["connected"] = self.connected
        return state

    def write_servo(self, angle: int):
        """Send a servo angle command to the Arduino (0-180)."""
        angle = max(0, min(180, int(angle)))
        if angle == self._last_servo_angle:
            return  # no change, skip write
        with self._ser_lock:
            if self._ser is None or not self.connected:
                return
            try:
                cmd = f"SERVO:{angle}\n"
                self._ser.write(cmd.encode("utf-8"))
                self._last_servo_angle = angle
            except Exception as exc:
                print(f"[Serial] servo write failed: {exc}")

    def _append_event(self, kind: str, text: str):
        self.events.append(
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": kind,
                "text": text,
            }
        )

    def _read_loop(self):
        while self.running:
            if self._ser is None:
                try:
                    with self._ser_lock:
                        self._ser = serial.Serial(self.port, self.baud, timeout=1)
                        self._ser.reset_input_buffer()
                    self.connected = True
                    self._append_event("status", f"Serial connected on {self.port}")
                except Exception:
                    self.connected = False
                    time.sleep(1.0)
                    continue
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").rstrip()
            except Exception:
                try:
                    with self._ser_lock:
                        self._ser.close()
                        self._ser = None
                except Exception:
                    self._ser = None
                self.connected = False
                time.sleep(0.5)
                continue

            if not raw:
                continue

            parsed = self._parse_line(raw)
            if self.log_raw:
                print(f"[Serial] raw: {raw}", flush=True)
            if parsed is None:
                if self.log_raw:
                    print(f"[Serial] ignored malformed line: {raw}", flush=True)
                continue
            hr, left, right, behind, speak_button = parsed

            with self.lock:
                prev_hr = self.latest.get("heart_rate")
                prev_left = int(self.latest.get("left") or 0)
                prev_right = int(self.latest.get("right") or 0)
                prev_behind = int(self.latest.get("behind") or 0)
                self.latest = {
                    "heart_rate": hr,
                    "left": left,
                    "right": right,
                    "behind": behind,
                    "speak_button": speak_button,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "raw": raw,
                }

            if prev_hr is not None and abs(int(hr) - int(prev_hr)) >= self.hr_delta_alert:
                self._append_event("heart_rate", f"Heart rate changed from {prev_hr} to {hr}")
            if prev_left == 0 and left == 1:
                self._append_event("direction", "Object detected on LEFT")
            if prev_right == 0 and right == 1:
                self._append_event("direction", "Object detected on RIGHT")
            if prev_behind == 0 and behind == 1:
                self._append_event("direction", "Object detected BEHIND")

    @staticmethod
    def _parse_line(raw):
        if isinstance(raw, (list, tuple)):
            parts = [str(p).strip() for p in raw]
        else:
            parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) < 4:
            return None
        try:
            nums = [int(float(p)) for p in parts]
        except ValueError:
            return None
        heart_rate = nums[0]
        left = 1 if int(nums[1]) > 0 else 0
        right = 1 if int(nums[2]) > 0 else 0
        behind = 1 if int(nums[3]) > 0 else 0
        speak_button = 1 if (len(nums) >= 5 and int(nums[4]) > 0) else 0
        return heart_rate, left, right, behind, speak_button


class MJPEGHandler(server.BaseHTTPRequestHandler):
    camera = None
    target_fps = 20
    gpt_feed = deque(maxlen=30)
    gpt_lock = threading.Lock()
    annotated_jpeg = None
    annotated_lock = threading.Lock()
    gpt_paused = False
    control_lock = threading.Lock()
    sensors = None
    rotation_tracker = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            frame_ready = self.camera.get_frame() is not None
            with self.gpt_lock:
                gpt_items = len(self.gpt_feed)
            with self.annotated_lock:
                annotated_ready = self.annotated_jpeg is not None
            with self.control_lock:
                paused = self.gpt_paused
            sensor_connected = False
            if self.sensors is not None:
                sensor_connected = bool(self.sensors.get_state().get("connected"))
            status = {
                "ok": True,
                "frame_ready": frame_ready,
                "annotated_ready": annotated_ready,
                "gpt_items": gpt_items,
                "gpt_paused": paused,
                "sensor_connected": sensor_connected,
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        if self.path == "/sensor-state":
            payload = {"ok": True, "sensor": {}}
            if self.sensors is not None:
                payload["sensor"] = self.sensors.get_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if self.path == "/gpt-feed":
            with self.gpt_lock:
                payload = {"ok": True, "entries": list(self.gpt_feed)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if self.path == "/gpt-control":
            with self.control_lock:
                paused = self.gpt_paused
            payload = {"ok": True, "paused": paused}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if self.path == "/compass-state":
            state = {"ok": True, "heading_deg": 0.0, "servo_angle": 90}
            if self.rotation_tracker is not None:
                state.update(self.rotation_tracker.get_state())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode("utf-8"))
            return

        if self.path not in ("/video", "/annotated-video"):
            self.send_error(404, "Not found")
            return

        prefer_annotated = self.path == "/annotated-video"
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        frame_sleep = 1.0 / float(max(1, self.target_fps))
        try:
            while True:
                jpeg = None
                if prefer_annotated:
                    with self.annotated_lock:
                        jpeg = self.annotated_jpeg
                if jpeg is None:
                    jpeg = self.camera.get_frame()
                if jpeg is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(frame_sleep)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        if self.path == "/gpt-log":
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self.send_error(400, "Missing request body")
                return

            raw_body = self.rfile.read(min(content_length, 65536))
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(400, "Body must be valid JSON")
                return

            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_error(400, "JSON field 'text' is required")
                return

            try:
                memory_conf = float(payload.get("memory_conf", 0.0))
            except (TypeError, ValueError):
                memory_conf = 0.0

            entry = {
                "time": payload.get("time") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase": str(payload.get("phase", "")).strip(),
                "memory_conf": memory_conf,
                "text": text[:4000],
            }
            with self.gpt_lock:
                self.gpt_feed.append(entry)
                count = len(self.gpt_feed)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "count": count}).encode("utf-8"))
            return

        if self.path == "/gpt-control":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(min(content_length, 4096))
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(400, "Body must be valid JSON")
                return

            paused = bool(payload.get("paused", False))
            with self.control_lock:
                self.gpt_paused = paused

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "paused": paused}).encode("utf-8"))
            return

        if self.path == "/annotated-frame":
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self.send_error(400, "Missing frame body")
                return
            max_bytes = 5 * 1024 * 1024
            if content_length > max_bytes:
                self.send_error(413, "Frame too large")
                return

            raw_body = self.rfile.read(content_length)
            if not raw_body:
                self.send_error(400, "Empty frame body")
                return

            with self.annotated_lock:
                self.annotated_jpeg = raw_body

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "bytes": len(raw_body)}).encode("utf-8"))
            return

        if self.path == "/compass-reset":
            if self.rotation_tracker is not None:
                self.rotation_tracker.reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "heading_deg": 0.0, "servo_angle": 90}).encode("utf-8"))
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        return


class ThreadedHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="Pi5 USB webcam MJPEG stream server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--camera-index", type=int, default=0, help="USB webcam index on Pi")
    parser.add_argument(
        "--camera-device",
        default="",
        help="Optional explicit camera device path (e.g. /dev/video0)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--fps", type=int, default=20, help="Capture/stream FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="MJPEG quality (20-95)")
    parser.add_argument("--serial-port", default="/dev/ttyACM0", help="Arduino serial port")
    parser.add_argument("--serial-baud", type=int, default=115200, help="Arduino serial baud rate")
    parser.add_argument(
        "--disable-serial",
        action="store_true",
        help="Disable Arduino serial sensor reader",
    )
    parser.add_argument(
        "--hr-delta-alert",
        type=int,
        default=8,
        help="Heart-rate change delta used for sensor event logging",
    )
    parser.add_argument(
        "--no-serial-log-raw",
        action="store_true",
        help="Disable printing raw Arduino serial lines to logs",
    )
    parser.add_argument(
        "--disable-compass",
        action="store_true",
        help="Disable camera rotation tracking and servo compass",
    )
    parser.add_argument(
        "--compass-smoothing",
        type=float,
        default=0.4,
        help="EMA smoothing factor for compass heading (0-1, lower = smoother)",
    )
    parser.add_argument(
        "--servo-update-hz",
        type=float,
        default=10.0,
        help="How often to send servo angle updates to Arduino (Hz)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Rotation tracker (compass)
    compass_enabled = not args.disable_compass
    rotation_tracker = RotationTracker(
        enabled=compass_enabled,
        smoothing=args.compass_smoothing,
    ) if compass_enabled else None

    camera = CameraSource(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        device_path=args.camera_device,
        rotation_tracker=rotation_tracker,
    )
    camera.start()
    sensors = SerialSensorSource(
        port=args.serial_port,
        baud=args.serial_baud,
        enabled=(not args.disable_serial),
        hr_delta_alert=args.hr_delta_alert,
        log_raw=(not args.no_serial_log_raw),
    )
    sensors.start()

    # Servo update loop: periodically push compass heading to Arduino servo
    servo_stop_event = threading.Event()

    def _servo_loop():
        interval = 1.0 / max(1.0, float(args.servo_update_hz))
        while not servo_stop_event.is_set():
            if rotation_tracker is not None:
                state = rotation_tracker.get_state()
                sensors.write_servo(state["servo_angle"])
            servo_stop_event.wait(interval)

    servo_thread = None
    if compass_enabled and not args.disable_serial:
        servo_thread = threading.Thread(target=_servo_loop, daemon=True)
        servo_thread.start()

    MJPEGHandler.camera = camera
    MJPEGHandler.target_fps = max(1, args.fps)
    MJPEGHandler.sensors = sensors
    MJPEGHandler.rotation_tracker = rotation_tracker

    httpd = ThreadedHTTPServer((args.host, args.port), MJPEGHandler)
    print(f"Pi5 stream server ready on http://{args.host}:{args.port}")
    if camera.active_source is not None:
        print(f"Camera source selected: {camera.active_source}")
    if compass_enabled:
        print("Compass rotation tracking enabled (servo output via serial).")
    print(
        "Endpoints: / (dashboard), /video (raw MJPEG), /annotated-video (YOLO MJPEG), "
        "/health, /sensor-state, /gpt-feed, /gpt-log, /gpt-control, /annotated-frame, "
        "/compass-state, /compass-reset"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servo_stop_event.set()
        if servo_thread is not None:
            servo_thread.join(timeout=1.0)
        httpd.server_close()
        camera.stop()
        sensors.stop()


if __name__ == "__main__":
    main()
