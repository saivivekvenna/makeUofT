#!/usr/bin/env python3
import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http import server
from socketserver import ThreadingMixIn

import cv2
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
      grid-template-columns: minmax(360px, 2fr) minmax(280px, 1fr);
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
  </script>
</body>
</html>
"""


class CameraSource:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        device_path: str = "",
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

    def _append_event(self, kind: str, text: str):
        self.events.append(
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": kind,
                "text": text,
            }
        )

    def _read_loop(self):
        ser = None
        while self.running:
            if ser is None:
                try:
                    ser = serial.Serial(self.port, self.baud, timeout=1)
                    ser.reset_input_buffer()
                    self.connected = True
                    self._append_event("status", f"Serial connected on {self.port}")
                except Exception:
                    self.connected = False
                    time.sleep(1.0)
                    continue
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").rstrip()
            except Exception:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
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
    parser.add_argument("--serial-baud", type=int, default=9600, help="Arduino serial baud rate")
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
    return parser.parse_args()


def main():
    args = parse_args()
    camera = CameraSource(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        device_path=args.camera_device,
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

    MJPEGHandler.camera = camera
    MJPEGHandler.target_fps = max(1, args.fps)
    MJPEGHandler.sensors = sensors

    httpd = ThreadedHTTPServer((args.host, args.port), MJPEGHandler)
    print(f"Pi5 stream server ready on http://{args.host}:{args.port}")
    if camera.active_source is not None:
        print(f"Camera source selected: {camera.active_source}")
    print(
        "Endpoints: / (dashboard), /video (raw MJPEG), /annotated-video (YOLO MJPEG), "
        "/health, /sensor-state, /gpt-feed, /gpt-log, /gpt-control, /annotated-frame"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        camera.stop()
        sensors.stop()


if __name__ == "__main__":
    main()
