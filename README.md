# Pi5 Webcam -> Laptop Stream + GPT Vision Assistant

This repo is set up for this architecture:

- **Raspberry Pi 5**: captures USB webcam and serves MJPEG stream at `http://<PI_IP>:8080/video`
- **Arduino on Pi**: Pi reads serial sensor packets like `heart_rate,left,right,behind,speak_button` from `/dev/ttyACM0`
- **Laptop**: receives that stream, runs YOLO + OpenAI GPT analysis, shows video windows, and prints GPT output in laptop terminal
- **Browser dashboard**: open `http://<PI_IP>:8080` to see live YOLO-boxed video and GPT outputs side-by-side
- **Speech on Mac**: GPT output is also spoken through your Mac audio output (speakers or Bluetooth)

## Files You Will Use

- `pi5_camera_stream.py`: Pi webcam stream server
- `vision_assistant.py`: Laptop YOLO + GPT assistant
- `scripts/deploy_to_pi5.sh`: Upload project to Pi and install Pi dependencies
- `scripts/pi5_run_stream.sh`: Start camera stream on Pi
- `scripts/laptop_run_assistant.sh`: Start laptop assistant against Pi stream
- `deploy/pi5-camera-stream.service`: Optional systemd service for auto-start on Pi boot

## 1) Laptop Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

`OPENAI_API_KEY` is required on the laptop (where GPT calls happen).

## 2) Upload Code To Pi 5

From your laptop in this project directory:

```bash
bash scripts/deploy_to_pi5.sh <PI_HOST_OR_IP>
```

Example:

```bash
bash scripts/deploy_to_pi5.sh 192.168.1.42
```

This syncs files to `/home/pi/makeUofT` and installs Pi dependencies into `/home/pi/makeUofT/.venv`.

## 3) Start Pi Camera Stream

### Manual start on Pi

```bash
ssh pi@<PI_HOST_OR_IP> 'bash /home/pi/makeUofT/scripts/pi5_run_stream.sh'
```

### Verify from laptop

```bash
curl http://<PI_HOST_OR_IP>:8080/health
```

Expected JSON includes `"ok": true`.

Open the dashboard in your browser:

```bash
open http://<PI_HOST_OR_IP>:8080
```

Dashboard controls:
- `Pause GPT` / `Resume GPT` button stops or resumes GPT analysis from the webpage.
- Video pane shows YOLO annotated frames when laptop assistant is running.

## 4) Run Assistant On Laptop (Video + GPT Output)

From laptop project root:

```bash
source .venv/bin/activate
bash scripts/laptop_run_assistant.sh <PI_HOST_OR_IP>
```

This opens:
- `Raw Camera Stream` window (direct Pi feed)
- `Vision Assistant (YOLO + GPT Context)` window (annotated feed)

GPT analysis is printed in the **laptop terminal**.
GPT analysis is also spoken aloud on your Mac by default.
Press/hold your Arduino `speak_button` (5th serial value), speak into your webcam mic, then release the button to send the voice prompt.

Use the cheaper model explicitly:

```bash
bash scripts/laptop_run_assistant.sh <PI_HOST_OR_IP> --model gpt-4o-mini
```

## 4.1) Audio Output (Mac Speakers / Bluetooth)

Default behavior:
- Speech goes to whichever output device macOS currently uses.

Optional auto-route to Bluetooth output by device name:

```bash
brew install switchaudio-osx
AUDIO_OUTPUT_DEVICE="Your Bluetooth Headphones" bash scripts/laptop_run_assistant.sh <PI_HOST_OR_IP> --model gpt-4o-mini
```

List available output device names:

```bash
python vision_assistant.py --list-audio-devices
```

## 5) Optional: Auto-start Pi stream on boot (systemd)

On Pi:

```bash
sudo cp /home/pi/makeUofT/deploy/pi5-camera-stream.service /etc/systemd/system/pi5-camera-stream.service
sudo systemctl daemon-reload
sudo systemctl enable --now pi5-camera-stream.service
sudo systemctl status pi5-camera-stream.service
```

## Useful Commands

List local cameras (laptop):

```bash
python vision_assistant.py --list-cameras
```

List microphone devices:

```bash
python vision_assistant.py --list-mics
```

Run laptop assistant with custom context:

```bash
bash scripts/laptop_run_assistant.sh <PI_HOST_OR_IP> --context "Indoor workspace with tools"
```

Manual direct run (without helper script):

```bash
python vision_assistant.py \
  --source-url "http://<PI_HOST_OR_IP>:8080/video" \
  --show-raw-stream \
  --gpt-push-url "http://<PI_HOST_OR_IP>:8080/gpt-log" \
  --gpt-control-url "http://<PI_HOST_OR_IP>:8080/gpt-control" \
  --sensor-url "http://<PI_HOST_OR_IP>:8080/sensor-state" \
  --annotated-push-url "http://<PI_HOST_OR_IP>:8080/annotated-frame" \
  --speak-output \
  --interrupt-speech \
  --push-to-talk \
  --sensor-button-ptt \
  --model gpt-4o-mini
```
