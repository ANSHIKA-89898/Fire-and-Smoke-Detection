# 🔥 Fire & Smoke Detection System

An AI-based local surveillance application that detects **fire** and **smoke** in
uploaded images, uploaded videos, and a live webcam stream, using a YOLO object
detection model, OpenCV, a small Flask backend, and a plain HTML/CSS/JavaScript
dashboard (no frontend framework, no build step).

---

## 1. Project Overview

This project is a self-contained, locally-run web application that:

- Accepts an **image**, a **video file**, or a **live webcam feed** as input.
- Runs a YOLO model (trained specifically for `Fire` and `Smoke` classes) on
  each frame.
- Draws bounding boxes with class name + confidence on detected regions.
- Raises clear visual alerts ("🔥 FIRE DETECTED!" / "⚠️ SMOKE DETECTED!"),
  optionally plays an alarm sound, and saves evidence frames to `alerts/`.
- Uses a cooldown mechanism so continuous detections don't flood your disk.
- Keeps a searchable, exportable (CSV) detection history.
- Displays everything on a single dashboard with status cards, metrics, and
  controls.

All processing is done **locally on your machine** — nothing is uploaded to
an external server.

---

## 2. Architecture

```
 Input (Image / Video / Webcam)
              |
              v
          OpenCV (frame capture / decoding / resizing)
              |
              v
      YOLO Fire/Smoke Model  (detector.py -> FireSmokeDetector)
              |
              v
        Detection Results (class, confidence, box)
              |
              v
       Bounding Boxes drawn on frame
              |
              v
   Alert Manager (alert_manager.py -> AlertManager)
     - cooldown / debounce per class
     - saves evidence image to alerts/
     - plays alarm sound (triggered by the frontend)
     - appends to detection history (CSV)
              |
              v
   Flask API (server.py) -> JSON / base64 responses
              |
              v
   HTML/CSS/JS Dashboard (static/index.html, style.css, app.js)
     - dashboard cards / metrics
     - live detection view (image, video, webcam)
     - alert panel
     - history table + CSV export
```

The browser never runs the YOLO model itself — it uploads images/video or
webcam frames to the Flask API, which performs detection server-side and
returns an annotated image/video plus structured JSON.

---

## 3. Folder Structure

```
fire-smoke-detection/
│
├── server.py                # Flask backend / entry point (API + static files)
├── config.py                 # All configuration in one place
├── detector.py                # YOLO model loading + inference
├── alert_manager.py            # Cooldown, evidence saving, history, alarm
├── utils.py                     # Shared helper functions
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/                       # Frontend (HTML/CSS/JS, no build step)
│   ├── index.html                  # Dashboard markup
│   ├── css/style.css                # Dashboard styling
│   ├── js/app.js                     # Frontend logic (calls the Flask API)
│   └── assets/alarm.wav               # Alarm sound served to the browser
│
├── models/
│   ├── fire_smoke.pt        # <-- YOU place your trained model here
│   └── README.txt
│
├── alerts/                  # Saved evidence images (auto-created)
│   └── .gitkeep
│
├── outputs/                 # detection_history.csv + annotated videos
│   └── .gitkeep
│
├── assets/
│   ├── alarm.wav             # source alarm file (copied into static/assets)
│   └── README.txt
│
└── sample_data/              # optional folder for your own test files
    └── README.txt
```

---

## 4. Tech Stack

- **Python** 3.10+
- **Ultralytics YOLO** (YOLOv8 / YOLO11 architecture, custom-trained weights)
- **OpenCV** (`opencv-python-headless`) — frame capture, resizing, drawing, video I/O
- **NumPy** — array/image manipulation
- **Flask** — lightweight JSON/multipart API backend, also serves the static frontend
- **HTML / CSS / vanilla JavaScript** — dashboard UI (no framework, no build tooling)
- **Pillow** — image handling utilities
- **Pandas** — detection history table + CSV export
- **playsound** — used only as an optional server-side beep hook; the primary
  alarm sound plays client-side via the browser's `<audio>` element so it's
  audible on whatever machine the dashboard is open on
- Browser **MediaDevices / getUserMedia API** — webcam capture on the client,
  frames are POSTed to the backend for detection

---

## 5. Installation

### 5.1 Prerequisites

- Python 3.10 or newer installed and available on your `PATH`.
- A webcam if you want to use Webcam mode (optional).

### 5.2 Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` will install `torch` as a dependency. On the first
> run, Ultralytics/PyTorch may download small helper files. This still works
> fully on CPU — no GPU is required. If a compatible NVIDIA GPU + CUDA is
> present, Ultralytics will use it automatically for faster inference.

---

## 6. Model Setup (IMPORTANT — read before running)

A **standard COCO-pretrained YOLO model (e.g. `yolov8n.pt`, `yolo11n.pt`)
CANNOT detect fire or smoke.** COCO's 80 classes do not include "fire" or
"smoke", so using a stock model would silently produce meaningless (or zero)
detections. This project therefore **refuses to run** until a real fire/smoke
model is provided — it will never silently substitute a generic model.

### 6.1 Where to put your model

Place your trained YOLO fire/smoke model file at:

```
models/fire_smoke.pt
```

The app checks for this file on startup. If it's missing, the dashboard
shows a clear error explaining exactly what to do instead of crashing or
producing fake results.

### 6.2 Where to get a model

You have two options:

1. **Use a publicly available fire/smoke-trained YOLO model.** Several
   community-trained fire/smoke YOLO checkpoints exist on model-sharing
   sites (e.g. Roboflow Universe, Hugging Face, GitHub). Search for
   "YOLOv8 fire smoke detection weights", verify the license permits your
   use case, download the `.pt` file, rename it `fire_smoke.pt`, and place
   it in `models/`.
2. **Train your own model** on a fire/smoke dataset — see Section 7 below
   for the full step-by-step guide.

### 6.3 Class name mapping

`config.py` defines the expected class map:

```python
CLASS_NAMES = {0: "Fire", 1: "Smoke"}
```

The app actually prefers the class names embedded in your `.pt` file
(`model.names`) automatically — if your model's class order/names differ,
just make sure the model's own `names` mapping in the checkpoint is correct;
you generally do not need to edit `config.py` unless you want to override it.

---

## 7. Model Training Guide

If you don't have a fire/smoke model yet, here's how to train one with
Ultralytics YOLO.

### 7.1 Dataset collection

Collect labeled images/video frames containing:
- Open flames / fire (various sizes, indoor & outdoor, day & night)
- Smoke (various densities, colors, backgrounds)
- Negative examples (no fire/smoke) to reduce false positives

Public sources to consider: Roboflow Universe fire/smoke datasets, Kaggle
fire/smoke datasets, or your own captured footage (always respect dataset
licenses).

### 7.2 Image annotation

Annotate each image with bounding boxes around fire and smoke regions using
a tool such as **Roboflow**, **CVAT**, or **LabelImg**, and export in
**YOLO format** (one `.txt` file per image with lines like
`class_id x_center y_center width height`, all normalized 0–1).

### 7.3 YOLO dataset directory structure

```
fire_smoke_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 7.4 data.yaml

Create a `data.yaml` describing the dataset:

```yaml
path: fire_smoke_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Fire
  1: Smoke
```

### 7.5 Train/validation/test split

A common split is roughly 70% train / 20% validation / 10% test. Make sure
each split has a reasonable mix of fire, smoke, and negative examples.

### 7.6 Training command

Install Ultralytics in your training environment (`pip install ultralytics`)
and run:

```bash
yolo detect train data=fire_smoke_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

- `model=yolov8n.pt` starts from a pretrained COCO checkpoint (transfer
  learning) — this is only the *starting point* for training; the resulting
  fine-tuned weights will actually know about `Fire`/`Smoke`.
- Adjust `epochs`, `batch`, and `imgsz` based on your dataset size and
  hardware.

### 7.7 Validation

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=fire_smoke_dataset/data.yaml
```

Review precision/recall/mAP metrics and confusion matrix output in the
`runs/detect/val/` folder.

### 7.8 Testing

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=fire_smoke_dataset/images/test
```

Manually review the predicted bounding boxes on your test images.

### 7.9 Exporting and placing the trained model

Once satisfied, copy the trained weights into this project:

```bash
cp runs/detect/train/weights/best.pt models/fire_smoke.pt
```

(On Windows, use `copy` instead of `cp`.)

Restart the server (`python server.py`) — it will now detect fire and smoke.

---

## 8. Custom Model Configuration

All tunable settings live in `config.py`:

```python
MODEL_PATH = MODELS_DIR / "fire_smoke.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_ALERT_COOLDOWN_SECONDS = 6.0
DEFAULT_FRAME_SKIP = 2
MAX_FRAME_WIDTH = 960
```

You can also override `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, alarm on/off,
and cooldown directly from the sidebar at runtime — no code changes needed
for day-to-day tuning.

---

## 9. How to Run

```bash
python server.py
```

This starts the Flask backend and serves the dashboard at
`http://localhost:5000`. Open that URL in your browser.

If `models/fire_smoke.pt` is missing, the dashboard will show an on-screen
banner explaining what to do instead of a stack trace — follow Section 6 to
fix it. The rest of the UI still loads so you can see the layout even
without a model.

### 9.1 Image mode

1. Select **Image** in the sidebar.
2. Choose a `.jpg` / `.jpeg` / `.png` / `.bmp` file — detection runs
   automatically on selection.
3. View the annotated result, class/confidence list, and any alerts raised.

### 9.2 Video mode

1. Select **Video** in the sidebar.
2. Choose an `.mp4` / `.avi` / `.mov` / `.mkv` file.
3. Click **Start Video Processing**. The backend processes the whole video
   synchronously (this can take a while for long clips — a progress
   indicator and status text will update).
4. When finished, review the summary stats (frames processed, fire/smoke
   detection counts, max confidence, duration) and play back the annotated
   video directly in the browser.

### 9.3 Webcam mode

1. Select **Webcam** in the sidebar.
2. Click **Start Webcam** (grant camera permission if your browser prompts
   you). The browser captures frames locally and sends each one to the
   Flask backend for detection roughly twice a second.
3. Watch the live annotated feed and alert banner update in real time.
4. Click **Stop Webcam** to release the camera.

---

## 10. Testing Procedure

1. Place your own test files as described in `sample_data/README.txt`
   (`sample_fire.jpg`, `sample_smoke.jpg`, `sample_video.mp4`) — this
   project does not ship fabricated test media.
2. Run `python server.py` and open `http://localhost:5000`.
3. Try Image mode with `sample_fire.jpg` and `sample_smoke.jpg` and confirm
   bounding boxes + alerts appear as expected.
4. Try Video mode with `sample_video.mp4` and confirm processing completes
   and stats look reasonable.
5. Try Webcam mode (optional) if a camera is available.
6. Check that:
   - `alerts/` contains newly saved evidence images.
   - `outputs/detection_history.csv` contains matching rows.
   - The dashboard's "Download History as CSV" button produces the same data.

---

## 11. Common Errors and Fixes

| Problem | Likely Cause | Fix |
|---|---|---|
| "Model not available" banner on launch | `models/fire_smoke.pt` missing | Add your trained model (Section 6) |
| `RuntimeError: Failed to load YOLO model` | Corrupted or invalid `.pt` file | Re-download/re-export the model file |
| Browser shows "Could not access the webcam" | No camera, camera in use by another app, or missing browser permission | Close other apps/tabs using the camera; grant camera permission for this site; try a different browser |
| Video won't process / "Could not open the uploaded video" | Corrupted file or codec unsupported by your OpenCV build | Convert the video to standard H.264 MP4 and retry |
| No sound on alerts | Browser blocked autoplay audio, or `static/assets/alarm.wav` missing | Interact with the page once (click anywhere) so the browser allows audio playback; confirm the `.wav` file exists |
| App is slow on CPU during video/webcam | Large frames / low frame skip | Increase "Frame Skip" in the sidebar, or reduce `MAX_FRAME_WIDTH` in `config.py` |
| `ModuleNotFoundError` for any package | Dependencies not installed in the active virtual environment | Re-run `pip install -r requirements.txt` inside the activated venv |
| Alerts folder filling up too fast | Cooldown too low | Increase "Alert Cooldown (seconds)" in the sidebar |
| Dashboard loads but stats never update | Backend not reachable at `/api/*` (e.g. wrong port, server not running) | Confirm `python server.py` is running and the page was opened at the same host/port it's serving on |

---

## 12. Future Improvements

- Add multi-camera / RTSP stream support.
- Add email/SMS/webhook notifications on alert.
- Add a background/headless monitoring mode (no browser required).
- Add model auto-download from a hosted release once a vetted public
  fire/smoke model is available.
- Add night-vision/low-light preprocessing for better night detection.
- Add a heatmap/analytics view of alert frequency over time.
- Containerize the app with Docker for easier deployment.

---

## 13. Privacy & Security Notes

This is a **local** application. Images, video, and webcam frames are
processed entirely on your machine using OpenCV and a local YOLO model —
nothing is sent to an external server by this codebase.
