"""
config.py
---------
Centralized configuration for the Fire & Smoke Detection System.

Change values here to tune detection behaviour, thresholds, and file
locations without touching the rest of the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories (cross-platform, relative to this file's location)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

MODELS_DIR: Path = BASE_DIR / "models"
ALERTS_DIR: Path = BASE_DIR / "alerts"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
ASSETS_DIR: Path = BASE_DIR / "assets"
SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"

# Make sure the runtime-writable directories exist.
for _dir in (MODELS_DIR, ALERTS_DIR, OUTPUTS_DIR, ASSETS_DIR, SAMPLE_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Path to the fire/smoke-specific YOLO model (.pt file).
# A standard COCO-pretrained YOLO model CANNOT detect fire/smoke because
# "fire" and "smoke" are not COCO classes. You must place a model trained
# on a fire/smoke dataset here. See the README "Model Setup" section.
MODEL_PATH: Path = MODELS_DIR / "firedetect-11s.pt"

# Expected class names for the fire/smoke model. If your custom model was
# trained with different class names/order, update this list to match
# the model's `model.names` mapping exactly.
CLASS_NAMES = {
    0: "Fire",
    1: "Smoke",
}

# Classes that should trigger an alert when detected above the threshold.
ALERT_CLASSES = {"Fire", "Smoke"}

# ---------------------------------------------------------------------------
# Detection thresholds (overridable at runtime from the web dashboard sidebar)
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_IOU_THRESHOLD: float = 0.45

# Inference image size (larger = more accurate, slower). Must be a multiple
# of 32 for YOLO models.
INFERENCE_IMAGE_SIZE: int = 640

# ---------------------------------------------------------------------------
# Alert system configuration
# ---------------------------------------------------------------------------
# Minimum number of seconds between two saved alert images for the SAME
# detection class. Prevents flooding the "alerts/" folder with near-
# identical frames when fire/smoke is continuously visible.
DEFAULT_ALERT_COOLDOWN_SECONDS: float = 6.0

# Path to the alarm sound played when an alert is triggered.
ALARM_SOUND_PATH: Path = ASSETS_DIR / "alarm.wav"

# ---------------------------------------------------------------------------
# Video / webcam processing configuration
# ---------------------------------------------------------------------------
# Only run detection on every Nth frame to keep the UI responsive on CPU.
# 1 = process every frame, 2 = process every other frame, etc.
DEFAULT_FRAME_SKIP: int = 2

# Maximum width (pixels) frames are resized to before inference/display.
# Keeps memory and CPU usage bounded for large video files/webcams.
MAX_FRAME_WIDTH: int = 960

# Reserved: minimum delay between webcam frame captures on the frontend.
WEBCAM_LOOP_SLEEP_SECONDS: float = 0.01

# ---------------------------------------------------------------------------
# Detection history configuration
# ---------------------------------------------------------------------------
HISTORY_CSV_PATH: Path = OUTPUTS_DIR / "detection_history.csv"
HISTORY_COLUMNS = [
    "timestamp",
    "detection_type",
    "confidence",
    "source",
    "alert_image_path",
]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
APP_TITLE: str = "🔥 Fire & Smoke Detection System"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
