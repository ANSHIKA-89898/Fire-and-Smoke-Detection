"""
server.py
---------
Flask backend for the Fire & Smoke Detection System.

DEPLOYMENT MODEL
    Frontend (static HTML/CSS/JS)  ->  Netlify
    Backend  (this file)           ->  Render / Railway / Fly.io / Hugging Face Spaces / VPS

Netlify cannot run a long-lived Flask + OpenCV + YOLO process, so this file is
made "cross-origin ready": it enables CORS, reads PORT / ALLOWED_ORIGINS from
environment variables, returns absolute URLs for generated videos, and exposes
a /api/health endpoint.

Local run:
    python server.py

Production run (e.g. Render / Railway start command):
    gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300

Environment variables:
    PORT             Port to listen on (set automatically by most hosts). Default 5000
    ALLOWED_ORIGINS  Comma-separated list of allowed frontend origins,
                     e.g. "https://my-site.netlify.app". Default "*" (allow all)
    MAX_UPLOAD_MB    Max upload size in MB. Default 100
    FLASK_DEBUG      "1" to enable debug mode locally. Default off
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

import config
import utils
from alert_manager import AlertManager
from detector import FireSmokeDetector, ModelNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = config.BASE_DIR / "static"
config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# Trust the reverse proxy headers (Render/Railway/etc.) so request.host_url is
# correct (https + real hostname) when building absolute URLs.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024

# ---------------------------------------------------------------------------
# CORS - lets the Netlify-hosted frontend call this backend.
# ---------------------------------------------------------------------------
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = "*" if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS}, r"/outputs/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=False,
    max_age=86400,
)

# ---------------------------------------------------------------------------
# Global (single-process) detector + alert manager.
# The detector is loaded lazily so the server can still start even if the
# model file is missing; detection endpoints return a clear JSON error.
# ---------------------------------------------------------------------------
_detector: FireSmokeDetector | None = None
_detector_error: str | None = None
_detector_lock = threading.Lock()
_alert_manager = AlertManager()


def get_detector() -> FireSmokeDetector | None:
    global _detector, _detector_error
    if _detector is not None:
        return _detector
    if _detector_error is not None:
        return None
    with _detector_lock:
        if _detector is not None:
            return _detector
        if _detector_error is not None:
            return None
        try:
            _detector = FireSmokeDetector()
            return _detector
        except ModelNotFoundError as exc:
            _detector_error = str(exc)
            return None
        except Exception as exc:  # noqa: BLE001
            _detector_error = f"Unexpected error while loading the model: {exc}"
            return None


def _frame_to_base64_jpeg(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def _base64_to_frame(data_url: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL prefixed) image into a BGR frame."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the submitted image data.")
    return frame


def _detections_to_json(detections) -> list[dict]:
    return [
        {
            "class_name": d.class_name,
            "confidence": round(d.confidence * 100, 2),
            "box": list(d.box),
        }
        for d in detections
    ]


def _alerts_to_json(alerts) -> list[dict]:
    return [
        {
            "timestamp": a.timestamp,
            "detection_type": a.detection_type,
            "confidence": a.confidence,
            "source": a.source,
            "alert_image_path": a.alert_image_path,
        }
        for a in alerts
    ]


def _thresholds_from_request(source) -> tuple[float, float]:
    conf = float(source.get("confidence", config.DEFAULT_CONFIDENCE_THRESHOLD))
    iou = float(source.get("iou", config.DEFAULT_IOU_THRESHOLD))
    return conf, iou


def _absolute_url(path: str) -> str:
    """Build a full URL so a frontend on another domain (Netlify) can load it."""
    return request.host_url.rstrip("/") + path


# ---------------------------------------------------------------------------
# Frontend (still served here for local use; on Netlify the static files are
# hosted by Netlify itself and only /api/* is used from this server).
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return jsonify({"service": "Fire & Smoke Detection API", "status": "running"})


# ---------------------------------------------------------------------------
# Health / status / config
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    """Cheap endpoint for uptime pings and for the frontend to test reachability."""
    return jsonify({"ok": True, "time": time.time()})


@app.route("/api/status")
def status():
    detector = get_detector()
    return jsonify(
        {
            "model_ready": detector is not None,
            "model_error": _detector_error,
            "model_path": str(config.MODEL_PATH),
            "defaults": {
                "confidence": config.DEFAULT_CONFIDENCE_THRESHOLD,
                "iou": config.DEFAULT_IOU_THRESHOLD,
                "cooldown": config.DEFAULT_ALERT_COOLDOWN_SECONDS,
                "frame_skip": config.DEFAULT_FRAME_SKIP,
            },
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update alarm on/off and cooldown for the shared AlertManager."""
    payload = request.get_json(force=True, silent=True) or {}
    if "cooldown" in payload:
        try:
            _alert_manager.cooldown_seconds = float(payload["cooldown"])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid cooldown value."}), 400
    if "alarm_enabled" in payload:
        _alert_manager.alarm_enabled = bool(payload["alarm_enabled"])
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------
@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    detector = get_detector()
    if detector is None:
        return jsonify({"error": _detector_error or "Model not available."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["image"]
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded image is empty."}), 400

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not read the uploaded file as a valid image."}), 400

    try:
        conf, iou = _thresholds_from_request(request.form)
    except ValueError:
        return jsonify({"error": "Invalid confidence/IoU value."}), 400

    try:
        annotated, detections = detector.predict(frame, conf, iou)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    new_alerts = _alert_manager.process_detections(annotated, detections, source="Image")

    return jsonify(
        {
            "annotated_image": _frame_to_base64_jpeg(annotated),
            "detections": _detections_to_json(detections),
            "new_alerts": _alerts_to_json(new_alerts),
        }
    )


# ---------------------------------------------------------------------------
# Webcam single-frame detection (browser captures frames and posts them)
# ---------------------------------------------------------------------------
@app.route("/api/detect/frame", methods=["POST"])
def detect_frame():
    detector = get_detector()
    if detector is None:
        return jsonify({"error": _detector_error or "Model not available."}), 503

    payload = request.get_json(force=True, silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No frame data was provided."}), 400

    try:
        frame = _base64_to_frame(image_data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid frame data: {exc}"}), 400

    frame = utils.resize_frame(frame)
    try:
        conf, iou = _thresholds_from_request(payload)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid confidence/IoU value."}), 400

    try:
        annotated, detections = detector.predict(frame, conf, iou)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    new_alerts = _alert_manager.process_detections(annotated, detections, source="Webcam")

    return jsonify(
        {
            "annotated_image": _frame_to_base64_jpeg(annotated),
            "detections": _detections_to_json(detections),
            "new_alerts": _alerts_to_json(new_alerts),
        }
    )


# ---------------------------------------------------------------------------
# Video detection (uploaded file, processed synchronously)
# ---------------------------------------------------------------------------
@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    detector = get_detector()
    if detector is None:
        return jsonify({"error": _detector_error or "Model not available."}), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file was uploaded."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Uploaded video has no filename."}), 400

    try:
        conf, iou = _thresholds_from_request(request.form)
    except ValueError:
        return jsonify({"error": "Invalid confidence/IoU value."}), 400
    try:
        frame_skip = max(1, int(request.form.get("frame_skip", config.DEFAULT_FRAME_SKIP)))
    except ValueError:
        frame_skip = config.DEFAULT_FRAME_SKIP

    suffix = Path(file.filename).suffix or ".mp4"
    temp_path = config.OUTPUTS_DIR / f"_upload_{uuid.uuid4().hex}{suffix}"
    file.save(str(temp_path))

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        utils.safe_delete(temp_path)
        return jsonify(
            {"error": "Could not open the uploaded video. It may be corrupted or in an unsupported format."}
        ), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_filename = f"annotated_{uuid.uuid4().hex}.mp4"
    out_path = config.OUTPUTS_DIR / out_filename

    writer = None
    frames_processed = 0
    fire_hits = 0
    smoke_hits = 0
    max_conf = 0.0
    all_new_alerts = []
    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            frame = utils.resize_frame(frame)

            if writer is None:
                h, w = frame.shape[:2]
                writer = utils.create_video_writer(out_path, (w, h), fps)

            if frame_index % frame_skip == 0:
                try:
                    annotated, detections = detector.predict(frame, conf, iou)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping a frame due to a detection error: %s", exc)
                    annotated, detections = frame, []

                frames_processed += 1
                for det in detections:
                    max_conf = max(max_conf, det.confidence * 100)
                    if det.class_name.lower() == "fire":
                        fire_hits += 1
                    elif det.class_name.lower() == "smoke":
                        smoke_hits += 1

                new_alerts = _alert_manager.process_detections(annotated, detections, source="Video")
                all_new_alerts.extend(new_alerts)
                write_frame = annotated
            else:
                write_frame = frame

            if writer is not None:
                writer.write(write_frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        utils.safe_delete(temp_path)

    duration = time.time() - start_time

    return jsonify(
        {
            "stats": {
                "frames_processed": frames_processed,
                "fire_detections": fire_hits,
                "smoke_detections": smoke_hits,
                "max_confidence": round(max_conf, 2),
                "duration_seconds": round(duration, 2),
            },
            "new_alerts": _alerts_to_json(all_new_alerts),
            # Absolute URL so it works when the frontend is on Netlify.
            "annotated_video_url": _absolute_url(f"/outputs/{out_filename}"),
        }
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(str(config.OUTPUTS_DIR), filename)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@app.route("/api/history")
def get_history():
    df = _alert_manager.history_dataframe()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/history/csv")
def download_history_csv():
    if not config.HISTORY_CSV_PATH.exists():
        return jsonify({"error": "No history recorded yet."}), 404
    return send_from_directory(
        str(config.HISTORY_CSV_PATH.parent),
        config.HISTORY_CSV_PATH.name,
        as_attachment=True,
        download_name="detection_history.csv",
    )


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    _alert_manager.clear_history()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "Uploaded file is too large."}), 413


@app.errorhandler(500)
def server_error(exc):  # noqa: ANN001
    logger.exception("Unhandled server error: %s", exc)
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
