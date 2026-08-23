"""
server.py
---------
Flask backend for the Fire & Smoke Detection System.

Serves the static HTML/CSS/JS frontend and exposes a small JSON/multipart
API that the frontend calls to run detection on images, video files, and
webcam frames, plus endpoints for the detection history.

Run with:
    python server.py
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response

import config
import utils
from alert_manager import AlertManager
from detector import FireSmokeDetector, ModelNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = config.BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# ---------------------------------------------------------------------------
# Global (single-process) detector + alert manager.
# The detector is loaded lazily so the server can still start and serve the
# frontend even if the model file is missing; every detection endpoint
# checks readiness first and returns a clear JSON error instead.
# ---------------------------------------------------------------------------
_detector: FireSmokeDetector | None = None
_detector_error: str | None = None
_alert_manager = AlertManager()


def get_detector() -> FireSmokeDetector | None:
    global _detector, _detector_error
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


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ---------------------------------------------------------------------------
# Status / config
# ---------------------------------------------------------------------------
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

    conf, iou = _thresholds_from_request(request.form)

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
    conf, iou = _thresholds_from_request(payload)

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

    conf, iou = _thresholds_from_request(request.form)
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
            "annotated_video_url": f"/outputs/{out_filename}",
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
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
