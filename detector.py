"""
detector.py
-----------
Wraps the Ultralytics YOLO model used for fire/smoke detection.

Responsibilities:
- Safely load the custom fire/smoke YOLO model (never silently falls back
  to a generic COCO model, since COCO has no "fire"/"smoke" classes).
- Run inference on a single frame (numpy BGR array) and return a
  structured list of detections plus an annotated frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the fire/smoke YOLO model file cannot be located."""


@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates


class FireSmokeDetector:
    """
    Loads a fire/smoke YOLO model and runs inference on frames.

    The model MUST be a YOLO model trained specifically to detect
    "Fire" and "Smoke" classes. A stock COCO-pretrained model (yolov8n.pt,
    yolo11n.pt, etc.) does NOT contain these classes and must never be
    used as a silent fallback.
    """

    def __init__(self, model_path: Path = config.MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names: dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Fire/Smoke model not found at '{self.model_path}'. "
                "Place your trained model file at models/fire_smoke.pt. "
                "See the README 'Model Setup' section for instructions."
            )

        try:
            # Imported lazily so the rest of the app can still start (and
            # show a friendly error) even if ultralytics/torch fail to
            # import on a broken environment.
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "The 'ultralytics' package is required. Install it with: "
                "pip install -r requirements.txt"
            ) from exc

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}'. "
                f"The file may be corrupted or is not a valid YOLO .pt file. "
                f"Original error: {exc}"
            ) from exc

        # Prefer the class names embedded in the model file itself; fall
        # back to config.CLASS_NAMES if the model has no names attribute.
        model_names = getattr(self.model, "names", None)
        if model_names:
            self.class_names = {int(k): str(v) for k, v in model_names.items()}
        else:
            self.class_names = config.CLASS_NAMES

        logger.info("Loaded fire/smoke model from %s with classes: %s",
                    self.model_path, self.class_names)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Run detection on a single BGR frame.

        Returns:
            annotated_frame: a copy of `frame` with bounding boxes drawn.
            detections: list of Detection objects found in the frame.
        """
        if self.model is None:
            raise ModelNotFoundError("Model is not loaded.")

        if frame is None or frame.size == 0:
            raise ValueError("Received an empty/invalid frame for detection.")

        results = self.model.predict(
            source=frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=config.INFERENCE_IMAGE_SIZE,
            verbose=False,
        )

        detections: List[Detection] = []
        annotated = frame.copy()

        if not results:
            return annotated, detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return annotated, detections

        for box in boxes:
            try:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            except (IndexError, AttributeError, ValueError) as exc:
                logger.warning("Skipping malformed detection box: %s", exc)
                continue

            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            detections.append(Detection(class_name, conf, (x1, y1, x2, y2)))
            _draw_box(annotated, class_name, conf, (x1, y1, x2, y2))

        return annotated, detections


def _draw_box(
    frame: np.ndarray,
    class_name: str,
    confidence: float,
    box: Tuple[int, int, int, int],
) -> None:
    """Draw a labeled bounding box on `frame` in-place, styled by class."""
    x1, y1, x2, y2 = box

    name_lower = class_name.lower()
    if "fire" in name_lower:
        color = (0, 0, 255)       # red (BGR) for fire
    elif "smoke" in name_lower:
        color = (128, 128, 128)   # grey for smoke
    else:
        color = (0, 255, 0)       # green fallback for other classes

    label = f"{class_name} | {confidence * 100:.1f}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    label_y1 = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 5 if y1 - 5 > 0 else y1 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
