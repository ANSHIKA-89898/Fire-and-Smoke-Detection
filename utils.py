"""
utils.py
--------
Small shared helper functions used across app.py, detector.py, and
alert_manager.py: image conversion, frame resizing, temp-file handling,
and video writer creation.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image (any mode) to an OpenCV BGR numpy array."""
    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB (for display in a browser <img> tag)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, max_width: int = config.MAX_FRAME_WIDTH) -> np.ndarray:
    """
    Resize a frame down to `max_width` while preserving aspect ratio.
    Leaves the frame unchanged if it is already narrower than max_width.
    Keeps memory/CPU usage bounded for large videos or webcam frames.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def save_uploaded_file_to_temp(uploaded_file) -> Path:
    """
    Persist an uploaded file-like object (video) to a temporary file on disk
    so OpenCV's VideoCapture can open it by path.
    """
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.read())
    finally:
        tmp.close()
    return Path(tmp.name)


def safe_delete(path: Optional[Path]) -> None:
    """Delete a temp file if it exists, ignoring any errors."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not delete temp file %s: %s", path, exc)


def create_video_writer(
    output_path: Path,
    frame_size: Tuple[int, int],
    fps: float,
) -> cv2.VideoWriter:
    """
    Create an mp4 VideoWriter. Falls back to a widely-supported codec if
    the preferred one is unavailable on the current platform.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), frame_size)
    if not writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fallback_path = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(fallback_path), fourcc, max(fps, 1.0), frame_size)
    return writer


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1m 23.4s' or '4.2s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s"
