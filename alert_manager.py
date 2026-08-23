"""
alert_manager.py
-----------------
Handles alert debouncing/cooldown, evidence saving, alarm sound playback,
and the persistent detection history log.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    timestamp: str
    detection_type: str
    confidence: float
    source: str
    alert_image_path: str


class AlertManager:
    """
    Tracks per-class cooldowns so a continuous fire/smoke detection does
    not spam the alerts folder or history table, saves evidence frames,
    optionally plays an alarm sound, and maintains an in-memory +
    CSV-backed detection history.
    """

    def __init__(
        self,
        cooldown_seconds: float = config.DEFAULT_ALERT_COOLDOWN_SECONDS,
        alerts_dir: Path = config.ALERTS_DIR,
        history_csv_path: Path = config.HISTORY_CSV_PATH,
        alarm_enabled: bool = True,
        alarm_path: Path = config.ALARM_SOUND_PATH,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = Path(history_csv_path)
        self.alarm_enabled = alarm_enabled
        self.alarm_path = Path(alarm_path)

        self._last_alert_time: Dict[str, float] = {}
        self.history: List[AlertRecord] = self._load_history()

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------
    def _load_history(self) -> List[AlertRecord]:
        if not self.history_csv_path.exists():
            return []
        try:
            df = pd.read_csv(self.history_csv_path)
            records = []
            for _, row in df.iterrows():
                records.append(
                    AlertRecord(
                        timestamp=str(row.get("timestamp", "")),
                        detection_type=str(row.get("detection_type", "")),
                        confidence=float(row.get("confidence", 0.0)),
                        source=str(row.get("source", "")),
                        alert_image_path=str(row.get("alert_image_path", "")),
                    )
                )
            return records
        except Exception as exc:
            logger.warning("Could not load existing history CSV: %s", exc)
            return []

    def _persist_history(self) -> None:
        try:
            df = pd.DataFrame([r.__dict__ for r in self.history],
                               columns=config.HISTORY_COLUMNS)
            self.history_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.history_csv_path, index=False)
        except Exception as exc:
            logger.warning("Could not persist history CSV: %s", exc)

    def history_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=config.HISTORY_COLUMNS)
        return pd.DataFrame([r.__dict__ for r in self.history],
                             columns=config.HISTORY_COLUMNS)

    def clear_history(self) -> None:
        self.history = []
        self._persist_history()

    # ------------------------------------------------------------------
    # Cooldown logic
    # ------------------------------------------------------------------
    def _is_on_cooldown(self, detection_type: str) -> bool:
        last_time = self._last_alert_time.get(detection_type)
        if last_time is None:
            return False
        return (time.time() - last_time) < self.cooldown_seconds

    # ------------------------------------------------------------------
    # Evidence saving
    # ------------------------------------------------------------------
    def _save_evidence(self, frame: np.ndarray, detection_type: str) -> Optional[str]:
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{detection_type.lower()}_{timestamp_str}.jpg"
            out_path = self.alerts_dir / filename
            success = cv2.imwrite(str(out_path), frame)
            if not success:
                logger.warning("cv2.imwrite failed for %s", out_path)
                return None
            return str(out_path)
        except Exception as exc:
            logger.warning("Failed to save alert evidence: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Alarm sound
    # ------------------------------------------------------------------
    def _play_alarm(self) -> None:
        if not self.alarm_enabled:
            return
        if not self.alarm_path.exists():
            logger.info(
                "Alarm sound file not found at %s; skipping playback. "
                "Place a .wav file there to enable audible alerts.",
                self.alarm_path,
            )
            return
        try:
            from playsound import playsound
            playsound(str(self.alarm_path), block=False)
        except Exception as exc:
            # Audio playback is best-effort only; never crash the app.
            logger.info("Alarm playback unavailable (%s). Continuing silently.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_detections(
        self,
        frame: np.ndarray,
        detections: List["object"],
        source: str,
    ) -> List[AlertRecord]:
        """
        Evaluate a frame's detections, trigger alerts (respecting the
        per-class cooldown), and append new records to history.

        `detections` is a list of detector.Detection objects (duck-typed
        here to avoid a circular import).

        Returns the list of newly created AlertRecord objects (may be
        empty if nothing new was detected or everything is on cooldown).
        """
        new_alerts: List[AlertRecord] = []

        # Group detections by class, keep only the highest confidence per class
        best_per_class: Dict[str, float] = {}
        for det in detections:
            class_name = det.class_name
            if class_name not in config.ALERT_CLASSES:
                continue
            if class_name not in best_per_class or det.confidence > best_per_class[class_name]:
                best_per_class[class_name] = det.confidence

        for detection_type, confidence in best_per_class.items():
            if self._is_on_cooldown(detection_type):
                continue

            self._last_alert_time[detection_type] = time.time()
            image_path = self._save_evidence(frame, detection_type)
            self._play_alarm()

            record = AlertRecord(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                detection_type=detection_type,
                confidence=round(confidence * 100, 2),
                source=source,
                alert_image_path=image_path or "",
            )
            self.history.append(record)
            new_alerts.append(record)

        if new_alerts:
            self._persist_history()

        return new_alerts
