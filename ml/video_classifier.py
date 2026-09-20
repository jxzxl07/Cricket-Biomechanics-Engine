"""Production inference adapters for cricket action classification."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import DATA_DIR

BATTING_CLASSES = [
    "cover_drive",
    "defence",
    "flick",
    "hook",
    "late_cut",
    "lofted_shot",
    "pull",
    "square_cut",
    "straight_drive",
    "sweep",
]

DISPLAY_LABELS = {
    "cover_drive": "Cover drive",
    "defence": "Defence",
    "flick": "Flick",
    "hook": "Hook",
    "late_cut": "Late cut",
    "lofted_shot": "Lofted shot",
    "pull": "Pull",
    "square_cut": "Square cut",
    "straight_drive": "Straight drive",
    "sweep": "Sweep",
    "left_arm_pace": "Left-arm pace",
    "right_arm_pace": "Right-arm pace",
    "left_arm_spin": "Left-arm spin",
    "right_arm_spin": "Right-arm spin",
    "unknown": "Unclear action",
}


def _resize_with_pad(frame: np.ndarray, size: int = 224) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def sample_video_frames(video_path: Path, count: int = 30) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError("The uploaded video contains no readable frames.")
    indices = np.linspace(0, max(total - 1, 0), count).astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        else:
            frames.append(_resize_with_pad(frame))
    capture.release()
    return np.expand_dims(np.stack(frames).astype(np.float32), axis=0)


class BattingVideoClassifier:
    model_id = "efficientnetb0-gru-cricshot10-v1"

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or DATA_DIR / "models" / "batting_video.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Batting model missing at {self.model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, video_path: Path) -> dict:
        probabilities = self.session.run(None, {self.input_name: sample_video_frames(video_path)})[0][0]
        ranked = sorted(
            zip(BATTING_CLASSES, probabilities.tolist()), key=lambda item: item[1], reverse=True
        )
        best_label, confidence = ranked[0]
        low_confidence = confidence < 0.55 or confidence - ranked[1][1] < 0.12
        display = "Unclear shot" if low_confidence else DISPLAY_LABELS[best_label]
        return {
            "label": best_label,
            "display_label": display,
            "confidence": float(confidence),
            "low_confidence": low_confidence,
            "probabilities": {label: float(value) for label, value in ranked},
            "model": {
                "id": self.model_id,
                "kind": "video_neural_network",
                "experimental": False,
                "note": "Confidence is the model score, not a guarantee of correctness.",
            },
        }


class BowlingPrototypeClassifier:
    """Transparent broad-family fallback until validated bowling weights are available.

    It deliberately avoids fine-grained spin labels. The score is capped because
    it is a pose prototype rather than a trained production neural network.
    """

    model_id = "pose-action-prototype-v1"

    def predict(self, features: dict) -> dict:
        arm = features.get("detected_bowling_arm") or "right"
        speed = float(features.get("peak_wrist_speed") or 0)
        duration = float(features.get("action_duration_seconds") or 1)
        vertical = float(features.get("arm_path_vertical_range") or 0)
        rotation = float(features.get("shoulder_rotation_range") or 0)
        pace_signal = 0.55 * _sigmoid((speed - 6.0) / 2.0)
        pace_signal += 0.25 * _sigmoid((vertical - 1.6) / 0.5)
        pace_signal += 0.20 * _sigmoid((0.9 - duration) / 0.25)
        spin_signal = 1 - pace_signal
        if rotation > 65:
            spin_signal += 0.08
        pace_probability = max(0.08, min(0.92, 1 - spin_signal / 1.08))
        family = "pace" if pace_probability >= 0.5 else "spin"
        raw_confidence = pace_probability if family == "pace" else 1 - pace_probability
        confidence = min(0.69, 0.5 + abs(raw_confidence - 0.5) * 0.38)
        label = f"{arm}_arm_{family}"
        counterpart = f"{arm}_arm_{'spin' if family == 'pace' else 'pace'}"
        return {
            "label": label,
            "display_label": DISPLAY_LABELS[label],
            "confidence": confidence,
            "low_confidence": True,
            "probabilities": {label: raw_confidence, counterpart: 1 - raw_confidence},
            "model": {
                "id": self.model_id,
                "kind": "transparent_pose_prototype",
                "experimental": True,
                "note": "Broad pace/spin estimate only. Treat it as experimental until a licensed bowling video model passes the evaluation gate.",
            },
        }


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, value))))
