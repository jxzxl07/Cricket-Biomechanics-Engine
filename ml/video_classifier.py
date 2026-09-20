"""Production inference adapters for cricket action classification.

Batting runs a converted temporal video network. Bowling uses a transparent
pose prototype until a licensed bowling video model passes evaluation. Both are
described by sidecar JSON specs so the preprocessing that produced the published
benchmark is exactly the preprocessing used at request time.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ml.model_spec import ModelSpec, load_spec

BATTING_SPEC_FILE = "batting_video.json"
BOWLING_SPEC_FILE = "bowling_prototype.json"


def _resize_with_pad(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinear resize into a zero-padded square canvas, matching the source pipeline."""
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    x, y = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized.astype(np.float32)
    return canvas[..., ::-1]  # OpenCV reads BGR; the model expects RGB.


def _frame_indices(total: int, count: int, window: tuple[int, int] | None) -> list[int]:
    if window is not None and window[1] > window[0] >= 0:
        first, last = window[0], min(window[1], total - 1)
    else:
        first, last = 0, total - 1
    return np.linspace(first, last, count).astype(int).tolist()


def sample_video_frames(
    video_path: Path,
    count: int = 30,
    size: int = 224,
    value_scale: float = 255.0,
    window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Sample ``count`` frames as an ``(1, count, size, size, 3)`` float batch.

    ``value_scale`` multiplies the 0-255 pixel values. The published model's
    graph contains ``Rescaling(1/255)`` and ImageNet normalisation, so it expects
    inputs in 0-255; the source demo incorrectly fed 0-1 frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError("The uploaded video contains no readable frames.")
    frames = []
    try:
        for index in _frame_indices(total, count, window):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((size, size, 3), dtype=np.uint8)
                frames.append(frame.astype(np.float32))
                continue
            frames.append(_resize_with_pad(frame, size) * value_scale)
    finally:
        capture.release()
    return np.expand_dims(np.stack(frames).astype(np.float32), axis=0)


class BattingVideoClassifier:
    """Temporal ONNX classifier for ten batting shot classes."""

    def __init__(self, spec: ModelSpec | None = None):
        self.spec = spec or load_spec(BATTING_SPEC_FILE)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.spec.artifact_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, video_path: Path, action_window: tuple[int, int] | None = None) -> dict:
        probabilities = self.session.run(
            None,
            {self.input_name: sample_video_frames(video_path, window=action_window)},
        )[0][0]
        return build_classification(probabilities, self.spec)


def build_classification(probabilities: np.ndarray, spec: ModelSpec) -> dict:
    """Turn a probability vector into the public classification contract."""
    classes = spec.classes
    if len(probabilities) != len(classes):
        raise ValueError(f"{spec.id} returned {len(probabilities)} scores for {len(classes)} classes")
    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    runner_up = ranked[1][1]
    confident = top_score >= spec.unknown_threshold and (top_score - runner_up) >= spec.unknown_margin
    display_label = spec.display_label(top_label)
    if not confident:
        top_label, display_label = "unknown", "Unclear action"
    shown_score = min(top_score, spec.confidence_cap) if spec.confidence_cap < 1 else top_score
    return {
        "label": top_label,
        "display_label": display_label,
        "confidence": float(shown_score),
        "raw_confidence": float(top_score),
        "unknown": not confident,
        "low_confidence": not confident,
        "top_alternatives": [
            {"label": name, "display_label": spec.display_label(name), "probability": float(score)}
            for name, score in ranked[1:4]
        ],
        "probabilities": {name: float(score) for name, score in ranked},
        "model": {
            "id": spec.id,
            "version": spec.version,
            "kind": spec.kind,
            "experimental": spec.experimental,
            "note": spec.note,
            "confidence_cap": spec.confidence_cap,
            "benchmark": spec.benchmark(),
            "limitations": spec.data.get("limitations", []),
        },
    }


class BowlingPrototypeClassifier:
    """Transparent broad-family fallback until validated bowling weights are available.

    It deliberately avoids fine-grained spin labels. The score is capped because
    it is a pose prototype rather than a trained production neural network, and
    it never makes a bowling-legality judgement.
    """

    def __init__(self, spec: ModelSpec | None = None):
        self.spec = spec or load_spec(BOWLING_SPEC_FILE)

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
        confidence = min(self.spec.confidence_cap, 0.5 + abs(pace_probability - 0.5) * 0.38)
        label = f"{arm}_arm_{family}"
        counterpart = f"{arm}_arm_{'spin' if family == 'pace' else 'pace'}"
        return {
            "label": label,
            "display_label": self.spec.display_label(label),
            "confidence": float(confidence),
            "raw_confidence": float(max(pace_probability, 1 - pace_probability)),
            "unknown": False,
            "low_confidence": True,
            "top_alternatives": [
                {"label": counterpart, "display_label": self.spec.display_label(counterpart), "probability": float(1 - pace_probability)}
            ],
            "probabilities": {label: float(pace_probability), counterpart: float(1 - pace_probability)},
            "model": {
                "id": self.spec.id,
                "version": self.spec.version,
                "kind": self.spec.kind,
                "experimental": True,
                "note": self.spec.note,
                "confidence_cap": self.spec.confidence_cap,
                "benchmark": {},
                "limitations": self.spec.data.get("limitations", []),
            },
        }


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, value))))
