"""Research-only adapter for the published broadcast batting video model.

Production classification lives in ``ml/pose_classifier.py``. This model is kept
because the evaluation scripts in ``research/evaluation`` compare against it; it
is not loaded by the API.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ml.classification import build_classification
from ml.model_spec import ModelSpec, load_spec

# Research-only comparison model. Production uses ml/pose_classifier.py.
VIDEO_SPEC_FILE = "batting_video.json"


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
        self.spec = spec or load_spec(VIDEO_SPEC_FILE)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's CPU arena grows to hundreds of MB and never returns it. Measured
        # inside the deploy container: peak 809 MB with the arena versus 497 MB
        # without, for about 2% extra latency. Worth it for a 512 MB-2 GB host.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
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


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, value))))
