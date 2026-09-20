"""Probe which frame preprocessing the batting weights were actually trained on.

The ONNX graph starts with ``Rescaling(1/255)`` and ImageNet normalization, so the
"intended" input is [0, 255]. The published Streamlit demo, however, feeds
``tf.image.resize_with_pad`` output, which is float [0, 1]. Only a measurement on
real clips can say which one produced the weights we have.

Run: python -m research.evaluation.batting_preprocessing_probe
"""

from __future__ import annotations

import itertools
import statistics
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import DATA_DIR
from research.evaluation.batting_benchmark import COMPATIBLE_LABELS, discover_clips, session_key  # noqa: F401

BATTING_CLASSES = [
    "cover_drive", "defence", "flick", "hook", "late_cut",
    "lofted_shot", "pull", "square_cut", "straight_drive", "sweep",
]

MODEL_PATH = DATA_DIR / "models" / "batting_video.onnx"


def _resize_with_pad(frame: np.ndarray, size: int = 224) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    x, y = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized.astype(np.float32)
    return canvas


def _read_indices(capture: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        frames.append(frame if ok else np.zeros((224, 224, 3), dtype=np.uint8))
    return frames


def build_batch(video_path: Path, strategy: str, scale: float, count: int = 30) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"no frames in {video_path}")
    try:
        if strategy == "uniform":
            indices = np.linspace(0, total - 1, count).astype(int).tolist()
        elif strategy == "head":
            indices = list(range(min(count, total)))
        elif strategy == "middle":
            start = max(0, (total - count) // 2)
            indices = list(range(start, min(start + count, total)))
        else:
            raise ValueError(strategy)
        raw = _read_indices(capture, indices)
    finally:
        capture.release()
    # Source app converts BGR -> RGB after padding.
    batch = np.stack([
        _resize_with_pad(frame)[..., ::-1] * scale for frame in raw
    ]).astype(np.float32)
    return np.expand_dims(batch, axis=0)


def main() -> None:
    clips = discover_clips(DATA_DIR / "raw" / "batting")
    supported = [(path, label) for path, label in clips if label in COMPATIBLE_LABELS]
    print(f"{len(supported)} supported clips, {len(clips)} total\n")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rows = []
    for strategy, scale in itertools.product(["uniform", "head", "middle"], [255.0, 1.0]):
        correct = 0
        top2 = 0
        predictions: dict[str, int] = {}
        confidences: list[float] = []
        for path, label in supported:
            try:
                batch = build_batch(path, strategy, scale)
            except ValueError as error:
                print("  skip", path.name, error)
                continue
            probabilities = session.run(None, {input_name: batch})[0][0]
            ranked = sorted(zip(BATTING_CLASSES, probabilities.tolist()), key=lambda item: item[1], reverse=True)
            predictions[ranked[0][0]] = predictions.get(ranked[0][0], 0) + 1
            confidences.append(ranked[0][1])
            correct += int(ranked[0][0] in COMPATIBLE_LABELS[label])
            top2 += int(any(name in COMPATIBLE_LABELS[label] for name, _ in ranked[:2]))
        rows.append((strategy, scale, correct / len(supported), top2 / len(supported),
                     statistics.mean(confidences), max(confidences), predictions))
        print(
            f"strategy={strategy:<8} scale={scale:<6} top1={correct / len(supported):.3f} "
            f"top2={top2 / len(supported):.3f} mean_conf={statistics.mean(confidences):.3f} "
            f"distinct_preds={len(predictions)} top={sorted(predictions.items(), key=lambda i: -i[1])[:5]}"
        )

    print("\n=== ranked ===")
    for strategy, scale, top1, top2, mean_conf, max_conf, _ in sorted(rows, key=lambda r: -r[2]):
        print(f"{strategy:<8} scale={scale:<6} top1={top1:.3f} top2={top2:.3f} mean_conf={mean_conf:.3f}")


if __name__ == "__main__":
    main()
