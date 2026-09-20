"""Test zero-training domain adaptation for the published batting model.

Both the model and the historical clips are fixed. We vary only how the frames
are chosen and framed:

* ``crop``: none, or a padded bounding box around the visible pose landmarks.
  Broadcast training footage is tightly framed on the batsman; phone clips
  often have a lot of room around the subject.
* ``window``: sample the whole clip, or only the detected batting action window.
* ``flip``: average probabilities with a horizontally mirrored input. The
  source project trained with horizontal flips.

Run: python -m research.evaluation.batting_domain_probe
"""

from __future__ import annotations

import itertools
import statistics
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import DATA_DIR
from research.evaluation.batting_benchmark import COMPATIBLE_LABELS, discover_clips
from vision.features import extract_batting_features_from_data, load_landmark_json

BATTING_CLASSES = [
    "cover_drive", "defence", "flick", "hook", "late_cut",
    "lofted_shot", "pull", "square_cut", "straight_drive", "sweep",
]

MODEL_PATH = DATA_DIR / "models" / "batting_video.onnx"
LANDMARKS_DIR = DATA_DIR / "landmarks" / "batting"
VISIBILITY = 0.35
PAD_RATIO = 0.25


def pose_bounding_box(landmarks: list[dict], frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    visible = [point for point in landmarks if point["visibility"] >= VISIBILITY]
    if len(visible) < 8:
        return None
    xs = [point["x"] for point in visible]
    ys = [point["y"] for point in visible]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    pad_x, pad_y = width * PAD_RATIO, height * PAD_RATIO
    x0 = max(0, int((min(xs) - pad_x) * frame_width))
    y0 = max(0, int((min(ys) - pad_y) * frame_height))
    x1 = min(frame_width, int((max(xs) + pad_x) * frame_width))
    y1 = min(frame_height, int((max(ys) + pad_y) * frame_height))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


def _resize_with_pad(frame: np.ndarray, size: int = 224) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    x, y = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized.astype(np.float32)
    return canvas


def build_batch(
    video_path: Path,
    landmark_path: Path | None,
    *,
    crop: bool,
    window: str,
    count: int = 30,
) -> np.ndarray:
    frame_by_index: dict[int, tuple[int, int, int, int] | None] = {}
    if landmark_path is not None and landmark_path.exists():
        data = load_landmark_json(landmark_path)
        for frame in data.get("frames", []):
            frame_by_index[frame["frame_index"]] = (
                pose_bounding_box(frame.get("landmarks") or [], data.get("frame_width") or 1, data.get("frame_height") or 1)
                if crop else None
            )
        features = extract_batting_features_from_data(data)
        start = int(features.get("action_start_frame") or 0)
        end = int(features.get("action_end_frame") or 0)
    else:
        start = end = 0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"no frames in {video_path}")
    try:
        if window == "action" and end > start:
            indices = np.linspace(start, end, count).astype(int).tolist()
        else:
            indices = np.linspace(0, total - 1, count).astype(int).tolist()
        raw_frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frame_by_index.setdefault(int(index), None)
            box = frame_by_index.get(int(index))
            if crop and box is not None:
                x0, y0, x1, y1 = box
                frame = frame[y0:y1, x0:x1]
            raw_frames.append(frame)
    finally:
        capture.release()

    batch = np.stack([_resize_with_pad(frame)[..., ::-1] * 255.0 for frame in raw_frames]).astype(np.float32)
    return np.expand_dims(batch, axis=0)


def main() -> None:
    clips = discover_clips(DATA_DIR / "raw" / "batting")
    supported = []
    for path, label in clips:
        if label not in COMPATIBLE_LABELS:
            continue
        landmark_path = LANDMARKS_DIR / label / f"{path.stem}.json"
        supported.append((path, label, landmark_path))
    print(f"{len(supported)} supported clips with landmarks\n")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rows = []
    for crop, window, flip in itertools.product([False, True], ["full", "action"], [False, True]):
        correct = top2 = 0
        predictions: dict[str, int] = {}
        confidences: list[float] = []
        for path, label, landmark_path in supported:
            probabilities = session.run(
                None,
                {input_name: build_batch(path, landmark_path, crop=crop, window=window)},
            )[0][0]
            if flip:
                flipped = build_batch(path, landmark_path, crop=crop, window=window)[..., ::-1, :]
                probabilities = (probabilities + session.run(None, {input_name: flipped})[0][0]) / 2
            ranked = sorted(zip(BATTING_CLASSES, probabilities.tolist()), key=lambda item: item[1], reverse=True)
            predictions[ranked[0][0]] = predictions.get(ranked[0][0], 0) + 1
            confidences.append(ranked[0][1])
            correct += int(ranked[0][0] in COMPATIBLE_LABELS[label])
            top2 += int(any(name in COMPATIBLE_LABELS[label] for name, _ in ranked[:2]))
        rows.append({
            "crop": crop, "window": window, "flip": flip,
            "top1": correct / len(supported), "top2": top2 / len(supported),
            "mean_conf": statistics.mean(confidences), "distinct": len(predictions),
            "top_predictions": sorted(predictions.items(), key=lambda item: -item[1])[:4],
        })
        print(rows[-1], flush=True)

    print("\n=== ranked by top-1 ===")
    for row in sorted(rows, key=lambda item: -item["top1"]):
        print(f"crop={row['crop']:<5} window={row['window']:<6} flip={row['flip']:<5} "
              f"top1={row['top1']:.3f} top2={row['top2']:.3f} mean_conf={row['mean_conf']:.3f} "
              f"distinct={row['distinct']} {row['top_predictions']}")


if __name__ == "__main__":
    main()
