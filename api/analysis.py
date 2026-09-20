"""Orchestrates video classification, pose metrics, replay timeline, and coaching.

The public response is versioned by ``ANALYSIS_SCHEMA_VERSION``. Every metric
carries the frame it was measured on so the replay can seek to the evidence.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from api.coaching import enhanced_coach, rules_coach
from ml.model_spec import ModelSpec, load_spec
from ml.video_classifier import (
    BATTING_SPEC_FILE,
    BOWLING_SPEC_FILE,
    BattingVideoClassifier,
    BowlingPrototypeClassifier,
)
from vision.features import extract_features_from_data, load_landmark_json
from vision.pose import extract_landmarks_from_video
from vision.quality import evaluate_quality

ANALYSIS_SCHEMA_VERSION = "1.0.0"

UNKNOWN_CLASSIFICATION = {
    "label": "unknown",
    "display_label": "Unclear action",
    "confidence": 0.0,
    "raw_confidence": 0.0,
    "unknown": True,
    "low_confidence": True,
    "top_alternatives": [],
    "probabilities": {},
}


class AnalysisEngine:
    def __init__(self):
        self.batting_spec: ModelSpec = load_spec(BATTING_SPEC_FILE)
        self.bowling_spec: ModelSpec = load_spec(BOWLING_SPEC_FILE)
        self.batting = BattingVideoClassifier(self.batting_spec)
        self.bowling = BowlingPrototypeClassifier(self.bowling_spec)

    def model_cards(self) -> dict:
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "batting": self.batting_spec.public(),
            "bowling": self.bowling_spec.public(),
            "pose": {
                "id": "mediapipe-pose-landmarker-lite",
                "kind": "pose_estimation",
                "classes": [],
                "experimental": False,
                "note": "33 2D landmarks with estimated depth. Used for metrics and the replay overlay.",
            },
        }

    def analyze(
        self,
        video_path: Path,
        mode: str,
        camera_angle: str,
        use_ai_coach: bool,
        deadline: float | None = None,
    ) -> dict:
        started = time.perf_counter()
        landmarks_path = video_path.with_suffix(".landmarks.json")

        pose_started = time.perf_counter()
        extract_landmarks_from_video(video_path, mode, output_path=landmarks_path, deadline=deadline)
        pose_ms = (time.perf_counter() - pose_started) * 1000

        data = load_landmark_json(landmarks_path)
        features_started = time.perf_counter()
        features = extract_features_from_data(data, mode)
        features_ms = (time.perf_counter() - features_started) * 1000

        fps = float(data.get("fps") or 30)
        total_frames = int(data.get("total_frames") or 0)
        quality = evaluate_quality(data, features, camera_angle, mode)

        classification_started = time.perf_counter()
        if not features.get("action_detected"):
            classification = dict(UNKNOWN_CLASSIFICATION)
            classification["model"] = {
                "id": "pose-action-gate",
                "version": ANALYSIS_SCHEMA_VERSION,
                "kind": "quality_gate",
                "experimental": False,
                "note": "Classification was skipped because the action quality gate did not pass.",
                "benchmark": {},
                "limitations": [],
            }
            status = "needs_better_clip"
        else:
            classification = self._classify(video_path, mode, features)
            status = "complete"
        classification_ms = (time.perf_counter() - classification_started) * 1000

        phases = _build_phases(features, mode, fps, total_frames)
        phase_frames = [phase["frame"] for phase in phases]

        coaching_started = time.perf_counter()
        coach = (
            enhanced_coach(video_path, mode, camera_angle, classification, features, phase_frames)
            if use_ai_coach
            else rules_coach(mode, classification["display_label"], features)
        )
        coaching_ms = (time.perf_counter() - coaching_started) * 1000

        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": status,
            "mode": mode,
            "camera_angle": camera_angle,
            "video": {
                "width": int(data.get("frame_width") or 0),
                "height": int(data.get("frame_height") or 0),
                "fps": round(fps, 2),
                "frames": total_frames,
                "duration_ms": round(total_frames / fps * 1000) if fps else 0,
                "orientation": "portrait"
                if int(data.get("frame_height") or 0) >= int(data.get("frame_width") or 0)
                else "landscape",
            },
            "classification": classification,
            "metrics": _metric_cards(features, mode, fps),
            "features": _public_features(features),
            "phases": phases,
            "timeline": _compact_timeline(data),
            "quality": quality.public(),
            "warnings": _warnings(classification, quality, features),
            "coach": coach,
            "timings": {
                "pose_ms": round(pose_ms),
                "features_ms": round(features_ms),
                "classification_ms": round(classification_ms),
                "coaching_ms": round(coaching_ms),
                "total_ms": round((time.perf_counter() - started) * 1000),
            },
            "privacy": "The uploaded clip was processed in a temporary directory and is not retained by the API.",
        }

    def _classify(self, video_path: Path, mode: str, features: dict) -> dict:
        if mode == "batting":
            window = (int(features["action_start_frame"]), int(features["action_end_frame"]))
            return self.batting.predict(video_path, action_window=window)
        return self.bowling.predict(features)


def _warnings(classification: dict, quality, features: dict) -> list[str]:
    warnings = list(quality.warnings)
    model = classification.get("model", {})
    if model.get("experimental"):
        benchmark = model.get("benchmark", {})
        accuracy = benchmark.get("top1_accuracy")
        if isinstance(accuracy, (int, float)):
            warnings.append(
                f"Experimental model: measured {accuracy * 100:.0f}% top-1 on this project's own phone clips."
            )
        else:
            warnings.append("Experimental model: no independent accuracy measurement exists for this mode.")
    if classification.get("unknown") and features.get("action_detected"):
        warnings.append("The model was not confident enough to name a shot, so no label is shown.")
    return warnings


def _time(frame: int | None, fps: float) -> int | None:
    return None if frame is None else round(frame / fps * 1000)


def _build_phases(features: dict, mode: str, fps: float, total_frames: int) -> list[dict]:
    last = max(0, total_frames - 1)
    start = int(features.get("action_start_frame") or 0)
    end = int(features.get("action_end_frame") or last)
    if mode == "batting":
        key = int(features.get("peak_speed_frame") or min(last, (start + end) // 2))
        acceleration = int(features.get("downswing_start_frame") or start)
        specs = [
            ("setup", "Setup", start),
            ("downswing", "Downswing", acceleration),
            ("contact", "Approximate contact", key),
            ("follow_through", "Follow-through", end),
        ]
    else:
        key = int(features.get("release_frame") or min(last, (start + end) // 2))
        acceleration = int(features.get("delivery_stride_frame") or start)
        specs = [
            ("gather", "Gather", start),
            ("delivery_stride", "Delivery stride", acceleration),
            ("release", "Approximate release", key),
            ("follow_through", "Follow-through", end),
        ]
    return [
        {"id": phase_id, "label": label, "frame": frame, "timestamp_ms": _time(frame, fps)}
        for phase_id, label, frame in specs
    ]


def _metric(key, label, value, unit, description, frame, fps):
    return {
        "key": key,
        "label": label,
        "value": None if value is None else round(float(value), 2),
        "unit": unit,
        "description": description,
        "frame": frame,
        "timestamp_ms": _time(frame, fps),
    }


def _metric_cards(features: dict, mode: str, fps: float) -> list[dict]:
    if mode == "batting":
        frame = features.get("peak_speed_frame")
        specs = [
            ("peak_hand_speed", "Peak hand speed", "body lengths/s", "2D speed normalised to torso size"),
            ("peak_speed_timing", "Peak timing", "of swing", "Where peak speed occurs from setup to follow-through"),
            ("shoulder_rotation_range", "Shoulder rotation", "°", "Estimated shoulder-line rotation across the action"),
            ("max_knee_bend", "Max knee flexion", "°", "Largest estimated knee flexion"),
            ("head_drop", "Head movement", "× torso", "Vertical head drop from the start of the action"),
            ("final_hand_height", "Finish height", "× torso", "Hands above the hip line at the finish"),
        ]
    else:
        frame = features.get("release_frame")
        specs = [
            ("peak_wrist_speed", "Peak wrist speed", "body lengths/s", "2D speed normalised to torso size"),
            ("release_height", "Release height", "× torso", "Wrist height above the hip line at release"),
            ("release_forward_reach", "Release reach", "× shoulders", "Horizontal wrist reach from the bowling shoulder"),
            ("shoulder_rotation_range", "Shoulder rotation", "°", "Estimated shoulder-line rotation across the action"),
            ("torso_lean_at_release", "Release lean", "°", "Torso lean from vertical at release"),
            ("elbow_angle_at_release", "Bowling-arm elbow angle", "°", "Estimated joint angle at release. A coaching observation, not a legality assessment."),
        ]
    return [_metric(key, label, features.get(key), unit, description, frame, fps) for key, label, unit, description in specs]


def _public_features(features: dict) -> dict:
    return {key: value for key, value in features.items() if key != "person_detection"}


def _compact_timeline(data: dict, max_frames: int = 180) -> dict:
    frames = data.get("frames", [])
    step = max(1, math.ceil(len(frames) / max_frames))
    compact = []
    for frame in frames[::step]:
        compact.append({
            "frame": frame["frame_index"],
            "timestamp_ms": frame["timestamp_ms"],
            "landmarks": [
                [point["index"], round(point["x"], 5), round(point["y"], 5), round(point["visibility"], 3)]
                for point in frame.get("landmarks", [])
            ],
        })
    return {
        "fps": data.get("fps"),
        "width": data.get("frame_width"),
        "height": data.get("frame_height"),
        "total_frames": data.get("total_frames"),
        "sample_step": step,
        "frames": compact,
    }
