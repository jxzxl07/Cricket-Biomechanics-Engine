"""Orchestrates video classification, pose metrics, replay timeline, and coaching."""

from __future__ import annotations

import math
from pathlib import Path

from api.coaching import enhanced_coach, rules_coach
from ml.video_classifier import BattingVideoClassifier, BowlingPrototypeClassifier
from vision.features import extract_features_from_data, load_landmark_json
from vision.pose import extract_landmarks_from_video


class AnalysisEngine:
    def __init__(self):
        self.batting = BattingVideoClassifier()
        self.bowling = BowlingPrototypeClassifier()

    def analyze(self, video_path: Path, mode: str, camera_angle: str, use_ai_coach: bool) -> dict:
        landmarks_path = video_path.with_suffix(".landmarks.json")
        extract_landmarks_from_video(video_path, mode, output_path=landmarks_path)
        data = load_landmark_json(landmarks_path)
        features = extract_features_from_data(data, mode)
        fps = float(data.get("fps") or 30)
        phases = _build_phases(features, mode, fps, int(data.get("total_frames") or 0))

        if not features.get("action_detected"):
            classification = {
                "label": "unknown",
                "display_label": "No clear action",
                "confidence": 0.0,
                "low_confidence": True,
                "probabilities": {},
                "model": {"id": "quality-gate", "kind": "pose_action_gate", "experimental": False, "note": "Classification was skipped because the action quality gate did not pass."},
            }
            status = "needs_better_clip"
        else:
            classification = self.batting.predict(video_path) if mode == "batting" else self.bowling.predict(features)
            status = "complete"

        phase_frames = [phase["frame"] for phase in phases]
        coach = enhanced_coach(video_path, mode, camera_angle, classification, features, phase_frames) if use_ai_coach else rules_coach(mode, classification["display_label"], features)
        return {
            "status": status,
            "mode": mode,
            "camera_angle": camera_angle,
            "classification": classification,
            "metrics": _metric_cards(features, mode, fps),
            "features": _public_features(features),
            "phases": phases,
            "timeline": _compact_timeline(data),
            "quality": _quality(features, camera_angle),
            "coach": coach,
            "privacy": "The uploaded clip was processed in a temporary directory and is not retained by the API.",
        }


def _time(frame: int | None, fps: float) -> int | None:
    return None if frame is None else round(frame / fps * 1000)


def _build_phases(features: dict, mode: str, fps: float, total_frames: int) -> list[dict]:
    last = max(0, total_frames - 1)
    start = int(features.get("action_start_frame") or 0)
    middle_key = "peak_speed_frame" if mode == "batting" else "release_frame"
    middle = int(features.get(middle_key) or min(last, total_frames // 2))
    end = int(features.get("action_end_frame") or last)
    labels = ("Setup", "Peak hand speed", "Follow-through") if mode == "batting" else ("Gather", "Release", "Follow-through")
    return [
        {"id": "start", "label": labels[0], "frame": start, "timestamp_ms": _time(start, fps)},
        {"id": "key", "label": labels[1], "frame": middle, "timestamp_ms": _time(middle, fps)},
        {"id": "finish", "label": labels[2], "frame": end, "timestamp_ms": _time(end, fps)},
    ]


def _metric(key, label, value, unit, description, frame, fps):
    return {"key": key, "label": label, "value": None if value is None else round(float(value), 2), "unit": unit, "description": description, "timestamp_ms": _time(frame, fps)}


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
            ("release_height", "Release height", "× torso", "Wrist height above the hip line"),
            ("release_forward_reach", "Release reach", "× shoulders", "Horizontal wrist reach from bowling shoulder"),
            ("shoulder_rotation_range", "Shoulder rotation", "°", "Estimated shoulder-line rotation"),
            ("torso_lean_at_release", "Release lean", "°", "Torso lean from vertical at release"),
            ("action_duration_seconds", "Action window", "s", "Duration of the detected high-motion window"),
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


def _quality(features: dict, camera_angle: str) -> dict:
    person = features.get("person_detection", {})
    ratio = float(person.get("detection_ratio") or 0)
    score = round(min(100, ratio * 80 + (20 if features.get("action_detected") else 0)))
    notes = []
    if ratio < 0.8:
        notes.append("Keep the full body visible throughout the clip.")
    if camera_angle == "unknown":
        notes.append("Choose the closest camera angle next time for more useful coaching context.")
    if not notes:
        notes.append("Pose coverage is strong enough for movement feedback.")
    return {"score": score, "pose_coverage": ratio, "notes": notes}
