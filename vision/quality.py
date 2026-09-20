"""Capture-quality gate for uploaded clips.

The gate explains *why* a clip cannot be trusted instead of returning a label
built on missing data. Every check is derived from the pose extraction output,
so it works for both batting and bowling and for both camera angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_DETECTION_RATIO = 0.6
MIN_KEY_JOINT_RATIO = 0.5
MIN_RESOLUTION_SHORT_SIDE = 360
MIN_DURATION_SECONDS = 0.8
MAX_DURATION_SECONDS = 12.0
FRAME_EDGE_MARGIN = 0.02

KEY_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee")
FULL_BODY_JOINTS = ("nose", "left_ankle", "right_ankle", "left_wrist", "right_wrist")


@dataclass
class QualityCheck:
    id: str
    label: str
    passed: bool
    detail: str
    blocking: bool = False


@dataclass
class QualityReport:
    score: int
    pose_coverage: float
    checks: list[QualityCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "score": self.score,
            "pose_coverage": round(self.pose_coverage, 3),
            "checks": [
                {"id": check.id, "label": check.label, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ],
            "warnings": self.warnings,
            "notes": self.notes,
        }


def _landmark_names(frame: dict) -> set[str]:
    return {landmark["name"] for landmark in frame.get("landmarks", []) if landmark.get("visibility", 0) >= 0.4}


def _frames(data: dict) -> list[dict]:
    return data.get("frames", [])


def evaluate_quality(
    data: dict,
    features: dict,
    camera_angle: str,
    mode: str,
) -> QualityReport:
    frames = _frames(data)
    detected = [frame for frame in frames if frame.get("person_detected") and frame.get("landmarks")]
    total = len(frames) or 1
    coverage = len(detected) / total
    fps = float(data.get("fps") or 30)
    duration = float(data.get("total_frames") or total) / fps if fps else 0.0

    checks: list[QualityCheck] = []

    checks.append(QualityCheck(
        id="person_visible",
        label="Athlete visible",
        passed=coverage >= MIN_DETECTION_RATIO,
        blocking=coverage < MIN_DETECTION_RATIO,
        detail=f"Pose detected in {coverage * 100:.0f}% of frames (needs {MIN_DETECTION_RATIO * 100:.0f}%).",
    ))

    second_pose_ratio = sum(1 for frame in frames if frame.get("pose_count", 0) > 1) / total
    checks.append(QualityCheck(
        id="single_athlete",
        label="Single athlete in frame",
        passed=second_pose_ratio < 0.2,
        detail=f"A second person was tracked in {second_pose_ratio * 100:.0f}% of frames.",
    ))

    key_joint_frames = 0
    for frame in frames:
        if _landmark_names(frame).issuperset(KEY_JOINTS):
            key_joint_frames += 1
    key_joint_ratio = key_joint_frames / total
    checks.append(QualityCheck(
        id="key_joints",
        label="Shoulders, hips and knees visible",
        passed=key_joint_ratio >= MIN_KEY_JOINT_RATIO,
        blocking=key_joint_ratio < MIN_KEY_JOINT_RATIO,
        detail=f"Tracked in {key_joint_ratio * 100:.0f}% of frames.",
    ))

    inside = 0
    tracked_full_body = 0
    for frame in detected:
        landmarks = [landmark for landmark in frame["landmarks"] if landmark.get("visibility", 0) >= 0.4]
        if not landmarks:
            continue
        tracked_full_body += 1
        if all(
            FRAME_EDGE_MARGIN <= landmark["x"] <= 1 - FRAME_EDGE_MARGIN
            and FRAME_EDGE_MARGIN <= landmark["y"] <= 1 - FRAME_EDGE_MARGIN
            for landmark in landmarks
        ):
            inside += 1
    framing_ratio = inside / max(tracked_full_body, 1)
    checks.append(QualityCheck(
        id="full_body_framed",
        label="Full body inside the frame",
        passed=framing_ratio >= 0.6,
        detail=f"No landmark touched the frame edge in {framing_ratio * 100:.0f}% of tracked frames.",
    ))

    motion = features.get("action_detected", False)
    checks.append(QualityCheck(
        id="action_detected",
        label="A clear movement was detected",
        passed=bool(motion),
        blocking=not motion,
        detail=str(features.get("action_status") or "Movement not measured."),
    ))

    short_side = min(int(data.get("frame_width") or 0), int(data.get("frame_height") or 0))
    checks.append(QualityCheck(
        id="resolution",
        label="Resolution",
        passed=short_side >= MIN_RESOLUTION_SHORT_SIDE,
        detail=f"Short side is {short_side}px (needs {MIN_RESOLUTION_SHORT_SIDE}px).",
    ))

    checks.append(QualityCheck(
        id="duration",
        label="Clip length",
        passed=MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS,
        detail=f"{duration:.1f}s (supported range {MIN_DURATION_SECONDS:.1f}-{MAX_DURATION_SECONDS:.0f}s).",
    ))

    weights = {
        "person_visible": 28,
        "key_joints": 18,
        "full_body_framed": 14,
        "action_detected": 24,
        "single_athlete": 6,
        "resolution": 5,
        "duration": 5,
    }
    earned = sum(weights.get(check.id, 0) for check in checks if check.passed)
    score = int(round(earned / sum(weights.values()) * 100))

    warnings = [check.detail for check in checks if check.blocking]
    if second_pose_ratio >= 0.2:
        warnings.append("More than one person was in frame, so the tracked athlete may not be you.")
    notes: list[str] = []
    if camera_angle == "unknown":
        notes.append("Choose the closest camera angle next time so side-on-only metrics can be used.")
    if mode == "bowling" and camera_angle == "front_on":
        notes.append("Front-on bowling clips give a weaker read of release reach and arm path than side-on.")
    if not warnings:
        notes.append("Capture quality is strong enough for movement feedback.")

    return QualityReport(score=score, pose_coverage=coverage, checks=checks, warnings=warnings, notes=notes)
