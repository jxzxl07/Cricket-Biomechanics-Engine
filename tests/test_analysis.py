"""Unit tests for pose features, the quality gate, phases, and coaching guards."""

from __future__ import annotations

import numpy as np
import pytest

from api.analysis import _build_phases, _compact_timeline, _metric_cards
from api.coaching import CoachResponse, Improvement, find_unsupported_claims, rules_coach, validate_coach_response
from vision.features import extract_batting_features_from_data, extract_bowling_features_from_data
from vision.quality import evaluate_quality


def _landmarks(**overrides) -> list[dict]:
    names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    ]
    defaults = {
        "nose": (0.50, 0.20), "left_shoulder": (0.45, 0.30), "right_shoulder": (0.55, 0.30),
        "left_elbow": (0.42, 0.40), "right_elbow": (0.58, 0.40),
        "left_wrist": (0.40, 0.50), "right_wrist": (0.60, 0.50),
        "left_hip": (0.46, 0.52), "right_hip": (0.54, 0.52),
        "left_knee": (0.46, 0.70), "right_knee": (0.54, 0.70),
        "left_ankle": (0.46, 0.88), "right_ankle": (0.54, 0.88),
    }
    points = []
    for index, name in enumerate(names):
        x, y = overrides.get(name, defaults[name])
        points.append({"index": index, "name": name, "x": x, "y": y, "z": 0.0, "visibility": 0.9, "presence": 0.9})
    return points


def _moving_hand_data(frames: int = 40, hand_shift: float = 0.02, mode: str = "batting") -> dict:
    data = {
        "fps": 30.0,
        "analysis_fps": 30.0,
        "total_frames": frames,
        "analysed_frames": frames,
        "frame_width": 720,
        "frame_height": 1280,
        "max_poses_seen": 1,
        "frames": [],
    }
    for index in range(frames):
        shift = index * hand_shift
        landmarks = _landmarks()
        for landmark in landmarks:
            if landmark["name"] == "left_wrist":
                landmark["x"] = min(0.95, 0.40 + shift)
            if landmark["name"] == "right_wrist":
                landmark["x"] = min(0.95, 0.60 + shift)
        data["frames"].append(
            {"frame_index": index, "timestamp_ms": int(index / 30 * 1000), "person_detected": True, "pose_count": 1, "landmarks": landmarks}
        )
    return data


def _empty_data(frames: int = 20) -> dict:
    return {
        "fps": 30.0,
        "analysis_fps": 30.0,
        "total_frames": frames,
        "analysed_frames": frames,
        "frame_width": 720,
        "frame_height": 1280,
        "max_poses_seen": 0,
        "frames": [
            {"frame_index": index, "timestamp_ms": int(index / 30 * 1000), "person_detected": False, "pose_count": 0, "landmarks": []}
            for index in range(frames)
        ],
    }


# --------------------------------------------------------------------------- features


def test_batting_features_detect_motion():
    features = extract_batting_features_from_data(_moving_hand_data())
    assert features["action_detected"] is True
    assert features["peak_hand_speed"] > 0
    assert features["action_start_frame"] <= features["peak_speed_frame"] <= features["action_end_frame"]
    assert features["downswing_start_frame"] <= features["peak_speed_frame"]


def test_batting_features_refuse_without_a_person():
    features = extract_batting_features_from_data(_empty_data())
    assert features["action_detected"] is False
    assert features["action_status"] == "No person detected during clip"


def test_bowling_features_detect_motion_and_name_an_arm():
    features = extract_bowling_features_from_data(_moving_hand_data())
    assert features["detected_bowling_arm"] in {"left", "right"}
    assert features["elbow_angle_at_release"] is None or 0 <= features["elbow_angle_at_release"] <= 180
    assert features["delivery_stride_frame"] <= features["release_frame"]


def test_unknown_mode_is_rejected():
    from vision.features import extract_features_from_data

    with pytest.raises(ValueError):
        extract_features_from_data(_moving_hand_data(), "hockey")


# --------------------------------------------------------------------------- quality gate


def test_quality_gate_passes_on_clean_pose_data():
    data = _moving_hand_data()
    features = extract_batting_features_from_data(data)
    report = evaluate_quality(data, features, "side_on", "batting")
    assert report.score > 80
    assert report.warnings == []
    assert all(check.passed for check in report.checks)


def test_quality_gate_blocks_when_nobody_is_visible():
    data = _empty_data()
    features = extract_batting_features_from_data(data)
    report = evaluate_quality(data, features, "side_on", "batting")
    assert report.score < 50
    assert report.warnings
    failed = {check.id for check in report.checks if not check.passed}
    assert {"person_visible", "action_detected"} <= failed


def test_quality_gate_flags_low_resolution():
    data = _moving_hand_data()
    data["frame_width"], data["frame_height"] = 160, 120
    features = extract_batting_features_from_data(data)
    report = evaluate_quality(data, features, "side_on", "batting")
    resolution = next(check for check in report.checks if check.id == "resolution")
    assert resolution.passed is False


def test_quality_gate_flags_a_second_person():
    data = _moving_hand_data()
    for frame in data["frames"]:
        frame["pose_count"] = 2
    features = extract_batting_features_from_data(data)
    report = evaluate_quality(data, features, "side_on", "batting")
    assert any("person" in warning for warning in report.warnings)


def test_quality_gate_notes_unknown_camera_angle():
    data = _moving_hand_data()
    features = extract_batting_features_from_data(data)
    report = evaluate_quality(data, features, "unknown", "batting")
    assert any("camera angle" in note for note in report.notes)


def test_quality_public_payload_shape():
    data = _moving_hand_data()
    features = extract_batting_features_from_data(data)
    payload = evaluate_quality(data, features, "side_on", "batting").public()
    assert set(payload) == {"score", "pose_coverage", "checks", "warnings", "notes"}
    assert all(set(check) == {"id", "label", "passed", "detail"} for check in payload["checks"])


# --------------------------------------------------------------------------- phases, metrics, timeline


def test_batting_phases_are_ordered_and_labelled():
    features = extract_batting_features_from_data(_moving_hand_data())
    phases = _build_phases(features, "batting", 30.0, 40)
    assert [phase["id"] for phase in phases] == ["setup", "downswing", "contact", "follow_through"]
    frames = [phase["frame"] for phase in phases]
    assert frames == sorted(frames)
    assert phases[0]["timestamp_ms"] == 0 or phases[0]["timestamp_ms"] > 0


def test_bowling_phases_are_ordered_and_labelled():
    features = extract_bowling_features_from_data(_moving_hand_data())
    phases = _build_phases(features, "bowling", 30.0, 40)
    assert [phase["id"] for phase in phases] == ["gather", "delivery_stride", "release", "follow_through"]
    assert [phase["frame"] for phase in phases] == sorted(phase["frame"] for phase in phases)


def test_phases_survive_missing_frames():
    phases = _build_phases({}, "batting", 30.0, 0)
    assert len(phases) == 4
    assert all(phase["frame"] == 0 for phase in phases)


def test_metric_cards_reference_a_frame():
    features = extract_batting_features_from_data(_moving_hand_data())
    cards = _metric_cards(features, "batting", 30.0)
    assert len(cards) == 6
    for card in cards:
        assert {"key", "label", "value", "unit", "description", "frame", "timestamp_ms"} == set(card)


def test_bowling_metric_cards_include_elbow_angle_without_legality():
    features = extract_bowling_features_from_data(_moving_hand_data())
    cards = _metric_cards(features, "bowling", 30.0)
    elbow = next(card for card in cards if card["key"] == "elbow_angle_at_release")
    assert "not a legality assessment" in elbow["description"]


def test_compact_timeline_is_bounded():
    data = _moving_hand_data(frames=400)
    timeline = _compact_timeline(data, max_frames=180)
    assert len(timeline["frames"]) <= 180
    assert timeline["sample_step"] == 3
    assert all(len(frame["landmarks"]) == 0 or len(frame["landmarks"][0]) == 4 for frame in timeline["frames"])


def test_compact_timeline_keeps_frame_indices():
    data = _moving_hand_data(frames=90)
    timeline = _compact_timeline(data)
    assert timeline["frames"][0]["frame"] == 0
    assert timeline["frames"][-1]["timestamp_ms"] <= 3000


# --------------------------------------------------------------------------- coaching


def test_rules_coach_is_grounded_in_measurements():
    features = extract_batting_features_from_data(_moving_hand_data())
    coach = rules_coach("batting", "Cover drive", features)
    assert coach["enhanced"] is False
    assert coach["uncertainty"]
    assert coach["disclaimer"]
    assert "peak" in coach["strengths"][0].lower()
    assert find_unsupported_claims(coach["summary"] + coach["drill"]) == []


def test_rules_coach_refuses_to_guess_without_an_action():
    features = {"action_detected": False}
    coach = rules_coach("bowling", "Unclear action", features)
    assert coach["strengths"] == []
    assert "Re-record" in coach["improvements"][0]["title"]


def test_unsupported_claims_are_detected():
    assert "bowling legality" in find_unsupported_claims("That looks like an illegal action.")
    assert "medical diagnosis" in find_unsupported_claims("You may have a rotator cuff tear.")
    assert "ball tracking" in find_unsupported_claims("The ball was delivered at 132 km/h.")
    assert find_unsupported_claims("Shoulder rotation of 40 degrees across the action.") == []


def test_validator_rejects_ungrounded_improvements():
    response = CoachResponse(
        summary="Solid session.",
        strengths=["Good balance."],
        improvements=[Improvement(title="Rotate more", evidence="Trust the process.", cue="Keep going.")],
        drill="Shadow bat for five minutes.",
        uncertainty="Estimates are approximate.",
        disclaimer="Coaching aid only.",
    )
    assert "improvement without measured evidence" in validate_coach_response(response)


def test_validator_accepts_grounded_responses():
    response = CoachResponse(
        summary="Shoulder rotation of 42 degrees through contact.",
        strengths=["Peak hand speed of 5.1 body lengths/s."],
        improvements=[Improvement(title="Hold the finish", evidence="Finish height was -0.1 torso lengths.", cue="Hold for two seconds.")],
        drill="3 x 8 shadow swings.",
        uncertainty="Single-camera estimates.",
        disclaimer="Coaching aid only.",
    )
    assert validate_coach_response(response) == []


def test_quality_gate_handles_numpy_values():
    """Feature values may be numpy scalars; the API must serialise them."""
    features = extract_batting_features_from_data(_moving_hand_data())
    cards = _metric_cards(features, "batting", 30.0)
    import json

    json.dumps(cards)
    assert isinstance(np.float64(1.0).item(), float)
