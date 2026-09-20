"""Pose-derived cricket movement metrics for the web analysis API.

These are coaching aids, not medical, officiating, or laboratory measurements.
This module intentionally contains no bowling legality judgement.
"""

import json

from vision.biomechanics import (
    angle_between_three_points,
    calculate_speed_series,
    circular_angle_range,
    detect_action_window,
    detect_bowling_arm,
    get_arm_landmark_names,
    get_landmark,
    get_landmark_series,
    get_shoulder_line_angles,
    get_torso_lean,
    index_of_max,
    normalise_distance,
    path_angle,
    safe_max,
    safe_mean,
    safe_min,
)

MIN_DETECTED_POSE_FRAMES = 5
MIN_DETECTED_POSE_RATIO = 0.1
MIN_ACTION_DURATION_FRAMES = 8
MIN_BOWLING_PEAK_WRIST_SPEED = 1.2
MIN_BOWLING_ARM_PATH_RANGE = 0.18
MIN_BATTING_PEAK_HAND_SPEED = 0.6
MIN_BATTING_HAND_PATH_RANGE = 0.12


def load_landmark_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_person_detection_summary(data):
    frames = data.get("frames", [])
    total = data.get("total_frames") or len(frames)
    detected = sum(1 for frame in frames if frame.get("person_detected") and frame.get("landmarks"))
    ratio = detected / total if total else 0.0
    return {
        "total_frames": total,
        "detected_frames": detected,
        "detection_ratio": ratio,
        "has_enough_person": detected >= MIN_DETECTED_POSE_FRAMES and ratio >= MIN_DETECTED_POSE_RATIO,
    }


def _clean(data):
    return [frame for frame in data.get("frames", []) if frame.get("person_detected") and frame.get("landmarks")]


def _point_range(points, axis):
    values = [point[axis] for point in points if point is not None]
    return max(values) - min(values) if values else None


def _first(points):
    return next((point for point in points if point is not None), None)


def _last(points):
    return next((point for point in reversed(points) if point is not None), None)


def _mid_hips(frame):
    left, right = get_landmark(frame, "left_hip"), get_landmark(frame, "right_hip")
    if left is None or right is None:
        return None
    return {
        "x": (left["x"] + right["x"]) / 2,
        "y": (left["y"] + right["y"]) / 2,
        "z": (left["z"] + right["z"]) / 2,
        "visibility": min(left["visibility"], right["visibility"]),
    }


def _average_speed(speeds, center, window=5):
    if center is None:
        return None
    return safe_mean(speeds[max(0, center - window) : center + window + 1])


def _knee_bend(frame, side):
    angle = angle_between_three_points(
        get_landmark(frame, f"{side}_hip"),
        get_landmark(frame, f"{side}_knee"),
        get_landmark(frame, f"{side}_ankle"),
    )
    return None if angle is None else 180 - angle


def _no_person(data):
    return {
        "action_detected": False,
        "action_status": "No person detected during clip",
        "classification_block_reason": "no_person",
        "person_detection": get_person_detection_summary(data),
    }


def extract_bowling_features_from_data(data):
    summary, frames = get_person_detection_summary(data), _clean(data)
    if not summary["has_enough_person"] or len(frames) < 5:
        return _no_person(data)

    fps = float(data.get("fps") or 30)
    arm = detect_bowling_arm(frames)
    names = get_arm_landmark_names(arm)
    points = get_landmark_series(frames, names["wrist"], should_mirror=arm == "left")
    speeds = calculate_speed_series(frames, points, fps)
    start, end = detect_action_window(speeds)
    action_frames, action_points, action_speeds = frames[start : end + 1], points[start : end + 1], speeds[start : end + 1]
    release = index_of_max(action_speeds)
    release = release if release is not None else len(action_frames) // 2
    release_frame, release_wrist = action_frames[release], action_points[release]
    shoulder = get_landmark(release_frame, names["shoulder"])
    elbow = get_landmark(release_frame, names["elbow"])
    hips = _mid_hips(release_frame)

    vertical_range = normalise_distance(_point_range(action_points, "y"), release_frame)
    horizontal_range = normalise_distance(_point_range(action_points, "x"), release_frame, scale="shoulder")
    release_height = normalise_distance(abs(hips["y"] - release_wrist["y"]), release_frame) if hips and release_wrist else None
    forward_reach = normalise_distance(abs(shoulder["x"] - release_wrist["x"]), release_frame, scale="shoulder") if shoulder and release_wrist else None
    peak_speed = safe_max(action_speeds)
    duration = end - start + 1
    movement_range = safe_max([vertical_range, horizontal_range])
    detected = bool(peak_speed is not None and peak_speed >= MIN_BOWLING_PEAK_WRIST_SPEED and movement_range is not None and movement_range >= MIN_BOWLING_ARM_PATH_RANGE and duration >= MIN_ACTION_DURATION_FRAMES)
    pre, post = max(0, release - 8), min(len(action_points) - 1, release + 8)

    return {
        "peak_wrist_speed": peak_speed,
        "mean_wrist_speed_near_release": _average_speed(action_speeds, release),
        "release_height": release_height,
        "release_forward_reach": forward_reach,
        "arm_path_vertical_range": vertical_range,
        "arm_path_horizontal_range": horizontal_range,
        "pre_release_path_angle": path_angle(action_points[pre], release_wrist),
        "post_release_path_angle": path_angle(release_wrist, action_points[post]),
        "shoulder_rotation_range": circular_angle_range(get_shoulder_line_angles(action_frames)),
        "torso_lean_at_release": get_torso_lean(release_frame),
        "elbow_angle_at_release": angle_between_three_points(shoulder, elbow, get_landmark(release_frame, names["wrist"])),
        "action_duration_frames": duration,
        "action_duration_seconds": duration / fps,
        "detected_bowling_arm": arm,
        "action_start_frame": action_frames[0]["frame_index"],
        "action_end_frame": action_frames[-1]["frame_index"],
        "release_frame": action_frames[release]["frame_index"],
        "action_detected": detected,
        "action_status": "Bowling action detected" if detected else "No clear bowling action detected",
        "classification_block_reason": None if detected else "no_action",
        "person_detection": summary,
    }


def _hand_midpoints(frames):
    result = []
    for frame in frames:
        left, right = get_landmark(frame, "left_wrist"), get_landmark(frame, "right_wrist")
        if left is None or right is None:
            result.append(None)
        else:
            result.append({
                "x": (left["x"] + right["x"]) / 2,
                "y": (left["y"] + right["y"]) / 2,
                "z": (left["z"] + right["z"]) / 2,
                "visibility": min(left["visibility"], right["visibility"]),
            })
    return result


def extract_batting_features_from_data(data):
    summary, frames = get_person_detection_summary(data), _clean(data)
    if not summary["has_enough_person"] or len(frames) < 5:
        return _no_person(data)

    fps = float(data.get("fps") or 30)
    hands = _hand_midpoints(frames)
    speeds = calculate_speed_series(frames, hands, fps)
    start, end = detect_action_window(speeds)
    action_frames, action_hands, action_speeds = frames[start : end + 1], hands[start : end + 1], speeds[start : end + 1]
    peak = index_of_max(action_speeds)
    peak_speed = safe_max(action_speeds)
    duration = end - start + 1
    horizontal_range = normalise_distance(_point_range(action_hands, "x"), action_frames[0], scale="shoulder")
    vertical_range = normalise_distance(_point_range(action_hands, "y"), action_frames[0])
    start_hand, end_hand = _first(action_hands), _last(action_hands)
    final_hips = _mid_hips(action_frames[-1])
    final_height = normalise_distance(final_hips["y"] - end_hand["y"], action_frames[-1]) if final_hips and end_hand else None
    heads = [get_landmark(frame, "nose") for frame in action_frames]
    start_head = _first(heads)
    low_head = max((point for point in heads if point is not None), key=lambda point: point["y"], default=None)
    head_drop = normalise_distance(low_head["y"] - start_head["y"], action_frames[0]) if start_head and low_head else None
    torso = [get_torso_lean(frame) for frame in action_frames]
    torso_min, torso_max = safe_min(torso), safe_max(torso)
    knees = [value for frame in action_frames for value in (_knee_bend(frame, "left"), _knee_bend(frame, "right")) if value is not None]
    movement_range = safe_max([horizontal_range, vertical_range])
    detected = bool(peak_speed is not None and peak_speed >= MIN_BATTING_PEAK_HAND_SPEED and movement_range is not None and movement_range >= MIN_BATTING_HAND_PATH_RANGE and duration >= MIN_ACTION_DURATION_FRAMES)

    return {
        "peak_hand_speed": peak_speed,
        "mean_hand_speed": safe_mean(action_speeds),
        "peak_speed_timing": peak / (duration - 1) if peak is not None and duration > 1 else None,
        "hand_horizontal_range": horizontal_range,
        "hand_vertical_range": vertical_range,
        "hand_path_angle": path_angle(start_hand, end_hand),
        "final_hand_height": final_height,
        "shoulder_rotation_range": circular_angle_range(get_shoulder_line_angles(action_frames)),
        "max_knee_bend": max(knees) if knees else None,
        "head_drop": head_drop,
        "torso_lean_range": torso_max - torso_min if torso_min is not None and torso_max is not None else None,
        "action_duration_frames": duration,
        "action_duration_seconds": duration / fps,
        "action_start_frame": action_frames[0]["frame_index"],
        "action_end_frame": action_frames[-1]["frame_index"],
        "peak_speed_frame": action_frames[peak]["frame_index"] if peak is not None else None,
        "action_detected": detected,
        "action_status": "Batting shot detected" if detected else "No clear batting shot detected",
        "classification_block_reason": None if detected else "no_action",
        "person_detection": summary,
    }


def extract_features_from_data(data, mode):
    if mode == "batting":
        return extract_batting_features_from_data(data)
    if mode == "bowling":
        return extract_bowling_features_from_data(data)
    raise ValueError(f"Unknown mode: {mode}")
