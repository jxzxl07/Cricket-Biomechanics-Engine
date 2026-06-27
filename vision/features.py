# Takes landmarks as input and outputs the feature vector
# Different features needed for bowling and batting models
# extract_bowling_features(), extract_batting_features(), aggregate_features(), normalise_features()

import json
from pathlib import Path
from config import DATA_DIR

from vision.biomechanics import (
    angle_between_three_points,
    circular_angle_range,
    calculate_speed_series,
    detect_action_window,
    detect_bowling_arm,
    get_arm_landmark_names,
    get_landmark,
    get_landmark_series,
    get_shoulder_line_angles,
    get_torso_lean,
    index_of_max,
    line_angle,
    mirror_point_if_needed,
    normalise_distance,
    path_angle,
    safe_max,
    safe_mean,
    safe_min,
    slice_by_window,
)


BOWLING_FEATURE_NAMES = [
    "peak_wrist_speed",
    "mean_wrist_speed_near_release",
    "release_height",
    "release_forward_reach",
    "arm_path_vertical_range",
    "arm_path_horizontal_range",
    "pre_release_path_angle",
    "post_release_path_angle",
    "shoulder_rotation_range",
    "torso_lean_at_release",
    "elbow_angle_at_release",
    "action_duration",
]

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


def get_person_detection_summary(landmark_data):
    frames = landmark_data.get("frames", [])
    total_frames = landmark_data.get("total_frames") or len(frames)

    detected_frames = sum(
        1
        for frame in frames
        if frame.get("person_detected") and frame.get("landmarks")
    )

    if total_frames:
        detection_ratio = detected_frames / total_frames
    else:
        detection_ratio = 0

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_ratio": detection_ratio,
        "has_enough_person": (
            detected_frames >= MIN_DETECTED_POSE_FRAMES
            and detection_ratio >= MIN_DETECTED_POSE_RATIO
        ),
    }


def clean_frames(landmark_data):
    return [
        frame
        for frame in landmark_data["frames"]
        if frame["person_detected"] and frame["landmarks"]
    ]


def get_point_range(points, axis):
    values = [
        point[axis]
        for point in points
        if point is not None
    ]

    if not values:
        return None

    return max(values) - min(values)


def get_point_at(points, index):
    if index is None:
        return None

    if index < 0 or index >= len(points):
        return None

    return points[index]


def average_speed_around_release(speeds, release_index, window_size=5):
    if release_index is None:
        return None

    start_index = max(0, release_index - window_size)
    end_index = min(len(speeds) - 1, release_index + window_size)

    return safe_mean(speeds[start_index : end_index + 1])


def largest_absolute_value(values):
    valid_values = [
        abs(value)
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return max(valid_values)


def build_no_person_features(mode):
    if mode == "bowling":
        features = {
            feature_name: None
            for feature_name in BOWLING_FEATURE_NAMES
        }
        features.update(
            {
                "detected_bowling_arm": None,
                "action_start_frame": None,
                "action_end_frame": None,
                "release_frame": None,
                "critical_phase_start_frame": None,
                "start_elbow_angle": None,
                "release_elbow_angle": None,
                "elbow_extension": None,
                "max_elbow_extension": None,
                "rapid_elbow_extension": None,
                "frames_over_15_degrees": 0,
                "legality": "Review Needed",
                "legality_confidence": "low",
            }
        )

    elif mode == "batting":
        features = {
            feature_name: None
            for feature_name in BATTING_FEATURE_NAMES
        }
        features.update(
            {
                "action_start_frame": None,
                "action_end_frame": None,
                "peak_speed_frame": None,
                "action_duration": None,
            }
        )

    else:
        features = {}

    features.update(
        {
            "action_detected": False,
            "action_status": "No person detected during clip",
            "should_classify": False,
            "classification_block_reason": "no_person",
        }
    )

    return features


def extract_bowling_features_from_data(landmark_data):
    person_summary = get_person_detection_summary(landmark_data)

    if not person_summary["has_enough_person"]:
        features = build_no_person_features("bowling")
        features["person_detection"] = person_summary
        return features

    frames = clean_frames(landmark_data)

    if len(frames) < 5:
        features = build_no_person_features("bowling")
        features["person_detection"] = person_summary
        return features

    fps = landmark_data["fps"]

    bowling_arm = detect_bowling_arm(frames)
    should_mirror = bowling_arm == "left"

    arm_names = get_arm_landmark_names(bowling_arm)

    wrist_points = get_landmark_series(
        frames,
        arm_names["wrist"],
        should_mirror=should_mirror,
    )

    wrist_speeds = calculate_speed_series(frames, wrist_points, fps)

    action_start, action_end = detect_action_window(wrist_speeds)

    action_frames = frames[action_start : action_end + 1]
    action_wrist_points = wrist_points[action_start : action_end + 1]
    action_wrist_speeds = wrist_speeds[action_start : action_end + 1]

    release_index_in_action = index_of_max(action_wrist_speeds)

    if release_index_in_action is None:
        raise ValueError("Could not find release frame from wrist speed.")

    release_frame = action_frames[release_index_in_action]
    release_wrist = action_wrist_points[release_index_in_action]

    peak_wrist_speed = safe_max(action_wrist_speeds)
    mean_wrist_speed_near_release = average_speed_around_release(
        action_wrist_speeds,
        release_index_in_action,
    )

    mid_hip = get_mid_hip_for_release(release_frame)

    if release_wrist is not None and mid_hip is not None:
        raw_release_height = abs(mid_hip["y"] - release_wrist["y"])
        release_height = normalise_distance(raw_release_height, release_frame)
    else:
        release_height = None

    release_shoulder = get_landmark(
        release_frame,
        arm_names["shoulder"],
    )
    release_shoulder = mirror_point_if_needed(release_shoulder, should_mirror)

    if release_wrist is not None and release_shoulder is not None:
        raw_release_forward_reach = abs(release_wrist["x"] - release_shoulder["x"])
        release_forward_reach = normalise_distance(
            raw_release_forward_reach,
            release_frame,
            scale="shoulder",
        )
    else:
        release_forward_reach = None

    raw_vertical_range = get_point_range(action_wrist_points, "y")
    arm_path_vertical_range = normalise_distance(raw_vertical_range, release_frame)

    raw_horizontal_range = get_point_range(action_wrist_points, "x")
    arm_path_horizontal_range = normalise_distance(
        raw_horizontal_range,
        release_frame,
        scale="shoulder",
    )

    pre_release_start = max(0, release_index_in_action - 8)
    post_release_end = min(len(action_wrist_points) - 1, release_index_in_action + 8)

    pre_release_path_angle = path_angle(
        get_point_at(action_wrist_points, pre_release_start),
        release_wrist,
    )

    post_release_path_angle = path_angle(
        release_wrist,
        get_point_at(action_wrist_points, post_release_end),
    )

    shoulder_angles = get_shoulder_line_angles(action_frames)
    shoulder_rotation_range = circular_angle_range(shoulder_angles)

    torso_lean_at_release = get_torso_lean(release_frame)

    release_shoulder = get_landmark(
        release_frame,
        arm_names["shoulder"],
    )
    release_elbow = get_landmark(
        release_frame,
        arm_names["elbow"],
    )
    release_wrist_raw = get_landmark(
        release_frame,
        arm_names["wrist"],
    )

    release_shoulder = mirror_point_if_needed(release_shoulder, should_mirror)
    release_elbow = mirror_point_if_needed(release_elbow, should_mirror)
    release_wrist_raw = mirror_point_if_needed(release_wrist_raw, should_mirror)

    elbow_angle_at_release = angle_between_three_points(
        release_shoulder,
        release_elbow,
        release_wrist_raw,
    )

    legality_result = calculate_bowling_legality(
        action_frames,
        arm_names,
        should_mirror,
        release_index_in_action,
    )

    action_duration = action_end - action_start + 1
    movement_range = largest_absolute_value(
        [
            arm_path_vertical_range,
            arm_path_horizontal_range,
        ]
    )
    action_detected = (
        peak_wrist_speed is not None
        and peak_wrist_speed >= MIN_BOWLING_PEAK_WRIST_SPEED
        and movement_range is not None
        and movement_range >= MIN_BOWLING_ARM_PATH_RANGE
        and action_duration >= MIN_ACTION_DURATION_FRAMES
    )

    if not action_detected:
        action_status = "No bowling action detected"
        should_classify = False
        classification_block_reason = "no_action"
    elif not legality_result["should_classify"]:
        action_status = "Illegal delivery detected"
        should_classify = False
        classification_block_reason = "illegal_delivery"
    else:
        action_status = "Bowling action detected"
        should_classify = True
        classification_block_reason = None

    features = {
        "peak_wrist_speed": peak_wrist_speed,
        "mean_wrist_speed_near_release": mean_wrist_speed_near_release,
        "release_height": release_height,
        "release_forward_reach": release_forward_reach,
        "arm_path_vertical_range": arm_path_vertical_range,
        "arm_path_horizontal_range": arm_path_horizontal_range,
        "pre_release_path_angle": pre_release_path_angle,
        "post_release_path_angle": post_release_path_angle,
        "shoulder_rotation_range": shoulder_rotation_range,
        "torso_lean_at_release": torso_lean_at_release,
        "elbow_angle_at_release": elbow_angle_at_release,
        "action_duration": action_duration,
        "detected_bowling_arm": bowling_arm,
        "action_start_frame": action_start,
        "action_end_frame": action_end,
        "release_frame": action_start + release_index_in_action,
        "critical_phase_start_frame": (
            action_start + legality_result["critical_phase_start_index"]
            if legality_result["critical_phase_start_index"] is not None
            else None
        ),
        "start_elbow_angle": legality_result["start_elbow_angle"],
        "release_elbow_angle": legality_result["release_elbow_angle"],
        "elbow_extension": legality_result["elbow_extension"],
        "max_elbow_extension": legality_result["max_elbow_extension"],
        "rapid_elbow_extension": legality_result["rapid_elbow_extension"],
        "frames_over_15_degrees": legality_result["frames_over_15_degrees"],
        "legality": legality_result["legality"],
        "legality_confidence": legality_result["legality_confidence"],
        "action_detected": action_detected,
        "action_status": action_status,
        "should_classify": should_classify,
        "classification_block_reason": classification_block_reason,
        "person_detection": person_summary,
    }

    return features


def get_mid_hip_for_release(frame):
    left_hip = get_landmark(frame, "left_hip")
    right_hip = get_landmark(frame, "right_hip")

    if left_hip is None or right_hip is None:
        return None

    return {
        "x": (left_hip["x"] + right_hip["x"]) / 2,
        "y": (left_hip["y"] + right_hip["y"]) / 2,
        "z": (left_hip["z"] + right_hip["z"]) / 2,
        "visibility": min(left_hip["visibility"], right_hip["visibility"]),
    }


def extract_bowling_features_from_json(json_path):
    landmark_data = load_landmark_json(json_path)
    features = extract_bowling_features_from_data(landmark_data)

    return {
        "clip_path": landmark_data["clip_path"],
        "mode": landmark_data["mode"],
        "label": landmark_data["label"],
        **features,
    }


def bowling_feature_vector(features):
    return [
        features[name]
        for name in BOWLING_FEATURE_NAMES
    ]


def save_features_json(features, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(features, file, indent=2)

    return output_path


def smooth_values(values, window_size=5):
    smoothed = []

    for index in range(len(values)):
        start_index = max(0, index - window_size // 2)
        end_index = min(len(values), index + window_size // 2 + 1)

        window = [
            value
            for value in values[start_index:end_index]
            if value is not None
        ]

        if not window:
            smoothed.append(None)
        else:
            smoothed.append(sum(window) / len(window))

    return smoothed


def count_frames_above(values, threshold):
    return sum(
        1
        for value in values
        if value is not None and value > threshold
    )




# BATTIING

BATTING_FEATURE_NAMES = [
    "peak_hand_speed",
    "mean_hand_speed",
    "peak_speed_timing",
    "hand_horizontal_range",
    "hand_vertical_range",
    "hand_path_angle",
    "lowest_hand_position",
    "final_hand_height",
    "shoulder_rotation_range",
    "max_knee_bend",
    "head_drop",
    "torso_lean_range",
]


def get_hand_range(points, axis):
    values = [
        point[axis]
        for point in points
        if point is not None
    ]

    if not values:
        return None

    return max(values) - min(values)


def get_lowest_point(points):
    valid_points = [
        point
        for point in points
        if point is not None
    ]

    if not valid_points:
        return None

    return max(valid_points, key=lambda point: point["y"])


def get_highest_point(points):
    valid_points = [
        point
        for point in points
        if point is not None
    ]

    if not valid_points:
        return None

    return min(valid_points, key=lambda point: point["y"])


def get_first_valid_point(points):
    for point in points:
        if point is not None:
            return point

    return None


def get_last_valid_point(points):
    for point in reversed(points):
        if point is not None:
            return point

    return None


def get_head_points(frames):
    return [
        get_landmark(frame, "nose")
        for frame in frames
    ]


def get_torso_lean_series(frames):
    return [
        get_torso_lean(frame)
        for frame in frames
    ]


def get_max_knee_bend(frames):
    left_bends = []
    right_bends = []

    for frame in frames:
        left_bends.append(get_knee_bend_for_features(frame, "left"))
        right_bends.append(get_knee_bend_for_features(frame, "right"))

    left_max = safe_max(left_bends)
    right_max = safe_max(right_bends)

    if left_max is None and right_max is None:
        return None

    if left_max is None:
        return right_max

    if right_max is None:
        return left_max

    return max(left_max, right_max)


def get_knee_bend_for_features(frame, side):
    if side == "left":
        hip = get_landmark(frame, "left_hip")
        knee = get_landmark(frame, "left_knee")
        ankle = get_landmark(frame, "left_ankle")
    else:
        hip = get_landmark(frame, "right_hip")
        knee = get_landmark(frame, "right_knee")
        ankle = get_landmark(frame, "right_ankle")

    knee_angle = angle_between_three_points(hip, knee, ankle)

    if knee_angle is None:
        return None

    return 180 - knee_angle


def extract_batting_features_from_data(landmark_data):
    person_summary = get_person_detection_summary(landmark_data)

    if not person_summary["has_enough_person"]:
        features = build_no_person_features("batting")
        features["person_detection"] = person_summary
        return features

    frames = clean_frames(landmark_data)

    if len(frames) < 5:
        features = build_no_person_features("batting")
        features["person_detection"] = person_summary
        return features

    fps = landmark_data["fps"]

    hand_points = get_hand_midpoint_series_for_features(frames)
    hand_speeds = calculate_speed_series(frames, hand_points, fps)

    action_start, action_end = detect_action_window(hand_speeds)

    action_frames = frames[action_start : action_end + 1]
    action_hand_points = hand_points[action_start : action_end + 1]
    action_hand_speeds = hand_speeds[action_start : action_end + 1]

    peak_speed_index = index_of_max(action_hand_speeds)

    peak_hand_speed = safe_max(action_hand_speeds)
    mean_hand_speed = safe_mean(action_hand_speeds)

    action_duration = action_end - action_start + 1

    if peak_speed_index is None or action_duration <= 1:
        peak_speed_timing = None
    else:
        peak_speed_timing = peak_speed_index / (action_duration - 1)

    raw_horizontal_range = get_hand_range(action_hand_points, "x")
    hand_horizontal_range = normalise_distance(
        raw_horizontal_range,
        action_frames[0],
        scale="shoulder",
    )

    raw_vertical_range = get_hand_range(action_hand_points, "y")
    hand_vertical_range = normalise_distance(
        raw_vertical_range,
        action_frames[0],
    )

    start_hand = get_first_valid_point(action_hand_points)
    end_hand = get_last_valid_point(action_hand_points)

    hand_path_angle = path_angle(start_hand, end_hand)

    lowest_hand = get_lowest_point(action_hand_points)

    if lowest_hand is not None:
        mid_hips = get_mid_hip_for_release(action_frames[0])
        raw_lowest_hand_position = lowest_hand["y"] - mid_hips["y"]
        lowest_hand_position = normalise_distance(
            raw_lowest_hand_position,
            action_frames[0],
        )
    else:
        lowest_hand_position = None

    final_hand = get_last_valid_point(action_hand_points)

    if final_hand is not None:
        mid_hips = get_mid_hip_for_release(action_frames[-1])
        raw_final_hand_height = mid_hips["y"] - final_hand["y"]
        final_hand_height = normalise_distance(
            raw_final_hand_height,
            action_frames[-1],
        )
    else:
        final_hand_height = None

    shoulder_angles = get_shoulder_line_angles(action_frames)
    shoulder_rotation_range = circular_angle_range(shoulder_angles)

    max_knee_bend = get_max_knee_bend(action_frames)

    head_points = get_head_points(action_frames)
    start_head = get_first_valid_point(head_points)
    lowest_head = get_lowest_point(head_points)

    if start_head is not None and lowest_head is not None:
        raw_head_drop = lowest_head["y"] - start_head["y"]
        head_drop = normalise_distance(raw_head_drop, action_frames[0])
    else:
        head_drop = None

    torso_leans = get_torso_lean_series(action_frames)

    torso_lean_min = safe_min(torso_leans)
    torso_lean_max = safe_max(torso_leans)

    if torso_lean_min is None or torso_lean_max is None:
        torso_lean_range = None
    else:
        torso_lean_range = torso_lean_max - torso_lean_min

    movement_range = largest_absolute_value(
        [
            hand_horizontal_range,
            hand_vertical_range,
        ]
    )
    action_detected = (
        peak_hand_speed is not None
        and peak_hand_speed >= MIN_BATTING_PEAK_HAND_SPEED
        and movement_range is not None
        and movement_range >= MIN_BATTING_HAND_PATH_RANGE
        and action_duration >= MIN_ACTION_DURATION_FRAMES
    )

    if action_detected:
        action_status = "Batting shot detected"
        should_classify = True
        classification_block_reason = None
    else:
        action_status = "No batting shot detected"
        should_classify = False
        classification_block_reason = "no_action"

    features = {
        "peak_hand_speed": peak_hand_speed,
        "mean_hand_speed": mean_hand_speed,
        "peak_speed_timing": peak_speed_timing,
        "hand_horizontal_range": hand_horizontal_range,
        "hand_vertical_range": hand_vertical_range,
        "hand_path_angle": hand_path_angle,
        "lowest_hand_position": lowest_hand_position,
        "final_hand_height": final_hand_height,
        "shoulder_rotation_range": shoulder_rotation_range,
        "max_knee_bend": max_knee_bend,
        "head_drop": head_drop,
        "torso_lean_range": torso_lean_range,
        "action_start_frame": action_start,
        "action_end_frame": action_end,
        "peak_speed_frame": action_start + peak_speed_index
        if peak_speed_index is not None
        else None,
        "action_duration": action_duration,
        "action_detected": action_detected,
        "action_status": action_status,
        "should_classify": should_classify,
        "classification_block_reason": classification_block_reason,
        "person_detection": person_summary,
    }

    return features


def get_hand_midpoint_series_for_features(frames):
    hand_points = []

    for frame in frames:
        left_wrist = get_landmark(frame, "left_wrist")
        right_wrist = get_landmark(frame, "right_wrist")

        if left_wrist is None or right_wrist is None:
            hand_points.append(None)
            continue

        hand_points.append(
            {
                "x": (left_wrist["x"] + right_wrist["x"]) / 2,
                "y": (left_wrist["y"] + right_wrist["y"]) / 2,
                "z": (left_wrist["z"] + right_wrist["z"]) / 2,
                "visibility": min(left_wrist["visibility"], right_wrist["visibility"]),
            }
        )

    return hand_points


def extract_batting_features_from_json(json_path):
    landmark_data = load_landmark_json(json_path)
    features = extract_batting_features_from_data(landmark_data)

    return {
        "clip_path": landmark_data["clip_path"],
        "mode": landmark_data["mode"],
        "label": landmark_data["label"],
        **features,
    }


def batting_feature_vector(features):
    return [
        features[name]
        for name in BATTING_FEATURE_NAMES
    ]



def build_features_path(landmarks_path):
    landmarks_path = Path(landmarks_path)

    try:
        relative_path = landmarks_path.relative_to(DATA_DIR / "landmarks")
    except ValueError:
        relative_path = Path(landmarks_path.name)

    return (DATA_DIR / "processed" / relative_path).with_suffix(".json")


def extract_features_from_landmarks_json(landmarks_path):
    landmark_data = load_landmark_json(landmarks_path)
    mode = landmark_data["mode"]

    if mode == "bowling":
        return extract_bowling_features_from_data(landmark_data)

    if mode == "batting":
        return extract_batting_features_from_data(landmark_data)

    raise ValueError(f"Unknown mode: {mode}")


def extract_and_save_features(landmarks_path, output_path=None):
    landmarks_path = Path(landmarks_path)

    if output_path is None:
        output_path = build_features_path(landmarks_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    landmark_data = load_landmark_json(landmarks_path)
    mode = landmark_data["mode"]

    if mode == "bowling":
        features = extract_bowling_features_from_data(landmark_data)
    elif mode == "batting":
        features = extract_batting_features_from_data(landmark_data)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    feature_data = {
        "clip_path": landmark_data["clip_path"],
        "landmarks_path": str(landmarks_path),
        "mode": mode,
        "label": landmark_data["label"],
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(feature_data, file, indent=2)

    return output_path



import csv


def build_training_row(feature_json_path, label=None):
    with open(feature_json_path, "r", encoding="utf-8") as file:
        feature_data = json.load(file)

    if label is None:
        label = feature_data.get("label")

    features = feature_data.get("features", {})

    return feature_data, label, features


def get_feature_names_for_mode(mode):
    if mode == "bowling":
        return BOWLING_FEATURE_NAMES

    if mode == "batting":
        return BATTING_FEATURE_NAMES

    raise ValueError(f"Unknown mode: {mode}")


def build_training_csv(mode):
    feature_names = get_feature_names_for_mode(mode)

    processed_mode_dir = DATA_DIR / "processed" / mode
    output_path = DATA_DIR / "processed" / f"{mode}_features.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if not processed_mode_dir.exists():
        raise ValueError(f"No processed data folder found: {processed_mode_dir}")

    def add_row(feature_json_path, label=None):
        feature_data, label, features = build_training_row(feature_json_path, label)
        
        if not label:
            return

        if not features.get("action_detected", True):
            print(
                f"Skipping {feature_json_path} because no valid "
                f"{mode} action was detected."
            )
            return

        if not features.get("should_classify", True):
            print(
                f"Skipping {feature_json_path} because it should not be "
                "used for model classification training."
            )
            return

        row = {
            "clip_path": feature_data.get("clip_path"),
            "landmarks_path": feature_data.get("landmarks_path"),
            "features_path": str(feature_json_path),
            "label": label,
        }

        missing_features = []

        for feature_name in feature_names:
            value = features.get(feature_name)

            if value is None:
                missing_features.append(feature_name)

            row[feature_name] = value

        if missing_features:
            print(
                f"Skipping {feature_json_path} because it is missing: "
                f"{', '.join(missing_features)}"
            )
            return

        rows.append(row)

    for feature_json_path in sorted(processed_mode_dir.rglob("*.json")):
        add_row(feature_json_path)

    from database.database import get_corrected_prediction_records

    for record in get_corrected_prediction_records(mode):
        add_row(record["features_path"], record["corrected_label"])

    fieldnames = [
        "clip_path",
        "landmarks_path",
        "features_path",
        "label",
        *feature_names,
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path, len(rows)


def build_all_training_csvs():
    bowling_path, bowling_count = build_training_csv("bowling")
    batting_path, batting_count = build_training_csv("batting")

    return {
        "bowling": {
            "path": bowling_path,
            "rows": bowling_count,
        },
        "batting": {
            "path": batting_path,
            "rows": batting_count,
        },
    }




def calculate_bowling_legality(action_frames, arm_names, should_mirror, release_index):
    elbow_angles = []
    elbow_points = []
    shoulder_points = []

    for frame in action_frames:
        shoulder = get_landmark(frame, arm_names["shoulder"])
        elbow = get_landmark(frame, arm_names["elbow"])
        wrist = get_landmark(frame, arm_names["wrist"])

        shoulder = mirror_point_if_needed(shoulder, should_mirror)
        elbow = mirror_point_if_needed(elbow, should_mirror)
        wrist = mirror_point_if_needed(wrist, should_mirror)

        elbow_angle = angle_between_three_points(
            shoulder,
            elbow,
            wrist,
        )

        elbow_angles.append(elbow_angle)
        elbow_points.append(elbow)
        shoulder_points.append(shoulder)

    if release_index is None:
        return build_legality_result(
            legality="Review Needed",
            legality_confidence="low",
            should_classify=True,
        )

    smoothed_angles = smooth_values(elbow_angles, window_size=5)

    critical_start_index = find_critical_phase_start_index(
        elbow_points,
        shoulder_points,
        release_index,
    )

    if critical_start_index is None:
        return build_legality_result(
            legality="Review Needed",
            legality_confidence="low",
            should_classify=True,
        )

    start_index = critical_start_index
    end_index = min(len(smoothed_angles) - 1, release_index + 3)

    legality_window = smoothed_angles[start_index : end_index + 1]
    critical_window = smoothed_angles[start_index : release_index + 1]

    release_elbow_angle = smoothed_angles[release_index]

    valid_critical_angles = [
        angle
        for angle in critical_window
        if angle is not None
    ]

    if release_elbow_angle is None or len(valid_critical_angles) < 4:
        return build_legality_result(
            critical_phase_start_index=critical_start_index,
            legality="Review Needed",
            legality_confidence="low",
            should_classify=True,
        )

    baseline_count = min(3, len(valid_critical_angles))
    start_elbow_angle = safe_mean(valid_critical_angles[:baseline_count])

    if start_elbow_angle is None:
        return build_legality_result(
            critical_phase_start_index=critical_start_index,
            legality="Review Needed",
            legality_confidence="low",
            should_classify=True,
        )

    elbow_extension = release_elbow_angle - start_elbow_angle

    extension_values = [
        angle - start_elbow_angle
        for angle in legality_window
        if angle is not None
    ]

    frames_over_15 = count_frames_above(extension_values, 15)
    frames_over_20 = count_frames_above(extension_values, 20)
    max_elbow_extension = safe_max(extension_values)
    rapid_elbow_extension = calculate_rapid_elbow_extension(
        smoothed_angles,
        start_index,
        end_index,
    )

    if max_elbow_extension is None:
        legality = "Review Needed"
        legality_confidence = "low"
        should_classify = True

    elif (
        max_elbow_extension >= 22
        and frames_over_15 >= 2
        and elbow_extension >= 10
    ) or (
        rapid_elbow_extension is not None
        and rapid_elbow_extension >= 16
        and max_elbow_extension >= 24
        and elbow_extension >= 10
    ):
        legality = "Illegal Delivery"
        legality_confidence = "high"
        should_classify = False

    elif (
        max_elbow_extension >= 12
        or elbow_extension >= 12
        or (
            rapid_elbow_extension is not None
            and rapid_elbow_extension >= 14
            and max_elbow_extension >= 12
        )
    ):
        legality = "Review Needed"
        legality_confidence = "medium"
        should_classify = True

    else:
        legality = "Fair"
        legality_confidence = "high"
        should_classify = True

    return build_legality_result(
        critical_phase_start_index=critical_start_index,
        start_elbow_angle=start_elbow_angle,
        release_elbow_angle=release_elbow_angle,
        elbow_extension=elbow_extension,
        max_elbow_extension=max_elbow_extension,
        rapid_elbow_extension=rapid_elbow_extension,
        frames_over_15_degrees=frames_over_15,
        frames_over_20_degrees=frames_over_20,
        legality=legality,
        legality_confidence=legality_confidence,
        should_classify=should_classify,
    )


def build_legality_result(
    *,
    critical_phase_start_index=None,
    start_elbow_angle=None,
    release_elbow_angle=None,
    elbow_extension=None,
    max_elbow_extension=None,
    rapid_elbow_extension=None,
    frames_over_15_degrees=0,
    frames_over_20_degrees=0,
    legality,
    legality_confidence,
    should_classify,
):
    return {
        "critical_phase_start_index": critical_phase_start_index,
        "start_elbow_angle": start_elbow_angle,
        "release_elbow_angle": release_elbow_angle,
        "elbow_extension": elbow_extension,
        "max_elbow_extension": max_elbow_extension,
        "rapid_elbow_extension": rapid_elbow_extension,
        "frames_over_15_degrees": frames_over_15_degrees,
        "frames_over_20_degrees": frames_over_20_degrees,
        "legality": legality,
        "legality_confidence": legality_confidence,
        "should_classify": should_classify,
    }


def find_critical_phase_start_index(elbow_points, shoulder_points, release_index):
    if release_index is None:
        return None

    latest_index = min(release_index, len(elbow_points) - 1)
    critical_start = None

    for index in range(latest_index, -1, -1):
        elbow = elbow_points[index]
        shoulder = shoulder_points[index]

        if elbow is None or shoulder is None:
            if critical_start is not None:
                break
            continue

        if upper_arm_has_reached_horizontal(elbow, shoulder):
            critical_start = index
        elif critical_start is not None:
            break

    return critical_start


def upper_arm_has_reached_horizontal(elbow, shoulder):
    # MediaPipe y increases down the frame. When the elbow is around shoulder
    # height or higher, the upper arm is in the critical delivery phase.
    return elbow["y"] <= shoulder["y"] + 0.04


def calculate_rapid_elbow_extension(smoothed_angles, start_index, end_index):
    best_extension = None

    for index in range(start_index, end_index + 1):
        current_angle = smoothed_angles[index]

        if current_angle is None:
            continue

        lookback_start = max(start_index, index - 3)

        previous_angles = [
            smoothed_angles[previous_index]
            for previous_index in range(lookback_start, index)
            if smoothed_angles[previous_index] is not None
        ]

        if not previous_angles:
            continue

        extension = current_angle - min(previous_angles)

        if best_extension is None or extension > best_extension:
            best_extension = extension

    return best_extension
