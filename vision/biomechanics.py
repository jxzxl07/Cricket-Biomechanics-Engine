# Maths Functions: calculate_angle(), calculate_velocity(), calculate_motion_score(), calculate_release_frame(), calculate_legality()

import math

MIN_VISIBILITY = 0.4

def get_landmark(frame, name, min_visibility=MIN_VISIBILITY):
    for landmark in frame["landmarks"]:
        if landmark["name"] == name:
            if landmark["visibility"] < min_visibility:
                return None

            return {
                "x": landmark["x"],
                "y": landmark["y"],
                "z": landmark["z"],
                "visibility": landmark["visibility"],
            }

    return None


def midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return {
        "x": (point_a["x"] + point_b["x"]) / 2,
        "y": (point_a["y"] + point_b["y"]) / 2,
        "z": (point_a["z"] + point_b["z"]) / 2,
        "visibility": min(point_a["visibility"], point_b["visibility"]),
    }


def distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    dx = point_a["x"] - point_b["x"]
    dy = point_a["y"] - point_b["y"]
    dz = point_a["z"] - point_b["z"]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def distance_2d(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    dx = point_a["x"] - point_b["x"]
    dy = point_a["y"] - point_b["y"]

    return math.sqrt(dx * dx + dy * dy)


def angle_between_three_points(point_a, point_b, point_c):
    if point_a is None or point_b is None or point_c is None:
        return None

    vector_ab = {
        "x": point_a["x"] - point_b["x"],
        "y": point_a["y"] - point_b["y"],
    }

    vector_cb = {
        "x": point_c["x"] - point_b["x"],
        "y": point_c["y"] - point_b["y"],
    }

    dot_product = (
        vector_ab["x"] * vector_cb["x"]
        + vector_ab["y"] * vector_cb["y"]
    )

    magnitude_ab = math.sqrt(
        vector_ab["x"] * vector_ab["x"]
        + vector_ab["y"] * vector_ab["y"]
    )

    magnitude_cb = math.sqrt(
        vector_cb["x"] * vector_cb["x"]
        + vector_cb["y"] * vector_cb["y"]
    )

    if magnitude_ab == 0 or magnitude_cb == 0:
        return None

    cosine_angle = dot_product / (magnitude_ab * magnitude_cb)
    cosine_angle = max(-1, min(1, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def line_angle(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    dx = point_b["x"] - point_a["x"]
    dy = point_b["y"] - point_a["y"]

    return math.degrees(math.atan2(dy, dx))


def path_angle(start_point, end_point):
    return line_angle(start_point, end_point)


def angle_range(angles):
    valid_angles = [angle for angle in angles if angle is not None]

    if not valid_angles:
        return None

    return max(valid_angles) - min(valid_angles)


def circular_angle_range(angles):
    valid_angles = [angle % 360 for angle in angles if angle is not None]

    if not valid_angles:
        return None

    if len(valid_angles) == 1:
        return 0

    valid_angles.sort()

    gaps = []

    for index in range(1, len(valid_angles)):
        gaps.append(valid_angles[index] - valid_angles[index - 1])

    wrap_gap = (valid_angles[0] + 360) - valid_angles[-1]
    gaps.append(wrap_gap)

    largest_gap = max(gaps)

    return 360 - largest_gap



def get_mid_shoulders(frame):
    left_shoulder = get_landmark(frame, "left_shoulder")
    right_shoulder = get_landmark(frame, "right_shoulder")
    return midpoint(left_shoulder, right_shoulder)


def get_mid_hips(frame):
    left_hip = get_landmark(frame, "left_hip")
    right_hip = get_landmark(frame, "right_hip")
    return midpoint(left_hip, right_hip)


def get_torso_length(frame):
    mid_shoulders = get_mid_shoulders(frame)
    mid_hips = get_mid_hips(frame)

    torso_length = distance_2d(mid_shoulders, mid_hips)

    if torso_length is None or torso_length == 0:
        return None

    return torso_length


def get_shoulder_width(frame):
    left_shoulder = get_landmark(frame, "left_shoulder")
    right_shoulder = get_landmark(frame, "right_shoulder")

    shoulder_width = distance_2d(left_shoulder, right_shoulder)

    if shoulder_width is None or shoulder_width == 0:
        return None

    return shoulder_width


def normalise_distance(raw_distance, frame, scale="torso"):
    if raw_distance is None:
        return None

    if scale == "shoulder":
        normaliser = get_shoulder_width(frame)
    else:
        normaliser = get_torso_length(frame)

    if normaliser is None or normaliser == 0:
        return None

    return raw_distance / normaliser


def calculate_normalised_speed(previous_point, current_point, current_frame, fps):
    raw_distance = distance_2d(previous_point, current_point)

    if raw_distance is None:
        return None

    normalised_distance = normalise_distance(raw_distance, current_frame)

    if normalised_distance is None:
        return None

    return normalised_distance * fps


def mirror_point_if_needed(point, should_mirror):
    if point is None or not should_mirror:
        return point

    return {
        "x": 1 - point["x"],
        "y": point["y"],
        "z": point["z"],
        "visibility": point["visibility"],
    }


def get_landmark_series(frames, landmark_name, should_mirror=False):
    points = []

    for frame in frames:
        point = get_landmark(frame, landmark_name)
        point = mirror_point_if_needed(point, should_mirror)
        points.append(point)

    return points


def calculate_speed_series(frames, points, fps):
    speeds = [None]

    for index in range(1, len(points)):
        previous_point = points[index - 1]
        current_point = points[index]
        current_frame = frames[index]

        speed = calculate_normalised_speed(
            previous_point,
            current_point,
            current_frame,
            fps,
        )

        speeds.append(speed)

    return speeds


def valid_numbers(values):
    return [value for value in values if value is not None]


def safe_max(values):
    values = valid_numbers(values)

    if not values:
        return None

    return max(values)


def safe_min(values):
    values = valid_numbers(values)

    if not values:
        return None

    return min(values)


def safe_mean(values):
    values = valid_numbers(values)

    if not values:
        return None

    return sum(values) / len(values)


def index_of_max(values):
    best_index = None
    best_value = None

    for index, value in enumerate(values):
        if value is None:
            continue

        if best_value is None or value > best_value:
            best_value = value
            best_index = index

    return best_index


def detect_action_window(speeds, padding_frames=5):
    valid_speeds = valid_numbers(speeds)

    if not valid_speeds:
        return 0, len(speeds) - 1

    peak_speed = max(valid_speeds)

    if peak_speed == 0:
        return 0, len(speeds) - 1

    threshold = peak_speed * 0.2

    active_indices = [
        index
        for index, speed in enumerate(speeds)
        if speed is not None and speed >= threshold
    ]

    if not active_indices:
        release_index = index_of_max(speeds)

        if release_index is None:
            return 0, len(speeds) - 1

        start_index = max(0, release_index - padding_frames)
        end_index = min(len(speeds) - 1, release_index + padding_frames)
        return start_index, end_index

    start_index = max(0, min(active_indices) - padding_frames)
    end_index = min(len(speeds) - 1, max(active_indices) + padding_frames)

    return start_index, end_index


def slice_by_window(values, start_index, end_index):
    return values[start_index : end_index + 1]


def detect_bowling_arm(frames):
    left_wrist_points = get_landmark_series(frames, "left_wrist")
    right_wrist_points = get_landmark_series(frames, "right_wrist")

    left_score = calculate_motion_score(left_wrist_points)
    right_score = calculate_motion_score(right_wrist_points)

    if right_score >= left_score:
        return "right"

    return "left"


def calculate_motion_score(points):
    total_motion = 0

    for index in range(1, len(points)):
        previous_point = points[index - 1]
        current_point = points[index]

        movement = distance_2d(previous_point, current_point)

        if movement is not None:
            total_motion += movement

    return total_motion


def get_arm_landmark_names(side):
    if side == "left":
        return {
            "shoulder": "left_shoulder",
            "elbow": "left_elbow",
            "wrist": "left_wrist",
        }

    return {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
    }


def get_hand_midpoint(frame, should_mirror=False):
    left_wrist = get_landmark(frame, "left_wrist")
    right_wrist = get_landmark(frame, "right_wrist")

    left_wrist = mirror_point_if_needed(left_wrist, should_mirror)
    right_wrist = mirror_point_if_needed(right_wrist, should_mirror)

    return midpoint(left_wrist, right_wrist)


def get_hand_midpoint_series(frames, should_mirror=False):
    return [
        get_hand_midpoint(frame, should_mirror=should_mirror)
        for frame in frames
    ]


def get_shoulder_line_angles(frames):
    angles = []

    for frame in frames:
        left_shoulder = get_landmark(frame, "left_shoulder")
        right_shoulder = get_landmark(frame, "right_shoulder")
        angles.append(line_angle(left_shoulder, right_shoulder))

    return angles


def get_hip_line_angles(frames):
    angles = []

    for frame in frames:
        left_hip = get_landmark(frame, "left_hip")
        right_hip = get_landmark(frame, "right_hip")
        angles.append(line_angle(left_hip, right_hip))

    return angles


def get_torso_lean(frame):
    mid_shoulders = get_mid_shoulders(frame)
    mid_hips = get_mid_hips(frame)

    if mid_shoulders is None or mid_hips is None:
        return None

    vertical_reference = {
        "x": mid_hips["x"],
        "y": mid_hips["y"] - 1,
        "z": mid_hips["z"],
        "visibility": mid_hips["visibility"],
    }

    return angle_between_three_points(
        vertical_reference,
        mid_hips,
        mid_shoulders,
    )


def get_knee_bend(frame, side):
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