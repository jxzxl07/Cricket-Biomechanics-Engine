"""MediaPipe pose extraction for uploaded clips.

Pose data is the shared source of truth for metrics, the replay overlay, phase
timestamps and the capture-quality gate. Extraction is bounded so a long clip
cannot exhaust the request budget.
"""

import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import DATA_DIR, FPS, MODELS_DIR

MODEL_PATH = str(MODELS_DIR / "pose_landmarker_lite.task")

# Two poses lets the quality gate notice a second athlete in frame.
NUM_POSES = 2
# Upper bound on analysed frames so a 12s 60fps clip cannot blow the request budget.
MAX_ANALYSED_FRAMES = 240

POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
]


def landmark_to_dict(index, landmark):
    return {
        "index": index,
        "name": POSE_LANDMARK_NAMES[index],
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": float(getattr(landmark, "visibility", 0.0)),
        "presence": float(getattr(landmark, "presence", 0.0)),
    }


def build_landmarks_path(clip_path):
    clip_path = Path(clip_path)

    try:
        relative_clip_path = clip_path.relative_to(DATA_DIR / "raw")
    except ValueError:
        relative_clip_path = Path(clip_path.name)

    return (DATA_DIR / "landmarks" / relative_clip_path).with_suffix(".json")


class PoseEstimator:
    def __init__(self, num_poses: int = NUM_POSES):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def detect_landmarks(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.pose_landmarks

    def close(self):
        self.landmarker.close()


def _frame_step(total_frames: int, max_frames: int = MAX_ANALYSED_FRAMES) -> int:
    return max(1, math.ceil(total_frames / max_frames))


def extract_landmarks_from_video(clip_path, mode, label=None, output_path=None, deadline=None, max_frames=MAX_ANALYSED_FRAMES):
    """Extract pose landmarks for every ``frame_step``-th frame of a clip.

    ``deadline`` is a ``time.monotonic()`` value. When it passes, extraction
    raises ``TimeoutError`` so the API can answer instead of hanging.
    """
    clip_path = Path(clip_path)

    if output_path is None:
        output_path = build_landmarks_path(clip_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(clip_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video clip: {clip_path}")

    clip_fps = capture.get(cv2.CAP_PROP_FPS)

    if clip_fps <= 0:
        clip_fps = FPS

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = _frame_step(decoded_frames) if decoded_frames > 0 else 1

    pose_estimator = PoseEstimator()
    frames = []
    max_poses_seen = 0

    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Pose extraction exceeded the processing budget")
            success, frame = capture.read()

            if not success:
                break

            frame_index = len(frames) * step
            # Skip the frames between samples without decoding them one by one.
            if step > 1:
                for _ in range(step - 1):
                    if not capture.grab():
                        break

            timestamp_ms = int((frame_index / clip_fps) * 1000)
            pose_landmarks = pose_estimator.detect_landmarks(frame, timestamp_ms)

            max_poses_seen = max(max_poses_seen, len(pose_landmarks))

            if pose_landmarks:
                landmarks = [
                    landmark_to_dict(index, landmark)
                    for index, landmark in enumerate(pose_landmarks[0])
                ]
            else:
                landmarks = []

            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "person_detected": bool(pose_landmarks),
                    "pose_count": len(pose_landmarks),
                    "landmarks": landmarks,
                }
            )
            if len(frames) >= max_frames:
                break

    finally:
        capture.release()
        pose_estimator.close()

    landmark_data = {
        "clip_path": str(clip_path),
        "mode": mode,
        "label": label,
        "fps": clip_fps,
        "analysis_fps": clip_fps / step,
        "analysis_frame_step": step,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames": decoded_frames if decoded_frames > 0 else len(frames) * step,
        "analysed_frames": len(frames),
        "max_poses_seen": max_poses_seen,
        "landmark_format": "mediapipe_pose_33",
        "frames": frames,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(landmark_data, file, indent=2)

    return output_path
