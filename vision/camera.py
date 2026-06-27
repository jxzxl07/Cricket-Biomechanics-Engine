import time
from pathlib import Path

import cv2

from PyQt6.QtCore import QThread, pyqtSignal

from config import FPS
from vision.pose import PoseEstimator


def crop_to_portrait(frame):
    height, width, _ = frame.shape

    target_aspect_ratio = 3 / 4
    new_width = int(height * target_aspect_ratio)

    start_x = (width - new_width) // 2
    end_x = start_x + new_width

    return frame[:, start_x:end_x]


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    camera_error = pyqtSignal(str)
    pose_status = pyqtSignal(bool)
    recording_finished = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()

        self.camera_index = camera_index
        self.running = False
        self.capture = None
        self.pose_estimator = None

        self.is_recording = False
        self.video_writer = None
        self.recording_end_time = None
        self.recording_path = None

    def run(self):
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.camera_error.emit("Could not open webcam.")
            return

        self.pose_estimator = PoseEstimator()
        self.running = True

        while self.running:
            success, frame = self.capture.read()

            if not success:
                self.camera_error.emit("Could not read frame from webcam.")
                break

            portrait_frame = crop_to_portrait(frame)

            if self.is_recording:
                self.write_recording_frame(portrait_frame)

            output_frame, pose_landmarks = self.pose_estimator.process_frame(
                portrait_frame
            )

            self.pose_status.emit(bool(pose_landmarks))
            self.frame_ready.emit(output_frame)

        self.cleanup()

    def start_recording(self, output_path, duration_seconds=3):
        self.recording_path = output_path
        self.recording_end_time = time.monotonic() + duration_seconds
        self.is_recording = True

    def write_recording_frame(self, frame):
        if self.video_writer is None:
            Path(self.recording_path).parent.mkdir(parents=True, exist_ok=True)

            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.video_writer = cv2.VideoWriter(
                self.recording_path,
                fourcc,
                FPS,
                (width, height),
            )

        self.video_writer.write(frame)

        if time.monotonic() >= self.recording_end_time:
            self.finish_recording()

    def finish_recording(self):
        self.is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.recording_finished.emit(self.recording_path)

        self.recording_path = None
        self.recording_end_time = None

    def stop(self):
        self.running = False
        self.wait()

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.pose_estimator is not None:
            self.pose_estimator.close()

        if self.capture is not None:
            self.capture.release()
