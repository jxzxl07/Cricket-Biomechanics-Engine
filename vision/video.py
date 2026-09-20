"""Robust video metadata probing.

Browser MediaRecorder output (WebM from ``MediaRecorder``) has no duration in the
container header, and OpenCV reports nonsense for it: a 6-second recording can
report ``CAP_PROP_FRAME_COUNT = -1.8e17``. Trusting that metadata rejected every
clip recorded in the browser, so the frame count is verified by decoding when the
metadata is unusable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from config import FPS as DEFAULT_FPS

MAX_PROBE_FRAMES = 6000


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frames: int
    width: int
    height: int
    duration_seconds: float
    metadata_reliable: bool

    @property
    def orientation(self) -> str:
        return "portrait" if self.height >= self.width else "landscape"


def _plausible_fps(fps: float) -> bool:
    return bool(fps) and math.isfinite(fps) and 1.0 <= fps <= 240.0


def _plausible_frames(frames: float) -> bool:
    return bool(frames) and math.isfinite(frames) and 1 <= frames <= 1_000_000


def probe_video(path: Path) -> VideoInfo:
    """Return usable metadata, decoding the clip when the container lies.

    Raises ``ValueError`` when the file cannot be opened at all.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be decoded")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reliable = _plausible_fps(fps) and _plausible_frames(frames)

        if not reliable:
            counted = 0
            last_msec = 0.0
            while counted < MAX_PROBE_FRAMES:
                ok, _ = capture.read()
                if not ok:
                    break
                counted += 1
                msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                if msec and math.isfinite(msec) and msec > last_msec:
                    last_msec = msec
            if counted == 0:
                raise ValueError("The uploaded video contains no readable frames")
            frames = float(counted)
            if not _plausible_fps(fps) and last_msec > 0:
                fps = counted / (last_msec / 1000)
            if not _plausible_fps(fps):
                fps = float(DEFAULT_FPS)
    finally:
        capture.release()

    duration = frames / fps if fps else 0.0
    return VideoInfo(
        path=str(path),
        fps=fps,
        frames=int(frames),
        width=width,
        height=height,
        duration_seconds=duration,
        metadata_reliable=reliable,
    )
