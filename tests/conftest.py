"""Shared fixtures.

Real clips live in the gitignored ``data/raw`` tree, so tests that need them skip
cleanly on a fresh clone while still running on the author's machine and in the
private golden-set job.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from config import DATA_DIR

SYNTHETIC_SIZE = (320, 240)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _write_video(path: Path, seconds: float, fps: int = 30, moving: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, SYNTHETIC_SIZE)
    if not writer.isOpened():  # pragma: no cover - environment without mp4v
        pytest.skip("OpenCV cannot encode mp4 in this environment")
    frames = int(seconds * fps)
    for index in range(frames):
        frame = np.full((SYNTHETIC_SIZE[1], SYNTHETIC_SIZE[0], 3), 24, dtype=np.uint8)
        offset = index * 3 if moving else 0
        cv2.rectangle(frame, (40 + offset, 80), (110 + offset, 150), (200, 200, 200), -1)
        cv2.circle(frame, (75 + offset, 60), 18, (220, 220, 220), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A moving, person-free clip: the quality gate must refuse to classify it."""
    return _write_video(tmp_path / "synthetic.mp4", seconds=3)


@pytest.fixture
def long_synthetic_clip(tmp_path: Path) -> Path:
    return _write_video(tmp_path / "long.mp4", seconds=13)


def _find_clip(mode: str) -> Path | None:
    root = DATA_DIR / "raw" / mode
    if not root.exists():
        return None
    for candidate in sorted(root.glob("*/*.mp4")):
        if ".landmarks" not in candidate.name:
            return candidate
    return None


def _real_clip_or_skip(mode: str) -> Path:
    clip = _find_clip(mode)
    if clip is None:
        pytest.skip(f"No {mode} clip available under data/raw/{mode}")
    return clip


@pytest.fixture(scope="session")
def real_batting_clip() -> Path:
    return _real_clip_or_skip("batting")


@pytest.fixture(scope="session")
def real_bowling_clip() -> Path:
    return _real_clip_or_skip("bowling")


@pytest.fixture
def upload(synthetic_clip: Path):
    def _upload(path: Path | None = None, filename: str | None = None, content_type: str = "video/mp4"):
        path = path or synthetic_clip
        return {"file": (filename or path.name, path.read_bytes(), content_type)}

    return _upload
