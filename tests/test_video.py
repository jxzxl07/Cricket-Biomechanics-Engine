"""Regression tests for video probing and upload validation.

The fixture is a real browser MediaRecorder capture (synthetic camera content,
no personal footage). It matters because MediaRecorder WebM has no duration in
the container header and OpenCV reports CAP_PROP_FRAME_COUNT = -1.8e17 for it,
which used to reject every clip recorded in the browser.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from vision.video import probe_video

FIXTURE = Path(__file__).parent / "fixtures" / "media-recorder.webm"
ANALYZE = "/api/v1/analyze"


@pytest.fixture(scope="module")
def recorded_webm() -> Path:
    if not FIXTURE.exists():
        pytest.skip("recording fixture missing")
    return FIXTURE


def test_probe_recovers_metadata_from_media_recorder_output(recorded_webm):
    info = probe_video(recorded_webm)
    assert info.metadata_reliable is False, "this fixture is expected to report bad container metadata"
    assert info.frames > 0
    assert 3.0 < info.duration_seconds < 12.0
    assert info.width > 0 and info.height > 0
    assert 1.0 <= info.fps <= 240.0


def test_probe_handles_a_normal_container(real_batting_clip):
    info = probe_video(real_batting_clip)
    assert info.metadata_reliable is True
    assert info.frames > 0 and info.duration_seconds > 0


def test_probe_rejects_a_non_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"definitely not a video")
    with pytest.raises(ValueError):
        probe_video(junk)


def test_api_accepts_a_browser_recording(client, recorded_webm):
    response = client.post(
        f"{ANALYZE}?mode=batting&camera_angle=side_on",
        files={"file": (recorded_webm.name, recorded_webm.read_bytes(), "video/webm")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # The synthetic camera contains no person, so the quality gate must refuse it
    # rather than inventing a label.
    assert body["status"] == "needs_better_clip"
    assert body["video"]["frames"] > 0
    assert body["video"]["duration_ms"] > 0
    assert body["classification"]["unknown"] is True


def test_api_rejects_a_truncated_recording(client, recorded_webm):
    payload = recorded_webm.read_bytes()[:2048]
    response = client.post(
        f"{ANALYZE}?mode=bowling",
        files={"file": ("clip.webm", io.BytesIO(payload), "video/webm")},
    )
    assert response.status_code in {400, 500}
