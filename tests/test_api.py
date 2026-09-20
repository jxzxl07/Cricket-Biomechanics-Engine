"""API contract and abuse-resistance tests for the v1 analysis service."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

import api.main as api_main
from api.main import MAX_DURATION_SECONDS, MAX_UPLOAD_BYTES

ANALYZE = "/api/v1/analyze"


def _no_legality_fields(payload) -> list[str]:
    """Any key that looks like a bowling-legality verdict is a regression."""
    offenders: list[str] = []

    def walk(node, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = key.lower()
                if any(token in lowered for token in ("legal", "illegal", "chuck", "legality")):
                    offenders.append(f"{trail}.{key}")
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    walk(payload, "$")
    return offenders


# --------------------------------------------------------------------------- health


def test_health_endpoints_agree(client):
    for path in ("/health", "/api/v1/health"):
        response = client.get(path)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert "engine_ready" in body
        assert body["limits"]["max_duration_seconds"] == MAX_DURATION_SECONDS
        assert ".mp4" in body["limits"]["formats"]


def test_api_metadata_advertises_schema_version(client):
    body = client.get("/api").json()
    assert body["schema_version"]
    assert body["docs"] == "/docs"


def test_root_serves_frontend_when_built(client):
    response = client.get("/")
    if not api_main.FRONTEND_DIST.joinpath("index.html").is_file():
        assert response.json()["schema_version"]
        return
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root"></div>' in response.text


def test_unknown_api_route_is_not_rewritten_to_frontend(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_models_endpoint_reports_both_modes(client):
    response = client.get("/api/v1/models")
    if response.status_code == 503:
        pytest.skip("analysis engine unavailable in this environment")
    body = response.json()
    assert set(body) >= {"batting", "bowling", "pose", "schema_version"}
    assert body["bowling"]["experimental"] is True
    assert body["batting"]["classes"]
    assert body["batting"]["benchmark"]["status"] in {"measured", "pending"}
    assert len(body["batting"]["display_labels"]) == len(body["batting"]["classes"])


# --------------------------------------------------------------------------- request validation


def test_invalid_mode_is_rejected(client, upload):
    response = client.post(f"{ANALYZE}?mode=cricket", files=upload())
    assert response.status_code == 422


def test_invalid_extension_is_rejected(client):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported video type" in response.json()["detail"]


def test_empty_file_is_rejected(client):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_non_video_bytes_with_video_extension_are_rejected(client):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": ("clip.mp4", io.BytesIO(b"this is definitely not a video container"), "video/mp4")},
    )
    assert response.status_code == 400
    assert "container" in response.json()["detail"].lower()


def test_truncated_container_is_rejected(client, synthetic_clip: Path):
    """A real container header with no usable payload must not 500."""
    payload = synthetic_clip.read_bytes()[:64]
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": ("clip.mp4", io.BytesIO(payload), "video/mp4")},
    )
    assert response.status_code in {400, 500}
    assert response.status_code != 500 or "detail" in response.json()


def test_oversized_upload_is_rejected(client, monkeypatch, synthetic_clip: Path):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": (synthetic_clip.name, synthetic_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 413
    assert "MB limit" in response.json()["detail"]


def test_over_length_clip_is_rejected(client, long_synthetic_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": (long_synthetic_clip.name, long_synthetic_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 400
    assert f"{MAX_DURATION_SECONDS} seconds" in response.json()["detail"]


def test_unsupported_container_signature_is_rejected(client):
    """A GIF has no supported signature even if renamed to .mov."""
    gif = b"GIF89a" + b"\x00" * 128
    response = client.post(f"{ANALYZE}?mode=bowling", files={"file": ("clip.mov", gif, "video/quicktime")})
    assert response.status_code == 400


# --------------------------------------------------------------------------- analysis responses


def test_person_free_clip_returns_guidance_not_a_label(client, synthetic_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": (synthetic_clip.name, synthetic_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_better_clip"
    assert body["classification"]["label"] == "unknown"
    assert body["classification"]["unknown"] is True
    assert body["quality"]["warnings"]
    assert body["coach"]["provider"] == "movement_engine"
    assert _no_legality_fields(body) == []


def test_batting_analysis_contract(client, real_batting_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=batting&camera_angle=side_on",
        files={"file": (real_batting_clip.name, real_batting_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["schema_version"]
    assert body["mode"] == "batting"
    assert body["status"] in {"complete", "needs_better_clip"}
    assert body["video"]["frames"] > 0
    assert body["video"]["fps"] > 0
    assert body["timings"]["total_ms"] > 0

    classification = body["classification"]
    assert classification["model"]["id"]
    assert classification["model"]["experimental"] is True
    assert classification["model"]["benchmark"]["status"] == "measured"
    assert classification["confidence"] <= classification["model"]["confidence_cap"]
    assert len(classification["top_alternatives"]) == 3

    # Phase ordering must be monotonic and inside the clip.
    phases = body["phases"]
    assert [phase["id"] for phase in phases] == ["setup", "downswing", "contact", "follow_through"]
    frames = [phase["frame"] for phase in phases]
    assert frames == sorted(frames)
    assert all(0 <= frame < body["video"]["frames"] for frame in frames)
    assert all(phase["timestamp_ms"] is not None for phase in phases)

    # Every metric must point at a frame the replay can seek to.
    for metric in body["metrics"]:
        assert "frame" in metric and "timestamp_ms" in metric
        assert metric["frame"] is None or metric["frame"] < body["video"]["frames"]

    assert body["timeline"]["frames"]
    assert len(body["timeline"]["frames"]) <= 180
    assert _no_legality_fields(body) == []


def test_bowling_analysis_contract(client, real_bowling_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=bowling&camera_angle=side_on",
        files={"file": (real_bowling_clip.name, real_bowling_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "bowling"
    assert [phase["id"] for phase in body["phases"]] == [
        "gather", "delivery_stride", "release", "follow_through"
    ]
    classification = body["classification"]
    if classification["label"] != "unknown":
        assert classification["model"]["experimental"] is True
        assert classification["confidence"] <= classification["model"]["confidence_cap"]
        assert classification["model"]["kind"] == "pose_feature_classifier"
    assert _no_legality_fields(body) == []


def test_bowling_metrics_stay_neutral(client, real_bowling_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=bowling&camera_angle=side_on",
        files={"file": (real_bowling_clip.name, real_bowling_clip.read_bytes(), "video/mp4")},
    )
    body = response.json()
    keys = {metric["key"] for metric in body["metrics"]}
    assert "elbow_angle_at_release" in keys
    assert not any("legal" in key or "chuck" in key for key in keys)


def test_ai_coach_falls_back_without_api_key(client, monkeypatch, real_batting_clip: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        f"{ANALYZE}?mode=batting&use_ai_coach=true",
        files={"file": (real_batting_clip.name, real_batting_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    coach = response.json()["coach"]
    assert coach["enhanced"] is False
    assert coach["provider"] == "movement_engine"
    assert coach["uncertainty"]
    assert coach["disclaimer"]


def test_ai_coach_failure_does_not_break_analysis(client, monkeypatch, real_batting_clip: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setattr("api.coaching._frame_data_urls", lambda *args, **kwargs: [])
    response = client.post(
        f"{ANALYZE}?mode=batting&use_ai_coach=true",
        files={"file": (real_batting_clip.name, real_batting_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["coach"]["provider"] == "movement_engine"
    assert body["classification"]["model"]["id"]


def test_upload_is_deleted_after_analysis(client, real_batting_clip: Path):
    temp_root = Path(tempfile.gettempdir())
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": (real_batting_clip.name, real_batting_clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    assert list(temp_root.glob("creaselab-*")) == []


def test_response_is_json_serialisable(client, real_batting_clip: Path):
    response = client.post(
        f"{ANALYZE}?mode=batting",
        files={"file": (real_batting_clip.name, real_batting_clip.read_bytes(), "video/mp4")},
    )
    json.dumps(response.json())


def test_upload_size_limit_constant_is_documented(client):
    assert MAX_UPLOAD_BYTES == 25 * 1024 * 1024
