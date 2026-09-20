"""Unit tests for model specs and the classification contract."""

from __future__ import annotations

import numpy as np
import pytest

from ml.model_spec import ModelSpec, ModelSpecError, load_spec
from ml.video_classifier import (
    BATTING_SPEC_FILE,
    BOWLING_SPEC_FILE,
    BattingVideoClassifier,
    BowlingPrototypeClassifier,
    build_classification,
)


@pytest.fixture(scope="module")
def batting_spec() -> ModelSpec:
    return load_spec(BATTING_SPEC_FILE)


def test_batting_spec_checksum_matches_artifact(batting_spec):
    assert batting_spec.verify_artifact() == batting_spec.data["sha256"]


def test_spec_checksum_mismatch_is_fatal(tmp_path):
    spec_path = tmp_path / "fake.json"
    (tmp_path / "weights.bin").write_bytes(b"not the real weights")
    spec_path.write_text(
        '{"id": "x", "artifact": "weights.bin", "sha256": "deadbeef", "classes": []}',
        encoding="utf-8",
    )
    with pytest.raises(ModelSpecError, match="checksum mismatch"):
        ModelSpec(path=spec_path, data={"id": "x", "artifact": "weights.bin", "sha256": "deadbeef", "classes": []}).verify_artifact()


def test_bowling_spec_needs_no_artifact():
    spec = load_spec(BOWLING_SPEC_FILE)
    assert spec.artifact_path is None
    assert spec.verify_artifact() == ""
    assert spec.experimental is True


def test_batting_spec_pins_preprocessing(batting_spec):
    input_spec = batting_spec.data["input"]
    assert input_spec["value_range"] == [0, 255]
    assert input_spec["frames"] == 30
    assert input_spec["channels"] == "RGB"


def test_probability_vector_is_normalised(batting_spec):
    probabilities = np.full(10, 0.1, dtype=np.float32)
    result = build_classification(probabilities, batting_spec)
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-6) == 1.0
    # A flat distribution is not a prediction, so the contract reports "unknown".
    assert result["unknown"] is True
    assert result["label"] == "unknown"


def test_low_margin_becomes_unknown(batting_spec):
    probabilities = np.zeros(10, dtype=np.float32)
    probabilities[0] = 0.6
    probabilities[1] = 0.55
    result = build_classification(probabilities, batting_spec)
    assert result["unknown"] is True
    assert result["label"] == "unknown"
    assert result["display_label"] == "Unclear action"


def test_low_score_becomes_unknown(batting_spec):
    probabilities = np.full(10, 0.1, dtype=np.float32)
    result = build_classification(probabilities, batting_spec)
    assert result["unknown"] is True
    assert result["raw_confidence"] == pytest.approx(0.1, abs=1e-6)


def test_confidence_is_capped_for_experimental_models(batting_spec):
    probabilities = np.zeros(10, dtype=np.float32)
    probabilities[3] = 0.999
    result = build_classification(probabilities, batting_spec)
    assert result["raw_confidence"] == pytest.approx(0.999, abs=1e-6)
    assert result["confidence"] == batting_spec.confidence_cap
    assert result["confidence"] < result["raw_confidence"]


def test_wrong_probability_count_is_rejected(batting_spec):
    with pytest.raises(ValueError):
        build_classification(np.zeros(7, dtype=np.float32), batting_spec)


def test_batting_classifier_output_shape(real_batting_clip):
    classifier = BattingVideoClassifier()
    result = classifier.predict(real_batting_clip)
    assert len(result["probabilities"]) == 10
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-4) == 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_batting_classifier_accepts_action_window(real_batting_clip):
    classifier = BattingVideoClassifier()
    result = classifier.predict(real_batting_clip, action_window=(10, 40))
    assert len(result["probabilities"]) == 10


def test_bowling_prototype_confidence_is_capped():
    classifier = BowlingPrototypeClassifier()
    features = {
        "detected_bowling_arm": "right",
        "peak_wrist_speed": 40.0,
        "action_duration_seconds": 0.4,
        "arm_path_vertical_range": 3.0,
        "shoulder_rotation_range": 90.0,
    }
    result = classifier.predict(features)
    assert result["confidence"] <= classifier.spec.confidence_cap
    assert result["model"]["experimental"] is True
    assert result["label"] in {"right_arm_pace", "right_arm_spin"}
    assert set(result["probabilities"]) == {"right_arm_pace", "right_arm_spin"}


def test_bowling_prototype_never_claims_legality():
    """Limitations may state that no legality judgement is made; the result never makes one."""
    classifier = BowlingPrototypeClassifier()
    result = classifier.predict({"detected_bowling_arm": "left", "peak_wrist_speed": 8.0})
    assert result["label"] not in {"illegal_delivery", "legal_delivery"}
    assert not any("legal" in key or "chuck" in key for key in result)
    assert result["probabilities"].keys() == {"left_arm_pace", "left_arm_spin"}
