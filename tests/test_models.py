"""Unit tests for model specs and the classification contract."""

from __future__ import annotations

import numpy as np
import pytest

from ml.classification import build_classification
from ml.model_spec import ModelSpec, ModelSpecError, load_spec
from ml.pose_classifier import (
    BATTING_SPEC_FILE,
    BOWLING_SPEC_FILE,
    PoseActionClassifier,
    load_batting_classifier,
    load_bowling_classifier,
)
from vision.features import extract_batting_features_from_data, extract_bowling_features_from_data, load_landmark_json


@pytest.fixture(scope="module")
def batting_spec() -> ModelSpec:
    return load_spec(BATTING_SPEC_FILE)


@pytest.fixture(scope="module")
def bowling_spec() -> ModelSpec:
    return load_spec(BOWLING_SPEC_FILE)


@pytest.fixture(scope="module")
def batting_classifier() -> PoseActionClassifier:
    return load_batting_classifier()


@pytest.fixture(scope="module")
def bowling_classifier() -> PoseActionClassifier:
    return load_bowling_classifier()


# --------------------------------------------------------------------------- specs


@pytest.mark.parametrize("spec_file", [BATTING_SPEC_FILE, BOWLING_SPEC_FILE])
def test_spec_checksum_matches_artifact(spec_file):
    spec = load_spec(spec_file)
    assert spec.verify_artifact() == spec.data["sha256"]
    assert spec.artifact_path is not None and spec.artifact_path.exists()


@pytest.mark.parametrize("spec_file", [BATTING_SPEC_FILE, BOWLING_SPEC_FILE])
def test_specs_publish_a_measured_benchmark(spec_file):
    benchmark = load_spec(spec_file).benchmark()
    assert benchmark["status"] == "measured"
    assert benchmark["scheme"] == "leave-one-recording-session-out"
    assert 0.0 <= benchmark["top1_accuracy"] <= 1.0
    # The leakage-inflated random split must always be shown next to the honest
    # number, never instead of it.
    assert benchmark["random_split_accuracy_for_reference"] > benchmark["top1_accuracy"]


@pytest.mark.parametrize("spec_file", [BATTING_SPEC_FILE, BOWLING_SPEC_FILE])
def test_specs_pin_their_feature_pipeline(spec_file):
    spec = load_spec(spec_file)
    assert spec.data["input"]["features"]
    assert spec.data["input"]["source"].startswith("vision.features")
    assert spec.data["kind"] == "pose_feature_classifier"
    assert spec.experimental is True


def test_spec_checksum_mismatch_is_fatal(tmp_path):
    spec_path = tmp_path / "fake.json"
    (tmp_path / "weights.bin").write_bytes(b"not the real weights")
    spec = ModelSpec(
        path=spec_path,
        data={"id": "x", "artifact": "weights.bin", "sha256": "deadbeef", "classes": []},
    )
    with pytest.raises(ModelSpecError, match="checksum mismatch"):
        spec.verify_artifact()


def test_missing_artifact_is_fatal(tmp_path):
    spec = ModelSpec(path=tmp_path / "fake.json", data={"id": "x", "artifact": "nope.onnx", "sha256": "aa", "classes": []})
    with pytest.raises(ModelSpecError, match="missing"):
        spec.verify_artifact()


# --------------------------------------------------------------------------- contract


def test_probability_vector_is_normalised(batting_spec):
    result = build_classification(np.full(len(batting_spec.classes), 1 / len(batting_spec.classes)), batting_spec)
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-6) == 1.0
    # A flat distribution is not a prediction.
    assert result["unknown"] is True
    assert result["label"] == "unknown"


def test_low_margin_becomes_unknown(batting_spec):
    probabilities = np.full(len(batting_spec.classes), 0.01)
    probabilities[0] = batting_spec.unknown_threshold + 0.1
    probabilities[1] = probabilities[0] - batting_spec.unknown_margin / 2
    assert build_classification(probabilities, batting_spec)["unknown"] is True


def test_low_score_becomes_unknown(batting_spec):
    probabilities = np.full(len(batting_spec.classes), 0.01)
    probabilities[0] = batting_spec.unknown_threshold / 2
    result = build_classification(probabilities, batting_spec)
    assert result["unknown"] is True
    assert result["raw_confidence"] == pytest.approx(batting_spec.unknown_threshold / 2, abs=1e-6)


def test_confidence_is_capped_for_experimental_models(batting_spec):
    probabilities = np.zeros(len(batting_spec.classes))
    probabilities[0] = 0.999
    result = build_classification(probabilities, batting_spec)
    assert result["raw_confidence"] == pytest.approx(0.999, abs=1e-6)
    assert result["confidence"] == batting_spec.confidence_cap
    assert result["confidence"] < result["raw_confidence"]


def test_wrong_probability_count_is_rejected(batting_spec):
    with pytest.raises(ValueError):
        build_classification(np.zeros(3), batting_spec)


def test_top_alternatives_exclude_the_winner(batting_spec):
    result = build_classification(np.full(len(batting_spec.classes), 1 / len(batting_spec.classes)), batting_spec)
    assert len(result["top_alternatives"]) == 3
    assert result["label"] not in {item["label"] for item in result["top_alternatives"]}


# --------------------------------------------------------------------------- pose classifiers


def test_batting_classifier_scores_a_real_clip(batting_classifier, real_batting_clip):
    from api.analysis import AnalysisEngine

    engine = AnalysisEngine()
    features = engine_features(engine, real_batting_clip, "batting")
    result = batting_classifier.predict(features)
    assert set(result["probabilities"]) == set(batting_classifier.spec.classes)
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-4) == 1.0
    assert 0.0 <= result["confidence"] <= batting_classifier.spec.confidence_cap


def test_bowling_classifier_scores_a_real_clip(bowling_classifier, real_bowling_clip):
    from api.analysis import AnalysisEngine

    engine = AnalysisEngine()
    features = engine_features(engine, real_bowling_clip, "bowling")
    result = bowling_classifier.predict(features)
    assert set(result["probabilities"]) == set(bowling_classifier.spec.classes)
    assert 0.0 <= result["confidence"] <= bowling_classifier.spec.confidence_cap


def engine_features(engine, clip, mode):
    """Run the production feature path for a clip, without a second video pass."""
    import tempfile
    from pathlib import Path

    from vision.features import extract_features_from_data, load_landmark_json
    from vision.pose import extract_landmarks_from_video

    with tempfile.TemporaryDirectory() as tmpdir:
        landmarks = Path(tmpdir) / "clip.landmarks.json"
        extract_landmarks_from_video(clip, mode, output_path=landmarks)
        return extract_features_from_data(load_landmark_json(landmarks), mode)


def test_feature_vector_handles_missing_values(batting_classifier):
    vector = batting_classifier.feature_vector({"peak_hand_speed": 4.0})
    assert vector.shape == (1, len(batting_classifier.feature_names))
    assert np.isnan(vector).any(), "absent features must reach the imputer as NaN, not as zeros"
    assert vector[0][batting_classifier.feature_names.index("peak_hand_speed")] == pytest.approx(4.0)


def test_bowling_feature_vector_encodes_handedness(bowling_classifier):
    left = bowling_classifier.feature_vector({"detected_bowling_arm": "left"})
    right = bowling_classifier.feature_vector({"detected_bowling_arm": "right"})
    index = bowling_classifier.feature_names.index("bowling_arm_is_left")
    assert left[0][index] == 1.0
    assert right[0][index] == 0.0


def test_classifier_abstains_on_a_flat_feature_vector(batting_classifier):
    """All-NaN features must not produce a confident label."""
    result = batting_classifier.predict({})
    assert result["unknown"] is True
    assert result["label"] == "unknown"


def test_batting_taxonomy_covers_the_labelled_shots(batting_spec):
    assert {"cut", "drive", "flick", "pull", "sweep", "reverse_sweep", "scoop"} == set(batting_spec.classes)


def test_bowling_taxonomy_is_six_action_classes(bowling_spec):
    assert {
        "left_arm_pace", "right_arm_pace", "left_arm_off", "right_arm_off", "left_arm_leg", "right_arm_leg"
    } == set(bowling_spec.classes)


def test_bowling_spec_never_claims_legality(bowling_spec):
    text = " ".join(bowling_spec.data.get("limitations", [])).lower()
    assert "illegal" not in text and "chucking" not in text
    assert "legality" not in " ".join(bowling_spec.classes)


def test_video_model_is_research_only():
    """The broadcast video model is kept for comparison, not loaded by the API."""
    from ml.video_classifier import BattingVideoClassifier

    import api.analysis as analysis

    assert "BattingVideoClassifier" not in dir(analysis)
    assert BattingVideoClassifier().spec.id == "efficientnetb0-gru-cricshot10-v1"
