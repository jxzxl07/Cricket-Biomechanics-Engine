"""ONNX inference for the trained pose-feature action classifiers.

Both action models consume the pose features the API already computes, so the
training pipeline and serving path share one feature implementation. The
preprocessing (median imputation, scaling where used) is embedded in the ONNX
graph, so serving cannot drift from what was benchmarked.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ml.classification import build_classification
from ml.model_spec import ModelSpec, load_spec

BATTING_SPEC_FILE = "batting_pose.json"
BOWLING_SPEC_FILE = "bowling_pose.json"

# Feature names that are derived rather than copied straight from the extractor.
DERIVED_FEATURES = {"bowling_arm_is_left"}


class PoseActionClassifier:
    """Classifies an action from its pose-feature dictionary."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.feature_names: list[str] = list(spec.data["input"]["features"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        artifact = spec.artifact_path
        if artifact is None:
            raise ValueError(f"{spec.id} has no artifact to load")
        self.session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.probability_output = self._probability_output_index()

    def _probability_output_index(self) -> int:
        """Find the output that holds per-class probabilities."""
        expected = len(self.spec.classes)
        for index, output in enumerate(self.session.get_outputs()):
            shape = output.shape
            if shape and isinstance(shape[-1], int) and shape[-1] == expected:
                return index
        raise ValueError(f"{self.spec.id}: no output with {expected} class scores")

    def feature_vector(self, features: dict) -> np.ndarray:
        row = []
        for name in self.feature_names:
            if name == "bowling_arm_is_left":
                row.append(1.0 if features.get("detected_bowling_arm") == "left" else 0.0)
                continue
            value = features.get(name)
            row.append(math.nan if value is None else float(value))
        return np.array([row], dtype=np.float32)

    def predict(self, features: dict) -> dict:
        # An imputer can turn a completely absent pose into a plausible-looking
        # median athlete. Never let that synthetic row become a classification.
        observed = [features.get(name) for name in self.feature_names if name not in DERIVED_FEATURES]
        if not any(value is not None for value in observed):
            return build_classification(np.full(len(self.spec.classes), 1 / len(self.spec.classes)), self.spec)
        probabilities = np.asarray(
            self.session.run(None, {self.input_name: self.feature_vector(features)})[self.probability_output]
        )[0]
        return build_classification(probabilities, self.spec)


def load_batting_classifier() -> PoseActionClassifier:
    return PoseActionClassifier(load_spec(BATTING_SPEC_FILE))


def load_bowling_classifier() -> PoseActionClassifier:
    return PoseActionClassifier(load_spec(BOWLING_SPEC_FILE))


def artifact_is_available(spec: ModelSpec) -> bool:
    path: Path | None = spec.artifact_path
    return path is not None and path.exists()
