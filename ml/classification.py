"""The public classification contract shared by every model adapter.

Keeping this in one place means the API response shape cannot drift between the
trained pose models and the research-only video model.
"""

from __future__ import annotations

import numpy as np

from ml.model_spec import ModelSpec


def build_classification(probabilities: np.ndarray, spec: ModelSpec) -> dict:
    """Turn a probability vector into the public classification contract."""
    classes = spec.classes
    if len(probabilities) != len(classes):
        raise ValueError(f"{spec.id} returned {len(probabilities)} scores for {len(classes)} classes")
    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    runner_up = ranked[1][1]
    confident = top_score >= spec.unknown_threshold and (top_score - runner_up) >= spec.unknown_margin
    display_label = spec.display_label(top_label)
    if not confident:
        top_label, display_label = "unknown", "Unclear action"
    shown_score = min(top_score, spec.confidence_cap) if spec.confidence_cap < 1 else top_score
    return {
        "label": top_label,
        "display_label": display_label,
        "confidence": float(shown_score),
        "raw_confidence": float(top_score),
        "unknown": not confident,
        "low_confidence": not confident,
        "top_alternatives": [
            {"label": name, "display_label": spec.display_label(name), "probability": float(score)}
            for name, score in ranked[1:4]
        ],
        "probabilities": {name: float(score) for name, score in ranked},
        "model": {
            "id": spec.id,
            "version": spec.version,
            "kind": spec.kind,
            "experimental": spec.experimental,
            "note": spec.note,
            "confidence_cap": spec.confidence_cap,
            "benchmark": spec.benchmark(),
            "limitations": spec.data.get("limitations", []),
        },
    }
