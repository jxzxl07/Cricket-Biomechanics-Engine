"""Turn a training report into the shipped model spec for a pose classifier.

Chooses the unknown/abstain threshold from the measured cross-validated
probabilities (the smallest threshold that keeps at least 60% of clips while
maximising accuracy on the ones it keeps), then writes the spec the API serves.

    python -m research.training.publish_pose_model --mode bowling
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from config import DATA_DIR
from research.training.pose_dataset import feature_names

MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "evaluation"

DISPLAY_LABELS = {
    "cut": "Cut",
    "drive": "Drive",
    "flick": "Flick",
    "pull": "Pull",
    "sweep": "Sweep",
    "reverse_sweep": "Reverse sweep",
    "scoop": "Scoop",
    "left_arm_pace": "Left-arm pace",
    "right_arm_pace": "Right-arm pace",
    "left_arm_off": "Left-arm off spin",
    "right_arm_off": "Right-arm off spin",
    "left_arm_leg": "Left-arm leg spin",
    "right_arm_leg": "Right-arm leg spin",
}

MIN_COVERAGE = 0.6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_abstention_policy(report: dict) -> dict:
    """Pick score and margin cutoffs from out-of-session predictions.

    Candidate cutoffs come from the measured probability distribution rather
    than a fixed 0.30--0.90 range.  Strong regularisation is valuable for this
    tiny dataset but intentionally produces conservative probabilities; a
    fixed range would therefore make the improved models abstain on every clip.
    """
    grouped = report["grouped_cross_validation"]
    probabilities = np.asarray(grouped["probabilities"], dtype=float)
    predictions = grouped["predictions"]
    truth = grouped["truth"]

    ordered = np.sort(probabilities, axis=1)
    scores = ordered[:, -1]
    margins = ordered[:, -1] - ordered[:, -2]
    quantiles = np.linspace(0, 1, 21)
    score_cutoffs = np.unique(np.quantile(scores, quantiles))
    margin_cutoffs = np.unique(np.quantile(margins, quantiles))
    curve = []
    for threshold in score_cutoffs:
        for margin in margin_cutoffs:
            kept = np.flatnonzero((scores >= threshold) & (margins >= margin))
            if not len(kept):
                continue
            correct = sum(predictions[index] == truth[index] for index in kept)
            curve.append({
                "threshold": round(float(threshold), 6),
                "margin": round(float(margin), 6),
                "coverage": round(len(kept) / len(probabilities), 3),
                "accuracy": round(correct / len(kept), 3),
            })

    eligible = [point for point in curve if point["coverage"] >= MIN_COVERAGE]
    selected = max(eligible, key=lambda point: (point["accuracy"], point["coverage"])) if eligible else curve[0]
    # Copy before attaching the curve: the selected point is itself in the curve.
    best = dict(selected)
    # The full grid is noisy in a model card. Keep the useful score-only slice
    # plus the selected joint policy for reproducibility.
    best["curve"] = [point for point in curve if point["margin"] == round(float(margin_cutoffs[0]), 6)]
    return best


def build_spec(mode: str, report: dict, artifact: Path) -> dict:
    grouped = report["grouped_cross_validation"]
    abstention = choose_abstention_policy(report)
    classes = sorted(report["classes"])
    seen = grouped.get("accuracy_on_seen_classes")
    raw = grouped["accuracy"]
    experimental = True
    cap = 0.6 if mode == "batting" else 0.75
    return {
        "id": f"pose-{mode}-{report['selected_model']}-v2",
        "version": "2.0.0",
        "kind": "pose_feature_classifier",
        "artifact": artifact.name,
        "sha256": sha256(artifact),
        "architecture": f"{report['selected_model']} over {len(report['feature_names'])} MediaPipe pose features, exported to ONNX",
        "input": {
            "features": feature_names(mode),
            "source": "vision.features (identical code path in training and serving)",
            "imputation": "median (fitted during training, embedded in the ONNX graph)",
        },
        "classes": classes,
        "display_labels": {name: DISPLAY_LABELS.get(name, name.replace("_", " ").title()) for name in classes},
        "experimental": experimental,
        "confidence_cap": cap,
        "unknown_threshold": abstention["threshold"],
        "unknown_margin": abstention["margin"],
        "note": (
            "Trained on this project's own clips. Small dataset, one athlete: it generalises to new "
            "sessions only as well as the measured number shows."
        ),
        "limitations": [
            f"Trained on {report['clips']} clips from {len(report['sessions'])} recording sessions of a single athlete.",
            f"Leave-one-session-out accuracy is {raw * 100:.0f}%"
            + (f" (rising to {seen * 100:.0f}% on clips whose class appears in the training sessions)." if seen is not None else "."),
            "Some classes only appear in one session, so a held-out session can contain classes the model never saw.",
            "A random split of this dataset reports a much higher number; that figure is leakage and is not published as accuracy.",
            "Not validated on other athletes, other cameras, or match conditions.",
        ],
        "benchmark": {
            "status": "measured",
            "dataset": f"This project's clips ({report['clips']} clips, {len(report['sessions'])} sessions)",
            "independent": False,
            "scheme": grouped["scheme"],
            "top1_accuracy": raw,
            "top1_accuracy_on_seen_classes": seen,
            "clips_on_seen_classes": grouped.get("clips_on_seen_classes"),
            "random_split_accuracy_for_reference": report["random_split_accuracy_for_reference"],
            "leakage_gap": report["leakage_gap"],
            "per_class_recall": grouped["per_class_recall"],
            "folds": grouped["folds"],
            "mean_confidence_correct": report["mean_confidence_correct"],
            "mean_confidence_incorrect": report["mean_confidence_incorrect"],
            "abstention_policy": abstention,
            "candidates": report["candidates"],
            "source": f"data/evaluation/{mode}_pose_training_report.json",
            "interpretation": (
                "Grouped by recording session. Treat this as the honest ceiling for a single-athlete dataset "
                "of this size, not as general shot-recognition accuracy."
            ),
        },
        "provenance": {
            "trained_by": "research/training/train_action_model.py",
            "trained_at": report["trained_at"],
            "data": "data/landmarks (gitignored personal footage)",
            "weights_license": "Trained here; no third-party weights",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batting", "bowling"], required=True)
    args = parser.parse_args()

    report = json.loads((REPORTS_DIR / f"{args.mode}_pose_training_report.json").read_text(encoding="utf-8"))
    artifact = MODELS_DIR / report["artifact"]
    spec = build_spec(args.mode, report, artifact)
    target = MODELS_DIR / f"{args.mode}_pose.json"
    target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(json.dumps({
        "id": spec["id"],
        "classes": spec["classes"],
        "unknown_threshold": spec["unknown_threshold"],
        "unknown_margin": spec["unknown_margin"],
        "confidence_cap": spec["confidence_cap"],
        "benchmark": {k: v for k, v in spec["benchmark"].items() if k not in {"folds", "per_class_recall", "candidates", "abstention_policy"}},
    }, indent=2))


if __name__ == "__main__":
    main()
