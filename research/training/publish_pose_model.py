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


def choose_threshold(report: dict) -> dict:
    """Pick the abstain threshold from measured probabilities."""
    grouped = report["grouped_cross_validation"]
    probabilities = grouped["probabilities"]
    predictions = grouped["predictions"]
    truth = grouped["truth"]

    curve = []
    for threshold in [round(0.30 + 0.05 * step, 2) for step in range(13)]:
        kept = [index for index, row in enumerate(probabilities) if max(row) >= threshold]
        if not kept:
            continue
        correct = sum(1 for index in kept if predictions[index] == truth[index])
        curve.append({
            "threshold": threshold,
            "coverage": round(len(kept) / len(probabilities), 3),
            "accuracy": round(correct / len(kept), 3),
        })

    eligible = [point for point in curve if point["coverage"] >= MIN_COVERAGE]
    selected = max(eligible, key=lambda point: point["accuracy"]) if eligible else curve[-1]
    # Copy before attaching the curve: the selected point is itself in the curve.
    best = dict(selected)
    best["curve"] = curve
    return best


def build_spec(mode: str, report: dict, artifact: Path) -> dict:
    grouped = report["grouped_cross_validation"]
    threshold = choose_threshold(report)
    classes = sorted(report["classes"])
    seen = grouped.get("accuracy_on_seen_classes")
    raw = grouped["accuracy"]
    experimental = True
    cap = 0.6 if mode == "batting" else 0.75
    return {
        "id": f"pose-{mode}-{report['selected_model']}-v1",
        "version": "1.0.0",
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
        "unknown_threshold": threshold["threshold"],
        "unknown_margin": 0.10,
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
            "abstain_threshold": threshold,
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
        "confidence_cap": spec["confidence_cap"],
        "benchmark": {k: v for k, v in spec["benchmark"].items() if k not in {"folds", "per_class_recall", "candidates", "abstain_threshold"}},
    }, indent=2))


if __name__ == "__main__":
    main()
