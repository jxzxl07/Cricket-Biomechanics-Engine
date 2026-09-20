"""Train the pose-feature action classifiers and export them to ONNX.

Honest evaluation is the point of this script:

* Leave-one-recording-session-out cross-validation. Clips from the same session
  are near-duplicates, so a random split would report leakage-inflated numbers.
* The random-split score is also printed, purely to show how much it flatters.
* Per-class recall and the confusion matrix are reported, because a single
  accuracy figure hides classes the model never gets right.

    python -m research.training.train_action_model --mode batting
    python -m research.training.train_action_model --mode bowling
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR
from research.training.pose_dataset import feature_names

MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "evaluation"


def load_dataset(path: Path, mode: str):
    names = feature_names(mode)
    X, y, groups, clips = [], [], [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            X.append([np.nan if row[name] == "" else float(row[name]) for name in names])
            y.append(row["label"])
            groups.append(row["session"])
            clips.append(row["clip"])
    return np.array(X, dtype=np.float64), np.array(y), np.array(groups), clips


def candidates() -> dict[str, object]:
    return {
        # With fewer than 50 clips per mode, the original C=1 model overfits
        # recording-session details.  This setting was added after comparing
        # whole-session holdouts; it deliberately favours a smoother boundary.
        "strongly_regularized_logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.001, class_weight="balanced")),
        ]),
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=3, random_state=0
            )),
        ]),
    }


def grouped_cross_validation(model, X, y, groups) -> dict:
    logo = LeaveOneGroupOut()
    predictions = np.empty(len(y), dtype=object)
    classes = sorted(set(y))
    probabilities = np.zeros((len(y), len(classes)))
    index = {label: position for position, label in enumerate(classes)}
    folds = []
    for train_idx, test_idx in logo.split(X, y, groups):
        clone = _clone(model)
        clone.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = clone.predict(X[test_idx])
        if hasattr(clone, "predict_proba"):
            proba = clone.predict_proba(X[test_idx])
            for row, label in enumerate(clone.classes_):
                probabilities[test_idx, index[label]] = proba[:, row]
        held_out = sorted(set(y[test_idx]))
        unseen = sorted(set(held_out) - set(y[train_idx]))
        folds.append({
            "held_out_session": str(groups[test_idx][0]),
            "held_out_clips": int(len(test_idx)),
            "accuracy": float(np.mean(predictions[test_idx] == y[test_idx])),
            "classes_in_held_out_session": held_out,
            # If a class never appears in the training sessions, no model can
            # predict it. Reported so a low score is not misread as a model bug.
            "classes_absent_from_training": unseen,
        })
    accuracy = float(np.mean(predictions == y))
    # Clips whose class never appeared in the training sessions cannot be
    # predicted by any model. The adjusted figure reports capability on classes
    # the model could actually have learned, and is always reported alongside
    # the raw figure rather than instead of it.
    seen_mask = np.array([
        label not in fold["classes_absent_from_training"]
        for label, fold in zip(y, _fold_for_each_clip(y, groups, folds))
    ]) if folds else np.ones(len(y), dtype=bool)
    accuracy_on_seen = float(np.mean(predictions[seen_mask] == y[seen_mask])) if seen_mask.any() else None
    return {
        "accuracy": accuracy,
        "accuracy_on_seen_classes": accuracy_on_seen,
        "clips_on_seen_classes": int(seen_mask.sum()),
        "classes": classes,
        "folds": folds,
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "per_class_recall": {
            label: float(np.mean(predictions[y == label] == label)) if np.any(y == label) else None
            for label in classes
        },
        "confusion": {
            actual: {
                predicted: int(np.sum((y == actual) & (predictions == predicted)))
                for predicted in classes
                if np.any((y == actual) & (predictions == predicted))
            }
            for actual in classes
        },
    }


def _fold_for_each_clip(y, groups, folds):
    """Map every clip back to the fold it was evaluated in."""
    lookup = {fold["held_out_session"]: fold for fold in folds}
    return [lookup[str(group)] for group in groups]


def _clone(model):
    from sklearn.base import clone as sklearn_clone

    return sklearn_clone(model)


def random_split_accuracy(model, X, y) -> float:
    """Leakage-inflated reference number, printed only for comparison."""
    scores = []
    for seed in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        clone = _clone(model)
        clone.fit(X_train, y_train)
        scores.append(float(np.mean(clone.predict(X_test) == y_test)))
    return statistics.mean(scores)


def export_onnx(model, mode: str, feature_count: int, target: Path, sample: np.ndarray) -> None:
    from skl2onnx import to_onnx

    onnx_model = to_onnx(model, sample.astype(np.float32), options={"zipmap": False})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_model.SerializeToString())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batting", "bowling"], required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="ONNX artifact path")
    args = parser.parse_args()

    mode = args.mode
    dataset = args.dataset or REPORTS_DIR / f"{mode}_pose_dataset.csv"
    X, y, groups, clips = load_dataset(dataset, mode)
    print(f"{mode}: {len(y)} clips, {len(set(y))} classes, {len(set(groups))} sessions")
    print("classes:", sorted(set(y)))
    print("sessions:", sorted(set(groups)))

    results = {}
    for name, model in candidates().items():
        grouped = grouped_cross_validation(model, X, y, groups)
        leaky = random_split_accuracy(model, X, y)
        results[name] = {"grouped": grouped, "random_split": leaky}
        print(f"\n{name}: grouped={grouped['accuracy']:.3f}  random_split={leaky:.3f}  (gap {leaky - grouped['accuracy']:+.3f})")
        print("  per-class recall:", {k: (None if v is None else round(v, 2)) for k, v in grouped["per_class_recall"].items()})

    best_name = max(results, key=lambda key: results[key]["grouped"]["accuracy"])
    best = results[best_name]
    print(f"\nselected: {best_name} (grouped accuracy {best['grouped']['accuracy']:.3f})")

    # Final model trained on every clip, for serving.
    final_model = _clone(candidates()[best_name])
    final_model.fit(X, y)
    artifact = args.out or MODELS_DIR / f"{mode}_pose.onnx"
    export_onnx(final_model, mode, X.shape[1], artifact, X[:1])

    classes = sorted(set(y))
    probabilities = np.array(best["grouped"]["probabilities"])
    correct = np.array(best["grouped"]["predictions"]) == y
    confident = probabilities.max(axis=1) if probabilities.size else np.zeros(len(y))

    report = {
        "mode": mode,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset.relative_to(DATA_DIR.parent)),
        "clips": int(len(y)),
        "classes": classes,
        "sessions": sorted(set(groups)),
        "feature_names": feature_names(mode),
        "selected_model": best_name,
        "grouped_cross_validation": {
            "scheme": "leave-one-recording-session-out",
            "accuracy": best["grouped"]["accuracy"],
            "accuracy_on_seen_classes": best["grouped"]["accuracy_on_seen_classes"],
            "clips_on_seen_classes": best["grouped"]["clips_on_seen_classes"],
            "per_class_recall": best["grouped"]["per_class_recall"],
            "confusion": best["grouped"]["confusion"],
            "folds": best["grouped"]["folds"],
            "classes": best["grouped"]["classes"],
            "truth": y.tolist(),
            "predictions": best["grouped"]["predictions"],
            "probabilities": best["grouped"]["probabilities"],
        },
        "random_split_accuracy_for_reference": best["random_split"],
        "leakage_gap": best["random_split"] - best["grouped"]["accuracy"],
        "mean_confidence_correct": float(confident[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(confident[~correct].mean()) if (~correct).any() else None,
        "candidates": {name: value["grouped"]["accuracy"] for name, value in results.items()},
        "artifact": artifact.name,
    }
    report_path = REPORTS_DIR / f"{mode}_pose_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {artifact} and {report_path}")
    print(json.dumps({k: v for k, v in report.items() if k not in {"feature_names"}}, indent=2)[:1600])


if __name__ == "__main__":
    main()
