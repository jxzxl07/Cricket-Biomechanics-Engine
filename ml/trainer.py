import csv
import json
import os
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import DATA_DIR
from database.database import get_accepted_training_records, mark_trained
from vision.features import (
    BATTING_FEATURE_NAMES,
    BOWLING_FEATURE_NAMES,
    build_training_csv,
)


def get_feature_names(mode):
    if mode == "bowling":
        return BOWLING_FEATURE_NAMES

    if mode == "batting":
        return BATTING_FEATURE_NAMES

    raise ValueError(f"Unknown mode: {mode}")


def get_model_path(mode):
    return DATA_DIR / "models" / f"{mode}.joblib"


def get_metrics_path(mode):
    return DATA_DIR / "evaluation" / f"{mode}_metrics.json"


def get_report_path(mode):
    return DATA_DIR / "evaluation" / f"{mode}_classification_report.txt"


def get_confusion_matrix_path(mode):
    return DATA_DIR / "evaluation" / f"{mode}_confusion_matrix.png"


def load_training_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def prepare_dataset(rows, feature_names):
    x_values = []
    y_values = []

    for row in rows:
        label = row["label"]

        features = []

        for feature_name in feature_names:
            value = row[feature_name]

            if value == "" or value is None:
                raise ValueError(
                    f"Missing feature {feature_name} for clip {row['clip_path']}"
                )

            features.append(float(value))

        x_values.append(features)
        y_values.append(label)

    return x_values, y_values


def choose_cv_folds(label_counts, requested_folds=5):
    smallest_class_count = min(label_counts.values())

    if smallest_class_count < 2:
        return 0

    return min(requested_folds, smallest_class_count)


def save_confusion_matrix_plot(confusion, labels, output_path):
    if confusion is None:
        return None

    matplotlib_cache_dir = DATA_DIR / "evaluation" / "matplotlib_cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_size = max(6, len(labels) * 1.2)
    figure, axes = plt.subplots(figsize=(figure_size, figure_size))

    image = axes.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axes)

    axes.set_title("Confusion Matrix")
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_xticks(range(len(labels)))
    axes.set_yticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticklabels(labels)

    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            axes.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max(max(r) for r in confusion) / 2 else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return output_path


def format_percent(value):
    if value is None:
        return "not available yet"

    return f"{value:.2%}"


def format_confusion_matrix(confusion, labels):
    if confusion is None:
        return "Not available yet. Add more clips per class, then train again."

    index_width = max(2, len(str(len(labels) - 1)))
    cell_width = 4

    label_key = [
        f"{index}: {label}"
        for index, label in enumerate(labels)
    ]

    header = "Actual \\ Pred".ljust(14)
    header += "".join(str(index).rjust(cell_width) for index in range(len(labels)))

    lines = [
        "Label key:",
        *label_key,
        "",
        header,
    ]

    for row_index, row in enumerate(confusion):
        line = str(row_index).rjust(index_width).ljust(14)
        line += "".join(str(value).rjust(cell_width) for value in row)
        lines.append(line)

    return "\n".join(lines)


def format_label_counts(label_counts):
    if not label_counts:
        return "-"

    return ", ".join(
        f"{label}: {count}"
        for label, count in label_counts.items()
    )


def load_evaluation_metrics(mode):
    metrics_path = get_metrics_path(mode)

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No saved {mode} evaluation found yet. Train the model first."
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    report_path = get_report_path(mode)

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as file:
            report_text = file.read().strip()
    else:
        report_text = "Classification report not available yet."

    metrics["report_text"] = report_text
    metrics["metrics_path"] = str(metrics_path)
    metrics["report_path"] = str(report_path)

    labels = metrics.get("confusion_matrix_labels")
    confusion = metrics.get("confusion_matrix")
    confusion_matrix_path = metrics.get("confusion_matrix_path")

    if confusion is not None and labels:
        if not confusion_matrix_path or not Path(confusion_matrix_path).exists():
            generated_path = save_confusion_matrix_plot(
                confusion,
                labels,
                get_confusion_matrix_path(mode),
            )

            if generated_path is not None:
                metrics["confusion_matrix_path"] = str(generated_path)

                with open(metrics_path, "w", encoding="utf-8") as file:
                    json.dump(metrics, file, indent=2)

    return metrics


def format_evaluation_summary(mode):
    metrics = load_evaluation_metrics(mode)
    labels = metrics.get("confusion_matrix_labels", [])
    confusion = metrics.get("confusion_matrix")
    cv_scores = metrics.get("cv_accuracy_scores", [])

    if cv_scores:
        cv_score_text = ", ".join(f"{score:.0%}" for score in cv_scores)
    else:
        cv_score_text = "not available yet"

    confusion_text = format_confusion_matrix(confusion, labels)

    return (
        f"Latest {mode} evaluation\n\n"
        "What clips were evaluated:\n"
        "A stratified 25% holdout split from the accepted labelled clips. "
        "After evaluation, the final model is refit on all accepted clips.\n\n"
        f"Rows: {metrics.get('row_count')}\n"
        f"Classes: {format_label_counts(metrics.get('label_counts', {}))}\n\n"
        f"5-fold CV accuracy: {format_percent(metrics.get('cv_accuracy_mean'))}\n"
        f"CV fold scores: {cv_score_text}\n"
        f"Holdout accuracy: {format_percent(metrics.get('holdout_accuracy'))}\n\n"
        f"Confusion matrix:\n{confusion_text}\n\n"
        f"Metrics saved:\n{metrics.get('metrics_path')}\n\n"
        f"Report saved:\n{metrics.get('report_path')}\n\n"
        f"Confusion matrix image:\n{metrics.get('confusion_matrix_path', '-')}"
    )


def train_model(mode):
    feature_names = get_feature_names(mode)

    csv_path, row_count = build_training_csv(mode)
    rows = load_training_rows(csv_path)
    model_path = get_model_path(mode)
    accepted_records = get_accepted_training_records(mode)
    untrained_records = get_accepted_training_records(mode, only_untrained=True)
    untrained_count = len(untrained_records)

    if model_path.exists() and untrained_count == 0:
        return {
            "success": False,
            "message": (
                f"No new accepted {mode} clips since the last training run. "
                "Existing model left unchanged."
            ),
            "csv_path": str(csv_path),
            "row_count": row_count,
            "new_clip_count": 0,
        }

    if row_count < 2:
        return {
            "success": False,
            "message": (
                f"Need at least 2 labelled {mode} clips to train. "
                f"Found {row_count}."
            ),
            "csv_path": str(csv_path),
            "row_count": row_count,
        }

    label_counts = Counter(row["label"] for row in rows)

    if len(label_counts) < 2:
        return {
            "success": False,
            "message": (
                f"Need at least 2 different {mode} classes to train. "
                f"Found: {dict(label_counts)}"
            ),
            "csv_path": str(csv_path),
            "row_count": row_count,
            "label_counts": dict(label_counts),
        }

    x_values, y_values = prepare_dataset(rows, feature_names)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    cv_folds = choose_cv_folds(label_counts)

    if cv_folds >= 2:
        cross_validator = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        cv_scores = cross_val_score(
            model,
            x_values,
            y_values,
            cv=cross_validator,
        )

        cv_accuracy_mean = float(cv_scores.mean())
        cv_accuracy_scores = [float(score) for score in cv_scores]

    else:
        cv_accuracy_mean = None
        cv_accuracy_scores = []

    can_make_test_split = all(count >= 2 for count in label_counts.values())

    if can_make_test_split and len(rows) >= 4:
        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.25,
            random_state=42,
            stratify=y_values,
        )

        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)

        holdout_accuracy = float(accuracy_score(y_test, y_predicted))
        labels = sorted(label_counts.keys())

        confusion = confusion_matrix(
            y_test,
            y_predicted,
            labels=labels,
        ).tolist()

        report_text = classification_report(
            y_test,
            y_predicted,
            labels=labels,
            zero_division=0,
        )

    else:
        model.fit(x_values, y_values)

        holdout_accuracy = None
        labels = sorted(label_counts.keys())
        confusion = None
        report_text = (
            "Not enough data for a holdout validation split yet.\n"
            "Model was trained on all available rows."
        )

    confusion_matrix_path = get_confusion_matrix_path(mode)
    saved_confusion_path = save_confusion_matrix_plot(
        confusion,
        labels,
        confusion_matrix_path,
    )

    model.fit(x_values, y_values)

    model_package = {
        "mode": mode,
        "model": model,
        "feature_names": feature_names,
        "labels": sorted(label_counts.keys()),
        "label_counts": dict(label_counts),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, model_path)

    metrics = {
        "mode": mode,
        "row_count": row_count,
        "label_counts": dict(label_counts),
        "feature_names": feature_names,
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "cv_folds": cv_folds,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_scores": cv_accuracy_scores,
        "holdout_accuracy": holdout_accuracy,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion,
        "confusion_matrix_path": str(saved_confusion_path)
        if saved_confusion_path is not None
        else None,
    }

    metrics_path = get_metrics_path(mode)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_path = get_report_path(mode)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    mark_trained([record["id"] for record in accepted_records])

    return {
        "success": True,
        "message": (
            f"Trained {mode} model with {row_count} rows "
            f"({untrained_count} new clips)."
        ),
        "mode": mode,
        "row_count": row_count,
        "new_clip_count": untrained_count,
        "label_counts": dict(label_counts),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_matrix_path": str(saved_confusion_path)
        if saved_confusion_path is not None
        else None,
        "cv_accuracy_mean": cv_accuracy_mean,
        "holdout_accuracy": holdout_accuracy,
    }
