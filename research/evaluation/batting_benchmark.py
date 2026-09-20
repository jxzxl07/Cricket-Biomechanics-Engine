"""Zero-training benchmark of the ONNX batting model on the historical phone clips.

This uses exactly the production preprocessing path (``ml.video_classifier``)
so the numbers describe what the API actually serves.

Usage:
    python -m research.evaluation.batting_benchmark
    python -m research.evaluation.batting_benchmark --json out.json --markdown out.md

Design notes
------------
* The stored folder labels are coarser or finer than the 10 model classes.
  ``reverse_sweep`` and ``scoop`` have no corresponding model class and are
  excluded from headline accuracy, then reported separately.
* Clips were recorded in a handful of sessions only minutes apart, so clips
  within a session are near-duplicates. Per-clip accuracy overstates
  independent evidence; we also report a session-grouped summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DATA_DIR
from ml.video_classifier import BattingVideoClassifier
from vision.features import extract_batting_features_from_data, load_landmark_json

BATTING_CLASSES = [
    "cover_drive", "defence", "flick", "hook", "late_cut",
    "lofted_shot", "pull", "square_cut", "straight_drive", "sweep",
]

# Stored folder label -> set of model classes considered correct.
COMPATIBLE_LABELS: dict[str, set[str]] = {
    "cut": {"square_cut", "late_cut"},
    "drive": {"cover_drive", "straight_drive"},
    "flick": {"flick"},
    "pull": {"pull", "hook"},
    "sweep": {"sweep"},
}

# Stored labels with no supported model class. Reported, never counted as hits.
UNSUPPORTED_LABELS = {"reverse_sweep", "scoop"}

SESSION_GAP_MINUTES = 30


@dataclass
class ClipResult:
    path: str
    stored_label: str
    top1: str
    top1_confidence: float
    top2: str
    top2_confidence: float
    probabilities: dict[str, float]
    expected: list[str] = field(default_factory=list)
    is_supported: bool = True
    correct: bool = False
    in_top2: bool = False
    session: str = ""
    latency_seconds: float = 0.0
    window: tuple[int, int] | None = None


def discover_clips(root: Path) -> list[tuple[Path, str]]:
    clips: list[tuple[Path, str]] = []
    for path in sorted(root.glob("*/*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
            continue
        clips.append((path, path.parent.name))
    return clips


def session_key(filename: str) -> str:
    """Cluster clips recorded within ``SESSION_GAP_MINUTES`` of each other."""
    stamp = filename.rsplit("_", 2)[-2:]
    try:
        recorded = datetime.strptime("_".join(stamp), "%Y%m%d_%H%M%S")
    except ValueError:
        return filename
    return recorded.strftime("%Y%m%d_%H") + f"_{(recorded.minute // 30) * 30:02d}"


def evaluate(clips: list[tuple[Path, str]], classifier: BattingVideoClassifier) -> list[ClipResult]:
    import time

    results: list[ClipResult] = []
    for index, (path, stored_label) in enumerate(clips, start=1):
        window = action_window_for(path, stored_label)
        started = time.perf_counter()
        classification = classifier.predict(path, action_window=window)
        latency = time.perf_counter() - started
        probabilities = classification["probabilities"]
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        expected = sorted(COMPATIBLE_LABELS.get(stored_label, set()))
        result = ClipResult(
            path=str(path),
            stored_label=stored_label,
            top1=ranked[0][0],
            top1_confidence=float(ranked[0][1]),
            top2=ranked[1][0],
            top2_confidence=float(ranked[1][1]),
            probabilities={label: float(value) for label, value in ranked},
            expected=expected,
            is_supported=bool(expected),
            correct=bool(expected) and ranked[0][0] in expected,
            in_top2=bool(expected) and any(label in expected for label, _ in ranked[:2]),
            session=session_key(path.stem),
            latency_seconds=latency,
            window=window,
        )
        results.append(result)
        print(
            f"[{index:>3}/{len(clips)}] {path.parent.name:<14} {path.stem:<32} "
            f"-> {result.top1:<14} {result.top1_confidence:.3f} "
            f"{'OK' if result.correct else ('MISS' if result.is_supported else 'UNSUPPORTED')} "
            f"({latency:.2f}s)",
            flush=True,
        )
    return results


def action_window_for(path: Path, stored_label: str) -> tuple[int, int] | None:
    """Production uses the detected action window when landmarks are available."""
    landmark_path = DATA_DIR / "landmarks" / "batting" / stored_label / f"{path.stem}.json"
    if not landmark_path.exists():
        return None
    try:
        features = extract_batting_features_from_data(load_landmark_json(landmark_path))
    except Exception:
        return None
    if not features.get("action_detected"):
        return None
    return int(features["action_start_frame"]), int(features["action_end_frame"])


def _macro_f1(results: list[ClipResult]) -> tuple[float, dict[str, float]]:
    labels = sorted({label for result in results for label in result.expected})
    per_class: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for r in results if label in r.expected and r.top1 == label)
        fp = sum(1 for r in results if label not in r.expected and r.top1 == label)
        fn = sum(1 for r in results if label in r.expected and r.top1 != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (statistics.mean(per_class.values()) if per_class else 0.0), per_class


def _selective_accuracy(results: list[ClipResult], threshold: float) -> dict[str, float | int]:
    kept = [r for r in results if r.top1_confidence >= threshold]
    return {
        "threshold": threshold,
        "coverage": round(len(kept) / len(results), 3) if results else 0.0,
        "accuracy": round(sum(1 for r in kept if r.correct) / len(kept), 3) if kept else 0.0,
        "retained": len(kept),
    }


def _confusion(results: list[ClipResult]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for result in results:
        row = matrix.setdefault(result.stored_label, defaultdict(int))
        row[result.top1] += 1
    return {stored: dict(sorted(row.items(), key=lambda item: -item[1])) for stored, row in sorted(matrix.items())}


def summarise(results: list[ClipResult]) -> dict:
    supported = [r for r in results if r.is_supported]
    unsupported = [r for r in results if not r.is_supported]
    correct = [r for r in supported if r.correct]
    incorrect = [r for r in supported if not r.correct]
    macro_f1, per_class_f1 = _macro_f1(supported)

    by_stored: dict[str, dict] = {}
    for label in sorted({r.stored_label for r in supported}):
        group = [r for r in supported if r.stored_label == label]
        by_stored[label] = {
            "clips": len(group),
            "accuracy": round(sum(1 for r in group if r.correct) / len(group), 3),
            "top2_accuracy": round(sum(1 for r in group if r.in_top2) / len(group), 3),
            "mean_confidence": round(statistics.mean(r.top1_confidence for r in group), 3),
            "expected_classes": sorted({cls for r in group for cls in r.expected}),
        }

    # Session-grouped vote: majority top1 per stored label within a session.
    sessions: dict[tuple[str, str], list[ClipResult]] = defaultdict(list)
    for result in supported:
        sessions[(result.session, result.stored_label)].append(result)
    session_votes = 0
    session_correct = 0
    for group in sessions.values():
        vote = statistics.mode([r.top1 for r in group])
        session_votes += 1
        session_correct += int(vote in group[0].expected)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    distribution: dict[str, int] = {}
    for result in supported:
        distribution[result.top1] = distribution.get(result.top1, 0) + 1
    return {
        "clips_total": len(results),
        "clips_supported": len(supported),
        "clips_unsupported": len(unsupported),
        "top1_accuracy": round(len(correct) / len(supported), 4) if supported else None,
        "top2_accuracy": round(sum(1 for r in supported if r.in_top2) / len(supported), 4) if supported else None,
        "macro_f1": round(macro_f1, 4),
        "chance_top1": round(1 / 10, 4),
        "prediction_distribution": dict(sorted(distribution.items(), key=lambda item: -item[1])),
        "per_class_f1": {label: round(value, 3) for label, value in per_class_f1.items()},
        "per_stored_label": by_stored,
        "mean_confidence_correct": round(statistics.mean(r.top1_confidence for r in correct), 3) if correct else None,
        "mean_confidence_incorrect": round(statistics.mean(r.top1_confidence for r in incorrect), 3) if incorrect else None,
        "mean_confidence_correct_top2": None,
        "selective_accuracy": [_selective_accuracy(supported, threshold) for threshold in thresholds],
        "session_grouped": {
            "sessions": session_votes,
            "majority_accuracy": round(session_correct / session_votes, 4) if session_votes else None,
        },
        "confusion": _confusion(supported),
        "unsupported_report": {
            label: {
                "clips": len([r for r in unsupported if r.stored_label == label]),
                "predicted": _confusion([r for r in unsupported if r.stored_label == label]),
            }
            for label in sorted({r.stored_label for r in unsupported})
        },
        "mean_latency_seconds": round(statistics.mean(r.latency_seconds for r in results), 3) if results else None,
    }


def render_markdown(summary: dict, results: list[ClipResult]) -> str:
    lines = [
        "# Batting model benchmark (historical phone clips)",
        "",
        "Model: `data/models/batting_video.onnx`  ",
        "Preprocessing: production path (`sample_video_frames`, 30 padded 224x224 RGB frames)",
        "",
        f"- Clips evaluated: **{summary['clips_total']}** ({summary['clips_supported']} supported, {summary['clips_unsupported']} unsupported)",
        f"- Top-1 accuracy (supported labels): **{summary['top1_accuracy']}**",
        f"- Top-2 accuracy (supported labels): **{summary['top2_accuracy']}**",
        f"- Macro F1: **{summary['macro_f1']}**",
        f"- Mean confidence, correct: {summary['mean_confidence_correct']}",
        f"- Mean confidence, incorrect: {summary['mean_confidence_incorrect']}",
        f"- Session-grouped majority accuracy: {summary['session_grouped']['majority_accuracy']} "
        f"over {summary['session_grouped']['sessions']} session/label groups",
        f"- Mean inference latency: {summary['mean_latency_seconds']} s/clip",
        "",
        "## Per stored label",
        "",
        "| Stored label | Clips | Top-1 | Top-2 | Mean confidence | Model classes counted correct |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, stats in summary["per_stored_label"].items():
        expected = ", ".join(stats["expected_classes"]) or "-"
        lines.append(
            f"| {label} | {stats['clips']} | {stats['accuracy']} | {stats['top2_accuracy']} | "
            f"{stats['mean_confidence']} | {expected} |"
        )

    lines += ["", "## Confusion (stored label -> model top-1)", "", "| Stored label | Predictions |", "| --- | --- |"]
    for stored, row in summary["confusion"].items():
        cells = ", ".join(f"{label} x{count}" for label, count in row.items())
        lines.append(f"| {stored} | {cells} |")

    lines += [
        "",
        "## Selective prediction (abstain below threshold)",
        "",
        "| Threshold | Coverage | Accuracy on retained | Clips retained |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["selective_accuracy"]:
        lines.append(
            f"| {entry['threshold']} | {entry['coverage']} | {entry['accuracy']} | {entry['retained']} |"
        )

    lines += ["", "## Unsupported labels (no model class exists)", ""]
    for label, report in summary["unsupported_report"].items():
        predictions = ", ".join(
            f"{name} x{count}" for row in report["predicted"].values() for name, count in row.items()
        )
        lines.append(f"- **{label}** ({report['clips']} clips) predicted as: {predictions}")

    lines += ["", "## Per-clip detail", "", "| Clip | Stored | Top-1 | Conf | Top-2 | Correct |", "| --- | --- | --- | ---: | --- | --- |"]
    for result in sorted(results, key=lambda r: (r.stored_label, r.path)):
        lines.append(
            f"| {Path(result.path).name} | {result.stored_label} | {result.top1} | "
            f"{result.top1_confidence:.3f} | {result.top2} | {'yes' if result.correct else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "raw" / "batting")
    parser.add_argument("--json", type=Path, default=DATA_DIR / "evaluation" / "revamp_batting_benchmark.json")
    parser.add_argument("--markdown", type=Path, default=DATA_DIR / "evaluation" / "revamp_batting_benchmark.md")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N clips (debugging).")
    args = parser.parse_args()

    clips = discover_clips(args.data_root)
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No clips found under {args.data_root}")

    classifier = BattingVideoClassifier()
    results = evaluate(clips, classifier)
    summary = summarise(results)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"summary": summary, "clips": [r.__dict__ for r in results]}, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(summary, results), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.json} and {args.markdown}")


if __name__ == "__main__":
    main()
