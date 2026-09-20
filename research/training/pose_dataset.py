"""Build labelled pose-feature tables from the extracted landmark JSON.

The classifier consumes the same pose features the API already computes, so
training and serving cannot drift: both call ``vision.features``.

    python -m research.training.pose_dataset                 # both modes
    python -m research.training.pose_dataset --mode bowling
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR
from research.sessions import session_key
from vision.features import extract_bowling_features_from_data, extract_batting_features_from_data, load_landmark_json

LANDMARKS_DIR = DATA_DIR / "landmarks"

# Features used by the batting model, in a fixed order.
BATTING_FEATURES = [
    "peak_hand_speed",
    "mean_hand_speed",
    "peak_speed_timing",
    "hand_horizontal_range",
    "hand_vertical_range",
    "hand_path_angle",
    "final_hand_height",
    "shoulder_rotation_range",
    "max_knee_bend",
    "head_drop",
    "torso_lean_range",
    "action_duration_seconds",
]

# Features used by the bowling model. Handedness is included because it is a
# property of the athlete, not of the delivery type, and the API reports it
# separately from the action family.
BOWLING_FEATURES = [
    "peak_wrist_speed",
    "mean_wrist_speed_near_release",
    "release_height",
    "release_forward_reach",
    "arm_path_vertical_range",
    "arm_path_horizontal_range",
    "pre_release_path_angle",
    "post_release_path_angle",
    "shoulder_rotation_range",
    "torso_lean_at_release",
    "elbow_angle_at_release",
    "action_duration_seconds",
    "bowling_arm_is_left",
]


@dataclass
class ClipRow:
    clip: str
    label: str
    session: str
    features: dict[str, float | None]


def feature_names(mode: str) -> list[str]:
    return BATTING_FEATURES if mode == "batting" else BOWLING_FEATURES


def extract_row(mode: str, landmark_path: Path, label: str) -> ClipRow:
    data = load_landmark_json(landmark_path)
    raw = extract_batting_features_from_data(data) if mode == "batting" else extract_bowling_features_from_data(data)

    features: dict[str, float | None] = {}
    for name in feature_names(mode):
        if name == "bowling_arm_is_left":
            features[name] = 1.0 if raw.get("detected_bowling_arm") == "left" else 0.0
            continue
        value = raw.get(name)
        features[name] = None if value is None else float(value)
    features["_action_detected"] = 1.0 if raw.get("action_detected") else 0.0
    return ClipRow(clip=landmark_path.stem, label=label, session=session_key(landmark_path.stem), features=features)


def collect(mode: str, require_action: bool = True) -> list[ClipRow]:
    root = LANDMARKS_DIR / mode
    rows: list[ClipRow] = []
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for landmark_path in sorted(label_dir.glob("*.json")):
            row = extract_row(mode, landmark_path, label_dir.name)
            if require_action and not row.features.get("_action_detected"):
                print(f"  skip (action gate failed): {mode}/{label_dir.name}/{landmark_path.name}")
                continue
            rows.append(row)
    return rows


def write_csv(rows: list[ClipRow], mode: str, target: Path) -> None:
    columns = ["clip", "label", "session", *feature_names(mode)]
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "clip": row.clip,
                "label": row.label,
                "session": row.session,
                **{name: ("" if row.features.get(name) is None else row.features[name]) for name in feature_names(mode)},
            })
    print(f"wrote {target} ({len(rows)} rows, {len({row.label for row in rows})} classes, {len({row.session for row in rows})} sessions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batting", "bowling", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "evaluation")
    args = parser.parse_args()

    modes = ["batting", "bowling"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"=== {mode} ===")
        rows = collect(mode)
        if not rows:
            print("  no landmark rows found")
            continue
        write_csv(rows, mode, args.out_dir / f"{mode}_pose_dataset.csv")
        summary: dict[str, int] = {}
        for row in rows:
            summary[row.label] = summary.get(row.label, 0) + 1
        print("  labels:", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
