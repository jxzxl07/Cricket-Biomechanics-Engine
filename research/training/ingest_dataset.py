"""Ingest an external clip corpus into the landmark store used for training.

Point it at any directory laid out as ``<class>/<clip>.mp4`` and it extracts pose
landmarks with the same MediaPipe configuration the API uses, writing them to
``data/landmarks/<mode>/<class>/<clip>.json``. From there the normal training
commands apply.

    # ActionBowl (Kaggle): unzip, then point at the class folders
    python -m research.training.ingest_dataset --source /data/actionbowl --mode bowling

    python -m research.training.pose_dataset --mode bowling
    python -m research.training.train_action_model --mode bowling

Landmarks are cached: a clip that already has a landmark file is skipped, so an
interrupted run can be resumed. Nothing is written into the repository history —
``data/landmarks`` is gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR
from vision.pose import extract_landmarks_from_video

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def discover(source: Path) -> list[tuple[str, Path]]:
    """Return ``(class_name, clip_path)`` pairs for a ``<class>/<clip>`` tree."""
    pairs: list[tuple[str, Path]] = []
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        for clip in sorted(class_dir.rglob("*")):
            if clip.suffix.lower() in VIDEO_SUFFIXES:
                pairs.append((class_dir.name, clip))
    return pairs


def normalise_label(label: str) -> str:
    """Map dataset folder names onto the project's class naming."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of <class>/<clip> folders")
    parser.add_argument("--mode", choices=["batting", "bowling"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only ingest the first N clips")
    parser.add_argument("--max-frames", type=int, default=240, help="Analysed frames per clip")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    pairs = discover(args.source)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"no clips found under {args.source} (expected <class>/<clip>.mp4)")

    target_root = DATA_DIR / "landmarks" / args.mode
    print(f"{len(pairs)} clips across {len({label for label, _ in pairs})} classes -> {target_root}")

    done = skipped = failed = 0
    for index, (label, clip) in enumerate(pairs, start=1):
        class_name = normalise_label(label)
        destination = target_root / class_name / f"{clip.stem}.json"
        if destination.exists():
            skipped += 1
            continue
        try:
            extract_landmarks_from_video(clip, args.mode, label=class_name, output_path=destination, max_frames=args.max_frames)
            done += 1
        except Exception as error:  # keep going: one corrupt clip must not stop a corpus
            failed += 1
            print(f"  failed {clip.name}: {type(error).__name__}: {error}")
        if index % 25 == 0 or index == len(pairs):
            print(f"  [{index}/{len(pairs)}] extracted={done} cached={skipped} failed={failed}", flush=True)

    print(f"\ndone: extracted={done} cached={skipped} failed={failed}")
    print("next: python -m research.training.pose_dataset --mode", args.mode)


if __name__ == "__main__":
    main()
