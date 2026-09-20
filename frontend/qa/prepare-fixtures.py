#!/usr/bin/env python3
"""Prepare browser-QA fixtures.

Browsers cannot decode MPEG-4 Part 2, which is what the historical OpenCV
clips use, so QA needs H.264 copies. Personal footage must never be committed,
so this script writes everything to /tmp and the generated no-person clip to
qa/fixtures (synthetic, safe to commit).

    python qa/prepare-fixtures.py
    python qa/prepare-fixtures.py --batting path/to/clip.mp4 --bowling path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def transcode(source: Path, target: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("this OpenCV build cannot encode H.264 (avc1)")
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
    writer.release()
    capture.release()
    print(f"{target}  {frames} frames  {width}x{height}  {fps:.0f} fps")


def write_no_person(target: Path, seconds: float = 3.0) -> None:
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"avc1"), 30, (640, 480))
    for index in range(int(seconds * 30)):
        frame = np.full((480, 640, 3), 26, dtype=np.uint8)
        x = 80 + index * 4
        cv2.rectangle(frame, (x, 200), (x + 90, 300), (190, 190, 190), -1)
        cv2.circle(frame, (x + 45, 170), 26, (215, 215, 215), -1)
        writer.write(frame)
    writer.release()
    print(f"{target}  synthetic, no person")


def first_clip(mode: str) -> Path | None:
    root = REPO / "data" / "raw" / mode
    if not root.exists():
        return None
    for candidate in sorted(root.glob("*/*.mp4")):
        if ".landmarks" not in candidate.name:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batting", type=Path, default=None)
    parser.add_argument("--bowling", type=Path, default=None)
    args = parser.parse_args()

    batting = args.batting or first_clip("batting")
    bowling = args.bowling or first_clip("bowling")
    if batting:
        transcode(batting, Path("/tmp/qa_batting_h264.mp4"))
        # Keep the original codec for the unplayable-codec path: the historical
        # OpenCV clips are MPEG-4 Part 2, which browsers refuse to decode.
        shutil.copyfile(batting, "/tmp/qa_batting_mp4v.mp4")
        print("/tmp/qa_batting_mp4v.mp4  original codec (browser-unplayable)")
    else:
        print("no batting clip found; pass --batting or skip that QA path")
    if bowling:
        transcode(bowling, Path("/tmp/qa_bowling_h264.mp4"))
    else:
        print("no bowling clip found; pass --bowling or skip that QA path")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    write_no_person(FIXTURES / "no-person.mp4")
    print("\nReady. Run: npm run qa")


if __name__ == "__main__":
    main()
