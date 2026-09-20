"""Copy measured benchmark results into the shipped model spec.

The spec is what the API serves, so the numbers a user sees always come from a
generated evaluation artifact rather than from hand-typed prose.

Run: python -m research.evaluation.publish_benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DATA_DIR

DEFAULT_BENCHMARK = DATA_DIR / "evaluation" / "revamp_batting_benchmark.json"
DEFAULT_SPEC = DATA_DIR / "models" / "batting_video.json"


def build_block(summary: dict, source: Path) -> dict:
    return {
        "status": "measured",
        "dataset": "This project's historical phone clips (single athlete, two recording sessions)",
        "independent": False,
        "clips": summary["clips_supported"],
        "unsupported_clips_excluded": summary["clips_unsupported"],
        "top1_accuracy": summary["top1_accuracy"],
        "top2_accuracy": summary["top2_accuracy"],
        "macro_f1": summary["macro_f1"],
        "chance_top1": summary["chance_top1"],
        "mean_confidence_correct": summary["mean_confidence_correct"],
        "mean_confidence_incorrect": summary["mean_confidence_incorrect"],
        "prediction_distribution": summary["prediction_distribution"],
        "session_grouped_accuracy": summary["session_grouped"]["majority_accuracy"],
        "selective_accuracy": summary["selective_accuracy"],
        "mean_latency_seconds": summary["mean_latency_seconds"],
        "source": str(source.relative_to(DATA_DIR.parent)),
        "interpretation": (
            "Broadcast-trained weights do not transfer to indoor phone clips. The model is "
            "deployed only as an explicitly experimental hint; it is not used for coaching claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()

    summary = json.loads(args.benchmark.read_text(encoding="utf-8"))["summary"]
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec["benchmark"] = build_block(summary, args.benchmark)
    args.spec.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {args.spec}")
    print(json.dumps(spec["benchmark"], indent=2)[:1200])


if __name__ == "__main__":
    main()
