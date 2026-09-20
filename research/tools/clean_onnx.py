"""Strip unused initializers from a converted ONNX model.

ONNX Runtime removes unused initializers at load time but logs a warning, so the
shipped artifact is cleaned once and its checksum and model card are updated.

Run: python -m research.tools.clean_onnx --model data/models/batting_video.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx


def unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input}
    return [init.name for init in model.graph.initializer if init.name not in used]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to in-place.")
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    before = sha256(args.model)
    unused = unused_initializers(model)
    if not unused:
        print("No unused initializers found.")
    else:
        print("Removing unused initializers:")
        for name in unused:
            print(" -", name)
    keep = [init for init in model.graph.initializer if init.name not in set(unused)]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model)

    output = args.output or args.model
    onnx.save(model, str(output))
    print(f"\nsha256 before: {before}")
    print(f"sha256 after:  {sha256(output)}")


if __name__ == "__main__":
    main()
