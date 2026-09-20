"""Load and validate the JSON specs that pin model preprocessing and provenance.

Every deployable artifact ships with a sidecar spec file. The spec is the single
source of truth for the class mapping, the exact preprocessing, the checksum of
the artifact, and the measured benchmark. Loading a model without its spec is a
hard error so preprocessing can never silently drift between evaluation and
production.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import MODELS_DIR

_CHUNK = 1024 * 1024


class ModelSpecError(RuntimeError):
    """Raised when a model artifact and its spec disagree."""


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    data: dict[str, Any]
    _digest: str | None = None

    # ---- identity -----------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def version(self) -> str:
        return str(self.data.get("version", "0"))

    @property
    def kind(self) -> str:
        return str(self.data.get("kind", "unknown"))

    @property
    def experimental(self) -> bool:
        return bool(self.data.get("experimental", False))

    @property
    def note(self) -> str:
        return str(self.data.get("note", ""))

    # ---- artifact -----------------------------------------------------
    @property
    def artifact_path(self) -> Path | None:
        artifact = self.data.get("artifact")
        return (self.path.parent / str(artifact)) if artifact else None

    def sha256(self) -> str:
        if self._digest is None:
            path = self.artifact_path
            if path is None:
                object.__setattr__(self, "_digest", "")
                return ""
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b""):
                    digest.update(chunk)
            object.__setattr__(self, "_digest", digest.hexdigest())
        return self._digest or ""

    def verify_artifact(self) -> str:
        path = self.artifact_path
        if path is None:
            return ""  # Weight-free model (for example the bowling pose prototype).
        if not path.exists():
            raise ModelSpecError(f"Model artifact missing: {path}")
        expected = self.data.get("sha256")
        if not expected:
            raise ModelSpecError(f"{self.path.name} does not declare a sha256")
        actual = self.sha256()
        if actual != expected:
            raise ModelSpecError(
                f"{path.name} checksum mismatch: expected {expected[:12]}…, found {actual[:12]}…"
            )
        return actual

    # ---- mapping ------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return [str(name) for name in self.data["classes"]]

    @property
    def display_labels(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.data.get("display_labels", {}).items()}

    def display_label(self, class_name: str) -> str:
        return self.display_labels.get(class_name, class_name.replace("_", " ").title())

    # ---- decisions ----------------------------------------------------
    @property
    def unknown_threshold(self) -> float:
        return float(self.data.get("unknown_threshold", 0.55))

    @property
    def unknown_margin(self) -> float:
        return float(self.data.get("unknown_margin", 0.12))

    @property
    def confidence_cap(self) -> float:
        return float(self.data.get("confidence_cap", 1.0))

    def benchmark(self) -> dict[str, Any]:
        return dict(self.data.get("benchmark", {}))

    def public(self) -> dict[str, Any]:
        """Metadata safe to expose over the API."""
        return {
            "id": self.id,
            "version": self.version,
            "kind": self.kind,
            "classes": self.classes,
            "display_labels": self.display_labels,
            "experimental": self.experimental,
            "note": self.note,
            "confidence_cap": self.confidence_cap,
            "unknown_threshold": self.unknown_threshold,
            "provenance": self.data.get("provenance", {}),
            "benchmark": self.benchmark(),
            "limitations": self.data.get("limitations", []),
        }


def load_spec(filename: str) -> ModelSpec:
    path = MODELS_DIR / filename
    if not path.exists():
        raise ModelSpecError(f"Model spec missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    spec = ModelSpec(path=path, data=data)
    spec.verify_artifact()
    return spec
