"""Shared paths and constants for the CreaseLab analysis service."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

# Fallback frame rate for clips that do not report their own.
FPS = 30
