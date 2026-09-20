from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cricket_biomechanics.sqlite3"

COUNTDOWN_SECONDS = 5
RECORD_SECONDS = 3
FPS = 30

BOWLING_LABELS = [
    "Left-arm pace",
    "Right-arm pace",
    "Left-arm spin",
    "Right-arm spin",
]

BATTING_LABELS = [
    "Cover Drive",
    "Defence",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted Shot",
    "Pull",
    "Square Cut",
    "Straight Drive",
    "Sweep",
]
