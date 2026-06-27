from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cricket_biomechanics.sqlite3"

COUNTDOWN_SECONDS = 5
RECORD_SECONDS = 3
FPS = 30

BOWLING_LABELS = [
    "Left Arm Leg",
    "Left Arm Off",
    "Left Arm Pace",
    "Right Arm Leg",
    "Right Arm Off",
    "Right Arm Pace",
]

BATTING_LABELS = [
    "Cut",
    "Drive",
    "Flick",
    "Pull",
    "Reverse Sweep",
    "Scoop",
    "Sweep",
]
