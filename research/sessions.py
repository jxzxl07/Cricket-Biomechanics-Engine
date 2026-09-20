"""Recording-session keys shared by evaluation and training.

Clips recorded minutes apart are near-duplicates. Any honest evaluation has to
keep a whole session on one side of the split, so the session key is derived from
the capture timestamp embedded in the filename.
"""

from __future__ import annotations

from datetime import datetime

# Clips less than this far apart belong to the same recording session.
SESSION_WINDOW_MINUTES = 30


def session_key(filename: str) -> str:
    """Cluster clips recorded within ``SESSION_WINDOW_MINUTES`` of each other."""
    parts = filename.rsplit("_", 2)[-2:]
    try:
        recorded = datetime.strptime("_".join(parts), "%Y%m%d_%H%M%S")
    except ValueError:
        return filename
    bucket = (recorded.minute // SESSION_WINDOW_MINUTES) * SESSION_WINDOW_MINUTES
    return recorded.strftime("%Y-%m-%d %H:") + f"{bucket:02d}"
