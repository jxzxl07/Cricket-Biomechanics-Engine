CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,
        purpose TEXT NOT NULL,
        clip_path TEXT UNIQUE NOT NULL,
        landmarks_path TEXT,
        features_path TEXT,
        label TEXT,
        predicted_label TEXT,
        confidence REAL,
        corrected_label TEXT,
        accepted_for_training INTEGER NOT NULL DEFAULT 0,
        trained INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id INTEGER NOT NULL,
        model_version TEXT,
        predicted_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        was_low_confidence INTEGER NOT NULL DEFAULT 0,
        explanation_json_path TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recording_id) REFERENCES recordings (id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id INTEGER NOT NULL,
        corrected_label TEXT NOT NULL,
        accepted_for_training INTEGER NOT NULL DEFAULT 1,
        trained INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recording_id) REFERENCES recordings (id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,
        model_path TEXT,
        dataset_size INTEGER NOT NULL DEFAULT 0,
        metrics_json_path TEXT,
        confusion_matrix_path TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
]
