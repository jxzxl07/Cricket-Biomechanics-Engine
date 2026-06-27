import sqlite3
from pathlib import Path

from config import DB_PATH
from database.schema import CREATE_TABLES


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database():
    with get_connection() as connection:
        for statement in CREATE_TABLES:
            connection.execute(statement)

        add_column_if_missing(connection, "recordings", "landmarks_path", "TEXT")
        add_column_if_missing(connection, "recordings", "features_path", "TEXT")


def add_column_if_missing(connection, table_name, column_name, column_type):
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }

    if column_name not in existing_columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def save_recording(
    *,
    mode,
    purpose,
    clip_path,
    landmarks_path=None,
    features_path=None,
    label=None,
    predicted_label=None,
    confidence=None,
    corrected_label=None,
    accepted_for_training=False,
    trained=False,
):
    init_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO recordings (
                mode,
                purpose,
                clip_path,
                landmarks_path,
                features_path,
                label,
                predicted_label,
                confidence,
                corrected_label,
                accepted_for_training,
                trained
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mode,
                purpose,
                str(clip_path),
                str(landmarks_path) if landmarks_path is not None else None,
                str(features_path) if features_path is not None else None,
                label,
                predicted_label,
                confidence,
                corrected_label,
                int(accepted_for_training),
                int(trained),
            ),
        )

        return cursor.lastrowid


def update_recording_paths(recording_id, *, landmarks_path=None, features_path=None):
    init_database()

    fields = []
    values = []

    if landmarks_path is not None:
        fields.append("landmarks_path = ?")
        values.append(str(landmarks_path))

    if features_path is not None:
        fields.append("features_path = ?")
        values.append(str(features_path))

    if not fields:
        return

    values.append(recording_id)

    with get_connection() as connection:
        connection.execute(
            f"""
            UPDATE recordings
            SET {", ".join(fields)}
            WHERE id = ?
            """,
            values,
        )


def update_recording_prediction(
    recording_id,
    *,
    predicted_label,
    confidence,
):
    init_database()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE recordings
            SET predicted_label = ?,
                confidence = ?
            WHERE id = ?
            """,
            (predicted_label, confidence, recording_id),
        )


def save_correction(recording_id, corrected_label):
    init_database()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE recordings
            SET corrected_label = ?,
                accepted_for_training = 1,
                trained = 0
            WHERE id = ?
            """,
            (corrected_label, recording_id),
        )

        connection.execute(
            """
            INSERT INTO corrections (
                recording_id,
                corrected_label,
                accepted_for_training,
                trained
            )
            VALUES (?, ?, 1, 0)
            """,
            (recording_id, corrected_label),
        )


def get_recent_recordings(limit=20):
    init_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM recordings
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_corrected_prediction_records(mode):
    init_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM recordings
            WHERE mode = ?
              AND purpose = 'prediction'
              AND accepted_for_training = 1
              AND corrected_label IS NOT NULL
              AND features_path IS NOT NULL
            ORDER BY created_at ASC
            """,
            (mode,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_untrained_training_recordings(mode):
    init_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM recordings
            WHERE mode = ?
              AND accepted_for_training = 1
              AND trained = 0
              AND label IS NOT NULL
              AND features_path IS NOT NULL
            ORDER BY created_at ASC
            """,
            (mode,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_accepted_training_records(mode, only_untrained=False):
    init_database()

    extra_filter = ""

    if only_untrained:
        extra_filter = "AND trained = 0"

    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM recordings
            WHERE mode = ?
              AND accepted_for_training = 1
              AND features_path IS NOT NULL
              AND (
                    label IS NOT NULL
                    OR corrected_label IS NOT NULL
                  )
              {extra_filter}
            ORDER BY created_at ASC
            """,
            (mode,),
        ).fetchall()

    return [dict(row) for row in rows]


def mark_trained(recording_ids):
    init_database()

    with get_connection() as connection:
        connection.executemany(
            """
            UPDATE recordings
            SET trained = 1
            WHERE id = ?
            """,
            [(recording_id,) for recording_id in recording_ids],
        )

        connection.executemany(
            """
            UPDATE corrections
            SET trained = 1
            WHERE recording_id = ?
            """,
            [(recording_id,) for recording_id in recording_ids],
        )
