"""SQLite storage for saved prediction history."""

import csv
import sqlite3
from pathlib import Path


class HistoryDatabase:
    """Small persistence layer used by the desktop app history screen."""

    def __init__(self, db_path):
        """Store the database path and create the predictions table if needed."""
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        """Open a short-lived SQLite connection for one database operation."""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create the history table while preserving existing records."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    matric_no TEXT NOT NULL,
                    sex TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    study_time INTEGER NOT NULL,
                    failures INTEGER NOT NULL,
                    activities TEXT NOT NULL,
                    absences INTEGER NOT NULL,
                    g1 INTEGER NOT NULL,
                    g2 INTEGER NOT NULL,
                    prediction_result TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add_prediction(self, student, prediction, confidence):
        """Insert one prediction result together with the input fields used."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    student_name, matric_no, sex, age, study_time, failures,
                    activities, absences, g1, g2, prediction_result, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student.name,
                    student.matric_no,
                    student.sex,
                    student.age,
                    student.study_time,
                    student.failures,
                    student.activities,
                    student.absences,
                    student.g1,
                    student.g2,
                    prediction,
                    confidence,
                ),
            )

    def list_predictions(self):
        """Return compact rows for the history table in the GUI."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, timestamp, student_name, matric_no, g1, g2,
                       prediction_result, confidence_score
                FROM predictions
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_prediction(self, prediction_id):
        """Return all stored fields for a selected history record."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, timestamp, student_name, matric_no, sex, age,
                       study_time, failures, activities, absences, g1, g2,
                       prediction_result, confidence_score
                FROM predictions
                WHERE id = ?
                """,
                (prediction_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_prediction(self, prediction_id):
        """Delete one record and keep visible IDs sequential for classroom demos."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            if cursor.rowcount:
                self._renumber_prediction_ids(conn)
            return cursor.rowcount

    def _renumber_prediction_ids(self, conn):
        """Rewrite IDs after deletion so the history table remains 1, 2, 3, ..."""
        rows = conn.execute("SELECT id FROM predictions ORDER BY id ASC").fetchall()

        # Use temporary negative ids first so updates never collide with existing primary keys.
        for new_id, (old_id,) in enumerate(rows, start=1):
            if old_id != new_id:
                conn.execute("UPDATE predictions SET id = ? WHERE id = ?", (-new_id, old_id))

        for new_id, _ in enumerate(rows, start=1):
            conn.execute("UPDATE predictions SET id = ? WHERE id = ?", (new_id, -new_id))

        max_id = len(rows)
        try:
            if max_id:
                conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", (max_id, "predictions"))
            else:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", ("predictions",))
        except sqlite3.OperationalError:
            pass

    def export_csv(self, path):
        """Export the same compact columns shown in the history table."""
        rows = self.list_predictions()
        fieldnames = [
            "id",
            "timestamp",
            "student_name",
            "matric_no",
            "g1",
            "g2",
            "prediction_result",
            "confidence_score",
        ]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
