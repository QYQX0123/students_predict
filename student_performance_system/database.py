"""SQLite storage for saved prediction history.

中文：封装预测记录的建表、新增、查询、删除、编号整理和 CSV 导出操作。
English: Encapsulates table creation, insertion, querying, deletion, ID
renumbering, and CSV export for prediction history.
"""

import csv
import sqlite3
from pathlib import Path


class HistoryDatabase:
    """Persistence layer used by the desktop application's history screen.

    中文：每个方法都使用短连接和 ``with`` 事务，成功时自动提交，异常时自动回滚。
    English: Each method uses a short-lived connection and a ``with`` transaction,
    which commits on success and rolls back automatically on failure.
    """

    def __init__(self, db_path):
        """Save the database path and initialize its schema / 保存路径并初始化表结构。"""
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        """Open a new SQLite connection / 为一次数据库操作创建新连接。"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create the predictions table only when absent / 不覆盖旧数据地创建预测表。"""
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
        """Insert a complete prediction record.

        中文：除预测类别和置信度外，同时保存全部输入特征，便于以后重新查看和解释。
        English: Stores every input feature together with the predicted class and
        confidence so the record can later be reconstructed and explained.
        """
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
        """Return summary rows for the GUI history table / 返回历史表格所需的摘要字段。"""
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
        """Return one complete record, or None when absent / 按编号返回完整记录，不存在则返回 None。"""
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
        """Delete one record and renumber remaining visible IDs.

        中文：删除成功后重新整理主键，使教学演示中的编号保持连续；返回受影响行数。
        English: After a successful deletion, primary keys are compacted for
        sequential classroom-facing IDs. The affected row count is returned.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            if cursor.rowcount:
                self._renumber_prediction_ids(conn)
            return cursor.rowcount

    def _renumber_prediction_ids(self, conn):
        """Rewrite IDs safely so they remain 1, 2, 3, ...

        中文：第一轮先写入临时负数 ID，第二轮再改为正数，避免更新过程中与现有主键冲突；
        最后同步 SQLite 的自增序列。
        English: The first pass assigns temporary negative IDs and the second restores
        positive IDs, preventing primary-key collisions during updates. The SQLite
        autoincrement sequence is synchronized afterward.
        """
        rows = conn.execute("SELECT id FROM predictions ORDER BY id ASC").fetchall()

        # 中文：先转为负数，保证新旧正数主键不会发生唯一性冲突。
        # English: Move IDs into negative space first to prevent uniqueness collisions.
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
            # 中文：某些 SQLite 配置可能没有 sqlite_sequence，此时无需额外处理。
            # English: Some SQLite configurations have no sqlite_sequence table.
            pass

    def export_csv(self, path):
        """Export history as an Excel-friendly UTF-8 CSV.

        中文：使用 UTF-8 BOM（utf-8-sig），让 Windows Excel 能正确识别中文。
        English: Uses UTF-8 with BOM (utf-8-sig) so Windows Excel detects text correctly.
        """
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
