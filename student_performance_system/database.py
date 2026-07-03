"""SQLite persistence for prediction history.

中文：这个模块负责把预测记录长期保存到本地 SQLite 文件中，并封装建表、旧库迁移、
新增、更新、查询、删除、重新编号和导出 CSV。这样写的原因是让界面只调用清晰的方法，
不需要在按钮回调中拼 SQL。

English: This module stores prediction history in a local SQLite file and wraps
table creation, legacy migration, insertion, update, lookup, deletion, ID
renumbering, and CSV export. The UI can call clear methods instead of embedding SQL
inside button callbacks.
"""

import csv
import sqlite3
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


class HistoryDatabase:
    """Small database access layer for the history screen.

    中文：类的职责是隔离 SQLite 细节。每个操作打开短连接，并利用连接上下文自动提交
    或回滚；这样写足够简单，也避免长时间持有数据库连接。

    English: This class isolates SQLite details. Each operation opens a short-lived
    connection and relies on the connection context to commit or roll back, keeping
    the design simple and avoiding long-held database handles.
    """

    def __init__(self, db_path):
        """Remember the database path and make sure the schema is ready.

        中文：对象创建时立即初始化表结构，保证后续调用 add/list/delete 时不必重复检查。

        English: The schema is prepared when the object is created so later add,
        list, and delete calls do not need their own setup checks.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        """Open a SQLite connection for one operation.

        中文：不复用全局连接，避免 GUI 长时间运行时连接状态变得不可控。

        English: A global connection is not reused, which keeps connection state
        predictable during a long-running GUI session.
        """
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create the history table and migrate older schemas if needed.

        中文：CREATE TABLE IF NOT EXISTS 不会覆盖已有数据；创建后继续调用迁移逻辑，
        让旧版 confidence_score 数据库也能被新版 pass_probability 使用。

        English: CREATE TABLE IF NOT EXISTS preserves existing data. The migration
        step then keeps older confidence_score databases compatible with the newer
        pass_probability field.
        """
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
                    pass_probability REAL NOT NULL,
                    predicted_g3 REAL NOT NULL DEFAULT 0.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn):
        """Upgrade legacy history tables without deleting records.

        中文：旧版本保存的是 confidence_score，新版本需要 pass_probability。迁移时只
        添加缺失列并复制旧值，因此用户历史不会丢失。

        English: Older versions stored confidence_score, while the newer UI needs
        pass_probability. Migration only adds the missing column and copies old
        values, preserving user history.
        """
        columns = self._prediction_columns(conn)
        if "pass_probability" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN pass_probability REAL NOT NULL DEFAULT 0.0")
            if "confidence_score" in columns:
                conn.execute("UPDATE predictions SET pass_probability = confidence_score")
        if "predicted_g3" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN predicted_g3 REAL NOT NULL DEFAULT 0.0")

    @staticmethod
    def _prediction_columns(conn):
        """Inspect current table columns for compatibility decisions.

        中文：SQLite 没有直接的“如果列不存在就添加”语法，因此先读取 PRAGMA 表结构。

        English: SQLite has no direct "add column if missing" syntax, so PRAGMA
        table_info is used before deciding how to migrate or insert.
        """
        return {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}

    def add_prediction(self, student, prediction, pass_probability, predicted_g3=0.0):
        """Insert one prediction and the inputs that produced it.

        中文：除了 Pass/Fail 和通过概率，还保存学生身份与所有特征。这样历史详情无需
        依赖当前表单状态，也能重新构造 StudentInput。

        English: Stores identity, all input features, Pass/Fail, and pass
        probability. History details can then reconstruct StudentInput without
        depending on the current form state.
        """
        with self._connect() as conn:
            columns = [
                "student_name",
                "matric_no",
                "sex",
                "age",
                "study_time",
                "failures",
                "activities",
                "absences",
                "g1",
                "g2",
                "prediction_result",
                "pass_probability",
                "predicted_g3",
            ]
            values = [
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
                pass_probability,
                predicted_g3,
            ]
            if "confidence_score" in self._prediction_columns(conn):
                columns.append("confidence_score")
                values.append(pass_probability)

            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def update_prediction_result(self, prediction_id, prediction, pass_probability, predicted_g3=0.0):
        """Update one saved result, including legacy confidence_score when present.

        中文：用于启动时把旧 Low/Medium/High 历史迁移为新的 Pass/Fail 结果；如果旧列
        仍存在，同时同步它以保持兼容。

        English: Used at startup to migrate old Low/Medium/High history to the new
        Pass/Fail result. If the legacy column still exists, it is kept in sync.
        """
        with self._connect() as conn:
            if "confidence_score" in self._prediction_columns(conn):
                conn.execute(
                    """
                    UPDATE predictions
                    SET prediction_result = ?, pass_probability = ?, predicted_g3 = ?, confidence_score = ?
                    WHERE id = ?
                    """,
                    (prediction, pass_probability, predicted_g3, pass_probability, prediction_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE predictions
                    SET prediction_result = ?, pass_probability = ?, predicted_g3 = ?
                    WHERE id = ?
                    """,
                    (prediction, pass_probability, predicted_g3, prediction_id),
                )

    def list_predictions(self):
        """Return the compact rows shown in the history table.

        中文：列表页只需要摘要字段，避免加载不必要的输入详情；详情窗口再按 ID 查询全量。

        English: The table needs only summary fields, so full input details are
        loaded later by ID when the detail window opens.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, timestamp, student_name, matric_no, g1, g2,
                       prediction_result, pass_probability, predicted_g3
                FROM predictions
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_prediction(self, prediction_id):
        """Return the complete record for one history ID.

        中文：删除或重新编号后可能找不到记录，因此返回 None 让界面能给出友好提示。

        English: A record may be missing after deletion or renumbering, so None lets
        the UI show a friendly message.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, timestamp, student_name, matric_no, sex, age,
                       study_time, failures, activities, absences, g1, g2,
                       prediction_result, pass_probability, predicted_g3
                FROM predictions
                WHERE id = ?
                """,
                (prediction_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_prediction(self, prediction_id):
        """Delete one row and compact visible IDs.

        中文：历史表格面向课堂或报告展示，连续编号更容易阅读；因此删除后重新编号。

        English: The history table is presentation-facing, so sequential IDs are
        easier to read. IDs are compacted after a successful deletion.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            if cursor.rowcount:
                self._renumber_prediction_ids(conn)
            return cursor.rowcount

    def delete_predictions(self, prediction_ids):
        """Delete several rows and renumber only once.

        中文：批量删除先一次性删除所有选中 ID，再统一整理编号，避免每删一条都移动主键。

        English: Multiple selected IDs are deleted in one statement and IDs are
        compacted once, avoiding repeated primary-key rewrites.
        """
        ids = [int(prediction_id) for prediction_id in prediction_ids]
        if not ids:
            return 0

        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM predictions WHERE id IN ({placeholders})", ids)
            if cursor.rowcount:
                self._renumber_prediction_ids(conn)
            return cursor.rowcount

    def _renumber_prediction_ids(self, conn):
        """Rewrite primary keys into a gap-free sequence.

        中文：直接把旧 ID 改成新 ID 可能撞上仍存在的主键，所以第一轮先写入临时负数，
        第二轮再写回正数。最后更新 sqlite_sequence，保证下次新增从正确编号继续。

        English: Directly changing old IDs to new IDs can collide with existing
        keys, so the first pass moves them to temporary negative values and the
        second pass restores positive IDs. sqlite_sequence is updated so future
        inserts continue from the correct number.
        """
        rows = conn.execute("SELECT id FROM predictions ORDER BY id ASC").fetchall()

        # 中文：临时负数主键为重新编号提供一个不会与现有正数 ID 冲突的过渡区。
        # English: Temporary negative IDs provide a collision-free staging area.
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
            # 中文：没有自增序列表时说明无需同步序列，删除或更新记录本身已经完成。
            # English: If the sequence table is absent, row changes are already complete.
            pass

    def export_csv(self, path):
        """Write history summaries to a CSV file.

        中文：导出列与历史表格保持一致，并使用 utf-8-sig，方便 Windows Excel 正确识别
        编码。这样用户看到的表格和导出的文件字段一致。

        English: Exported columns match the history table and use utf-8-sig so
        Windows Excel detects the encoding correctly. The visible table and exported
        file therefore share the same field set.
        """
        rows = self.list_predictions()
        fieldnames = [
            "id",
            "timestamp",
            "student_name",
            "matric_no",
            "g1",
            "g2",
            "predicted_g3",
            "prediction_result",
            "pass_probability",
        ]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def export_xlsx(self, path):
        """Write history summaries to a styled XLSX workbook."""
        rows = self.list_predictions()
        headers = [
            ("id", "ID"),
            ("timestamp", "Timestamp"),
            ("student_name", "Student"),
            ("matric_no", "Matric No."),
            ("g1", "G1"),
            ("g2", "G2"),
            ("predicted_g3", "Predicted G3"),
            ("prediction_result", "Result"),
            ("pass_probability", "Pass Probability"),
        ]
        sheet_rows = [self._xlsx_row_xml(1, [label for _key, label in headers], header=True)]
        for row_number, row in enumerate(rows, start=2):
            values = []
            for key, _label in headers:
                value = row[key]
                if key == "pass_probability":
                    value = f"{float(value):.2%}"
                elif key == "predicted_g3":
                    value = f"{float(value):.1f}"
                values.append(value)
            style = 2 if row["prediction_result"] == "Pass" else 3
            sheet_rows.append(self._xlsx_row_xml(row_number, values, style=style))

        worksheet_xml = self._worksheet_xml("".join(sheet_rows), len(headers), max(len(rows) + 1, 1))
        with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
            workbook.writestr("[Content_Types].xml", self._content_types_xml())
            workbook.writestr("_rels/.rels", self._root_relationships_xml())
            workbook.writestr("xl/workbook.xml", self._workbook_xml())
            workbook.writestr("xl/_rels/workbook.xml.rels", self._workbook_relationships_xml())
            workbook.writestr("xl/styles.xml", self._styles_xml())
            workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    @classmethod
    def _xlsx_row_xml(cls, row_number, values, header=False, style=0):
        cells = []
        for column_index, value in enumerate(values, start=1):
            reference = f"{cls._xlsx_column_letters(column_index)}{row_number}"
            cell_style = 1 if header else style
            style_attr = f' s="{cell_style}"' if cell_style else ""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{reference}"{style_attr}><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{reference}" t="inlineStr"{style_attr}><is><t>{escape(str(value))}</t></is></c>'
                )
        return f'<row r="{row_number}">{"".join(cells)}</row>'

    @staticmethod
    def _xlsx_column_letters(index):
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    @classmethod
    def _worksheet_xml(cls, rows_xml, column_count, row_count):
        last_column = cls._xlsx_column_letters(column_count)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_column}{row_count}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>
    <col min="1" max="1" width="8" customWidth="1"/>
    <col min="2" max="2" width="21" customWidth="1"/>
    <col min="3" max="4" width="18" customWidth="1"/>
    <col min="5" max="7" width="14" customWidth="1"/>
    <col min="8" max="9" width="18" customWidth="1"/>
  </cols>
  <sheetData>{rows_xml}</sheetData>
</worksheet>"""

    @staticmethod
    def _styles_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Times New Roman"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Times New Roman"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAD3"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF4CCCC"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2EC"/></left><right style="thin"><color rgb="FFD9E2EC"/></right><top style="thin"><color rgb="FFD9E2EC"/></top><bottom style="thin"><color rgb="FFD9E2EC"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>"""

    @staticmethod
    def _content_types_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

    @staticmethod
    def _root_relationships_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    @staticmethod
    def _workbook_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Prediction History" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

    @staticmethod
    def _workbook_relationships_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
