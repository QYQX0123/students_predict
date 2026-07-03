"""Command-line regression test for the prediction system.

中文：这个脚本不打开 Tkinter 窗口，而是直接调用数据校验、模型服务、数据库和批量导入
辅助函数。它实现了一次轻量端到端检查：训练、预测、评估、保存、删除、旧库迁移、
CSV/XLSX 导入和错误提示。这样写是为了在修改代码后快速发现主流程是否被破坏。

English: This script does not open the Tkinter window. It calls validation, the
prediction service, the database, and batch-import helpers directly. It implements
a lightweight end-to-end check: training, prediction, evaluation, saving, deletion,
legacy migration, CSV/XLSX import, and error reporting. This catches broken core
flows quickly after code changes.
"""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZipFile

from student_performance_system.app import StudentPerformanceApp
from student_performance_system.data_utils import STUDY_TIME_COLUMN, validate_student_input
from student_performance_system.database import HistoryDatabase
from student_performance_system.model_service import PredictionService


def write_minimal_xlsx(path, headers, rows):
    """Create a tiny XLSX file for import tests.

    中文：测试不依赖 openpyxl 等外部库，而是用 zipfile 写出 Excel 所需的最小 XML。
    这样写可以验证应用自己的 XLSX 解析逻辑，同时保持测试环境轻量。

    English: The test avoids external libraries such as openpyxl and writes the
    minimal XML parts with zipfile. This validates the app's own XLSX parser while
    keeping the test environment lightweight.
    """
    def column_letters(index):
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    def cell_xml(row_number, column_index, value):
        reference = f"{column_letters(column_index)}{row_number}"
        if isinstance(value, (int, float)):
            return f'<c r="{reference}"><v>{value}</v></c>'
        return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    sheet_rows = []
    for row_number, values in enumerate([headers, *rows], start=1):
        cells = "".join(cell_xml(row_number, column_index, value) for column_index, value in enumerate(values, start=1))
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')

    with ZipFile(path, "w") as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        workbook.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{''.join(sheet_rows)}</sheetData>
</worksheet>""",
        )


def main():
    """Run the full smoke-test workflow and print representative metrics.

    中文：每个 assert 都对应一个用户可见功能或关键兼容性要求；失败时脚本直接报错退出，
    便于定位最近修改造成的问题。

    English: Every assertion maps to a user-visible feature or compatibility
    requirement. A failure exits immediately, making regressions from recent edits
    easy to locate.
    """
    service = PredictionService(Path("dataset.csv"))
    normalized_choices = StudentPerformanceApp._normalize_choice_values(
        {"study_time": "2 (5 - 10 hours)", "failures": "4+"}
    )
    assert normalized_choices["study_time"] == "2"
    assert normalized_choices["failures"] == "4"
    student = validate_student_input(
        {
            "name": "Test Student",
            "matric_no": "T001",
            "sex": "F",
            "age": "17",
            "study_time": "2",
            "failures": "0",
            "activities": "yes",
            "absences": "4",
            "g1": "60",
            "g2": "65",
        }
    )
    result = service.predict(student)
    evaluation = service.metrics["evaluation"]
    comparison = service.metrics["model_comparison"]
    # 中文：评估结果必须覆盖完整数据集，并同时报告 Fail 与 Pass 两个类别。
    # English: Evaluation must cover the whole dataset and report both Fail and Pass.
    assert len(evaluation["fold_accuracies"]) == 5
    record_count = sum(service.metrics["class_distribution"].values())
    assert sum(sum(row.values()) for row in evaluation["confusion_matrix"].values()) == record_count
    assert set(evaluation["per_class"]) == {"Fail", "Pass"}
    assert 0.0 <= evaluation["macro_average"]["f1"] <= 1.0
    assert [row["model"] for row in comparison] == [
        "Majority Baseline",
        "Single Decision Tree",
        "Random Forest (Final)",
    ]
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in comparison)
    assert all(0.0 <= row["macro_f1"] <= 1.0 for row in comparison)
    assert 0.0 <= result["pass_probability"] <= 1.0
    assert 0.0 <= result["predicted_g3"] <= 100.0
    assert len(result["key_factors"]) == 3
    for factor in result["key_factors"]:
        assert factor["feature"]
        assert factor["value"]
        assert isinstance(factor["impact"], float)
        assert isinstance(factor["abs_impact"], float)
        assert isinstance(factor["importance"], float)
        assert factor["message"]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # 中文：所有数据库和导入文件都放在临时目录，保证测试不会污染用户真实数据。
        # English: Databases and import files live in a temp directory so real user data is untouched.
        db = HistoryDatabase(Path(tmp) / "history.db")
        db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        rows = db.list_predictions()
        assert [row["id"] for row in rows] == [1, 2]
        assert rows[0]["predicted_g3"] == result["predicted_g3"]
        detail = db.get_prediction(rows[0]["id"])
        assert detail["student_name"] == student.name
        assert detail["sex"] == student.sex
        assert detail["predicted_g3"] == result["predicted_g3"]
        assert db.delete_prediction(rows[0]["id"]) == 1
        assert [row["id"] for row in db.list_predictions()] == [1]
        db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        assert [row["id"] for row in db.list_predictions()] == [1, 2]
        db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        assert [row["id"] for row in db.list_predictions()] == [1, 2, 3, 4]
        assert db.delete_predictions([2, 4]) == 2
        assert [row["id"] for row in db.list_predictions()] == [1, 2]
        export_path = Path(tmp) / "history.xlsx"
        db.export_xlsx(export_path)
        with ZipFile(export_path) as workbook:
            sheet_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
            styles_xml = workbook.read("xl/styles.xml").decode("utf-8")
        assert "Predicted G3" in sheet_xml
        assert 's="2"' in sheet_xml or 's="3"' in sheet_xml
        assert "FFD9EAD3" in styles_xml
        assert "FFF4CCCC" in styles_xml

        legacy_path = Path(tmp) / "legacy_history.db"
        with sqlite3.connect(legacy_path) as conn:
            conn.execute(
                """
                CREATE TABLE predictions (
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
        legacy_db = HistoryDatabase(legacy_path)
        legacy_db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
        legacy_row = legacy_db.list_predictions()[0]
        assert legacy_row["pass_probability"] == result["pass_probability"]
        assert legacy_row["predicted_g3"] == result["predicted_g3"]
        with sqlite3.connect(legacy_path) as conn:
            confidence_score, pass_probability = conn.execute(
                "SELECT confidence_score, pass_probability FROM predictions WHERE id = 1"
            ).fetchone()
        assert confidence_score == pass_probability == result["pass_probability"]

        csv_path = Path(tmp) / "batch_students.csv"
        csv_path.write_text(
            (
                f'name,matric_no,sex,age,"{STUDY_TIME_COLUMN}",failures,activities,absences,G1,G2\n'
                "CSV Student,C001,F,17,3,0,yes,5,80,85\n"
            ),
            encoding="utf-8",
        )
        csv_students = StudentPerformanceApp._load_batch_students(csv_path)
        assert len(csv_students) == 1
        assert csv_students[0].name == "CSV Student"
        assert csv_students[0].study_time == 3
        assert csv_students[0].g1 == 80

        xlsx_path = Path(tmp) / "batch_students.xlsx"
        batch_headers = [
            "Name",
            "Matric No.",
            "Gender",
            "Age",
            "Study Time",
            "Failures",
            "Activities",
            "Absences",
            "G1",
            "G2",
        ]
        write_minimal_xlsx(
            xlsx_path,
            batch_headers,
            [["XLSX Student", "X001", "M", 18, 2, 0, "no", 2, 70, 75]],
        )
        xlsx_students = StudentPerformanceApp._load_batch_students(xlsx_path)
        assert len(xlsx_students) == 1
        assert xlsx_students[0].name == "XLSX Student"
        assert xlsx_students[0].matric_no == "X001"
        assert xlsx_students[0].g2 == 75

        for imported_student in [*csv_students, *xlsx_students]:
            imported_result = service.predict(imported_student)
            db.add_prediction(
                imported_student,
                imported_result["prediction"],
                imported_result["pass_probability"],
                imported_result["predicted_g3"],
            )
        assert [row["student_name"] for row in db.list_predictions()][-2:] == ["CSV Student", "XLSX Student"]

        invalid_path = Path(tmp) / "invalid_batch.csv"
        invalid_path.write_text(
            (
                "name,matric_no,sex,age,study_time,failures,activities,absences,G1,G2\n"
                "No Id,,F,17,2,0,yes,4,80,65\n"
                "Bad Grade,B001,F,17,2,0,yes,4,101,65\n"
            ),
            encoding="utf-8",
        )
        try:
            StudentPerformanceApp._load_batch_students(invalid_path)
        except ValueError as exc:
            message = str(exc)
            assert "Row 2" in message
            assert "Matric No." in message
            assert "Row 3" in message
            assert "Previous grade G1 must be between 0 and 100" in message
        else:
            raise AssertionError("Invalid batch input should fail validation.")

    print("Self-test passed")
    print(f"Holdout accuracy: {service.metrics['accuracy']:.2%}")
    print(
        f"5-fold accuracy: {evaluation['mean_accuracy']:.2%} "
        f"(+/- {evaluation['std_accuracy']:.2%})"
    )
    print(f"Macro F1: {evaluation['macro_average']['f1']:.2%}")
    print(
        f"Prediction: {result['prediction']} "
        f"(pass probability {result['pass_probability']:.2%}, predicted G3 {result['predicted_g3']:.1f})"
    )
    print("Top features:")
    for name, value in service.feature_importances()[:5]:
        print(f"  {name}: {value:.2%}")


if __name__ == "__main__":
    main()
