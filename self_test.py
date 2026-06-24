"""Smoke test for model training, evaluation, explanation, and history storage.

中文：这是无需打开 Tkinter 界面的轻量端到端测试，用断言快速检查主要业务流程。
English: This lightweight end-to-end test verifies core workflows with assertions
without opening the Tkinter interface.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZipFile

from student_performance_system.app import StudentPerformanceApp
from student_performance_system.data_utils import STUDY_TIME_COLUMN, validate_student_input
from student_performance_system.database import HistoryDatabase
from student_performance_system.model_service import PredictionService


def write_minimal_xlsx(path, headers, rows):
    """Create a tiny one-sheet XLSX fixture with inline strings and numeric values."""
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
    """Train, predict, evaluate, persist, delete, and print representative results.

    中文：任何断言失败都会使脚本以错误状态退出，适合修改代码后的快速回归检查。
    English: Any failed assertion exits with an error, making this useful as a quick
    regression check after code changes.
    """
    service = PredictionService(Path("dataset.csv"))
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
    local_importances = service.local_feature_importances(student)
    evaluation = service.metrics["evaluation"]
    # 中文：交叉验证应包含五折、覆盖全部数据并为三个目标类别产生指标。
    # English: Cross-validation must contain five folds, cover all records, and report all classes.
    assert len(evaluation["fold_accuracies"]) == 5
    record_count = sum(service.metrics["class_distribution"].values())
    assert sum(sum(row.values()) for row in evaluation["confusion_matrix"].values()) == record_count
    assert set(evaluation["per_class"]) == {"Low", "Medium", "High"}
    assert 0.0 <= evaluation["macro_average"]["f1"] <= 1.0
    # 中文：局部解释必须非空，并且解释类别应与普通预测结果一致。
    # English: Local explanations must be non-empty and target the same predicted class.
    assert local_importances["items"]
    assert local_importances["prediction"] == result["prediction"]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # 中文：使用临时数据库，测试结束后自动清理，不修改真实历史记录。
        # English: A disposable database is removed afterward and never changes real history.
        db = HistoryDatabase(Path(tmp) / "history.db")
        db.add_prediction(student, result["prediction"], result["confidence"])
        db.add_prediction(student, result["prediction"], result["confidence"])
        rows = db.list_predictions()
        assert [row["id"] for row in rows] == [1, 2]
        detail = db.get_prediction(rows[0]["id"])
        assert detail["student_name"] == student.name
        assert detail["sex"] == student.sex
        assert db.delete_prediction(rows[0]["id"]) == 1
        assert [row["id"] for row in db.list_predictions()] == [1]
        db.add_prediction(student, result["prediction"], result["confidence"])
        assert [row["id"] for row in db.list_predictions()] == [1, 2]
        db.add_prediction(student, result["prediction"], result["confidence"])
        db.add_prediction(student, result["prediction"], result["confidence"])
        assert [row["id"] for row in db.list_predictions()] == [1, 2, 3, 4]
        assert db.delete_predictions([2, 4]) == 2
        assert [row["id"] for row in db.list_predictions()] == [1, 2]

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
            db.add_prediction(imported_student, imported_result["prediction"], imported_result["confidence"])
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
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print("Top features:")
    for name, value in service.feature_importances()[:5]:
        print(f"  {name}: {value:.2%}")
    print("Top local influences:")
    for item in local_importances["items"][:5]:
        print(f"  {item['feature']}: {item['impact']:+.2%}")


if __name__ == "__main__":
    main()
