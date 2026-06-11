"""Smoke test for model training, evaluation, explanation, and history storage.

中文：这是无需打开 Tkinter 界面的轻量端到端测试，用断言快速检查主要业务流程。
English: This lightweight end-to-end test verifies core workflows with assertions
without opening the Tkinter interface.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from student_performance_system.data_utils import validate_student_input
from student_performance_system.database import HistoryDatabase
from student_performance_system.model_service import PredictionService


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
