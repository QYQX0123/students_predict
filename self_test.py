"""Smoke test for model training, prediction explanation, and history storage."""

from pathlib import Path
from tempfile import TemporaryDirectory

from student_performance_system.data_utils import validate_student_input
from student_performance_system.database import HistoryDatabase
from student_performance_system.model_service import PredictionService


def main():
    """Run a lightweight end-to-end check without opening the Tkinter UI."""
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
    # Local explanations should be non-empty and describe the same predicted class.
    assert local_importances["items"]
    assert local_importances["prediction"] == result["prediction"]
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # Use a disposable database so the test never changes the user's real history.
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
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print("Top features:")
    for name, value in service.feature_importances()[:5]:
        print(f"  {name}: {value:.2%}")
    print("Top local influences:")
    for item in local_importances["items"][:5]:
        print(f"  {item['feature']}: {item['impact']:+.2%}")


if __name__ == "__main__":
    main()
