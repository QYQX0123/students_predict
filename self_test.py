from pathlib import Path
from tempfile import TemporaryDirectory

from student_performance_system.data_utils import validate_student_input
from student_performance_system.database import HistoryDatabase
from student_performance_system.model_service import PredictionService


def main():
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
            "g1": "12",
            "g2": "13",
        }
    )
    result = service.predict(student)
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = HistoryDatabase(Path(tmp) / "history.db")
        db.add_prediction(student, result["prediction"], result["confidence"])
        assert db.list_predictions()

    print("Self-test passed")
    print(f"Holdout accuracy: {service.metrics['accuracy']:.2%}")
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print("Top features:")
    for name, value in service.feature_importances()[:5]:
        print(f"  {name}: {value:.2%}")


if __name__ == "__main__":
    main()
