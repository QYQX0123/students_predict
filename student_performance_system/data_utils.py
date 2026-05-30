import csv
from dataclasses import dataclass
from pathlib import Path


STUDY_TIME_COLUMN = (
    "weeklystudytime(numeric: 1 - <5 hours, 2 - 5 to 10 hours, "
    "3 - 10 to 20 hours, or 4 - >20 hours)"
)

FEATURES = [
    "sex",
    "age",
    STUDY_TIME_COLUMN,
    "failures",
    "activities",
    "absences",
    "G1",
    "G2",
]

DISPLAY_FEATURES = {
    "sex": "Gender",
    "age": "Age",
    STUDY_TIME_COLUMN: "Study Time",
    "failures": "Failures",
    "activities": "Activities",
    "absences": "Absences",
    "G1": "Previous Grade (G1)",
    "G2": "Midterm Grade (G2)",
}


@dataclass
class StudentInput:
    name: str
    matric_no: str
    sex: str
    age: int
    study_time: int
    failures: int
    activities: str
    absences: int
    g1: int
    g2: int

    def to_feature_row(self):
        return {
            "sex": self.sex,
            "age": self.age,
            STUDY_TIME_COLUMN: self.study_time,
            "failures": self.failures,
            "activities": self.activities,
            "absences": self.absences,
            "G1": self.g1,
            "G2": self.g2,
        }


def performance_category(score):
    score = float(score)
    if score < 10:
        return "Low"
    if score < 15:
        return "Medium"
    return "High"


def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Dataset is empty.")

    missing = [col for col in FEATURES + ["G3"] if col not in rows[0]]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    x_rows = []
    y = []
    for row in rows:
        item = {}
        for feature in FEATURES:
            value = row[feature]
            if feature in {"sex", "activities"}:
                item[feature] = value.strip()
            else:
                item[feature] = float(value)
        x_rows.append(item)
        y.append(performance_category(row["G3"]))

    return x_rows, y, rows


def validate_student_input(values):
    name = values.get("name", "").strip()
    matric_no = values.get("matric_no", "").strip()
    sex = values.get("sex", "").strip()
    activities = values.get("activities", "").strip()

    if sex not in {"F", "M"}:
        raise ValueError("Gender must be F or M.")
    if activities not in {"yes", "no"}:
        raise ValueError("Activities must be yes or no.")

    def int_in_range(key, label, low, high):
        try:
            value = int(values.get(key, ""))
        except ValueError as exc:
            raise ValueError(f"{label} must be a whole number.") from exc
        if value < low or value > high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    return StudentInput(
        name=name or "Unnamed Student",
        matric_no=matric_no or "-",
        sex=sex,
        age=int_in_range("age", "Age", 10, 25),
        study_time=int_in_range("study_time", "Study time", 1, 4),
        failures=int_in_range("failures", "Failures", 0, 4),
        activities=activities,
        absences=int_in_range("absences", "Absences", 0, 100),
        g1=int_in_range("g1", "Previous grade G1", 0, 20),
        g2=int_in_range("g2", "Midterm grade G2", 0, 20),
    )
