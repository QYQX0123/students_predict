"""Shared data definitions, CSV loading, and user-input validation.

中文：本模块统一定义学生输入结构、模型特征名称、CSV 数据集读取规则和表单校验规则。
English: This module centralizes student input structures, model feature names,
CSV dataset loading rules, and form validation.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


STUDY_TIME_COLUMN = (
    "weeklystudytime(numeric: 1 - <5 hours, 2 - 5 to 10 hours, "
    "3 - 10 to 20 hours, or 4 - >20 hours)"
)

# 中文：模型使用的原始 CSV 列名；顺序也决定后续编码和特征重要性的展示顺序。
# English: Raw CSV columns used by the model; this order also controls encoding
# and feature-importance display order.
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

# 中文：将较长或技术性的字段名转换为界面中更易理解的名称。
# English: Convert technical dataset column names into user-friendly UI labels.
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
    """Store validated student information shared by all application layers.

    中文：字段已经过类型转换和范围校验，可安全地用于界面、数据库和预测服务。
    English: Fields have already been type-converted and range-validated, so the
    object can safely be shared by the UI, database, and prediction service.
    """

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
        """Convert this object to the exact feature dictionary expected by the model.

        中文：界面采用 0-100 分制，而训练数据的 G1/G2 使用 0-20 分制，因此这里
        在保留其他字段的同时完成成绩换算。
        English: The UI uses a 0-100 scale while G1/G2 in the training dataset use
        0-20, so this method converts grades while preserving all other fields.
        """
        return {
            "sex": self.sex,
            "age": self.age,
            STUDY_TIME_COLUMN: self.study_time,
            "failures": self.failures,
            "activities": self.activities,
            "absences": self.absences,
            "G1": score_100_to_20(self.g1),
            "G2": score_100_to_20(self.g2),
        }


def performance_category(score):
    """Map final grade G3 (0-20) to Low, Medium, or High.

    中文：低于 10 为 Low，10 至低于 15 为 Medium，15 及以上为 High。
    English: Scores below 10 are Low, 10 to below 15 are Medium, and 15 or above
    are High.
    """
    score = float(score)
    if score < 10:
        return "Low"
    if score < 15:
        return "Medium"
    return "High"


def score_100_to_20(score):
    """Convert a 0-100 score to the dataset's 0-20 scale / 将百分制换算为二十分制。"""
    return round(float(score) / 5, 2)


def load_dataset(path):
    """Load and validate the training CSV.

    中文：返回三个对象：模型特征行列表、由 G3 生成的分类标签列表，以及未经转换的
    原始行。文件不存在、没有数据或缺少必要列时会立即抛出清晰的异常。
    English: Returns model feature rows, class labels derived from G3, and untouched
    raw rows. Clear exceptions are raised for a missing file, empty data, or schema
    mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Dataset is empty.")

    # 中文：训练前先检查结构，避免在模型内部出现难以定位的 KeyError。
    # English: Validate the schema early to avoid obscure KeyError failures in training.
    missing = [col for col in FEATURES + ["G3"] if col not in rows[0]]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    x_rows = []
    y = []
    for row in rows:
        item = {}
        for feature in FEATURES:
            value = row[feature]
            # 中文：分类字段暂时保留字符串，由 PredictionService 统一数值编码。
            # English: Keep categorical values as strings until PredictionService
            # performs consistent numeric encoding.
            if feature in {"sex", "activities"}:
                item[feature] = value.strip()
            else:
                item[feature] = float(value)
        x_rows.append(item)
        y.append(performance_category(row["G3"]))

    return x_rows, y, rows


def validate_student_input(values, require_identity=False):
    """Validate raw form strings and return a typed StudentInput.

    中文：Tkinter 输入均为字符串。本函数检查分类选项、整数格式及允许范围，并为
    空姓名和空学号提供默认值；任何错误都以 ValueError 返回给界面显示。
    English: Tkinter inputs arrive as strings. This function validates categories,
    integer syntax, and allowed ranges, supplies defaults for blank identity fields,
    and reports failures as ValueError messages for the UI.
    """
    name = values.get("name", "").strip()
    matric_no = values.get("matric_no", "").strip()
    sex = values.get("sex", "").strip()
    activities = values.get("activities", "").strip()

    if require_identity:
        if not name:
            raise ValueError("Name is required.")
        if not matric_no:
            raise ValueError("Matric No. is required.")

    if sex not in {"F", "M"}:
        raise ValueError("Gender must be F or M.")
    if activities not in {"yes", "no"}:
        raise ValueError("Activities must be yes or no.")

    def int_in_range(key, label, low, high):
        """Parse and range-check one integer / 解析整数并检查闭区间范围。"""
        try:
            raw_value = str(values.get(key, "")).strip()
            number = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a whole number.") from exc
        if not number.is_integer():
            raise ValueError(f"{label} must be a whole number.")
        value = int(number)
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
        g1=int_in_range("g1", "Previous grade G1", 0, 100),
        g2=int_in_range("g2", "Midterm grade G2", 0, 100),
    )
