"""Data contracts, dataset loading, and form validation.

中文：这个模块定义“学生记录在程序里长什么样”，并把训练 CSV 与界面输入统一成模型
可以消费的格式。这样写的原因是让数据清洗、字段命名、分数换算和输入校验集中在一处，
避免界面、数据库和模型服务各自重复一套规则。

English: This module defines the in-program shape of a student record and converts
both the training CSV and UI form values into the format consumed by the model. The
rules live here so cleaning, naming, score conversion, and validation are not
duplicated across the UI, database, and prediction service.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


STUDY_TIME_COLUMN = (
    "weeklystudytime(numeric: 1 - <5 hours, 2 - 5 to 10 hours, "
    "3 - 10 to 20 hours, or 4 - >20 hours)"
)

# 中文：训练和预测共同使用的特征列；把列表集中定义可以保证训练、预测和重要性展示
# 使用同一组字段与顺序。
# English: Feature columns shared by training and prediction. Keeping them in one
# list guarantees that training, inference, and importance display use the same
# fields in the same order.
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

# 中文：界面展示名与数据集列名分离，既保留原始 CSV 兼容性，又让用户看到更短的标签。
# English: Display names are separated from dataset column names so CSV compatibility
# is preserved while the UI can show shorter, friendlier labels.
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

PASS_THRESHOLD = 10.0
TARGET_CLASSES = ("Fail", "Pass")


@dataclass
class StudentInput:
    """Validated student input used as the boundary object between layers.

    中文：这个数据类保存一次预测所需的全部学生字段。它实现的功能是把松散的表单值
    变成有类型的对象；这样写是为了让数据库保存、模型预测和历史详情都使用同一种
    结构，减少字段遗漏或拼写不一致。

    English: This dataclass stores every field required for one prediction. It turns
    loose form values into a typed object so database writes, model prediction, and
    history details all share one structure and avoid field-name drift.
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
        """Build the model feature dictionary for this student.

        中文：界面让用户输入 0-100 分，训练集中的 G1/G2 是 0-20 分，所以这里完成
        分数换算并使用原始 CSV 的列名作为键。这样写可以让界面保持直观，同时不改变
        模型训练时依赖的数据格式。

        English: The UI accepts 0-100 grades, while the dataset stores G1/G2 on a
        0-20 scale. This method converts scores and uses the raw CSV column names as
        keys, keeping the UI intuitive without changing the model's training schema.
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


def pass_status(score):
    """Convert final grade G3 into the binary target label.

    中文：实现的功能是把 0-20 分制的最终成绩转成 Fail/Pass；阈值集中使用
    PASS_THRESHOLD，便于以后修改及在代码中保持同一标准。

    English: Converts the final 0-20 G3 grade into the Fail/Pass target. The cutoff
    is centralized in PASS_THRESHOLD so the rule can be changed once and remain
    consistent everywhere.
    """
    score = float(score)
    return "Pass" if score >= PASS_THRESHOLD else "Fail"


def score_100_to_20(score):
    """Convert a UI grade to the dataset grade scale.

    中文：实现百分制到二十分制的换算；保留两位小数是为了让批量导入中的非整型成绩
    也能稳定进入模型。

    English: Converts a 0-100 UI score to the 0-20 dataset scale. Rounding to two
    decimals keeps imported non-integer grades stable for model input.
    """
    return round(float(score) / 5, 2)


def load_dataset(path):
    """Read the training CSV and split it into features, labels, and raw rows.

    中文：这个函数实现三件事：检查文件和列是否存在、把可训练字段提取出来、用 G3
    生成二分类标签。这样写是为了在模型训练开始前就暴露数据问题，而不是让错误在
    随机森林内部变得难以定位。

    English: This function validates the file and required columns, extracts
    trainable fields, and derives binary labels from G3. Doing this before training
    makes dataset problems explicit instead of surfacing later inside the forest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Dataset is empty.")

    # 中文：先做结构检查，缺列时给出列名，方便用户修复数据文件。
    # English: Check the schema first and name missing columns so the dataset can be fixed quickly.
    missing = [col for col in FEATURES + ["G3"] if col not in rows[0]]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    x_rows = []
    y = []
    for row in rows:
        item = {}
        for feature in FEATURES:
            value = row[feature]
            # 中文：分类字段先保留文本，让模型服务在一个地方统一完成数值编码。
            # English: Keep categorical fields as text so numeric encoding happens in one service.
            if feature in {"sex", "activities"}:
                item[feature] = value.strip()
            else:
                item[feature] = float(value)
        x_rows.append(item)
        y.append(pass_status(row["G3"]))

    return x_rows, y, rows


def validate_student_input(values, require_identity=False):
    """Validate raw UI or batch values and return a StudentInput.

    中文：实现的功能包括必填身份检查、分类选项检查、整数解析和范围检查。这里接收
    字符串字典，是因为 Tkinter 与 CSV/XLSX 导入都会先得到文本；集中校验可以让单条
    预测和批量导入使用完全相同的错误规则。

    English: Performs identity, category, integer, and range validation. It accepts
    a string dictionary because both Tkinter and CSV/XLSX import start as text;
    centralized validation gives single prediction and batch import identical rules.
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
        """Parse one integer field and enforce its inclusive range.

        中文：允许 "2.0" 这类来自电子表格的数字文本，但拒绝真正的小数。这样写可以
        兼容 Excel 导出，同时保护模型不收到不合法的离散选项。

        English: Accepts spreadsheet-style integer text such as "2.0" but rejects
        real decimals, making Excel import tolerant while protecting discrete fields.
        """
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
