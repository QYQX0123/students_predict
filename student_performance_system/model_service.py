"""Prediction service that connects data preparation, model training, and the UI.

中文：这个模块实现“应用如何使用模型”：加载数据、编码特征、训练随机森林、计算评估
指标、进行单个学生预测，并整理界面需要的结果格式。这样写可以让 Tkinter 界面不直接
接触训练细节，也让算法模块保持通用。

English: This module implements how the application uses the model: loading data,
encoding features, training the forest, computing evaluation metrics, predicting a
single student, and shaping results for the UI. This keeps Tkinter away from
training details and keeps the algorithm module reusable.
"""

import math
import random
from collections import Counter

from .data_utils import DISPLAY_FEATURES, FEATURES, STUDY_TIME_COLUMN, TARGET_CLASSES, load_dataset
from .random_forest import DecisionTreeClassifier, RandomForestClassifier, RandomForestRegressor


STUDY_TIME_LABELS = {
    1: "<5 hours",
    2: "5 to 10 hours",
    3: "10 to 20 hours",
    4: ">20 hours",
}


class PredictionService:
    """Application-facing wrapper around the custom RandomForestClassifier.

    中文：类的职责是把原始业务字段转成模型字段，并把模型输出转成界面可显示的字典。
    这样写的原因是隔离“机器学习流程”和“窗口控件逻辑”。

    English: This class translates business fields into model fields and model
    output into dictionaries the UI can display. It separates the machine-learning
    workflow from widget logic.
    """

    def __init__(self, dataset_path):
        """Store the dataset path and train immediately.

        中文：桌面程序启动后应立即可预测，所以构造服务时直接训练模型，而不是等到用户
        第一次点击 Predict。

        English: The desktop app should be ready to predict after startup, so the
        model is trained during service construction instead of on the first click.
        """
        self.dataset_path = dataset_path
        self.model = None
        self.metrics = {}
        self.reference_values = {}
        self.train()

    def train(self):
        """Train the main forest and collect evaluation summaries.

        中文：实现流程为：读取 CSV、编码特征、固定随机种子切分 80/20 留出集、训练主模型、
        计算留出准确率，并额外执行五折交叉验证。固定种子是为了每次运行报告相同结果，
        便于调试和论文/报告复现。

        English: The workflow loads the CSV, encodes features, performs a seeded
        80/20 holdout split, trains the main model, computes holdout accuracy, and
        also runs five-fold cross-validation. The fixed seed makes reported results
        reproducible for debugging and documentation.
        """
        x_rows, y, raw_rows = load_dataset(self.dataset_path)
        x_rows = [self._encode_row(row) for row in x_rows]
        g3_scores = [float(row["G3"]) for row in raw_rows]
        self.reference_values = self._reference_values(x_rows)

        # 中文：固定随机种子，使训练/测试划分在每次运行时一致。
        # English: The fixed seed keeps the train/test split identical across runs.
        indexes = list(range(len(y)))
        rng = random.Random(42)
        rng.shuffle(indexes)
        split_at = int(len(indexes) * 0.8)
        train_idx = indexes[:split_at]
        test_idx = indexes[split_at:]

        x_train = [x_rows[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        g3_train = [g3_scores[i] for i in train_idx]
        x_test = [x_rows[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        self.model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        self.model.fit(x_train, y_train)
        self.grade_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=43)
        self.grade_model.fit(x_train, g3_train)

        predictions = [self.model.predict_one(row) for row in x_test]
        accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
        model_comparison = self._compare_models(x_train, y_train, x_test, y_test, predictions)
        self.metrics = {
            "accuracy": accuracy,
            "train_size": len(x_train),
            "test_size": len(x_test),
            "class_distribution": dict(Counter(y)),
            "evaluation": self._cross_validate(x_rows, y, folds=5),
            "model_comparison": model_comparison,
        }
        self.raw_rows = raw_rows
        return self.metrics

    @staticmethod
    def _reference_values(x_rows):
        """Return the median encoded value for each feature.

        English: Key factors compare the current student with a typical training
        value. Medians are deterministic and work for the numeric feature encoding
        used by the custom forest.
        """
        references = {}
        for feature in x_rows[0]:
            values = sorted(row[feature] for row in x_rows)
            midpoint = len(values) // 2
            if len(values) % 2:
                references[feature] = values[midpoint]
            else:
                references[feature] = (values[midpoint - 1] + values[midpoint]) / 2
        return references

    def _compare_models(self, x_train, y_train, x_test, y_test, forest_predictions):
        """Compare simple benchmark models on the same holdout split.

        English: These models are for evaluation evidence only. The application still
        uses the custom Random Forest as its final prediction model.
        """
        majority_label = Counter(y_train).most_common(1)[0][0]
        baseline_predictions = [majority_label for _ in x_test]

        tree = DecisionTreeClassifier(max_depth=7, random_state=42)
        tree.fit(x_train, y_train, list(x_train[0].keys()))
        tree_predictions = [tree.predict_one(row) for row in x_test]

        rows = [
            (
                "Majority Baseline",
                baseline_predictions,
            ),
            (
                "Single Decision Tree",
                tree_predictions,
            ),
            (
                "Random Forest (Final)",
                forest_predictions,
            ),
        ]
        comparison = []
        for name, predictions in rows:
            metrics = self._classification_metrics(y_test, predictions)
            comparison.append(
                {
                    "model": name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_average"]["f1"],
                }
            )
        return comparison

    def _cross_validate(self, x_rows, y, folds=5):
        """Run stratified k-fold validation and aggregate out-of-fold predictions.

        中文：实现了分层五折评估：同一类别的样本先打乱，再轮流放入不同折，减少类别
        不均衡对某一折的影响。每一折都重新训练模型，测试样本不参与该折训练，因此
        得到的预测更接近未见数据表现。

        English: Implements stratified k-fold evaluation. Samples of each class are
        shuffled and distributed across folds to reduce class-imbalance effects. A
        fresh model is trained for every fold, and test samples are never used in
        that fold's training, so predictions better approximate unseen performance.
        """
        # 中文：fold_indexes[i] 保存第 i 折要作为测试集的样本下标。
        # English: fold_indexes[i] stores the sample indexes used as fold i's test set.
        fold_indexes = [[] for _ in range(folds)]
        indexes_by_class = {}
        for index, label in enumerate(y):
            indexes_by_class.setdefault(label, []).append(index)

        rng = random.Random(42)
        for indexes in indexes_by_class.values():
            # 中文：按类别轮流分配样本，让每折尽量拥有相近的类别比例。
            # English: Round-robin assignment keeps class proportions similar per fold.
            rng.shuffle(indexes)
            for position, index in enumerate(indexes):
                fold_indexes[position % folds].append(index)

        all_indexes = set(range(len(y)))
        predictions = [None] * len(y)
        fold_accuracies = []
        for test_indexes in fold_indexes:
            # 中文：当前折只用于测试，其余样本训练一个独立森林。
            # English: The current fold is test-only; all remaining samples train a separate forest.
            train_indexes = sorted(all_indexes - set(test_indexes))
            model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
            model.fit([x_rows[i] for i in train_indexes], [y[i] for i in train_indexes])

            correct = 0
            for index in test_indexes:
                prediction = model.predict_one(x_rows[index])
                predictions[index] = prediction
                correct += prediction == y[index]
            fold_accuracies.append(correct / len(test_indexes))

        evaluation = self._classification_metrics(y, predictions)
        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        # 中文：标准差越小，表示不同折之间的准确率波动越小。
        # English: A smaller standard deviation means accuracy is more stable across folds.
        variance = sum((value - mean_accuracy) ** 2 for value in fold_accuracies) / len(fold_accuracies)
        evaluation.update(
            {
                "folds": folds,
                "fold_accuracies": fold_accuracies,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": math.sqrt(variance),
            }
        )
        return evaluation

    @staticmethod
    def _classification_metrics(actual, predicted):
        """Build confusion matrix, precision, recall, F1, macro average, and accuracy.

        中文：混淆矩阵按“真实类别 -> 预测类别”组织；每个类别再按一对多方式计算 TP、
        FP、FN。宏平均不按样本数加权，适合查看 Fail 与 Pass 是否都被模型照顾到。

        English: The confusion matrix is organized as actual class to predicted
        class. Each class then gets one-vs-rest TP, FP, and FN. Macro averages are
        unweighted, which helps show whether both Fail and Pass are handled well.
        """
        observed = set(actual) | set(predicted)
        classes = [klass for klass in TARGET_CLASSES if klass in observed]
        classes.extend(sorted(observed - set(classes)))
        matrix = {
            actual_class: {
                predicted_class: sum(
                    1
                    for target, prediction in zip(actual, predicted)
                    if target == actual_class and prediction == predicted_class
                )
                for predicted_class in classes
            }
            for actual_class in classes
        }

        per_class = {}
        for klass in classes:
            # 中文：把当前类别当作正类，其余类别当作负类来计算指标。
            # English: Treat the current class as positive and all other classes as negative.
            true_positive = matrix[klass][klass]
            false_positive = sum(matrix[other][klass] for other in classes if other != klass)
            false_negative = sum(matrix[klass][other] for other in classes if other != klass)
            support = sum(matrix[klass].values())
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[klass] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        macro = {
            metric: sum(values[metric] for values in per_class.values()) / len(per_class)
            for metric in ("precision", "recall", "f1")
        }
        macro["support"] = len(actual)
        accuracy = sum(target == prediction for target, prediction in zip(actual, predicted)) / len(actual)
        return {
            "classes": classes,
            "confusion_matrix": matrix,
            "per_class": per_class,
            "macro_average": macro,
            "accuracy": accuracy,
        }

    def predict(self, student_input):
        """Predict one validated student.

        中文：先把 StudentInput 转成模型特征，再用随机森林投票。返回值同时包含最终
        Pass/Fail、获胜类别置信度、Pass 概率和完整概率字典，满足结果卡片、概率条和
        历史保存的不同需求。

        English: Converts StudentInput into model features and lets the forest vote.
        The result includes Pass/Fail, winning-class confidence, Pass probability,
        and the full probability dictionary for the result card, bars, and history.
        """
        row = self._encode_row(student_input.to_feature_row())
        prediction = self.model.predict_one(row)
        probabilities = self._complete_probabilities(self.model.predict_proba_one(row))
        pass_probability = probabilities["Pass"]
        confidence = probabilities.get(prediction, 0.0)
        predicted_g3 = self._grade_20_to_100(self.grade_model.predict_one(row))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "pass_probability": pass_probability,
            "predicted_g3": predicted_g3,
            "probabilities": probabilities,
            "key_factors": self._key_factors(student_input, row, pass_probability),
        }

    @staticmethod
    def _grade_20_to_100(score):
        """Convert the model's 0-20 G3 estimate to the UI's 0-100 scale."""
        return round(max(0.0, min(20.0, float(score))) * 5, 1)

    def _key_factors(self, student_input, row, pass_probability):
        """Build local key factors by perturbing one feature at a time.

        English: Each feature is replaced with its training-reference value and the
        change in Pass probability is measured. The largest changes become the
        user-facing key factors.
        """
        factors = []
        for feature in FEATURES:
            encoded_feature = self._encoded_feature_name(feature)
            altered = dict(row)
            altered[encoded_feature] = self.reference_values.get(encoded_feature, row[encoded_feature])
            reference_probability = pass_probability
            if altered[encoded_feature] != row[encoded_feature]:
                reference_probability = self._complete_probabilities(self.model.predict_proba_one(altered))["Pass"]
            impact = pass_probability - reference_probability
            factors.append(
                {
                    "feature": DISPLAY_FEATURES[feature],
                    "value": self._student_feature_value(student_input, feature),
                    "impact": impact,
                    "abs_impact": abs(impact),
                    "importance": self.model.feature_importances_.get(encoded_feature, 0.0),
                    "message": self._factor_message(impact),
                }
            )
        return sorted(factors, key=lambda item: (item["abs_impact"], item["importance"]), reverse=True)[:3]

    @staticmethod
    def _student_feature_value(student_input, feature):
        """Format one current student value for display."""
        if feature == "sex":
            return "Male" if student_input.sex == "M" else "Female"
        if feature == STUDY_TIME_COLUMN:
            return STUDY_TIME_LABELS.get(student_input.study_time, str(student_input.study_time))
        if feature == "activities":
            return "Yes" if student_input.activities == "yes" else "No"
        if feature == "G1":
            return f"{student_input.g1}/100"
        if feature == "G2":
            return f"{student_input.g2}/100"
        if feature == "age":
            return f"{student_input.age} years"
        if feature == "absences":
            return f"{student_input.absences} absence(s)"
        if feature == "failures":
            return f"{student_input.failures} previous failure(s)"
        return str(getattr(student_input, feature, ""))

    @staticmethod
    def _factor_message(impact):
        """Describe how the current value affects Pass probability."""
        if impact > 0.005:
            return "This value supports a higher pass probability than the training reference."
        if impact < -0.005:
            return "This value lowers pass probability compared with the training reference."
        return "This factor is one of the stronger signals used by the trained model."

    def feature_importances(self):
        """Return global feature importance sorted for the evaluation chart.

        中文：随机森林内部使用编码后的特征名，界面需要可读标签，所以这里完成名称转换
        并按重要性降序排列。

        English: The forest stores encoded feature names, while the UI needs readable
        labels. This method converts names and sorts them in descending importance.
        """
        items = []
        for feature in FEATURES:
            encoded_feature = self._encoded_feature_name(feature)
            items.append((DISPLAY_FEATURES[feature], self.model.feature_importances_.get(encoded_feature, 0.0)))
        return sorted(items, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _complete_probabilities(probabilities):
        """Ensure both target classes are always present.

        中文：某些输入可能让所有树都投给同一类；补齐缺失类别可以让界面固定绘制 Fail
        和 Pass 两条概率条，不需要额外判断。

        English: Some inputs may receive votes for only one class. Filling missing
        classes lets the UI always draw both Fail and Pass bars without special cases.
        """
        return {klass: probabilities.get(klass, 0.0) for klass in TARGET_CLASSES}

    def _encode_row(self, row):
        """Turn mixed categorical/numeric fields into numeric tree features.

        中文：自定义决策树只比较数字阈值，所以性别和活动被编码成 0/1，其余字段转成
        float。编码集中在这里，可以保证训练集和用户输入使用完全相同的表示。

        English: The custom tree compares only numeric thresholds, so sex and
        activities become 0/1 and all other fields become floats. Centralizing this
        encoding guarantees identical representation for training and user input.
        """
        encoded = {}
        for feature in FEATURES:
            value = row[feature]
            if feature == "sex":
                encoded[self._encoded_feature_name(feature)] = 1.0 if value == "M" else 0.0
            elif feature == "activities":
                encoded[self._encoded_feature_name(feature)] = 1.0 if value == "yes" else 0.0
            else:
                encoded[self._encoded_feature_name(feature)] = float(value)
        return encoded

    @staticmethod
    def _encoded_feature_name(feature):
        """Map dataset column names to internal numeric feature names.

        中文：二元分类字段编码后语义会变化，例如 sex 变成 sex_M；显式映射能让重要性
        统计和界面展示知道如何对应回原始字段。

        English: Binary categorical fields change meaning after encoding, such as
        sex becoming sex_M. Explicit mapping lets importance statistics and UI labels
        trace values back to original fields.
        """
        if feature == "sex":
            return "sex_M"
        if feature == "activities":
            return "activities_yes"
        return feature
