"""High-level prediction workflow that prepares data and owns the model.

中文：连接数据处理层、自定义随机森林和 Tkinter 界面，负责训练、预测、交叉验证、
全局特征重要性和单个学生的局部解释。
English: Connects data processing, the custom Random Forest, and the Tkinter UI.
It owns training, prediction, cross-validation, global importance, and local
per-student explanations.
"""

import math
import random
from collections import Counter

from .data_utils import DISPLAY_FEATURES, FEATURES, load_dataset
from .random_forest import RandomForestClassifier


class PredictionService:
    """Train the model and expose UI-friendly prediction and evaluation methods.

    中文：界面不直接操作算法细节，只通过该服务获取统一格式的结果。
    English: The UI does not manipulate algorithm internals directly; it receives
    consistently structured results through this service.
    """

    def __init__(self, dataset_path):
        """Initialize state and train immediately / 初始化状态并立即训练以供界面使用。"""
        self.dataset_path = dataset_path
        self.model = None
        self.metrics = {}
        self.reference_row = {}
        self.train()

    def train(self):
        """Train the production model and calculate evaluation metrics.

        中文：先固定随机种子打乱数据，按 80%/20% 划分训练集和留出测试集；正式模型
        在训练集上拟合。除留出准确率外，还对完整数据执行分层五折交叉验证。
        English: Data is deterministically shuffled and split 80/20 into training
        and holdout sets. The production model is fitted on the training set. In
        addition to holdout accuracy, stratified 5-fold cross-validation evaluates
        the full dataset.
        """
        x_rows, y, raw_rows = load_dataset(self.dataset_path)
        x_rows = [self._encode_row(row) for row in x_rows]

        # 中文：固定随机种子，让每次启动得到相同的数据划分和可复现实验结果。
        # English: A fixed seed makes the split and reported results reproducible.
        indexes = list(range(len(y)))
        rng = random.Random(42)
        rng.shuffle(indexes)
        split_at = int(len(indexes) * 0.8)
        train_idx = indexes[:split_at]
        test_idx = indexes[split_at:]

        x_train = [x_rows[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        x_test = [x_rows[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        self.model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        self.model.fit(x_train, y_train)
        self.reference_row = self._build_reference_row(x_train)

        predictions = [self.model.predict_one(row) for row in x_test]
        accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
        self.metrics = {
            "accuracy": accuracy,
            "train_size": len(x_train),
            "test_size": len(x_test),
            "class_distribution": dict(Counter(y)),
            "evaluation": self._cross_validate(x_rows, y, folds=5),
        }
        self.raw_rows = raw_rows
        return self.metrics

    def _cross_validate(self, x_rows, y, folds=5):
        """Run deterministic stratified k-fold cross-validation.

        中文：先按类别分别打乱并轮流分配到各折，尽量保持每折的类别比例。每一折都用其余
        数据重新训练独立模型，并为当前测试折生成 out-of-fold 预测，最终汇总准确率、
        标准差、混淆矩阵以及 Precision/Recall/F1。
        English: Samples are shuffled within each class and distributed round-robin
        to preserve class balance. Each fold trains an independent model on the
        remaining data and produces out-of-fold predictions used to aggregate
        accuracy, standard deviation, confusion matrix, precision, recall, and F1.
        """
        # 中文：每个内部列表保存一折的测试样本下标。
        # English: Each inner list stores the test indexes for one fold.
        fold_indexes = [[] for _ in range(folds)]
        indexes_by_class = {}
        for index, label in enumerate(y):
            indexes_by_class.setdefault(label, []).append(index)

        rng = random.Random(42)
        for indexes in indexes_by_class.values():
            # 中文：分层后轮流发牌，避免少数类别集中在某一折。
            # English: Round-robin assignment prevents minority classes clustering.
            rng.shuffle(indexes)
            for position, index in enumerate(indexes):
                fold_indexes[position % folds].append(index)

        all_indexes = set(range(len(y)))
        predictions = [None] * len(y)
        fold_accuracies = []
        for test_indexes in fold_indexes:
            # 中文：当前折仅作测试，其余折组成训练集，杜绝测试样本参与本折训练。
            # English: The current fold is test-only; all remaining folds form training data.
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
        # 中文：这里计算总体标准差，用于观察模型在不同折之间是否稳定。
        # English: Population standard deviation indicates stability between folds.
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
        """Calculate confusion matrix and standard classification metrics.

        中文：对每个类别计算 TP、FP、FN，再得到 Precision、Recall 和 F1；Macro Avg
        对各类别等权平均，因此不会让样本较多的类别完全主导结果。
        English: Computes TP, FP, and FN per class, then derives precision, recall,
        and F1. Macro averages weight every class equally, preventing a large class
        from completely dominating the result.
        """
        observed = set(actual) | set(predicted)
        classes = [klass for klass in ("Low", "Medium", "High") if klass in observed]
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
            # 中文：从混淆矩阵中读取该类别的一对多统计量。
            # English: Derive one-vs-rest statistics for the current class.
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
        """Predict one student and return class, confidence, and all probabilities.

        中文：confidence 是获胜类别的投票比例；probabilities 保留 Low/Medium/High
        的完整分布，供界面绘制进度条。
        English: Confidence is the winning class's vote ratio, while probabilities
        keeps the complete Low/Medium/High distribution for UI bars.
        """
        row = self._encode_row(student_input.to_feature_row())
        prediction = self.model.predict_one(row)
        probabilities = self.model.predict_proba_one(row)
        confidence = probabilities.get(prediction, 0.0)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def feature_importances(self):
        """Return normalized global feature importances in descending order.

        中文：重要性来自所有树节点的加权基尼不纯度下降，并转换为友好的显示名称。
        English: Importance comes from weighted Gini decrease across all tree nodes
        and is converted to user-friendly display names.
        """
        items = []
        for feature in FEATURES:
            encoded_feature = self._encoded_feature_name(feature)
            items.append((DISPLAY_FEATURES[feature], self.model.feature_importances_.get(encoded_feature, 0.0)))
        return sorted(items, key=lambda item: item[1], reverse=True)

    def local_feature_importances(self, student_input):
        """Estimate local influence through one-feature-at-a-time replacement.

        中文：先得到学生当前预测概率，再逐个把一个特征替换成训练集典型值。替换前后的
        预测类别概率差值作为局部影响。该方法是直观的扰动解释，不等同于 SHAP 因果解释。
        English: Starting from the student's predicted probability, one feature at a
        time is replaced with its training-set reference value. The probability
        difference is used as local influence. This is an intuitive perturbation
        explanation, not a causal or SHAP explanation.
        """
        feature_row = student_input.to_feature_row()
        encoded_row = self._encode_row(feature_row)
        prediction = self.model.predict_one(encoded_row)
        probabilities = self.model.predict_proba_one(encoded_row)
        baseline_probability = probabilities.get(prediction, 0.0)

        items = []
        for feature in FEATURES:
            # 中文：一次只改变一个特征，其他输入保持不变，以隔离该特征的局部影响。
            # English: Change one feature only, holding all others fixed to isolate its effect.
            encoded_feature = self._encoded_feature_name(feature)
            changed_row = dict(encoded_row)
            changed_row[encoded_feature] = self.reference_row.get(encoded_feature, encoded_row[encoded_feature])
            changed_probabilities = self.model.predict_proba_one(changed_row)
            changed_probability = changed_probabilities.get(prediction, 0.0)
            impact = baseline_probability - changed_probability
            items.append(
                {
                    "feature": DISPLAY_FEATURES[feature],
                    "current_value": feature_row[feature],
                    "reference_value": self._display_reference_value(feature),
                    "impact": impact,
                    "direction": "supports" if impact >= 0 else "reduces",
                }
            )

        items.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {
            "prediction": prediction,
            "confidence": baseline_probability,
            "probabilities": probabilities,
            "items": items,
        }

    def _encode_row(self, row):
        """Encode mixed feature values as numbers required by decision trees.

        中文：性别编码为 sex_M（M=1，F=0），活动编码为 activities_yes
        （yes=1，no=0），其余字段转换为浮点数。
        English: Sex becomes sex_M (M=1, F=0), activities becomes activities_yes
        (yes=1, no=0), and all remaining fields become floats.
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

    def _build_reference_row(self, rows):
        """Build the reference student used in local explanations.

        中文：二元分类特征使用众数，连续数值特征使用训练集均值。
        English: Binary categorical features use the mode; continuous numeric
        features use the training-set mean.
        """
        reference = {}
        for feature in self.model.features:
            values = [row[feature] for row in rows]
            if feature in {"sex_M", "activities_yes"}:
                reference[feature] = Counter(values).most_common(1)[0][0]
            else:
                reference[feature] = sum(values) / len(values)
        return reference

    def _display_reference_value(self, feature):
        """Decode a reference value for display / 将内部参考值还原为界面可读格式。"""
        encoded_feature = self._encoded_feature_name(feature)
        value = self.reference_row.get(encoded_feature, 0.0)
        if feature == "sex":
            return "M" if value >= 0.5 else "F"
        if feature == "activities":
            return "yes" if value >= 0.5 else "no"
        if feature in {"G1", "G2"}:
            return round(value * 5, 1)
        if feature == "age":
            return round(value, 1)
        return round(value, 2)

    @staticmethod
    def _encoded_feature_name(feature):
        """Map raw names to encoded model names / 将原始字段名映射为模型内部名称。"""
        if feature == "sex":
            return "sex_M"
        if feature == "activities":
            return "activities_yes"
        return feature
