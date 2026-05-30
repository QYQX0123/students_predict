import random
from collections import Counter

from .data_utils import DISPLAY_FEATURES, FEATURES, load_dataset
from .random_forest import RandomForestClassifier


class PredictionService:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.metrics = {}
        self.reference_row = {}
        self.train()

    def train(self):
        x_rows, y, raw_rows = load_dataset(self.dataset_path)
        x_rows = [self._encode_row(row) for row in x_rows]

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
        }
        self.raw_rows = raw_rows
        return self.metrics

    def predict(self, student_input):
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
        items = []
        for feature in FEATURES:
            encoded_feature = self._encoded_feature_name(feature)
            items.append((DISPLAY_FEATURES[feature], self.model.feature_importances_.get(encoded_feature, 0.0)))
        return sorted(items, key=lambda item: item[1], reverse=True)

    def local_feature_importances(self, student_input):
        feature_row = student_input.to_feature_row()
        encoded_row = self._encode_row(feature_row)
        prediction = self.model.predict_one(encoded_row)
        probabilities = self.model.predict_proba_one(encoded_row)
        baseline_probability = probabilities.get(prediction, 0.0)

        items = []
        for feature in FEATURES:
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
        reference = {}
        for feature in self.model.features:
            values = [row[feature] for row in rows]
            if feature in {"sex_M", "activities_yes"}:
                reference[feature] = Counter(values).most_common(1)[0][0]
            else:
                reference[feature] = sum(values) / len(values)
        return reference

    def _display_reference_value(self, feature):
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
        if feature == "sex":
            return "sex_M"
        if feature == "activities":
            return "activities_yes"
        return feature
