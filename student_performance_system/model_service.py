import random
from collections import Counter

from .data_utils import DISPLAY_FEATURES, FEATURES, load_dataset
from .random_forest import RandomForestClassifier


class PredictionService:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.metrics = {}
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

    @staticmethod
    def _encoded_feature_name(feature):
        if feature == "sex":
            return "sex_M"
        if feature == "activities":
            return "activities_yes"
        return feature
