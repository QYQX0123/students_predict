"""A compact from-scratch Decision Tree and Random Forest implementation."""

import math
import random
from collections import Counter, defaultdict


class DecisionTreeClassifier:
    """CART-style classifier used as the base learner inside the Random Forest."""

    def __init__(self, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features=None, random_state=None):
        """Store tree hyperparameters and a seeded random generator."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root = None
        self.feature_importances_ = defaultdict(float)

    def fit(self, x_rows, y, features):
        """Build the tree recursively from feature dictionaries and labels."""
        self.features = list(features)
        self.classes_ = sorted(set(y))
        self.root = self._build(list(range(len(y))), x_rows, y, depth=0)
        return self

    def predict_one(self, row):
        """Traverse the fitted tree until a leaf prediction is reached."""
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            if value <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict_proba_one(self, row):
        """Return class probabilities from the training-label counts in the leaf."""
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        total = sum(node["counts"].values()) or 1
        return {klass: node["counts"].get(klass, 0) / total for klass in self.classes_}

    def _build(self, indexes, x_rows, y, depth):
        """Recursively create leaf nodes or split nodes for the current sample indexes."""
        counts = Counter(y[i] for i in indexes)
        prediction = counts.most_common(1)[0][0]

        # Stop when the node is pure, too small, or has reached the configured depth.
        if (
            depth >= self.max_depth
            or len(indexes) < self.min_samples_split
            or len(counts) == 1
        ):
            return {"prediction": prediction, "counts": counts}

        split = self._best_split(indexes, x_rows, y)
        if split is None:
            return {"prediction": prediction, "counts": counts}

        feature, threshold, left, right, gain = split
        self.feature_importances_[feature] += gain * len(indexes)
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(left, x_rows, y, depth + 1),
            "right": self._build(right, x_rows, y, depth + 1),
        }

    def _best_split(self, indexes, x_rows, y):
        """Search candidate feature thresholds and keep the highest Gini gain split."""
        parent_impurity = self._gini(indexes, y)
        best = None
        best_gain = 0.0

        features = self._candidate_features()
        for feature in features:
            values = sorted({x_rows[i][feature] for i in indexes})
            if len(values) < 2:
                continue

            # Midpoints between sorted unique values are the possible numeric thresholds.
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left = [i for i in indexes if x_rows[i][feature] <= threshold]
                right = [i for i in indexes if x_rows[i][feature] > threshold]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                weighted = (len(left) * self._gini(left, y) + len(right) * self._gini(right, y)) / len(indexes)
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold, left, right, gain)

        return best

    def _candidate_features(self):
        """Pick the random feature subset used at each split."""
        if self.max_features is None or self.max_features >= len(self.features):
            return self.features
        return self.random.sample(self.features, self.max_features)

    @staticmethod
    def _gini(indexes, y):
        """Calculate Gini impurity for a group of sample indexes."""
        counts = Counter(y[i] for i in indexes)
        total = len(indexes)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())


class RandomForestClassifier:
    """Ensemble classifier that averages many bootstrapped decision trees."""

    def __init__(
        self,
        n_estimators=80,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
    ):
        """Store forest hyperparameters and initialize fitted attributes."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.trees = []
        self.classes_ = []
        self.feature_importances_ = {}

    def fit(self, x_rows, y):
        """Train each tree on a bootstrap sample and aggregate importances."""
        self.features = list(x_rows[0].keys())
        self.classes_ = sorted(set(y))
        feature_count = self._max_feature_count()
        self.trees = []

        for tree_index in range(self.n_estimators):
            # Bootstrap sampling gives each tree a different view of the same dataset.
            sample_indexes = [self.random.randrange(len(x_rows)) for _ in range(len(x_rows))]
            sample_x = [x_rows[i] for i in sample_indexes]
            sample_y = [y[i] for i in sample_indexes]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=feature_count,
                random_state=self.random.randrange(1_000_000),
            )
            tree.fit(sample_x, sample_y, self.features)
            self.trees.append(tree)

        self._compute_importances()
        return self

    def predict_one(self, row):
        """Predict by majority vote across all trees."""
        votes = Counter(tree.predict_one(row) for tree in self.trees)
        return votes.most_common(1)[0][0]

    def predict_proba_one(self, row):
        """Approximate probabilities by counting tree votes per class."""
        totals = Counter()
        for tree in self.trees:
            totals[tree.predict_one(row)] += 1
        total = len(self.trees) or 1
        return {klass: totals.get(klass, 0) / total for klass in self.classes_}

    def score(self, x_rows, y):
        """Return accuracy for a labeled dataset."""
        if not y:
            return 0.0
        correct = sum(1 for row, target in zip(x_rows, y) if self.predict_one(row) == target)
        return correct / len(y)

    def _max_feature_count(self):
        """Translate max_features into the number of features sampled per split."""
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(len(self.features))))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, len(self.features)))
        return len(self.features)

    def _compute_importances(self):
        """Normalize impurity-gain totals collected from every tree."""
        totals = defaultdict(float)
        for tree in self.trees:
            for feature, value in tree.feature_importances_.items():
                totals[feature] += value
        total = sum(totals.values())
        if total <= 0:
            self.feature_importances_ = {feature: 0.0 for feature in self.features}
        else:
            self.feature_importances_ = {feature: totals.get(feature, 0.0) / total for feature in self.features}
