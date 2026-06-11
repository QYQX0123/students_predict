"""A compact Decision Tree and Random Forest implementation built from scratch.

中文：本模块不依赖 scikit-learn，使用基尼不纯度、递归二叉切分、Bootstrap 抽样和
随机特征子集实现分类随机森林，便于展示算法内部原理。
English: This module avoids scikit-learn and implements a classification forest
with Gini impurity, recursive binary splits, bootstrap sampling, and random
feature subsets so the algorithm remains inspectable.
"""

import math
import random
from collections import Counter, defaultdict


class DecisionTreeClassifier:
    """CART-style classification tree used as the forest's base learner.

    中文：每个内部节点保存特征和阈值，叶节点保存多数类别及类别计数。
    English: Each internal node stores a feature and threshold; each leaf stores
    the majority class and class counts.
    """

    def __init__(self, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features=None, random_state=None):
        """Store stopping rules, feature sampling settings, and random state.

        中文：max_depth 限制深度；min_samples_split 控制是否允许继续分裂；
        min_samples_leaf 防止产生样本过少的叶节点；max_features 控制每次候选特征数。
        English: max_depth limits depth, min_samples_split controls whether a node
        may split, min_samples_leaf prevents tiny leaves, and max_features controls
        how many features are considered at each node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root = None
        self.feature_importances_ = defaultdict(float)

    def fit(self, x_rows, y, features):
        """Fit the tree recursively and return self / 递归训练决策树并返回自身。"""
        self.features = list(features)
        self.classes_ = sorted(set(y))
        self.root = self._build(list(range(len(y))), x_rows, y, depth=0)
        return self

    def predict_one(self, row):
        """Follow threshold branches to a leaf / 根据阈值分支遍历到叶节点。"""
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            if value <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict_proba_one(self, row):
        """Estimate probabilities from label proportions inside the reached leaf.

        中文：叶节点内某类别样本数除以叶节点总样本数，得到该树的概率估计。
        English: A class count divided by total samples in the reached leaf becomes
        that tree's probability estimate.
        """
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        total = sum(node["counts"].values()) or 1
        return {klass: node["counts"].get(klass, 0) / total for klass in self.classes_}

    def _build(self, indexes, x_rows, y, depth):
        """Recursively construct a leaf or split node for the current samples."""
        counts = Counter(y[i] for i in indexes)
        prediction = counts.most_common(1)[0][0]

        # 中文：达到最大深度、样本不足或类别完全纯净时停止生长并生成叶节点。
        # English: Stop at maximum depth, insufficient samples, or a pure class node.
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
        """Find the feature and threshold with the largest Gini impurity reduction.

        中文：枚举随机候选特征及相邻唯一值的中点，过滤过小叶节点，然后选择
        parent_impurity - weighted_child_impurity 最大的切分。
        English: Enumerates random candidate features and midpoints between adjacent
        unique values, rejects undersized leaves, and maximizes parent impurity minus
        weighted child impurity.
        """
        parent_impurity = self._gini(indexes, y)
        best = None
        best_gain = 0.0

        features = self._candidate_features()
        for feature in features:
            values = sorted({x_rows[i][feature] for i in indexes})
            if len(values) < 2:
                continue

            # 中文：相邻唯一值的中点足以覆盖所有可能产生不同分组的数值切分。
            # English: Adjacent-value midpoints cover every distinct numeric partition.
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
        """Select the random feature subset considered at one split / 为当前节点抽取候选特征。"""
        if self.max_features is None or self.max_features >= len(self.features):
            return self.features
        return self.random.sample(self.features, self.max_features)

    @staticmethod
    def _gini(indexes, y):
        """Calculate Gini impurity, where 0 means a pure node / 计算基尼不纯度，0 表示纯节点。"""
        counts = Counter(y[i] for i in indexes)
        total = len(indexes)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())


class RandomForestClassifier:
    """Ensemble classifier based on bootstrapped randomized decision trees.

    中文：每棵树看到不同的有放回样本，并在每个节点只考虑部分特征，借此降低树之间
    的相关性；最终通过多数投票提升泛化能力。
    English: Every tree sees a different sample drawn with replacement and considers
    only a subset of features per node, reducing correlation between trees. Majority
    voting then improves generalization.
    """

    def __init__(
        self,
        n_estimators=80,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
    ):
        """Store forest hyperparameters and initialize fitted state / 保存参数并初始化训练状态。"""
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
        """Train all trees on bootstrap samples and aggregate feature importance."""
        self.features = list(x_rows[0].keys())
        self.classes_ = sorted(set(y))
        feature_count = self._max_feature_count()
        self.trees = []

        for tree_index in range(self.n_estimators):
            # 中文：有放回抽取与原数据等量的样本，同一行可能重复，也可能完全未被抽到。
            # English: Draw an equal-sized sample with replacement; rows may repeat or be omitted.
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
        """Return the class receiving the most tree votes / 返回所有树多数投票的类别。"""
        votes = Counter(tree.predict_one(row) for tree in self.trees)
        return votes.most_common(1)[0][0]

    def predict_proba_one(self, row):
        """Approximate class probabilities using tree-vote proportions.

        中文：这里统计每棵树的最终类别投票，而不是平均单棵树叶节点概率。
        English: This counts each tree's final class vote rather than averaging its
        leaf-level probabilities.
        """
        totals = Counter()
        for tree in self.trees:
            totals[tree.predict_one(row)] += 1
        total = len(self.trees) or 1
        return {klass: totals.get(klass, 0) / total for klass in self.classes_}

    def score(self, x_rows, y):
        """Return classification accuracy / 返回分类准确率。"""
        if not y:
            return 0.0
        correct = sum(1 for row, target in zip(x_rows, y) if self.predict_one(row) == target)
        return correct / len(y)

    def _max_feature_count(self):
        """Resolve max_features to a concrete count / 将特征抽样配置转换为具体数量。"""
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(len(self.features))))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, len(self.features)))
        return len(self.features)

    def _compute_importances(self):
        """Aggregate and normalize weighted impurity reductions from all trees.

        中文：所有特征重要性归一化后总和为 1；若没有产生有效切分，则全部设为 0。
        English: Normalized importances sum to 1. If no valid split was produced,
        every feature receives 0.
        """
        totals = defaultdict(float)
        for tree in self.trees:
            for feature, value in tree.feature_importances_.items():
                totals[feature] += value
        total = sum(totals.values())
        if total <= 0:
            self.feature_importances_ = {feature: 0.0 for feature in self.features}
        else:
            self.feature_importances_ = {feature: totals.get(feature, 0.0) / total for feature in self.features}
