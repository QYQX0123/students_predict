"""Custom decision-tree and random-forest classifiers.

中文：这个模块从零实现分类树和随机森林，功能包括基尼不纯度切分、递归建树、
Bootstrap 抽样、随机特征子集、多数投票和特征重要性统计。项目选择自定义实现，是
为了让算法逻辑可阅读、可解释，并避免依赖外部机器学习库。

English: This module implements classification trees and a random forest from
scratch: Gini splits, recursive tree building, bootstrap sampling, random feature
subsets, majority voting, and feature-importance aggregation. A custom
implementation keeps the algorithm readable and explainable without requiring an
external ML library.
"""

import math
import random
from collections import Counter, defaultdict


class DecisionTreeClassifier:
    """One CART-style classification tree used inside the forest.

    中文：树的内部节点保存“选择哪个特征、用哪个阈值切分、左右子树是什么”；叶节点
    保存当前样本中的多数类别和类别计数。这样写的数据结构简单，预测时只需要沿阈值
    一路走到叶子。

    English: Internal nodes store the selected feature, threshold, and left/right
    child nodes; leaves store the majority class and class counts. This structure is
    simple and makes prediction a threshold walk from root to leaf.
    """

    def __init__(self, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features=None, random_state=None):
        """Configure tree growth and randomness.

        中文：max_depth 限制树过深，min_samples_split 决定节点是否还能继续分裂，
        min_samples_leaf 避免产生过小叶子，max_features 控制每个节点考虑多少特征。
        这些限制共同减少过拟合，并让森林中不同树有差异。

        English: max_depth prevents overly deep trees, min_samples_split controls
        whether a node may split, min_samples_leaf avoids tiny leaves, and
        max_features controls how many features each node considers. Together these
        limits reduce overfitting and diversify trees in the forest.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root = None
        self.feature_importances_ = defaultdict(float)

    def fit(self, x_rows, y, features):
        """Train the tree from feature dictionaries and labels.

        中文：保存特征名和类别列表后，从包含所有样本的根节点开始递归构建。返回 self
        是常见模型 API 习惯，方便链式调用或测试。

        English: Saves feature names and classes, then recursively builds from a
        root node containing all samples. Returning self follows common model-API
        style and helps chaining or testing.
        """
        self.features = list(features)
        self.classes_ = sorted(set(y))
        self.root = self._build(list(range(len(y))), x_rows, y, depth=0)
        return self

    def predict_one(self, row):
        """Predict one row by walking from the root to a leaf.

        中文：每个内部节点只做一次“是否小于等于阈值”的判断，因此预测过程快速且容易
        理解。

        English: Each internal node performs only a "less than or equal threshold"
        check, making prediction fast and easy to follow.
        """
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            if value <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict_proba_one(self, row):
        """Estimate class probabilities from the reached leaf.

        中文：叶节点保存训练时落入该叶子的类别计数，所以概率可以用“该类别计数 / 叶子
        总计数”得到。这样实现简单，也符合决策树常见的概率估计方式。

        English: A leaf stores counts of training labels that reached it, so a class
        probability is class count divided by total leaf count. This is simple and
        matches a common decision-tree probability estimate.
        """
        node = self.root
        while "prediction" not in node:
            value = row[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        total = sum(node["counts"].values()) or 1
        return {klass: node["counts"].get(klass, 0) / total for klass in self.classes_}

    def _build(self, indexes, x_rows, y, depth):
        """Recursively choose whether to stop or split the current node.

        中文：函数接收当前节点包含的样本下标，先计算多数类作为兜底预测，再判断是否达到
        停止条件；如果还能分裂，就寻找最佳切分并递归构造左右子树。

        English: Receives the sample indexes for the current node, computes the
        majority class as a fallback prediction, checks stopping rules, then finds
        the best split and recursively builds children when splitting is allowed.
        """
        counts = Counter(y[i] for i in indexes)
        prediction = counts.most_common(1)[0][0]

        # 中文：深度、样本量和纯度是三个停止条件，用来防止无限递归和过拟合。
        # English: Depth, sample count, and purity stop recursion and reduce overfitting.
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
        """Search all valid candidate thresholds and keep the best Gini gain.

        中文：对候选特征取唯一值排序，再用相邻值中点作为阈值。只测试这些中点就能覆盖
        所有会产生不同左右分组的切分；然后选择加权子节点不纯度下降最多的方案。

        English: For each candidate feature, unique values are sorted and adjacent
        midpoints are used as thresholds. Those midpoints cover every split that
        changes the left/right partition, and the split with the largest weighted
        impurity reduction is selected.
        """
        parent_impurity = self._gini(indexes, y)
        best = None
        best_gain = 0.0

        features = self._candidate_features()
        for feature in features:
            values = sorted({x_rows[i][feature] for i in indexes})
            if len(values) < 2:
                continue

            # 中文：中点阈值既不遗漏有效切分，也避免测试大量等价阈值。
            # English: Midpoint thresholds avoid redundant equivalent split tests.
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
        """Return the feature subset considered by one split.

        中文：随机森林的关键之一是每个节点只看部分特征，这会降低不同树之间的相似度；
        如果 max_features 覆盖所有特征，则退化为普通决策树的切分搜索。

        English: A random forest gains diversity by letting each node consider only
        some features. If max_features covers all features, this becomes ordinary
        decision-tree split search.
        """
        if self.max_features is None or self.max_features >= len(self.features):
            return self.features
        return self.random.sample(self.features, self.max_features)

    @staticmethod
    def _gini(indexes, y):
        """Calculate Gini impurity for a group of samples.

        中文：基尼不纯度越低，节点内类别越集中；0 表示节点内只有一个类别。使用它是
        因为计算简单，适合在大量候选切分中反复评估。

        English: Lower Gini impurity means labels are more concentrated; 0 means the
        node contains only one class. It is cheap to compute, which suits repeated
        split evaluation.
        """
        counts = Counter(y[i] for i in indexes)
        total = len(indexes)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())


class RandomForestClassifier:
    """A forest of randomized decision trees combined by voting.

    中文：随机森林实现两个核心功能：每棵树用有放回抽样得到不同训练集，每个节点只
    考虑随机特征子集。这样写是为了让单棵树的错误不完全相同，最后通过多数投票降低
    方差并提升泛化能力。

    English: The forest uses two ideas: each tree trains on a bootstrap sample, and
    each node considers a random feature subset. This makes tree errors less
    correlated, so majority voting reduces variance and improves generalization.
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
        """Store forest hyperparameters and initialize trained attributes.

        中文：构造函数只保存配置，不做训练；fit 才会根据数据生成特征名、类别列表和树。

        English: The constructor stores configuration only. fit later derives
        feature names, class labels, and trained trees from data.
        """
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
        """Train every tree on its own bootstrap sample.

        中文：每棵树看到与原数据同样数量、但有重复和遗漏的样本；再给每棵树不同随机
        种子，进一步增加差异。训练完后汇总所有树的特征重要性。

        English: Each tree sees the same number of rows as the original data, but
        with repeats and omissions. A different seed per tree increases diversity,
        and feature importance is aggregated after all trees are fitted.
        """
        self.features = list(x_rows[0].keys())
        self.classes_ = sorted(set(y))
        feature_count = self._max_feature_count()
        self.trees = []

        for tree_index in range(self.n_estimators):
            # 中文：Bootstrap 抽样让每棵树学习到略有不同的数据分布。
            # English: Bootstrap sampling gives every tree a slightly different data view.
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
        """Predict one row by majority vote.

        中文：每棵树给出一个类别，出现次数最多的类别作为森林输出。这样比依赖单棵树
        更稳健。

        English: Each tree returns one class, and the most common class becomes the
        forest output. This is more stable than relying on a single tree.
        """
        votes = Counter(tree.predict_one(row) for tree in self.trees)
        return votes.most_common(1)[0][0]

    def predict_proba_one(self, row):
        """Estimate probabilities from forest vote shares.

        中文：这里把投给某类别的树数量除以树总数作为概率。项目需要的是清晰可解释的
        通过概率，所以使用投票比例而不是更复杂的概率校准。

        English: The probability is the number of trees voting for a class divided
        by the number of trees. The app needs an explainable pass probability, so
        vote share is used instead of more complex calibration.
        """
        totals = Counter()
        for tree in self.trees:
            totals[tree.predict_one(row)] += 1
        total = len(self.trees) or 1
        return {klass: totals.get(klass, 0) / total for klass in self.classes_}

    def score(self, x_rows, y):
        """Return accuracy on labeled rows.

        中文：用于快速评估模型，空标签列表时返回 0，避免除以零。

        English: Used for quick model evaluation. An empty label list returns 0 to
        avoid division by zero.
        """
        if not y:
            return 0.0
        correct = sum(1 for row, target in zip(x_rows, y) if self.predict_one(row) == target)
        return correct / len(y)

    def _max_feature_count(self):
        """Convert max_features configuration into an integer.

        中文：支持 "sqrt" 和整数两种配置；"sqrt" 是随机森林常用默认策略，可在特征数
        不多时仍保留随机性。

        English: Supports "sqrt" and integer configuration. "sqrt" is a common
        random-forest default that keeps randomness even with a modest feature count.
        """
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(len(self.features))))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, len(self.features)))
        return len(self.features)

    def _compute_importances(self):
        """Aggregate tree-level split gains into normalized feature importance.

        中文：每棵树在切分时记录“基尼下降 × 当前节点样本数”。森林把这些值累加后归一化，
        得到总和为 1 的全局重要性；如果没有有效切分，则所有特征重要性为 0。

        English: Each tree records Gini gain multiplied by node sample count. The
        forest sums and normalizes those values into importances that add to 1. If
        no valid split exists, every feature receives 0.
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


class DecisionTreeRegressor:
    """One CART-style regression tree used for predicted G3 scores."""

    def __init__(self, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root = None

    def fit(self, x_rows, y, features):
        self.features = list(features)
        self.root = self._build(list(range(len(y))), x_rows, y, depth=0)
        return self

    def predict_one(self, row):
        node = self.root
        while "value" not in node:
            value = row[node["feature"]]
            node = node["left"] if value <= node["threshold"] else node["right"]
        return node["value"]

    def _build(self, indexes, x_rows, y, depth):
        value = sum(y[i] for i in indexes) / len(indexes)
        if (
            depth >= self.max_depth
            or len(indexes) < self.min_samples_split
            or self._variance(indexes, y) <= 0
        ):
            return {"value": value}

        split = self._best_split(indexes, x_rows, y)
        if split is None:
            return {"value": value}

        feature, threshold, left, right = split
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(left, x_rows, y, depth + 1),
            "right": self._build(right, x_rows, y, depth + 1),
        }

    def _best_split(self, indexes, x_rows, y):
        parent_error = self._variance(indexes, y) * len(indexes)
        best = None
        best_gain = 0.0

        for feature in self._candidate_features():
            values = sorted({x_rows[i][feature] for i in indexes})
            if len(values) < 2:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left = [i for i in indexes if x_rows[i][feature] <= threshold]
                right = [i for i in indexes if x_rows[i][feature] > threshold]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                child_error = self._variance(left, y) * len(left) + self._variance(right, y) * len(right)
                gain = parent_error - child_error
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold, left, right)
        return best

    def _candidate_features(self):
        if self.max_features is None or self.max_features >= len(self.features):
            return self.features
        return self.random.sample(self.features, self.max_features)

    @staticmethod
    def _variance(indexes, y):
        if not indexes:
            return 0.0
        mean = sum(y[i] for i in indexes) / len(indexes)
        return sum((y[i] - mean) ** 2 for i in indexes) / len(indexes)


class RandomForestRegressor:
    """A small random forest regressor used to estimate final G3 grade."""

    def __init__(
        self,
        n_estimators=80,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.trees = []

    def fit(self, x_rows, y):
        self.features = list(x_rows[0].keys())
        feature_count = self._max_feature_count()
        self.trees = []
        for _tree_index in range(self.n_estimators):
            sample_indexes = [self.random.randrange(len(x_rows)) for _ in range(len(x_rows))]
            sample_x = [x_rows[i] for i in sample_indexes]
            sample_y = [y[i] for i in sample_indexes]
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=feature_count,
                random_state=self.random.randrange(1_000_000),
            )
            tree.fit(sample_x, sample_y, self.features)
            self.trees.append(tree)
        return self

    def predict_one(self, row):
        if not self.trees:
            return 0.0
        return sum(tree.predict_one(row) for tree in self.trees) / len(self.trees)

    def _max_feature_count(self):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(len(self.features))))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, len(self.features)))
        return len(self.features)
