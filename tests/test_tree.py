"""Tests for the streaming decision tree classifier."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.tree import (
    DecisionTreeClassifier,
    _best_split_feature,
    _best_split_feature_loop,
    _entropy,
    _gini,
    _midpoint,
)


def _separable_data(n=300, seed=0):
    """Two classes split by the line x0 + x1 = 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestImpurity(unittest.TestCase):
    def test_gini_known_values(self):
        self.assertEqual(_gini([3, 0]), 0.0)
        self.assertEqual(_gini([2, 2]), 0.5)
        assert_allclose(_gini([3, 2]), 0.48)

    def test_entropy_known_values(self):
        self.assertEqual(_entropy([3, 0]), 0.0)
        self.assertEqual(_entropy([2, 2]), 1.0)
        assert_allclose(_entropy([1, 1, 1, 1]), 2.0)

    def test_empty_node_is_zero_without_warnings(self):
        with np.errstate(all="raise"):
            self.assertEqual(_gini([0, 0]), 0.0)
            self.assertEqual(_entropy([0, 0]), 0.0)

    def test_vectorised_over_rows(self):
        assert_allclose(_gini(np.array([[1, 1], [4, 0], [0, 0]])),
                        [0.5, 0.0, 0.0])
        assert_allclose(_entropy(np.array([[1, 1], [4, 0]])), [1.0, 0.0])


class TestSplitSearch(unittest.TestCase):
    def test_worked_example(self):
        x = np.array([2, 7, 3, 9, 5.0])
        y = np.array([0, 1, 0, 1, 0])
        self.assertEqual(_best_split_feature(x, y, 2, _gini), (0.0, 6.0))
        self.assertEqual(_best_split_feature_loop(x, y, 2, _gini), (0.0, 6.0))

    def test_zero_variance_feature_has_no_split(self):
        y = np.array([0, 1, 0, 1])
        self.assertIsNone(_best_split_feature(np.ones(4), y, 2, _gini))
        self.assertIsNone(_best_split_feature_loop(np.ones(4), y, 2, _gini))

    def test_single_sample_has_no_split(self):
        self.assertIsNone(_best_split_feature(np.array([1.0]),
                                              np.array([0]), 1, _gini))

    def test_loop_and_vectorised_agree(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = int(rng.integers(1, 40))
            x = rng.integers(0, 6, n).astype(float)  # many ties
            y = rng.integers(0, 3, n)
            for impurity in (_gini, _entropy):
                for leaf in (1, 2, 3):
                    fast = _best_split_feature(x, y, 3, impurity, leaf)
                    slow = _best_split_feature_loop(x, y, 3, impurity, leaf)
                    if fast is None or slow is None:
                        self.assertIsNone(fast)
                        self.assertIsNone(slow)
                    else:
                        assert_allclose(fast[0], slow[0])
                        self.assertEqual(fast[1], slow[1])

    def test_min_samples_leaf_respected(self):
        x = np.arange(6.0)
        y = np.array([1, 0, 0, 0, 0, 0])
        # Best split isolates sample 0, but each leaf needs 2 samples.
        _, threshold = _best_split_feature(x, y, 2, _gini, min_samples_leaf=2)
        self.assertGreaterEqual(np.sum(x <= threshold), 2)
        self.assertGreaterEqual(np.sum(x > threshold), 2)

    def test_midpoint_separates_adjacent_floats(self):
        lower = 1.0
        upper = np.nextafter(1.0, 2.0)
        threshold = _midpoint(lower, upper)
        self.assertTrue(lower <= threshold < upper)
        self.assertEqual(_midpoint(5.0, 7.0), 6.0)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _separable_data(50)

    def test_invalid_hyperparameters(self):
        bad = [
            dict(criterion="foo"),
            dict(buffer_size=0),
            dict(max_depth=0),
            dict(min_samples_split=1),
            dict(min_samples_leaf=0),
            dict(max_features=1.5),
            dict(max_features="half"),
            dict(max_features=10),
        ]
        for params in bad:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier(**params).fit(self.X, self.y)

    def test_wrong_hyperparameter_types(self):
        for params in (dict(max_depth=2.5), dict(max_features=True),
                       dict(criterion=1), dict(buffer_size="10")):
            with self.subTest(params=params):
                with self.assertRaises(TypeError):
                    DecisionTreeClassifier(**params).fit(self.X, self.y)

    def test_nan_and_inf_rejected(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(X, self.y)
        X[0, 0] = np.inf
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(X, self.y)

    def test_shape_errors(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.fit(self.X, self.y[:-1])  # length mismatch
        with self.assertRaises(ValueError):
            tree.fit(self.X[:, 0], self.y)  # 1D X
        with self.assertRaises(ValueError):
            tree.fit(np.empty((0, 4)), [])  # no samples
        with self.assertRaises(ValueError):
            tree.fit(self.X, np.c_[self.y, self.y])  # 2D y

    def test_column_vector_y_accepted(self):
        tree = DecisionTreeClassifier().fit(self.X, self.y.reshape(-1, 1))
        self.assertEqual(tree.predict(self.X).shape, (50,))

    def test_nan_label_rejected(self):
        y = self.y.astype(float)
        y[3] = np.nan
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(self.X, y)

    def test_predict_feature_mismatch(self):
        tree = DecisionTreeClassifier().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            tree.predict(self.X[:, :2])

if __name__ == "__main__":
    unittest.main()
