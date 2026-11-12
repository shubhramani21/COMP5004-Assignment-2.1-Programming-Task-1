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


class TestFitPredict(unittest.TestCase):
    def test_separable_data_fits_perfectly(self):
        X, y = _separable_data()
        tree = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(tree.score(X, y), 1.0)
        self.assertEqual(tree.n_retrains_, 1)

    def test_xor_needs_zero_gain_split(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.0]])
        y = np.array([0, 1, 1, 0])
        tree = DecisionTreeClassifier().fit(X, y)
        assert_array_equal(tree.predict(X), y)

    def test_max_depth_limits_tree(self):
        X, y = _separable_data()
        for depth in (1, 2, 3):
            tree = DecisionTreeClassifier(max_depth=depth).fit(X, y)
            self.assertLessEqual(tree.get_depth(), depth)

    def test_min_samples_split_stops_growth(self):
        X, y = _separable_data(20)
        tree = DecisionTreeClassifier(min_samples_split=50).fit(X, y)
        self.assertEqual(tree.get_n_leaves(), 1)

    def test_string_labels_and_entropy(self):
        X, y = _separable_data()
        labels = np.where(y == 1, "yes", "no")
        tree = DecisionTreeClassifier(criterion="entropy").fit(X, labels)
        assert_array_equal(tree.classes_, ["no", "yes"])
        self.assertEqual(tree.score(X, labels), 1.0)

    def test_predict_proba_rows_sum_to_one(self):
        X, y = _separable_data()
        proba = DecisionTreeClassifier(max_depth=2).fit(X, y).predict_proba(X)
        self.assertEqual(proba.shape, (300, 2))
        assert_allclose(proba.sum(axis=1), 1.0)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_single_class_gives_single_leaf(self):
        X, _ = _separable_data(30)
        tree = DecisionTreeClassifier().fit(X, np.full(30, 7))
        self.assertEqual(tree.get_n_leaves(), 1)
        assert_array_equal(tree.predict(X[:3]), [7, 7, 7])

    def test_zero_variance_features_give_single_leaf(self):
        tree = DecisionTreeClassifier().fit(np.ones((10, 3)),
                                            [0, 1] * 5)
        self.assertEqual(tree.get_n_leaves(), 1)
        self.assertEqual(tree.get_depth(), 0)

    def test_tie_goes_to_first_class(self):
        tree = DecisionTreeClassifier().fit(np.ones((4, 1)), [1, 0, 1, 0])
        assert_array_equal(tree.predict([[1.0]]), [0])

    def test_max_features_reproducible(self):
        X, y = _separable_data()
        a = DecisionTreeClassifier(max_features="sqrt", random_state=3)
        b = DecisionTreeClassifier(max_features="sqrt", random_state=3)
        assert_array_equal(a.fit(X, y).predict(X), b.fit(X, y).predict(X))

    def test_predict_empty_input(self):
        X, y = _separable_data()
        tree = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(tree.predict(np.empty((0, 4))).shape, (0,))


class TestStreaming(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _separable_data()

    def test_predict_before_first_retrain_raises(self):
        tree = DecisionTreeClassifier(buffer_size=100)
        tree.partial_fit(self.X[:30], self.y[:30])
        with self.assertRaisesRegex(RuntimeError, "not trained yet"):
            tree.predict(self.X)

    def test_retrain_schedule_and_buffer_clearing(self):
        tree = DecisionTreeClassifier(buffer_size=100)
        counts, retrains = [], []
        for start in range(0, 300, 30):
            tree.partial_fit(self.X[start:start + 30],
                             self.y[start:start + 30])
            counts.append(tree.buffer_count_)
            retrains.append(tree.n_retrains_)
        self.assertEqual(counts, [30, 60, 90, 0, 30, 60, 90, 0, 30, 60])
        self.assertEqual(retrains, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(tree.last_train_size_, 120)

    def test_predictions_fixed_between_retrains(self):
        tree = DecisionTreeClassifier(buffer_size=100)
        tree.partial_fit(self.X[:100], self.y[:100])
        before = tree.predict(self.X)
        tree.partial_fit(self.X[100:150], 1 - self.y[100:150])
        assert_array_equal(tree.predict(self.X), before)

    def test_large_chunk_triggers_immediate_retrain(self):
        tree = DecisionTreeClassifier(buffer_size=50)
        tree.partial_fit(self.X, self.y)
        self.assertEqual(tree.n_retrains_, 1)
        self.assertEqual(tree.buffer_count_, 0)
        self.assertEqual(tree.last_train_size_, 300)

    def test_empty_chunk_is_ignored(self):
        tree = DecisionTreeClassifier(buffer_size=10)
        self.assertIs(tree.partial_fit(np.empty((0, 4)), []), tree)
        self.assertEqual(tree.buffer_count_, 0)

    def test_chunk_feature_mismatch(self):
        tree = DecisionTreeClassifier(buffer_size=100)
        tree.partial_fit(self.X[:10], self.y[:10])
        with self.assertRaises(ValueError):
            tree.partial_fit(self.X[:10, :3], self.y[:10])

    def test_new_class_in_later_buffer(self):
        tree = DecisionTreeClassifier(buffer_size=20)
        tree.partial_fit(self.X[:20], np.zeros(20, int))
        assert_array_equal(tree.classes_, [0])
        tree.partial_fit(self.X[20:40], np.arange(20) % 3)
        assert_array_equal(tree.classes_, [0, 1, 2])

    def test_fit_clears_pending_buffer(self):
        tree = DecisionTreeClassifier(buffer_size=100)
        tree.partial_fit(self.X[:40], self.y[:40])
        tree.fit(self.X, self.y)
        self.assertEqual(tree.buffer_count_, 0)

    def test_stream_accuracy_reasonable(self):
        tree = DecisionTreeClassifier(max_depth=5, buffer_size=150)
        for start in range(0, 300, 25):
            tree.partial_fit(self.X[start:start + 25],
                             self.y[start:start + 25])
        X_test, y_test = _separable_data(200, seed=1)
        self.assertGreater(tree.score(X_test, y_test), 0.85)


if __name__ == "__main__":
    unittest.main()
