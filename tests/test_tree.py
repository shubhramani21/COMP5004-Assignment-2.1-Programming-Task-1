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

if __name__ == "__main__":
    unittest.main()
