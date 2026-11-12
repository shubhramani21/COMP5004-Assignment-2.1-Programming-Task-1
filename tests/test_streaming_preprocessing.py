"""Tests for streaming (partial_fit) preprocessing."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _random_with_nans(seed=0, shape=(300, 3)):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=[5, -2, 100], scale=[1, 3, 0.5], size=shape)
    X[rng.random(shape) < 0.1] = np.nan
    return X


def _stream(transformer, X, size):
    for start in range(0, X.shape[0], size):
        transformer.partial_fit(X[start:start + size])
    return transformer


class TestStreamingImputer(unittest.TestCase):
    def test_running_mean_matches_full_data(self):
        X = _random_with_nans()
        imputer = _stream(SimpleImputer(strategy="mean"), X, size=37)
        assert_allclose(imputer.statistics_, np.nanmean(X, axis=0))
        assert_array_equal(imputer.n_samples_seen_, np.sum(~np.isnan(X), axis=0))

    def test_fills_with_running_mean(self):
        imputer = SimpleImputer(strategy="mean")
        imputer.partial_fit([[1.0, 10.0]]).partial_fit([[3.0, np.nan]])
        assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[2.0, 10.0]])

    def test_unseen_column_uses_fill_value(self):
        imputer = SimpleImputer(strategy="mean", fill_value=-1)
        imputer.partial_fit([[np.nan, 4.0]])
        assert_array_equal(imputer.statistics_, [-1.0, 4.0])

    def test_infinity_ignored_by_mean(self):
        imputer = SimpleImputer(strategy="mean").fit([[np.inf], [2.0], [4.0]])
        assert_array_equal(imputer.statistics_, [3.0])

    def test_constant_strategy_partial_fit(self):
        imputer = SimpleImputer(fill_value=7).partial_fit([[np.nan, 1.0]])
        assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[7, 7]])

    def test_fit_resets_state(self):
        imputer = SimpleImputer(strategy="mean").partial_fit([[100.0]])
        imputer.fit([[2.0]])
        assert_array_equal(imputer.statistics_, [2.0])

    def test_chunk_errors_keep_state(self):
        imputer = SimpleImputer(strategy="mean").partial_fit([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            imputer.partial_fit([[1.0, 2.0, 3.0]])
        assert_array_equal(imputer.statistics_, [1.0, 2.0])
        imputer.partial_fit(np.empty((0, 2)))
        assert_array_equal(imputer.n_samples_seen_, [1, 1])

if __name__ == "__main__":
    unittest.main()
