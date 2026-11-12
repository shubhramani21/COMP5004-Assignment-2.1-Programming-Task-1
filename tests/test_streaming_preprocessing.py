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


class TestStreamingStandardScaler(unittest.TestCase):
    def test_matches_full_fit(self):
        X = _random_with_nans(1)
        streamed = _stream(StandardScaler(), X, size=23)
        full = StandardScaler().fit(X)
        assert_allclose(streamed.mean_, full.mean_)
        assert_allclose(streamed.scale_, full.scale_)
        assert_allclose(streamed.mean_, np.nanmean(X, axis=0))
        assert_allclose(streamed.scale_, np.nanstd(X, axis=0))

    def test_chunk_size_does_not_matter(self):
        X = _random_with_nans(2)
        results = [_stream(StandardScaler(), X, size).mean_
                   for size in (1, 7, 50, 300)]
        for result in results[1:]:
            assert_allclose(result, results[0])

    def test_numerically_stable_with_large_offset(self):
        # Naive sum-of-squares variance loses all precision here.
        rng = np.random.default_rng(3)
        X = 1e9 + rng.normal(size=(1000, 1))
        scaler = _stream(StandardScaler(), X, size=100)
        assert_allclose(scaler.scale_, np.std(X, axis=0), rtol=1e-6)

    def test_zero_variance_chunks_and_columns(self):
        scaler = StandardScaler()
        scaler.partial_fit(np.full((5, 2), 3.0))
        assert_array_equal(scaler.scale_, [1.0, 1.0])
        assert_array_equal(scaler.transform([[3.0, 3.0]]), [[0.0, 0.0]])
        scaler.partial_fit([[5.0, 3.0]])
        self.assertGreater(scaler.scale_[0], 0)
        self.assertEqual(scaler.scale_[1], 1.0)

    def test_all_nan_chunk_column_keeps_state(self):
        scaler = StandardScaler().partial_fit([[1.0, 1.0], [3.0, 5.0]])
        scaler.partial_fit([[np.nan, 9.0]])
        assert_allclose(scaler.mean_, [2.0, 5.0])
        assert_array_equal(scaler.n_samples_seen_, [2, 3])

    def test_errors(self):
        scaler = StandardScaler().partial_fit([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            scaler.partial_fit([[np.inf, 1.0]])
        with self.assertRaises(ValueError):
            scaler.partial_fit([[1.0]])
        assert_array_equal(scaler.mean_, [1.0, 2.0])


class TestStreamingMinMaxScaler(unittest.TestCase):
    def test_matches_full_fit(self):
        X = _random_with_nans(4)
        streamed = _stream(MinMaxScaler(), X, size=40)
        full = MinMaxScaler().fit(X)
        assert_array_equal(streamed.data_min_, full.data_min_)
        assert_array_equal(streamed.data_max_, full.data_max_)
        assert_allclose(streamed.transform(X), full.transform(X))

    def test_unseen_column_passes_through(self):
        scaler = MinMaxScaler().partial_fit([[np.nan, 0.0], [np.nan, 4.0]])
        assert_allclose(scaler.transform([[2.0, 2.0]]), [[2.0, 0.5]])
        scaler.partial_fit([[10.0, 4.0], [20.0, 4.0]])
        assert_allclose(scaler.transform([[15.0, 2.0]]), [[0.5, 0.5]])

if __name__ == "__main__":
    unittest.main()
