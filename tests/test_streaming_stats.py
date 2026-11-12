"""Tests for streaming quantiles, histograms and per-column summaries."""

import doctest
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import stats
from numcompute_stream.stats import (
    StreamingHistogram,
    StreamingQuantile,
    StreamingSummary,
    histogram,
)


def _chunks(array, size):
    for start in range(0, len(array), size):
        yield array[start:start + size]


class TestStreamingQuantile(unittest.TestCase):
    def test_close_to_exact_on_large_stream(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20000)
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        sq = StreamingQuantile(levels)
        for chunk in _chunks(x, 500):
            sq.update(chunk)
        assert_allclose(sq.result(), np.quantile(x, levels), atol=0.03)
        self.assertEqual(sq.count, 20000)

    def test_skewed_distribution(self):
        x = np.random.default_rng(1).exponential(size=10000)
        sq = StreamingQuantile(0.5)
        for chunk in _chunks(x, 333):
            sq.update(chunk)
        assert_allclose(sq.result(), [np.median(x)], rtol=0.05)

    def test_exact_for_small_streams_and_extremes(self):
        sq = StreamingQuantile([0.0, 0.5, 1.0]).update([3, 1, 2])
        assert_array_equal(sq.result(), [1.0, 2.0, 3.0])
        sq.update([10, -5, 4, 4, 7])
        self.assertEqual(sq.result()[0], -5.0)
        self.assertEqual(sq.result()[2], 10.0)

    def test_constant_stream(self):
        sq = StreamingQuantile([0.25, 0.5]).update(np.full(100, 7.0))
        assert_array_equal(sq.result(), [7.0, 7.0])

    def test_nan_handling(self):
        sq = StreamingQuantile(0.5).update([1.0, np.nan, 3.0])
        self.assertEqual(sq.count, 2)
        self.assertTrue(np.isnan(StreamingQuantile().result()[0]))
        with self.assertRaises(ValueError):
            StreamingQuantile(nan_policy="raise").update([1.0, np.nan])

    def test_invalid_arguments(self):
        for q in (1.5, -0.1, [], [[0.5]]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                StreamingQuantile(q)
        with self.assertRaises(ValueError):
            StreamingQuantile(nan_policy="propagate")
        with self.assertRaises(ValueError):
            StreamingQuantile().update([[1.0, 2.0]])

if __name__ == "__main__":
    unittest.main()
