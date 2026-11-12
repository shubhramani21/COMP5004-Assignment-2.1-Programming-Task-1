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


class TestStreamingHistogram(unittest.TestCase):
    def test_matches_batch_histogram(self):
        x = np.random.default_rng(2).normal(size=1000)
        h = StreamingHistogram(bins=20, range=(-4, 4))
        for chunk in _chunks(x, 77):
            h.update(chunk)
        expected, edges = histogram(x, bins=20, range=(-4, 4))
        counts, got_edges = h.result()
        assert_array_equal(counts, expected)
        assert_array_equal(got_edges, edges)

    def test_out_of_range_values_tallied(self):
        h = StreamingHistogram(bins=2, range=(0, 2)).update([-1, 0.5, 1.5, 9, 9])
        assert_array_equal(h.counts_, [1, 1])
        self.assertEqual(h.n_below_, 1)
        self.assertEqual(h.n_above_, 2)

    def test_edges_learned_from_first_chunk(self):
        h = StreamingHistogram(bins=4)
        h.update([])  # empty chunk: edges still unknown
        self.assertIsNone(h.edges_)
        h.update([0.0, 4.0]).update([2.0, 10.0])
        assert_array_equal(h.edges_, [0, 1, 2, 3, 4])
        self.assertEqual(h.n_above_, 1)

    def test_sliding_window_drops_old_chunks(self):
        h = StreamingHistogram(bins=[0, 1, 2], window=2)
        h.update([0.5, 0.5]).update([1.5]).update([1.5, 1.5])
        assert_array_equal(h.counts_, [0, 3])
        h.reset()
        assert_array_equal(h.counts_, [0, 0])

    def test_quantile_from_bins(self):
        x = np.random.default_rng(3).uniform(0, 10, size=5000)
        h = StreamingHistogram(bins=100, range=(0, 10))
        for chunk in _chunks(x, 500):
            h.update(chunk)
        assert_allclose(h.quantile([0.0, 0.5, 1.0]),
                        np.quantile(x, [0.0, 0.5, 1.0]), atol=0.1)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            StreamingHistogram().result()
        with self.assertRaises(ValueError):
            StreamingHistogram(bins=0)
        with self.assertRaises(ValueError):
            StreamingHistogram(window=0)
        with self.assertRaises(ValueError):
            StreamingHistogram(bins=[0, 0, 1])
        with self.assertRaises(ValueError):
            StreamingHistogram(bins=2, range=(0, 1)).update([np.inf])

if __name__ == "__main__":
    unittest.main()
