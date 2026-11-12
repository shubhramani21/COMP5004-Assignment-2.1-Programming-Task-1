"""Tests for the streaming benchmark baselines and helpers."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from benchmark.run_stream_benchmarks import (
    bench_buffer_sizes,
    bench_split_search,
    iter_chunks,
    make_stream_data,
)
from benchmark.stream_baselines import (
    loop_confusion_matrix,
    loop_tree_predict,
    loop_welford,
)
from numcompute_stream.metrics import StreamingConfusionMatrix
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


class TestBaselinesAgree(unittest.TestCase):
    def test_loop_tree_predict(self):
        X, y = make_stream_data(500, seed=0)
        tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
        assert_array_equal(loop_tree_predict(tree, X), tree.predict(X))

    def test_loop_welford(self):
        X, _ = make_stream_data(350, seed=1)
        chunks = [X[i:i + 100] for i in range(0, 350, 100)]
        scaler = StandardScaler()
        for chunk in chunks:
            scaler.partial_fit(chunk)
        mean, std = loop_welford(chunks)
        assert_allclose(mean, scaler.mean_)
        assert_allclose(std, scaler.scale_)

    def test_loop_confusion_matrix(self):
        rng = np.random.default_rng(2)
        y_true, y_pred = rng.integers(0, 3, 250), rng.integers(0, 3, 250)
        chunks = list(iter_chunks(y_true, y_pred, 60))
        cm = StreamingConfusionMatrix(labels=[0, 1, 2])
        for t, p in chunks:
            cm.update(t, p)
        assert_array_equal(loop_confusion_matrix(chunks, 3), cm.matrix_)


class TestBenchmarkHelpers(unittest.TestCase):
    def test_stream_data_and_drift(self):
        X, y = make_stream_data(1000, n_features=4, noise=0.0, seed=3)
        self.assertEqual(X.shape, (1000, 4))
        self.assertEqual(set(np.unique(y)), {0, 1})
        _, drifted = make_stream_data(1000, n_features=4, drift_at=500,
                                      noise=0.0, seed=3)
        assert_array_equal(drifted[:500], y[:500])
        self.assertTrue(np.any(drifted[500:] != y[500:]))

    def test_iter_chunks_covers_everything(self):
        X, y = make_stream_data(250)
        chunks = list(iter_chunks(X, y, 100))
        self.assertEqual([len(c[1]) for c in chunks], [100, 100, 50])

    def test_split_search_rows(self):
        rows = bench_split_search([50], repeats=1)
        self.assertEqual(rows[0]["component"], "split_search")
        self.assertGreater(rows[0]["speedup"], 0)

    def test_buffer_size_rows(self):
        rows, curves = bench_buffer_sizes([100, 300], n_samples=600,
                                          chunk_size=100, seeds=[0],
                                          drift_at=300)
        self.assertEqual(len(rows), 4)  # 2 scenarios x 2 buffer sizes
        small = next(r for r in rows if r["scenario"] == "stationary"
                     and r["buffer_size"] == 100)
        large = next(r for r in rows if r["scenario"] == "stationary"
                     and r["buffer_size"] == 300)
        self.assertEqual(small["n_retrains"], 6)
        self.assertEqual(large["n_retrains"], 2)
        self.assertGreater(small["n_scored"], large["n_scored"])
        self.assertEqual(curves[("drift", 100)].shape, (6,))


if __name__ == "__main__":
    unittest.main()
