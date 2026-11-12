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

if __name__ == "__main__":
    unittest.main()
