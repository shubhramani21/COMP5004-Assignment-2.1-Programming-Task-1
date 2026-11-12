"""Tests for the streaming classification metrics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.metrics import (
    RollingWindowMetric,
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
)


def _chunks(*arrays, size):
    """Yield aligned chunks of the given arrays."""
    for start in range(0, len(arrays[0]), size):
        yield tuple(a[start:start + size] for a in arrays)


def _brute_force_auc(y_true, scores):
    """AUC by comparing every positive with every negative."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (wins + 0.5 * ties) / (pos.size * neg.size)


class TestStreamingConfusionMatrix(unittest.TestCase):
    def test_accumulates_over_chunks(self):
        cm = StreamingConfusionMatrix()
        cm.update([0, 1], [0, 0]).update([1, 1], [1, 0])
        matrix, labels = cm.result()
        assert_array_equal(labels, [0, 1])
        assert_array_equal(matrix, [[1, 0], [2, 1]])
        self.assertEqual(cm.n_samples_, 4)

    def test_grows_when_new_label_appears(self):
        cm = StreamingConfusionMatrix()
        cm.update([0, 0], [0, 0])
        self.assertEqual(cm.matrix_.shape, (1, 1))
        cm.update([2, 1], [1, 2])
        matrix, labels = cm.result()
        assert_array_equal(labels, [0, 1, 2])
        assert_array_equal(matrix, [[2, 0, 0], [0, 0, 1], [0, 1, 0]])

    def test_matches_batch_confusion_matrix(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, 500)
        y_pred = rng.integers(0, 4, 500)
        cm = StreamingConfusionMatrix(labels=[0, 1, 2, 3])
        for t, p in _chunks(y_true, y_pred, size=37):
            cm.update(t, p)
        expected, _ = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        assert_array_equal(cm.result()[0], expected)

    def test_string_labels(self):
        cm = StreamingConfusionMatrix()
        cm.update(["cat", "dog"], ["dog", "dog"])
        matrix, labels = cm.result()
        assert_array_equal(labels, ["cat", "dog"])
        assert_array_equal(matrix, [[0, 1], [0, 1]])

    def test_fixed_labels_reject_unknown(self):
        cm = StreamingConfusionMatrix(labels=[0, 1])
        with self.assertRaises(ValueError):
            cm.update([0, 2], [0, 1])

    def test_reset_and_empty_chunk(self):
        cm = StreamingConfusionMatrix()
        cm.update([0, 1], [1, 1]).update([], [])
        self.assertEqual(cm.n_samples_, 2)
        cm.reset()
        self.assertEqual(cm.n_samples_, 0)
        with self.assertRaises(RuntimeError):
            cm.result()

if __name__ == "__main__":
    unittest.main()
