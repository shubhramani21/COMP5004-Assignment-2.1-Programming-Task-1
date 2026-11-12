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


class TestStreamingScores(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y_true = rng.integers(0, 3, 400)
        self.y_pred = np.where(rng.random(400) < 0.7, self.y_true,
                               rng.integers(0, 3, 400))

    def _stream(self, metric, size=33):
        for t, p in _chunks(self.y_true, self.y_pred, size=size):
            metric.update(t, p)
        return metric.result()

    def test_accuracy_matches_batch(self):
        assert_allclose(self._stream(StreamingAccuracy()),
                        accuracy(self.y_true, self.y_pred))

    def test_averaged_metrics_match_batch(self):
        pairs = [(StreamingPrecision, precision),
                 (StreamingRecall, recall),
                 (StreamingF1, f1)]
        for streaming, batch in pairs:
            for average in ("macro", "micro", "weighted"):
                with self.subTest(metric=batch.__name__, average=average):
                    got = self._stream(streaming(average=average))
                    want = batch(self.y_true, self.y_pred, average=average)
                    assert_allclose(got, want)

    def test_binary_metrics_match_batch(self):
        y_true, y_pred = self.y_true % 2, self.y_pred % 2
        for streaming, batch in [(StreamingPrecision, precision),
                                 (StreamingRecall, recall),
                                 (StreamingF1, f1)]:
            metric = streaming()
            for t, p in _chunks(y_true, y_pred, size=50):
                metric.update(t, p)
            assert_allclose(metric.result(), batch(y_true, y_pred))

    def test_binary_before_positive_seen_uses_zero_division(self):
        metric = StreamingPrecision(zero_division=1)
        self.assertEqual(metric.update([0, 0], [0, 0]).result(), 1.0)

    def test_binary_with_three_classes_raises(self):
        with self.assertRaises(ValueError):
            StreamingF1().update([0, 1, 2], [0, 1, 2]).result()

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            StreamingF1(average="median")
        with self.assertRaises(ValueError):
            StreamingRecall(zero_division=2)

    def test_result_before_update_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no data"):
            StreamingAccuracy().result()

    def test_input_validation(self):
        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.update([0, 1], [0])  # unequal lengths
        with self.assertRaises(ValueError):
            metric.update([[0, 1]], [[0, 1]])  # 2D
        with self.assertRaises(ValueError):
            metric.update([0.0, np.nan], [0.0, 1.0])  # NaN label
        with self.assertRaises(TypeError):
            metric.update([True, False], [True, True])  # booleans
        with self.assertRaises(TypeError):
            metric.update([0, 1], ["a", "b"])  # mixed kinds
        self.assertEqual(metric.n_samples_, 0)


class TestStreamingAUC(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, 300)
        # Rounded scores create many ties across chunks.
        scores = np.round(0.3 * y_true + 0.7 * rng.random(300), 1)
        auc = StreamingAUC()
        for t, s in _chunks(y_true, scores, size=40):
            auc.update(t, s)
        assert_allclose(auc.result(), _brute_force_auc(y_true, scores))

    def test_perfect_and_reversed_ranking(self):
        self.assertEqual(StreamingAUC().update([0, 0, 1, 1],
                                               [.1, .2, .8, .9]).result(), 1.0)
        self.assertEqual(StreamingAUC().update([0, 0, 1, 1],
                                               [.9, .8, .2, .1]).result(), 0.0)

    def test_all_tied_scores_give_half(self):
        auc = StreamingAUC().update([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(auc.result(), 0.5)

    def test_single_class_is_nan(self):
        auc = StreamingAUC().update([1, 1], [0.2, 0.9])
        self.assertTrue(np.isnan(auc.result()))

    def test_string_pos_label(self):
        auc = StreamingAUC(pos_label="yes")
        auc.update(["no", "yes"], [0.3, 0.6])
        self.assertEqual(auc.result(), 1.0)

    def test_score_validation(self):
        with self.assertRaises(ValueError):
            StreamingAUC().update([0, 1], [0.5, np.nan])
        with self.assertRaises(ValueError):
            StreamingAUC().update([0, 1, 1], [0.5, 0.2])

if __name__ == "__main__":
    unittest.main()
