"""Tests for the StreamTrainer training and logging loop."""

import contextlib
import io
import unittest

import numpy as np
from numpy.testing import assert_allclose

from numcompute_stream.metrics import accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer, estimate_nbytes
from numcompute_stream.tree import DecisionTreeClassifier


def _stream(n=1000, chunk=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return [(X[i:i + chunk], y[i:i + chunk]) for i in range(0, n, chunk)]


def _pipe(buffer_size=200):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)),
    ])


class _NoRetrainModel:
    """Minimal model without n_retrains_ or root_ attributes."""

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestStreamTrainerLoop(unittest.TestCase):
    def test_test_then_train_skips_until_first_retrain(self):
        trainer = StreamTrainer(_pipe(buffer_size=200))
        trainer.run(_stream())
        scored = [entry["scored"] for entry in trainer.history_]
        # Chunks 0-1 fill the buffer; chunk 2 is the first one scored.
        self.assertEqual(scored, [False, False] + [True] * 8)
        self.assertEqual(trainer.n_chunks_, 10)
        self.assertEqual(trainer.n_samples_seen_, 1000)
        self.assertEqual(trainer.summary()["n_scored"], 800)

    def test_retrain_flags_follow_buffer(self):
        trainer = StreamTrainer(_pipe(buffer_size=200))
        trainer.run(_stream())
        retrained = [entry["retrained"] for entry in trainer.history_]
        self.assertEqual(retrained, [False, True] * 5)
        self.assertEqual(trainer.history_[-1]["n_retrains"], 5)
        self.assertEqual(trainer.history_[0]["buffer_count"], 100)

    def test_cumulative_accuracy_matches_scored_predictions(self):
        model = DecisionTreeClassifier(max_depth=4, buffer_size=100)
        trainer = StreamTrainer(model)
        y_true, y_pred = [], []
        for X, y in _stream():
            if model.root_ is not None:
                y_true.append(y)
                y_pred.append(model.predict(X))
            trainer.process_chunk(X, y)
        expected = accuracy(np.concatenate(y_true), np.concatenate(y_pred))
        assert_allclose(trainer.summary()["accuracy"], expected)
        assert_allclose(trainer.get_metric("cumulative_accuracy")[-1], expected)

    def test_score_and_fit_chunk_separately(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
        X, y = _stream()[0]
        self.assertFalse(trainer.score_chunk(X, y)["scored"])
        info = trainer.fit_chunk(X, y)
        self.assertTrue(info["retrained"])
        result = trainer.score_chunk(X, y)
        self.assertTrue(result["scored"])
        self.assertGreater(result["chunk_accuracy"], 0.9)
        np.testing.assert_array_equal(trainer.last_y_true_, y)

if __name__ == "__main__":
    unittest.main()
