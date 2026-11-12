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


class TestStreamTrainerLogging(unittest.TestCase):
    def test_history_keys_and_get_metric(self):
        trainer = StreamTrainer(_pipe(), rolling_window=150, auc_pos_label=1)
        trainer.run(_stream())
        expected_keys = {
            "chunk", "n_samples", "scored", "chunk_accuracy",
            "cumulative_accuracy", "cumulative_f1_macro", "rolling_accuracy",
            "cumulative_auc", "fit_time", "predict_time", "retrained",
            "n_retrains", "buffer_count", "model_bytes",
        }
        self.assertEqual(set(trainer.history_[0]), expected_keys)
        auc = trainer.get_metric("cumulative_auc")
        self.assertTrue(np.all(np.isnan(auc[:2])))
        self.assertTrue(np.all((auc[2:] > 0.8) & (auc[2:] <= 1.0)))
        self.assertTrue(np.all(np.isfinite(trainer.get_metric("rolling_accuracy")[2:])))
        with self.assertRaises(KeyError):
            trainer.get_metric("nope")

    def test_memory_tracks_buffer(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=300))
        trainer.run(_stream(chunk=100))
        memory = trainer.get_metric("model_bytes")
        X, y = _stream(chunk=100)[0]
        chunk_bytes = X.nbytes + y.nbytes
        # Each buffered chunk adds at least its own data to the model.
        # Retrains happen at chunks 2, 5, 8; chunks 3 -> 4 share one tree.
        self.assertGreaterEqual(memory[1] - memory[0], chunk_bytes)
        self.assertGreaterEqual(memory[4] - memory[3], chunk_bytes)
        self.assertTrue(trainer.history_[2]["retrained"])
        self.assertEqual(trainer.history_[2]["buffer_count"], 0)
        no_memory = StreamTrainer(DecisionTreeClassifier(), track_memory=False)
        no_memory.process_chunk(*_stream()[0])
        self.assertTrue(np.isnan(no_memory.history_[0]["model_bytes"]))

    def test_confusion_matrix_and_reset(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
        trainer.run(_stream())
        matrix, labels = trainer.confusion_matrix()
        self.assertEqual(matrix.sum(), 900)
        self.assertEqual(list(labels), [0, 1])
        trainer.reset()
        self.assertEqual(trainer.history_, [])
        self.assertTrue(np.isnan(trainer.summary()["accuracy"]))

    def test_verbose_prints_each_chunk(self):
        trainer = StreamTrainer(_pipe(), verbose=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            trainer.run(_stream()[:3])
        lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("retrained", lines[1])

    def test_model_without_retrain_info(self):
        trainer = StreamTrainer(_NoRetrainModel())
        entry = trainer.process_chunk(*_stream()[0])
        self.assertTrue(entry["scored"])
        self.assertIsNone(entry["retrained"])
        self.assertTrue(np.isnan(trainer.get_metric("retrained")[0]))


class TestStreamTrainerValidation(unittest.TestCase):
    def test_model_must_support_streaming(self):
        with self.assertRaises(TypeError):
            StreamTrainer(StandardScaler())
        with self.assertRaises(TypeError):
            StreamTrainer(_NoRetrainModel(), auc_pos_label=1)
        with self.assertRaises(ValueError):
            StreamTrainer(_pipe(), rolling_window=0)

    def test_bad_labels(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=10))
        X, y = _stream()[0]
        trainer.fit_chunk(X, y)
        with self.assertRaises(ValueError):
            trainer.score_chunk(X, y.reshape(-1, 1))

    def test_estimate_nbytes(self):
        small = estimate_nbytes(np.zeros(10))
        large = estimate_nbytes(np.zeros(1000))
        self.assertGreaterEqual(large - small, 990 * 8)
        shared = np.zeros(1000)
        self.assertLess(estimate_nbytes([shared, shared]),
                        2 * estimate_nbytes(shared))


if __name__ == "__main__":
    unittest.main()
