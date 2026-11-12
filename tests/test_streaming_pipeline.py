"""Tests for streaming (partial_fit) pipelines and feature unions."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import (
    ColumnSelector,
    MinMaxScaler,
    SimpleImputer,
    StandardScaler,
)
from numcompute_stream.tree import DecisionTreeClassifier


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=[10, -5, 0], scale=[2, 1, 1], size=(n, 3))
    y = ((X[:, 0] - 10) / 2 + (X[:, 1] + 5) > 0).astype(int)
    return X, y


def _chunks(X, y, size):
    for start in range(0, len(X), size):
        yield X[start:start + size], y[start:start + size]


class _CountingTransformer:
    """Transformer without partial_fit that records how often it is fitted."""

    def __init__(self):
        self.n_fits = 0

    def fit(self, X, y=None):
        self.n_fits += 1
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


class TestPipelinePartialFit(unittest.TestCase):
    def test_spec_example_trains_and_predicts(self):
        X, y = _data()
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=5, buffer_size=200)),
        ])
        for X_chunk, y_chunk in _chunks(X, y, 50):
            self.assertIs(pipe.partial_fit(X_chunk, y_chunk), pipe)
        self.assertEqual(pipe.named_steps["model"].n_retrains_, 3)
        X_test, y_test = _data(300, seed=1)
        self.assertGreater(pipe.score(X_test, y_test), 0.85)
        self.assertEqual(pipe.predict(X_test).shape, (300,))
        assert_allclose(pipe.predict_proba(X_test).sum(axis=1), 1.0)

    def test_scaler_state_matches_all_seen_data(self):
        X, y = _data()
        scaler = StandardScaler()
        pipe = Pipeline([("scale", scaler),
                         ("model", DecisionTreeClassifier(buffer_size=10_000))])
        for X_chunk, y_chunk in _chunks(X, y, 64):
            pipe.partial_fit(X_chunk, y_chunk)
        assert_allclose(scaler.mean_, X.mean(axis=0))
        assert_allclose(scaler.scale_, X.std(axis=0))

    def test_model_receives_transformed_chunk(self):
        X, y = _data(100)
        model = DecisionTreeClassifier(buffer_size=100)
        pipe = Pipeline([("minmax", MinMaxScaler()), ("model", model)])
        pipe.partial_fit(X, y)
        # Thresholds learned on [0, 1]-scaled data must lie in [0, 1].
        self.assertTrue(0.0 <= model.root_.threshold <= 1.0)

    def test_nan_chunks_with_streaming_imputer(self):
        X, y = _data()
        X[::7, 0] = np.nan
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=4, buffer_size=150)),
        ])
        for X_chunk, y_chunk in _chunks(X, y, 75):
            pipe.partial_fit(X_chunk, y_chunk)
        self.assertEqual(pipe.predict(X).shape, (600,))

    def test_step_without_partial_fit_is_fitted_once(self):
        X, y = _data(90)
        counter = _CountingTransformer()
        pipe = Pipeline([("count", counter),
                         ("model", DecisionTreeClassifier(buffer_size=30))])
        for X_chunk, y_chunk in _chunks(X, y, 30):
            pipe.partial_fit(X_chunk, y_chunk)
        self.assertEqual(counter.n_fits, 1)

    def test_predict_before_first_retrain_raises_model_error(self):
        X, y = _data(40)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", DecisionTreeClassifier(buffer_size=100))])
        pipe.partial_fit(X, y)
        with self.assertRaisesRegex(RuntimeError, "not trained yet"):
            pipe.predict(X)

    def test_unfitted_and_unsupported(self):
        pipe = Pipeline([("model", DecisionTreeClassifier())])
        with self.assertRaises(RuntimeError):
            pipe.score([[0.0]], [0])
        no_partial = Pipeline([("scale", StandardScaler()),
                               ("select", ColumnSelector([0]))])
        with self.assertRaisesRegex(TypeError, "partial_fit"):
            no_partial.partial_fit([[1.0, 2.0]])

    def test_fit_then_partial_fit_continues(self):
        X, y = _data()
        scaler = StandardScaler()
        pipe = Pipeline([("scale", scaler),
                         ("model", DecisionTreeClassifier(buffer_size=100))])
        pipe.fit(X[:300], y[:300])
        pipe.partial_fit(X[300:], y[300:])
        assert_allclose(scaler.mean_, X.mean(axis=0))

if __name__ == "__main__":
    unittest.main()
