"""Benchmarks for the streaming decision tree framework.

Part A compares loop-based and vectorised implementations (and checks
that they give the same answer). Part B streams data through trees with
different ``buffer_size`` values and records accuracy, retrain count and
retrain time.

Run from the repository root:

    python -m benchmark.run_stream_benchmarks            # full run
    python -m benchmark.run_stream_benchmarks --quick    # smaller, faster

Results are written to ``benchmark/results/stream_<timestamp>/``:
``vectorisation.csv``, ``buffer_size.csv``, ``environment.json`` and PNG
plots.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from benchmark.stream_baselines import (  # noqa: E402
    loop_confusion_matrix,
    loop_tree_predict,
    loop_welford,
)
from numcompute_stream import tree as tree_module  # noqa: E402
from numcompute_stream.benchmarking import benchmark, environment_info  # noqa: E402
from numcompute_stream.metrics import StreamingConfusionMatrix  # noqa: E402
from numcompute_stream.preprocessing import StandardScaler  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import (  # noqa: E402
    DecisionTreeClassifier,
    _best_split_feature,
    _best_split_feature_loop,
    _gini,
)
from numcompute_stream.visualise import compare_models  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_stream_data(n_samples, n_features=5, drift_at=None, noise=0.05,
                     seed=0):
    """Synthetic binary classification stream.

    The label depends non-linearly on the first three features. With
    ``drift_at`` set, the decision rule changes at that sample index
    (concept drift), which favours trees that retrain more often.
    Labels of ``noise`` fraction of samples are flipped.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    rule_a = (X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] > 0.3)
    rule_b = (-X[:, 0] + X[:, 2] * X[:, 1] > 0.0)
    y = rule_a.copy()
    if drift_at is not None:
        y[drift_at:] = rule_b[drift_at:]
    flip = rng.random(n_samples) < noise
    y = np.where(flip, ~y, y).astype(int)
    return X, y


def iter_chunks(X, y, chunk_size):
    for start in range(0, X.shape[0], chunk_size):
        yield X[start:start + chunk_size], y[start:start + chunk_size]


# ---------------------------------------------------------------------------
# Part A: loop vs vectorised
# ---------------------------------------------------------------------------

def _row(name, size, loop_result, vector_result):
    loop_time = loop_result["median_seconds"]
    vector_time = vector_result["median_seconds"]
    return {
        "component": name,
        "size": size,
        "loop_seconds": loop_time,
        "vectorised_seconds": vector_time,
        "speedup": loop_time / vector_time if vector_time > 0 else float("inf"),
    }


def bench_split_search(sizes, repeats):
    """Best threshold on one feature: O(n^2) loop vs O(n log n) cumsum."""
    rows = []
    for n in sizes:
        rng = np.random.default_rng(n)
        x = rng.normal(size=n)
        y = rng.integers(0, 3, n)
        loop_answer = _best_split_feature_loop(x, y, 3, _gini)
        fast_answer = _best_split_feature(x, y, 3, _gini)
        assert_allclose(fast_answer[0], loop_answer[0])
        assert fast_answer[1] == loop_answer[1]
        rows.append(_row(
            "split_search", n,
            benchmark(_best_split_feature_loop, x, y, 3, _gini, repeats=repeats),
            benchmark(_best_split_feature, x, y, 3, _gini, repeats=repeats),
        ))
    return rows
