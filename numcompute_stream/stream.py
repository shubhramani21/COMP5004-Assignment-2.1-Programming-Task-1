"""Streaming training loop with per-chunk logging.

``StreamTrainer`` wraps a model or pipeline that supports
``partial_fit`` and ``predict``. It trains chunk by chunk, scores each
chunk, and keeps a log of per-chunk and cumulative metrics, timings and
the model's memory footprint.

Example
-------
>>> trainer = StreamTrainer(pipe, rolling_window=500, auc_pos_label=1)
>>> for X_chunk, y_chunk in stream:
...     trainer.process_chunk(X_chunk, y_chunk)   # score, then train
>>> trainer.get_metric("cumulative_accuracy")
"""

import sys
import time

import numpy as np

from .metrics import (
    RollingWindowMetric,
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    _validate_stream_labels,
)
from .utils import _validate_integer


def estimate_nbytes(obj):
    """Approximate the memory held by an object and everything it references.

    NumPy arrays count their data buffer (``nbytes``); lists, tuples, sets,
    dicts and objects with ``__dict__`` or ``__slots__`` are walked
    recursively. Shared objects are counted once.

    Parameters
    ----------
    obj : object

    Returns
    -------
    int
        Estimated size in bytes.
    """
    seen = set()
    total = 0
    stack = [obj]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, np.ndarray):
            total += current.nbytes + sys.getsizeof(np.empty(0))
            if current.dtype == object:
                stack.extend(current.ravel().tolist())
            continue

        total += sys.getsizeof(current)
        if isinstance(current, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
        else:
            if hasattr(current, "__dict__"):
                stack.append(vars(current))
            for slot in getattr(type(current), "__slots__", ()):
                if hasattr(current, slot):
                    stack.append(getattr(current, slot))
    return total


def _final_estimator(model):
    """Return the last step of a Pipeline, or the model itself."""
    steps = getattr(model, "steps", None)
    if steps:
        return steps[-1][1]
    return model


def _is_trained(model):
    """True if the model can predict.

    An unfitted Pipeline, or a final estimator whose ``root_`` is None
    (a buffered tree that has not retrained yet), cannot.
    """
    if getattr(model, "_is_fitted", True) is False:
        return False
    return getattr(_final_estimator(model), "root_", True) is not None
