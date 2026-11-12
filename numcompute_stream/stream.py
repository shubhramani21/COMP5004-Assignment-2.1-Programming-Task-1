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


class StreamTrainer:
    """Train and evaluate a streaming model chunk by chunk.

    Parameters
    ----------
    model : object
        Estimator or Pipeline with ``partial_fit(X, y)`` and ``predict(X)``
        (e.g. ``DecisionTreeClassifier`` or ``Pipeline([..., tree])``).
    rolling_window : int or None, default=None
        If set, also log accuracy over the last ``rolling_window`` scored
        samples.
    auc_pos_label : label or None, default=None
        If set and the model has ``predict_proba``, log cumulative ROC AUC
        for this positive class (binary problems).
    track_memory : bool, default=True
        Log an estimate of the model's memory footprint after each chunk.
    verbose : bool, default=False
        Print one line per processed chunk.

    Attributes
    ----------
    history_ : list of dict
        One entry per chunk. Keys: "chunk", "n_samples", "scored",
        "chunk_accuracy", "cumulative_accuracy", "cumulative_f1_macro",
        "rolling_accuracy", "cumulative_auc", "fit_time", "predict_time",
        "retrained", "n_retrains", "buffer_count", "model_bytes".
        Values that do not apply are NaN (or None for "retrained").
    n_chunks_ : int
    n_samples_seen_ : int
    last_y_true_, last_y_pred_ : numpy.ndarray or None
        Labels and predictions of the most recently scored chunk.

    Notes
    -----
    ``process_chunk`` follows the test-then-train (prequential) protocol:
    each chunk is predicted by the current model before the model learns
    from it, so every score is on unseen data. Chunks that arrive before
    the model's first training (e.g. while a tree fills its buffer) are
    trained on but not scored.
    """

    def __init__(self, model, rolling_window=None, auc_pos_label=None,
                 track_memory=True, verbose=False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model must provide a {method} method.")
        if rolling_window is not None:
            rolling_window = _validate_integer(
                rolling_window, name="rolling_window", minimum=1)
        if auc_pos_label is not None and not callable(
                getattr(model, "predict_proba", None)):
            raise TypeError("auc_pos_label needs a model with predict_proba.")

        self.model = model
        self.rolling_window = rolling_window
        self.auc_pos_label = auc_pos_label
        self.track_memory = bool(track_memory)
        self.verbose = bool(verbose)
        self.reset()

    def reset(self):
        """Clear the log and metrics (the model is not reset). Returns self."""
        self.history_ = []
        self.n_chunks_ = 0
        self.n_samples_seen_ = 0
        self.last_y_true_ = None
        self.last_y_pred_ = None
        self._accuracy = StreamingAccuracy()
        self._f1 = StreamingF1(average="macro")
        self._confusion = StreamingConfusionMatrix()
        self._rolling = (None if self.rolling_window is None else
                         RollingWindowMetric(StreamingAccuracy(),
                                             self.rolling_window))
        self._auc = (None if self.auc_pos_label is None else
                     StreamingAUC(pos_label=self.auc_pos_label))
        self._pending = None
        return self

    def _stage(self, values):
        """Collect values for the log entry being built."""
        if self._pending is None:
            self._pending = {}
        self._pending.update(values)
        return dict(values, **self._cumulative())

    def _cumulative(self):
        return {
            "cumulative_accuracy": self._result_or_nan(self._accuracy),
            "cumulative_f1_macro": self._result_or_nan(self._f1),
            "rolling_accuracy": self._result_or_nan(self._rolling),
            "cumulative_auc": self._result_or_nan(self._auc),
        }

    @staticmethod
    def _result_or_nan(metric):
        if metric is None or metric.n_samples_ == 0:
            return np.nan
        return metric.result()
