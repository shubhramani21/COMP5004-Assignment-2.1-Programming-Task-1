"""assess the metrics for classification and regression using numpy.

The functions (``accuracy``, ``precision``, ...) score one batch of
predictions. The ``Streaming*`` classes score a stream of chunks: call
``update(y_true_chunk, y_pred_chunk)`` for each chunk, ``result()`` to
read the metric over everything seen so far, and ``reset()`` to start
again. ``RollingWindowMetric`` scores only the most recent samples.
"""

import copy
import numpy as np
from numbers import Real
from .utils import _as_float_array, _validate_integer

def _validate_label_values(array, name):
    """Validate label values and return their common kind."""
    label_kind = None

    for value in array:
        if value is None:
            raise ValueError(
                f"{name} must not contain missing labels."
            )

        if isinstance(value, (str, np.str_)):
            current_kind = "string"

        elif isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"{name} labels must be strings or real numbers."
            )

        elif isinstance(
            value,
            (Real, np.integer, np.floating),
        ):
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must not contain NaN or infinity."
                )

            current_kind = "numeric"

        else:
            raise TypeError(
                f"{name} labels must be strings or real numbers."
            )

        if label_kind is None:
            label_kind = current_kind

        elif current_kind != label_kind:
            raise TypeError(
                f"{name} must contain one homogeneous type."
            )

    return label_kind


def _validate_classification_inputs(y_true, y_pred):
    """Validate and return one-dimensional classification labels."""
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    true_kind = _validate_label_values(true, "y_true")
    pred_kind = _validate_label_values(pred, "y_pred")

    if true_kind != pred_kind:
        raise TypeError(
            "y_true and y_pred labels must have "
            "the same homogeneous type."
        )

    return true, pred

def _validate_metric_options(average, zero_division):
    """Validate shared classification metric options."""
    if not isinstance(average, str):
        raise TypeError("average must be a string.")

    valid_averages = (
        "binary",
        "macro",
        "micro",
        "weighted",
    )

    if average not in valid_averages:
        raise ValueError(
            "average must be 'binary', 'macro', "
            "'micro' or 'weighted'."
        )

    if isinstance(zero_division, (bool, np.bool_)):
        raise TypeError("zero_division must be 0 or 1.")

    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1.")
    
def accuracy(y_true, y_pred):
    """Return the fraction of correctly predicted labels.

    y_true and y_pred must be non-empty 1D array-like inputs with
    equal lengths. Labels must be homogeneous strings or finite
    real numbers, with the same label kind in both inputs.

    Return a float between 0 and 1. Inputs are not modified, and
    non-contiguous arrays are supported.

    Raise TypeError for unsupported or mixed label types.
    Raise ValueError for invalid shapes, unequal lengths, empty
    inputs, missing labels or non-finite numeric labels.

    Time complexity: O(n). Additional space: O(n).
    """
    true, pred = _validate_classification_inputs(
        y_true,
        y_pred,
    )

    return float(np.mean(true == pred))

def confusion_matrix(y_true, y_pred, labels=None):
    """Return a confusion matrix and its ordered label array.

    y_true and y_pred must be non-empty 1D array-like inputs with
    equal lengths and homogeneous string or finite real labels.

    labels is an optional non-empty 1D sequence of unique labels.
    It must match the observed label kind and include every observed
    class; additional unobserved classes are allowed. By default,
    order follows first appearance in y_true, then unseen labels
    from y_pred.

    Return (matrix, labels_array). The int64 matrix has shape (c, c),
    with true classes in rows and predicted classes in columns.
    The object label array has shape (c,) and preserves label values.
    Inputs are not modified; non-contiguous arrays are supported.

    Raise TypeError for unsupported or incompatible label types.
    Raise ValueError for invalid shapes, empty inputs, unequal
    lengths, missing or non-finite labels, duplicate explicit labels,
    or explicit labels that omit observed classes.

    For n samples and c classes, default label discovery takes
    O(n*c) time. Matrix construction takes expected O(n + c**2)
    time using dictionary lookup. Additional space: O(n + c**2).
    """
    true, pred = _validate_classification_inputs(
        y_true,
        y_pred,
    )

    if labels is None:
        ordered_labels = []

        for label in np.concatenate((true, pred)):
            if label not in ordered_labels:
                ordered_labels.append(label)

        label_array = np.asarray(ordered_labels, dtype=object)
    else:
        label_array = np.asarray(labels, dtype=object)

        if label_array.ndim != 1:
            raise ValueError("labels must be 1D.")

        if label_array.size == 0:
            raise ValueError("labels must not be empty.")

        label_kind = _validate_label_values(
            label_array,
            "labels",
        )
        observed_kind = _validate_label_values(
            true,
            "y_true",
        )

        if label_kind != observed_kind:
            raise TypeError(
                "labels must have the same type as "
                "the observed labels."
            )

        label_list = label_array.tolist()

        if len(set(label_list)) != len(label_list):
            raise ValueError(
                "labels must contain unique values."
            )
        
    label_to_index = {
        label: index
        for index, label in enumerate(label_array.tolist())
    }

    observed_labels = np.concatenate((true, pred)).tolist()

    if any(
        label not in label_to_index
        for label in observed_labels
    ):
        raise ValueError(
            "labels must include all observed labels."
        )

    matrix = np.zeros(
        (label_array.size, label_array.size),
        dtype=np.int64,
    )

    for actual, predicted in zip(true.tolist(), pred.tolist()):
        matrix[
            label_to_index[actual],
            label_to_index[predicted],
        ] += 1

    return matrix, label_array

def precision(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification precision.

    y_true and y_pred are non-empty 1D array-like labels of equal
    length: homogeneous strings or finite real numbers of matching
    kind. Inputs are not modified; non-contiguous arrays are allowed.

    average accepts 'binary', 'macro', 'micro' or 'weighted'.
    Binary averaging permits at most two observed classes and uses
    the observed class pos_label, which defaults to 1.
    Other averages ignore pos_label.

    Per-class precision is TP / (TP + FP). Macro averages classes
    equally; micro aggregates counts before division; weighted
    uses true-label support. zero_division is 0 or 1 and supplies
    the result when a denominator is zero.

    Return a scalar float. Raise TypeError for unsupported label
    types, non-string average or boolean zero_division. Raise
    ValueError for invalid data shapes or values, unsupported
    options, or an invalid binary-class configuration.

    For n samples and c observed classes, time is O(n*c + c**2)
    and additional space is O(n + c**2), including confusion-matrix
    construction.
    """
    _validate_metric_options(average, zero_division)

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = np.sum(
        matrix,
        axis=0,
        dtype=np.float64,
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        true_positive,
        predicted_positive,
        out=per_class,
        where=predicted_positive != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        denominator = float(np.sum(predicted_positive))

        if denominator == 0:
            return float(zero_division)

        return float(
            np.sum(true_positive) / denominator
        )

    support = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    return float(
        np.average(per_class, weights=support)
    )

def recall(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification recall.

    y_true and y_pred are non-empty 1D array-like labels of equal
    length: homogeneous strings or finite real numbers of matching
    kind. Inputs are not modified; non-contiguous arrays are allowed.

    average accepts 'binary', 'macro', 'micro' or 'weighted'.
    Binary averaging permits at most two observed classes and uses
    the observed class pos_label, which defaults to 1.
    Other averages ignore pos_label.

    Per-class recall is TP / (TP + FN). Macro averages classes
    equally; micro aggregates counts before division; weighted
    uses true-label support. zero_division is 0 or 1 and supplies
    the result when a denominator is zero.

    Return a scalar float. Raise TypeError for unsupported label
    types, non-string average or boolean zero_division. Raise
    ValueError for invalid data shapes or values, unsupported
    options, or an invalid binary-class configuration.

    For n samples and c observed classes, time is O(n*c + c**2)
    and additional space is O(n + c**2), including confusion-matrix
    construction.
    """
    _validate_metric_options(average, zero_division)

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)
    actual_positive = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        true_positive,
        actual_positive,
        out=per_class,
        where=actual_positive != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        denominator = float(np.sum(actual_positive))

        if denominator == 0:
            return float(zero_division)

        return float(
            np.sum(true_positive) / denominator
        )

    return float(
        np.average(per_class, weights=actual_positive)
    )

def f1(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification F1 score.

    y_true and y_pred are non-empty 1D array-like labels of equal
    length: homogeneous strings or finite real numbers of matching
    kind. Inputs are not modified; non-contiguous arrays are allowed.

    average accepts 'binary', 'macro', 'micro' or 'weighted'.
    Binary averaging permits at most two observed classes and uses
    the observed class pos_label, which defaults to 1.
    Other averages ignore pos_label.

    Per-class F1 is 2*TP / (2*TP + FP + FN). Macro averages these
    scores equally; weighted uses true-label support. Micro
    aggregates counts before calculating F1. zero_division is
    0 or 1 and supplies the result when a denominator is zero.

    Return a scalar float. Raise TypeError for unsupported label
    types, non-string average or boolean zero_division. Raise
    ValueError for invalid data shapes or values, unsupported
    options, or an invalid binary-class configuration.

    For n samples and c observed classes, time is O(n*c + c**2)
    and additional space is O(n + c**2), including confusion-matrix
    construction.
    """
    _validate_metric_options(average, zero_division)

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)

    false_positive = (
        np.sum(matrix, axis=0, dtype=np.float64)
        - true_positive
    )

    false_negative = (
        np.sum(matrix, axis=1, dtype=np.float64)
        - true_positive
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        2 * true_positive,
        denominator,
        out=per_class,
        where=denominator != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        numerator = 2 * float(np.sum(true_positive))
        micro_denominator = (
            numerator
            + float(np.sum(false_positive))
            + float(np.sum(false_negative))
        )

        if micro_denominator == 0:
            return float(zero_division)

        return float(numerator / micro_denominator)

    support = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    return float(
        np.average(per_class, weights=support)
    )

def mse(y_true, y_pred):
    """Return the scalar mean squared error over all elements."""
    true = _as_float_array(y_true, name="y_true", finite=True)
    pred = _as_float_array(y_pred, name="y_pred", finite=True)

    if true.ndim not in (1, 2):
        raise ValueError("y_true must be 1D or 2D.")

    if pred.ndim not in (1, 2):
        raise ValueError("y_pred must be 1D or 2D.")

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if true.size == 0:
        raise ValueError(
            "y_true and y_pred must not be empty."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        residuals = true - pred

    if not np.all(np.isfinite(residuals)):
        raise ValueError(
            "Mean squared error exceeds the numeric range."
        )

    scale = np.max(np.abs(residuals))

    if scale == 0:
        return 0.0

    normalized = residuals / scale
    mean_square = np.mean(normalized * normalized, dtype=np.float64)

    # Recover the root mean squared error before the final square.
    # Avoid computing scale**2, which could overflow prematurely.
    with np.errstate(over="ignore", invalid="ignore"):
        rms = scale * np.sqrt(mean_square)
        result = rms * rms

    if not np.isfinite(result):
        raise ValueError(
            "Mean squared error exceeds the numeric range."
        )

    return float(result)

# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------

def _validate_stream_labels(y_true, y_pred):
    """Validate one chunk of labels and return them as 1D arrays.

    Labels may be numbers or strings, but both inputs must use the same
    kind. Empty chunks are allowed (they leave a metric unchanged).

    Raises
    ------
    TypeError
        For boolean labels or numbers mixed with strings.
    ValueError
        For non-1D input, unequal lengths, NaN/inf or None labels.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have equal lengths; "
            f"got {true.shape[0]} and {pred.shape[0]}."
        )

    kinds = []
    for name, array in (("y_true", true), ("y_pred", pred)):
        kind = array.dtype.kind
        if kind == "b":
            raise TypeError(f"{name} labels must not be boolean.")
        if kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must not contain NaN or infinity.")
        if kind == "O" and any(label is None for label in array.tolist()):
            raise ValueError(f"{name} must not contain missing labels.")
        if kind in "iuf":
            kinds.append("numeric")
        elif kind in "US":
            kinds.append("string")
        else:
            kinds.append(None)  # object arrays: checked when sorting

    if None not in kinds and true.size and kinds[0] != kinds[1]:
        raise TypeError(
            "y_true and y_pred labels must be the same kind "
            "(both numeric or both strings)."
        )
    return true, pred


class StreamingMetric:
    """Base class for metrics that are updated chunk by chunk.

    Subclasses implement ``_reset_state``, ``_update`` and ``_result``.

    Attributes
    ----------
    n_samples_ : int
        Number of samples seen since the last ``reset()``.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget all data seen so far.

        Returns
        -------
        self
        """
        self.n_samples_ = 0
        self._reset_state()
        return self

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted values.

        Parameters
        ----------
        y_true : array-like, shape (n_chunk,)
        y_pred : array-like, shape (n_chunk,)

        Returns
        -------
        self
        """
        n_added = self._update(y_true, y_pred)
        self.n_samples_ += n_added
        return self

    def result(self):
        """Return the metric over all data seen since the last reset.

        Raises
        ------
        RuntimeError
            If no samples have been seen yet.
        """
        if self.n_samples_ == 0:
            raise RuntimeError(
                f"{type(self).__name__} has no data; call update() first."
            )
        return self._result()

    def _reset_state(self):
        raise NotImplementedError

    def _update(self, y_true, y_pred):
        raise NotImplementedError

    def _result(self):
        raise NotImplementedError


class StreamingConfusionMatrix(StreamingMetric):
    """Confusion matrix accumulated over a stream of chunks.

    Rows are true classes and columns are predicted classes. The matrix
    grows when a chunk contains labels not seen before.

    Parameters
    ----------
    labels : array-like or None, default=None
        Fixed label order. If given, any other label raises ValueError.
        If None, labels are added in order of first appearance (labels
        that first appear in the same chunk are added in sorted order).

    Attributes
    ----------
    matrix_ : numpy.ndarray of int64, shape (n_classes, n_classes)
    labels_ : numpy.ndarray of object, shape (n_classes,)
    """

    def __init__(self, labels=None):
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.size == 0:
                raise ValueError("labels must be a non-empty 1D sequence.")
            if np.unique(labels).size != labels.size:
                raise ValueError("labels must contain unique values.")
        self.labels = labels
        super().__init__()

    def _reset_state(self):
        if self.labels is None:
            self._label_list = []
        else:
            self._label_list = self.labels.tolist()
        self._index = {label: i for i, label in enumerate(self._label_list)}
        n = len(self._label_list)
        self.matrix_ = np.zeros((n, n), dtype=np.int64)

    @property
    def labels_(self):
        return np.asarray(self._label_list, dtype=object)

    def _update(self, y_true, y_pred):
        true, pred = _validate_stream_labels(y_true, y_pred)
        n = true.shape[0]
        if n == 0:
            return 0

        try:
            unique, inverse = np.unique(np.concatenate((true, pred)),
                                        return_inverse=True)
        except TypeError as exc:
            raise TypeError(
                "y_true and y_pred labels must be the same kind "
                "(both numeric or both strings)."
            ) from exc

        # Register new labels (only the unique values are looped over).
        new_labels = [label for label in unique.tolist()
                      if label not in self._index]
        if new_labels and self.labels is not None:
            raise ValueError(
                f"Labels {new_labels} are not in the fixed labels list."
            )
        for label in new_labels:
            self._index[label] = len(self._label_list)
            self._label_list.append(label)

        n_classes = len(self._label_list)
        if self.matrix_.shape[0] < n_classes:
            grown = np.zeros((n_classes, n_classes), dtype=np.int64)
            old = self.matrix_.shape[0]
            grown[:old, :old] = self.matrix_
            self.matrix_ = grown

        # Map every sample to its class index, then count all
        # (true, pred) pairs at once with bincount.
        unique_index = np.array([self._index[label]
                                 for label in unique.tolist()])
        codes = unique_index[inverse.ravel()]
        pairs = codes[:n] * n_classes + codes[n:]
        self.matrix_ += np.bincount(
            pairs, minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes)
        return n

    def _result(self):
        """Return ``(matrix, labels)`` as copies."""
        return self.matrix_.copy(), self.labels_


def _score_from_matrix(matrix, labels, metric, average, pos_label,
                       zero_division):
    """Compute precision, recall or F1 from a confusion matrix.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_classes, n_classes)
    labels : numpy.ndarray, shape (n_classes,)
    metric : {"precision", "recall", "f1"}
    average : {"binary", "macro", "micro", "weighted"}
    pos_label : label used when ``average="binary"``
    zero_division : 0 or 1, value used when a denominator is zero

    Returns
    -------
    float
    """
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp  # predicted as the class, but wrong
    fn = matrix.sum(axis=1) - tp  # the class, but predicted as another

    if metric == "precision":
        numerator, denominator = tp, tp + fp
    elif metric == "recall":
        numerator, denominator = tp, tp + fn
    else:
        numerator, denominator = 2 * tp, 2 * tp + fp + fn

    per_class = np.full(tp.shape, float(zero_division))
    np.divide(numerator, denominator, out=per_class,
              where=denominator != 0)

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes; "
                f"saw {labels.size}. Use average='macro' or 'weighted'."
            )
        matches = np.flatnonzero(labels == pos_label)
        if matches.size == 0:
            # pos_label not seen yet: no true or predicted positives.
            return float(zero_division)
        return float(per_class[matches[0]])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        total = float(np.sum(denominator))
        if total == 0:
            return float(zero_division)
        return float(np.sum(numerator) / total)

    support = matrix.sum(axis=1).astype(np.float64)
    return float(np.sum(per_class * support) / np.sum(support))


class _ConfusionMatrixMetric(StreamingMetric):
    """Base for metrics computed from an accumulated confusion matrix."""

    def __init__(self, labels=None):
        self._confusion = StreamingConfusionMatrix(labels=labels)
        super().__init__()

    def _reset_state(self):
        self._confusion.reset()

    def _update(self, y_true, y_pred):
        before = self._confusion.n_samples_
        self._confusion.update(y_true, y_pred)
        return self._confusion.n_samples_ - before

    @property
    def confusion_matrix_(self):
        """The accumulated ``StreamingConfusionMatrix``."""
        return self._confusion


class StreamingAccuracy(_ConfusionMatrixMetric):
    """Cumulative accuracy over a stream.

    Examples
    --------
    >>> acc = StreamingAccuracy()
    >>> acc.update([0, 1], [0, 0]).update([1, 1], [1, 1]).result()
    0.75
    """

    def _result(self):
        matrix = self._confusion.matrix_
        return float(np.trace(matrix) / matrix.sum())


class _AveragedMetric(_ConfusionMatrixMetric):
    """Shared options for precision, recall and F1."""

    _metric = None

    def __init__(self, average="binary", pos_label=1, zero_division=0,
                 labels=None):
        _validate_metric_options(average, zero_division)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        super().__init__(labels=labels)

    def _result(self):
        return _score_from_matrix(
            self._confusion.matrix_, self._confusion.labels_,
            self._metric, self.average, self.pos_label, self.zero_division,
        )
