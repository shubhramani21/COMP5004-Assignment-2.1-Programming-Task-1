"""assess the metrics for classification and regression using numpy."""

import numpy as np
from numbers import Real
from .utils import _as_float_array

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