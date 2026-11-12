"""Reusable matplotlib plots for streaming experiments.

Every function draws one figure and returns ``(fig, ax)``. Common options:

- ``ax``: draw into an existing matplotlib Axes (e.g. a subplot grid).
- ``save_path``: also write the figure to this file (PNG, PDF, SVG...).
- ``show``: call ``plt.show()``. In Jupyter, the returned figure is
  displayed inline anyway, so the default is False.

Example
-------
>>> fig, ax = plot_metric_over_time(trainer.get_metric("cumulative_accuracy"),
...                                 title="Accuracy", ylabel="Accuracy")
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _as_series(values, name):
    """Validate a 1D numeric series (NaN allowed for gaps)."""
    array = np.asarray(values, dtype=np.float64) if not isinstance(
        values, np.ndarray) else values.astype(np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {array.ndim}D.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinity.")
    return array


def _get_axes(ax, figsize):
    """Return (fig, ax), creating a new figure when ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _finish(fig, ax, title, xlabel, ylabel, save_path, show, legend):
    """Apply labels, then optionally save and show."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # whole chunks
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        if not isinstance(save_path, (str, os.PathLike)):
            raise TypeError("save_path must be a string or path.")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(metric_values, title="Metric over time",
                          ylabel="Value", xlabel="Chunk", label=None,
                          marks=None, ax=None, save_path=None, show=False,
                          figsize=(8, 4)):
    """Plot a metric (e.g. accuracy) across chunks.

    Parameters
    ----------
    metric_values : array-like, shape (n_chunks,) or dict of name -> array
        One series, or several named series drawn on the same axes.
        NaN values (e.g. chunks before the first training) leave gaps.
    title, ylabel, xlabel : str
    label : str or None
        Legend label for a single series.
    marks : array-like of int or None
        Chunk indices to highlight with vertical dashed lines, e.g. the
        chunks where the tree retrained.
    ax, save_path, show, figsize
        See the module docstring.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    series = (metric_values if isinstance(metric_values, dict)
              else {label: metric_values})
    if not series:
        raise ValueError("metric_values must not be empty.")

    fig, ax = _get_axes(ax, figsize)
    for name, values in series.items():
        values = _as_series(values, "metric_values")
        ax.plot(np.arange(values.size), values, marker="o", markersize=3,
                label=name)

    if marks is not None:
        for index, chunk in enumerate(np.asarray(marks).ravel()):
            ax.axvline(chunk, color="grey", linestyle="--", linewidth=0.8,
                       alpha=0.6, label="retrain" if index == 0 else None)

    legend = any(name is not None for name in series) or marks is not None
    return _finish(fig, ax, title, xlabel, ylabel, save_path, show, legend)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model comparison", ylabel="Value", xlabel="Chunk",
                   ax=None, save_path=None, show=False, figsize=(8, 4)):
    """Compare two models on the same streaming metric.

    Typical use: cumulative accuracy of two trees with different
    ``buffer_size`` values. The series may have different lengths.

    Parameters
    ----------
    metric1, metric2 : array-like, shape (n_chunks,)
    labels : sequence of two str

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    labels = list(labels)
    if len(labels) != 2 or not all(isinstance(l, str) for l in labels):
        raise ValueError("labels must contain exactly two strings.")
    first = _as_series(metric1, "metric1")
    second = _as_series(metric2, "metric2")

    fig, ax = _get_axes(ax, figsize)
    ax.plot(np.arange(first.size), first, marker="o", markersize=3,
            label=labels[0])
    ax.plot(np.arange(second.size), second, marker="s", markersize=3,
            label=labels[1])
    return _finish(fig, ax, title, xlabel, ylabel, save_path, show, True)


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs ground truth",
                                     ax=None, save_path=None, show=False,
                                     figsize=(8, 4)):
    """Show true and predicted labels for each sample of a chunk.

    True labels are drawn as circles and predictions as crosses; wrongly
    predicted samples are highlighted in red. Works for numeric or string
    class labels. The chunk accuracy is added to the title.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    # NumPy would silently turn numbers into strings when mixed.
    if (true.dtype.kind in "US") != (pred.dtype.kind in "US"):
        raise TypeError("y_true and y_pred must use the same label type.")

    try:
        classes = np.unique(np.concatenate((true, pred)))
    except TypeError as exc:
        raise TypeError("y_true and y_pred must use the same label type.") from exc
    true_index = np.searchsorted(classes, true)
    pred_index = np.searchsorted(classes, pred)
    wrong = true_index != pred_index
    samples = np.arange(true.size)

    fig, ax = _get_axes(ax, figsize)
    ax.scatter(samples, true_index, marker="o", facecolors="none",
               edgecolors="tab:blue", label="ground truth")
    ax.scatter(samples[~wrong], pred_index[~wrong], marker="x",
               color="tab:green", label="correct prediction")
    if np.any(wrong):
        ax.scatter(samples[wrong], pred_index[wrong], marker="x",
                   color="tab:red", label="wrong prediction")
    ax.set_yticks(np.arange(classes.size))
    ax.set_yticklabels([str(c) for c in classes])

    accuracy = 1.0 - wrong.mean()
    full_title = f"{title} (accuracy {accuracy:.1%})"
    return _finish(fig, ax, full_title, "Sample", "Class", save_path, show,
                   True)
