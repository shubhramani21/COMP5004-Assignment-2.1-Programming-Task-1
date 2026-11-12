"""Tests for the plotting helpers (run headless with the Agg backend)."""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_confusion_matrix,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


class _CloseFigures(unittest.TestCase):
    def tearDown(self):
        plt.close("all")


class TestPlotMetricOverTime(_CloseFigures):
    def test_single_series_with_gaps(self):
        values = [np.nan, np.nan, 0.8, 0.85, 0.9]
        fig, ax = plot_metric_over_time(values, title="Accuracy",
                                        ylabel="Accuracy")
        self.assertEqual(ax.get_title(), "Accuracy")
        self.assertEqual(ax.get_ylabel(), "Accuracy")
        self.assertEqual(ax.get_xlabel(), "Chunk")
        x, y = ax.lines[0].get_data()
        assert_array_equal(x, np.arange(5))
        assert_array_equal(y, values)

    def test_multiple_series_and_retrain_marks(self):
        _, ax = plot_metric_over_time(
            {"cumulative": [0.5, 0.6], "rolling": [0.4, 0.7]}, marks=[1])
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["cumulative", "rolling", "retrain"])

    def test_draw_into_existing_axes(self):
        fig, axes = plt.subplots(1, 2)
        returned_fig, returned_ax = plot_metric_over_time([1, 2], ax=axes[1])
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, axes[1])
        self.assertEqual(len(axes[0].lines), 0)

    def test_invalid_values(self):
        for bad in ([], [[1, 2]], [1, np.inf], {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plot_metric_over_time(bad)


class TestCompareModels(_CloseFigures):
    def test_two_lines_with_labels(self):
        _, ax = compare_models([0.7, 0.8, 0.9], [0.6, 0.9],
                               labels=["buffer 100", "buffer 500"])
        self.assertEqual(len(ax.lines), 2)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["buffer 100", "buffer 500"])

    def test_bad_labels(self):
        with self.assertRaises(ValueError):
            compare_models([1], [2], labels=["only one"])
        with self.assertRaises(ValueError):
            compare_models([1], [2], labels=["a", 2])

if __name__ == "__main__":
    unittest.main()
