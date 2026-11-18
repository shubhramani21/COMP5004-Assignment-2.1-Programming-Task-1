# NumCompute Stream

A streaming, decision-tree-based machine learning framework built from scratch with **NumPy** and **matplotlib** for COMP 5004 Assignment 2.2. It extends the group's NumCompute package (Assignment 2.1) so that every component learns from data arriving in chunks:

- a **decision tree classifier** (Gini or entropy) that streams via **buffered retraining**;
- **streaming preprocessing**, **statistics** and **metrics** that update with each chunk;
- a **pipeline** with `partial_fit`, a **stream trainer** that logs metrics, timing and memory per chunk;
- a **visualisation** module, a **demo notebook** and **benchmarks**.

No scikit-learn, pandas or other ML/data libraries are used.

## Requirements and installation

- Python **3.10 or later** (developed and tested with Python 3.14, NumPy 2.5, matplotlib 3.11).
- NumPy ≥ 1.24 and matplotlib ≥ 3.7, installed automatically.
- Jupyter, only for the demo notebook.

From the repository root (the folder containing `pyproject.toml`):

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install notebook ipykernel   # for the demo
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install notebook ipykernel
```

Check the installation:

```bash
python -c "import numcompute_stream.tree as t; print(t.DecisionTreeClassifier(max_depth=5, buffer_size=500))"
```

## Quick start

### A streaming decision tree

```python
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

tree = DecisionTreeClassifier(max_depth=5, buffer_size=200)
for start in range(0, 1000, 100):                 # chunks of 100 rows
    tree.partial_fit(X[start:start + 100], y[start:start + 100])
    print(start, tree.buffer_count_, tree.n_retrains_)

print(tree.predict(X[:5]), tree.score(X, y))
```

Each `partial_fit` call adds the chunk to an internal buffer. When the buffer holds at least `buffer_size` samples, the tree is **retrained from scratch** on the buffer and the buffer is cleared. Between retrains, `predict` uses the most recently trained tree. Before the first retrain, `predict` raises a `RuntimeError`.

### A streaming pipeline with logging and plots

```python
from numcompute_stream.io import iter_csv_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import plot_metric_over_time

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),   # running column means
    ("scale", StandardScaler()),                  # running Welford mean/variance
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

def stream():
    for chunk in iter_csv_chunks("demo/data/sensor_stream.csv",
                                 chunk_size=200, skip_header=1):
        yield chunk[:, :5], chunk[:, 5].astype(int)

trainer = StreamTrainer(pipe, rolling_window=500, auc_pos_label=1, verbose=True)
trainer.run(stream())                 # test-then-train on every chunk
print(trainer.summary())

plot_metric_over_time(trainer.get_metric("cumulative_accuracy"),
                      title="Cumulative accuracy", ylabel="Accuracy",
                      save_path="accuracy.png")
```

`StreamTrainer.process_chunk` (used by `run`) first **scores** each chunk with the current model and then **trains** on it, so every reported score is on data the model has not yet seen. Use `fit_chunk` and `score_chunk` to control the two steps separately.

## Package overview

All modules are in `numcompute_stream/`. Modules copied from the group package keep their batch API; the streaming additions are listed here.

| Module | Streaming components |
| --- | --- |
| `tree` | `DecisionTreeClassifier(max_depth, min_samples_split, max_features, criterion, buffer_size, min_samples_leaf, random_state)` with `fit`, buffered `partial_fit`, `predict`, `predict_proba`, `score`, `get_depth`, `get_n_leaves`. The split search is vectorised (sorted cumulative class counts). A loop version is kept for testing and benchmarking. |
| `stream` | `StreamTrainer` with `fit_chunk`, `score_chunk`, `process_chunk`, `run`, `get_metric`, `summary`, `confusion_matrix`. It logs per-chunk and cumulative accuracy, macro F1, rolling accuracy, AUC, fit/predict time, retrains, buffer size and memory footprint. `estimate_nbytes` gives the memory estimate. |
| `preprocessing` | `partial_fit` on `StandardScaler` (Chan/Welford merge), `MinMaxScaler` (running min/max), `SimpleImputer` (new `strategy="mean"` with running means) and `OneHotEncoder` (categories added as they appear). |
| `stats` | `StreamingQuantile` (P² algorithm, O(1) memory), `StreamingHistogram` (fixed edges, optional sliding window over the last N chunks), `StreamingSummary.update_stats(X_chunk)` (per-column count, mean, variance, min, max, quantiles). The existing `StreamingStats` is unchanged. |
| `metrics` | `StreamingAccuracy`, `StreamingPrecision`, `StreamingRecall`, `StreamingF1`, `StreamingConfusionMatrix`, `StreamingAUC` (exact), each with `update` / `reset` / `result`, plus `RollingWindowMetric` for any of them. |
| `pipeline` | `Pipeline.partial_fit`, `Pipeline.predict_proba`, `Pipeline.score`, `FeatureUnion.partial_fit`. Steps without `partial_fit` are fitted on the first chunk only. |
| `visualise` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix`. Each returns `(fig, ax)` and accepts `save_path=`, `show=` and `ax=`. |
| `io` | `iter_csv_chunks` reads a CSV lazily, chunk by chunk (from the group package). |

Every public class and function has a docstring describing parameters, shapes and errors, e.g. `help(DecisionTreeClassifier.partial_fit)`.

## Demo notebook

`demo/stream_demo.ipynb` runs the whole workflow on `demo/data/sensor_stream.csv`: 6,000 synthetic machine-sensor readings with about 3% missing values and a concept drift at row 4,000. It:

1. loads the CSV in chunks with `iter_csv_chunks`;
2. computes streaming statistics with `update_stats`;
3. trains an imputer → scaler → tree pipeline with `partial_fit`, printing a log line per chunk;
4. plots accuracy, AUC, fit time and memory over time with `visualise`;
5. shows predictions vs ground truth for the latest chunk and the confusion matrix;
6. compares `buffer_size` = 200, 600 and 2000.

```bash
cd demo
jupyter notebook stream_demo.ipynb
```

Then choose **Run All**. The notebook is stored with its outputs, so it can also be read without running it. To regenerate the dataset, run `python demo/generate_stream_data.py`.

## Tests

Run all tests from the repository root (install with `pip install -e .` first; one group test checks the installed package version):

```bash
python -m unittest discover -s tests -v
```

The streaming framework adds **139 tests** (the group package tests are also included in the suite):

| File | Tests | Covers |
| --- | ---: | --- |
| `test_tree.py` | 37 | impurity, loop vs vectorised split search, validation, fit/predict, buffered streaming |
| `test_streaming_metrics.py` | 25 | streaming metrics vs batch functions, AUC vs brute force, rolling windows |
| `test_streaming_preprocessing.py` | 20 | `partial_fit` equals a full fit, numerical stability, NaN/zero-variance chunks |
| `test_streaming_stats.py` | 17 | P² accuracy, histogram windows, per-column summaries, docstring examples |
| `test_stream.py` | 12 | test-then-train schedule, logging, memory, validation |
| `test_visualise.py` | 11 | plot contents, saving, invalid inputs (headless backend) |
| `test_streaming_pipeline.py` | 10 | chained `partial_fit`, imputation of NaN streams, feature unions |
| `test_stream_benchmarks.py` | 7 | loop baselines match vectorised code, benchmark helpers |

Run only the streaming tests:

```bash
python -m unittest tests.test_tree tests.test_streaming_metrics tests.test_streaming_preprocessing \
    tests.test_streaming_stats tests.test_streaming_pipeline tests.test_stream \
    tests.test_visualise tests.test_stream_benchmarks
```

Edge cases covered include NaN and infinite inputs, empty chunks, single-class chunks, zero-variance features and chunks, ties (in thresholds, features and class probabilities), labels that first appear in later chunks, prediction before the first retrain, feature-count changes between chunks, and huge offsets in variance computations.

## Benchmarks

```bash
python -m benchmark.run_stream_benchmarks          # about 45 s
python -m benchmark.run_stream_benchmarks --quick  # about 10 s
```

Results are written to a new `benchmark/results/stream_<timestamp>/` folder (`vectorisation.csv`, `buffer_size.csv`, `environment.json` and plots). The recorded run is [`stream_20261112T160811Z`](benchmark/results/stream_20261112T160811Z/) (Python 3.14.8, NumPy 2.5.3, Linux).

**(a) Loop vs vectorised.** Outputs are checked to be identical before timing. Median of 5 runs (2 for full tree training):

| Component | n | Loop | Vectorised | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| Split search (one feature) | 2000 | 122 ms | 0.71 ms | 172× |
| Full tree training (depth 6) | 1000 | 871 ms | 23 ms | 39× |
| Tree prediction | 2000 | 6.1 ms | 0.95 ms | 6.4× |
| Scaler `partial_fit` | 2000 | 6.3 ms | 1.2 ms | 5.2× |
| Confusion matrix update | 2000 | 0.80 ms | 0.58 ms | 1.4× |

At 100 samples, prediction and confusion-matrix updates are no faster vectorised (0.7× and 1.0×) because fixed NumPy and validation overhead dominates.

**(b) Buffer size.** 20,000 samples in chunks of 100, mean of 3 seeds, with and without concept drift halfway:

| `buffer_size` | Accuracy (stationary) | Accuracy (drift) | Retrains | Mean retrain time |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 77.2% | 75.2% | 200 | 5.8 ms |
| 250 | 81.6% | 79.5% | 66 | 12.1 ms |
| 500 | 83.2% | 80.7% | 40 | 16.6 ms |
| 1000 | 85.4% | 81.8% | 20 | 23.0 ms |
| 2000 | 86.8% | 80.5% | 10 | 31.5 ms |
| 4000 | 87.8% | 73.3% | 5 | 45.2 ms |

Larger buffers give more accurate trees on stationary data but react slowly to drift. Each retrain costs more, but there are fewer retrains. Larger buffers also start predicting later, so they score fewer samples. Measured times depend on hardware and system load.

## Design notes and limitations

- **Missing values:** the tree rejects NaN and infinity in `X` with a clear `ValueError`. Put `SimpleImputer` before it in a pipeline.
- **Ties:** split thresholds are midpoints between distinct sorted values. Ties in score go to the smallest threshold, then the lowest feature index. Probability ties predict the first class in `classes_`.
- **Zero-gain splits** are allowed (needed for XOR-like data). Growth always stops because both children are non-empty.
- **Buffered retraining** forgets earlier buffers: classes missing from the latest buffer are not predicted until they reappear.
- **Pipeline statistics evolve:** a scaler inside a pipeline updates with each chunk, so buffered rows are scaled with slightly older statistics than the newest rows.
- **`OneHotEncoder.partial_fit`** grows the output width as new categories appear. Fit it on all categories first when a downstream model needs a fixed width.
- **Approximations:** `StreamingQuantile` (P²) and `StreamingHistogram.quantile` are approximate. P² processes values in a Python loop because the algorithm is sequential. `StreamingAUC` is exact and stores one row per distinct score.
- **Memory** reported by `StreamTrainer` is an estimate of the objects referenced by the model, not process memory.

## Repository contents

| Path | Purpose |
| --- | --- |
| `numcompute_stream/` | Streaming framework package (this assignment) |
| `numcompute/` | Original group package from Assignment 2.1 (unchanged) |
| `tests/` | Unit tests for both packages |
| `demo/stream_demo.ipynb` | Streaming demo notebook |
| `demo/generate_stream_data.py`, `demo/data/sensor_stream.csv` | Demo dataset and its generator |
| `benchmark/run_stream_benchmarks.py`, `benchmark/stream_baselines.py` | Streaming benchmarks and loop baselines |
| `benchmark/results/stream_*/` | Recorded benchmark results and plots |
| `demo/quickstart.ipynb`, `benchmark/run_benchmarks.py` | Assignment 2.1 demo and benchmarks |

## Base package

`numcompute_stream` starts from the group's NumCompute package (Assignment 2.1: I/O, preprocessing, sorting and ranking, statistics, metrics, numerical differentiation, pipelines and benchmarking). The original package and its documentation are unchanged on the `main` branch.
