"""Decision tree classifier with buffered streaming support.

Implements a depth-limited CART-style classification tree (Gini or
entropy) using only NumPy. Streaming is supported through buffered
retraining: ``partial_fit`` appends chunks to an internal buffer and
retrains the tree from scratch once the buffer holds at least
``buffer_size`` samples.

Example
-------
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> for X_chunk, y_chunk in stream:
...     tree.partial_fit(X_chunk, y_chunk)
>>> y_pred = tree.predict(X_test)
"""

from numbers import Integral, Real
import time

import numpy as np

from .utils import _as_float_array, _validate_integer


# ---------------------------------------------------------------------------
# Impurity functions
# ---------------------------------------------------------------------------

def _proportions(counts):
    """Return class proportions and node sizes for count arrays.

    Empty nodes (zero total count) get all-zero proportions instead of
    NaN, so callers never divide by zero.
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=-1, keepdims=True)
    proportions = counts / np.maximum(totals, 1.0)
    return proportions, totals[..., 0]


def _gini(counts):
    """Gini impurity from class counts.

    Parameters
    ----------
    counts : array-like, shape (..., n_classes)
        Class counts. Leading dimensions allow many candidate splits to be
        evaluated at once (vectorised).

    Returns
    -------
    numpy.ndarray, shape (...)
        Gini impurity ``1 - sum(p_k ** 2)``. Empty nodes return 0.

    Examples
    --------
    >>> float(_gini([2, 2]))
    0.5
    """
    proportions, totals = _proportions(counts)
    impurity = 1.0 - np.sum(proportions ** 2, axis=-1)
    # Empty nodes have all-zero proportions, which would give 1.0.
    return np.where(totals > 0, impurity, 0.0)


def _entropy(counts):
    """Entropy impurity (in bits) from class counts.

    Parameters
    ----------
    counts : array-like, shape (..., n_classes)

    Returns
    -------
    numpy.ndarray, shape (...)
        ``-sum(p_k * log2(p_k))`` using the convention ``0 * log 0 = 0``.
        Empty nodes return 0.

    Examples
    --------
    >>> float(_entropy([2, 2]))
    1.0
    """
    proportions, _ = _proportions(counts)
    # Only take the log where p > 0; elsewhere the term is defined as 0.
    log_p = np.zeros_like(proportions)
    np.log2(proportions, out=log_p, where=proportions > 0)
    # Adding 0.0 turns any -0.0 into 0.0.
    return -np.sum(proportions * log_p, axis=-1) + 0.0


_CRITERIA = {"gini": _gini, "entropy": _entropy}


# ---------------------------------------------------------------------------
# Split search
# ---------------------------------------------------------------------------

def _midpoint(lower, upper):
    """Threshold strictly between two distinct sorted values.

    Uses ``lower + (upper - lower) / 2`` to avoid overflow, and falls back
    to ``lower`` when floating-point rounding makes the midpoint equal to
    ``upper`` (adjacent floats), so ``x <= threshold`` still separates them.
    """
    threshold = lower + (upper - lower) / 2.0
    if threshold >= upper:
        threshold = lower
    return float(threshold)


def _sorted_feature(x, y_encoded):
    """Sort one feature column and its labels together (stable sort)."""
    order = np.argsort(x, kind="stable")
    return x[order], y_encoded[order]


def _best_split_feature(x, y_encoded, n_classes, impurity_fn,
                        min_samples_leaf=1):
    """Find the best threshold for a single feature (vectorised).

    The feature is sorted once, labels are one-hot encoded, and a cumulative
    sum gives the left-child class counts for every split position at once.
    Right-child counts are ``total - left``. All candidate scores are then
    computed in one vectorised impurity call.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_samples,)
        One feature column (finite values).
    y_encoded : numpy.ndarray of int, shape (n_samples,)
        Labels encoded as ``0 .. n_classes - 1``.
    n_classes : int
    impurity_fn : callable
        ``_gini`` or ``_entropy``.
    min_samples_leaf : int, default=1
        Minimum number of samples each child must receive.

    Returns
    -------
    tuple of (float, float) or None
        ``(weighted_child_impurity, threshold)`` for the best split, or
        None if the feature has no valid split (e.g. zero variance).
        Ties are broken by the smallest threshold.

    Notes
    -----
    Time O(n log n + n * n_classes); extra space O(n * n_classes).
    """
    n_samples = x.shape[0]
    if n_samples < 2 * min_samples_leaf or n_samples < 2:
        return None

    x_sorted, y_sorted = _sorted_feature(x, y_encoded)

    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_sorted] = 1.0
    cumulative = np.cumsum(one_hot, axis=0)

    # Row i of `left` = class counts when the first i + 1 sorted samples go
    # left. The last row is dropped because it would leave the right empty.
    left = cumulative[:-1]
    right = cumulative[-1] - left
    n_left = np.arange(1, n_samples, dtype=np.float64)
    n_right = n_samples - n_left

    scores = (n_left * impurity_fn(left)
              + n_right * impurity_fn(right)) / n_samples

    # A threshold can only sit between two different values, and each
    # child must satisfy min_samples_leaf.
    valid = ((x_sorted[:-1] < x_sorted[1:])
             & (n_left >= min_samples_leaf)
             & (n_right >= min_samples_leaf))
    if not np.any(valid):
        return None

    scores = np.where(valid, scores, np.inf)
    best = int(np.argmin(scores))  # first minimum -> smallest threshold
    return float(scores[best]), _midpoint(x_sorted[best], x_sorted[best + 1])


def _best_split_feature_loop(x, y_encoded, n_classes, impurity_fn,
                             min_samples_leaf=1):
    """Loop-based reference version of ``_best_split_feature``.

    Same inputs and outputs. Recounts the classes on each side of every
    candidate threshold with a Python loop, giving O(n^2) time. Used for
    correctness tests and the loop-vs-vectorised benchmark.
    """
    n_samples = x.shape[0]
    if n_samples < 2 * min_samples_leaf or n_samples < 2:
        return None

    x_sorted, y_sorted = _sorted_feature(x, y_encoded)

    best_score = np.inf
    best_index = None
    for i in range(n_samples - 1):
        if x_sorted[i] == x_sorted[i + 1]:
            continue
        n_left = i + 1
        n_right = n_samples - n_left
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        left_counts = np.bincount(y_sorted[:n_left], minlength=n_classes)
        right_counts = np.bincount(y_sorted[n_left:], minlength=n_classes)
        score = (n_left * impurity_fn(left_counts)
                 + n_right * impurity_fn(right_counts)) / n_samples

        if score < best_score:  # strict: keep the first (smallest) threshold
            best_score = float(score)
            best_index = i

    if best_index is None:
        return None
    return best_score, _midpoint(x_sorted[best_index],
                                 x_sorted[best_index + 1])


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    """A single tree node.

    Internal nodes use ``feature``, ``threshold``, ``left``, ``right``
    (samples with ``x[feature] <= threshold`` go left).
    Leaf nodes have ``left is None`` and are predicted from ``value``.

    Attributes
    ----------
    feature : int or None
    threshold : float or None
    left, right : _Node or None
    value : numpy.ndarray, shape (n_classes,)
        Class counts of the training samples that reached this node.
    n_samples : int
    impurity : float
    """

    __slots__ = ("feature", "threshold", "left", "right",
                 "value", "n_samples", "impurity")

    def __init__(self, value, n_samples, impurity,
                 feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """Depth-limited decision tree classifier with buffered streaming.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree (root has depth 0). None means grow until
        leaves are pure or another stopping rule applies.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    max_features : int, float, "sqrt", "log2" or None, default=None
        Number of features considered at each split. An int is a count, a
        float in (0, 1] is a fraction of the features, None uses all.
    criterion : {"gini", "entropy"}, default="gini"
    buffer_size : int, default=500
        Number of buffered samples that triggers a retrain in
        ``partial_fit``.
    min_samples_leaf : int, default=1
        Minimum number of samples each child of a split must receive.
    random_state : int or None, default=None
        Seed for feature subsampling (only used with ``max_features``).

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted class labels seen during the last training.
    n_features_in_ : int
        Number of features seen during training.
    root_ : _Node or None
        Root of the most recently trained tree.
    n_retrains_ : int
        Number of times the tree has been (re)trained.
    last_train_size_ : int
        Number of samples used in the most recent training.
    last_train_time_ : float
        Wall-clock seconds taken by the most recent training.

    Notes
    -----
    Input ``X`` must be finite: NaN or infinite values raise ``ValueError``.
    Use an imputer earlier in the pipeline to handle missing values.

    With buffered retraining each new tree is trained only on the latest
    buffer, so classes absent from that buffer are not predicted until
    they appear again.
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 max_features=None, criterion="gini", buffer_size=500,
                 min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_retrains_ = 0
        self.last_train_size_ = 0
        self.last_train_time_ = 0.0
        self._rng = None
        self._clear_buffer()

    def __repr__(self):
        return (f"DecisionTreeClassifier(max_depth={self.max_depth!r}, "
                f"min_samples_split={self.min_samples_split!r}, "
                f"max_features={self.max_features!r}, "
                f"criterion={self.criterion!r}, "
                f"buffer_size={self.buffer_size!r}, "
                f"min_samples_leaf={self.min_samples_leaf!r}, "
                f"random_state={self.random_state!r})")

    # -- validation ---------------------------------------------------------

    def _validate_params(self):
        """Check hyperparameters; raise TypeError/ValueError if invalid."""
        if self.max_depth is not None:
            _validate_integer(self.max_depth, name="max_depth", minimum=1)
        _validate_integer(self.min_samples_split,
                          name="min_samples_split", minimum=2)
        _validate_integer(self.min_samples_leaf,
                          name="min_samples_leaf", minimum=1)
        _validate_integer(self.buffer_size, name="buffer_size", minimum=1)

        if not isinstance(self.criterion, str):
            raise TypeError("criterion must be a string.")
        if self.criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(_CRITERIA)}; "
                f"got {self.criterion!r}."
            )

        mf = self.max_features
        if mf is None:
            pass
        elif isinstance(mf, str):
            if mf not in ("sqrt", "log2"):
                raise ValueError(
                    "max_features string must be 'sqrt' or 'log2'; "
                    f"got {mf!r}."
                )
        elif isinstance(mf, (bool, np.bool_)):
            raise TypeError("max_features must not be a boolean.")
        elif isinstance(mf, Integral):
            _validate_integer(mf, name="max_features", minimum=1)
        elif isinstance(mf, Real):
            if not 0.0 < float(mf) <= 1.0:
                raise ValueError(
                    "max_features as a float must be in (0, 1]; "
                    f"got {mf!r}."
                )
        else:
            raise TypeError(
                "max_features must be None, an int, a float, "
                "'sqrt' or 'log2'."
            )

        if self.random_state is not None:
            _validate_integer(self.random_state, name="random_state",
                              minimum=0)

    @staticmethod
    def _validate_X_y(X, y, *, n_features=None, allow_empty=False):
        """Validate training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) or (n_samples, 1)
        n_features : int or None
            Required number of columns, or None to accept any.
        allow_empty : bool
            Whether zero samples are accepted.

        Returns
        -------
        X : numpy.ndarray of float64, shape (n_samples, n_features)
        y : numpy.ndarray, shape (n_samples,)
        """
        X = _as_float_array(X, name="X", ndim=2, finite=True)

        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D with shape (n_samples,); got shape {y.shape}."
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples; "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        if y.dtype.kind == "f" and np.any(np.isnan(y)):
            raise ValueError("y must not contain NaN labels.")
        if y.dtype.kind == "O" and any(label is None for label in y):
            raise ValueError("y must not contain None labels.")

        if X.shape[0] == 0 and not allow_empty:
            raise ValueError("X and y must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but {n_features} were "
                "expected from earlier data."
            )
        return X, y

    def _validate_X_predict(self, X):
        """Validate prediction input against the trained tree."""
        self._check_is_fitted()
        X = _as_float_array(X, name="X", ndim=2, finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was trained "
                f"with {self.n_features_in_}."
            )
        return X

    def _resolve_max_features(self, n_features):
        """Convert ``max_features`` into an int in ``[1, n_features]``."""
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, Integral):
            if mf > n_features:
                raise ValueError(
                    f"max_features={mf} exceeds the number of features "
                    f"({n_features})."
                )
            return int(mf)
        return max(1, int(float(mf) * n_features))

    # -- training -----------------------------------------------------------

    def fit(self, X, y):
        """Train the tree from scratch on ``X``, ``y``.

        Any samples waiting in the streaming buffer are discarded, so the
        model reflects exactly the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Finite numeric features.
        y : array-like, shape (n_samples,)
            Class labels (any sortable type, e.g. ints or strings).

        Returns
        -------
        self
        """
        self._validate_params()
        X, y = self._validate_X_y(X, y)
        self._fit(X, y)
        self._clear_buffer()
        return self

    def _fit(self, X, y):
        """Train on already-validated data."""
        start = time.perf_counter()

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        y_encoded = y_encoded.ravel()
        self.n_features_in_ = X.shape[1]
        self._n_classes = self.classes_.shape[0]
        self._impurity_fn = _CRITERIA[self.criterion]
        self._n_split_features = self._resolve_max_features(X.shape[1])
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)

        self.root_ = self._build(X, y_encoded, 0, self._rng)

        self.n_retrains_ += 1
        self.last_train_size_ = X.shape[0]
        self.last_train_time_ = time.perf_counter() - start

    def _best_split(self, X, y_encoded, rng):
        """Search the (sub-sampled) features for the best split.

        Splits with zero impurity gain are allowed, because problems such
        as XOR need a "useless-looking" first split. Every split still
        produces two non-empty children, so growth always terminates.

        Returns
        -------
        tuple of (int, float) or None
            ``(feature_index, threshold)`` or None if no feature has a
            valid split. Ties go to the lowest feature index.
        """
        n_features = X.shape[1]
        if self._n_split_features < n_features:
            features = np.sort(rng.choice(n_features, self._n_split_features,
                                          replace=False))
        else:
            features = np.arange(n_features)

        best_score = np.inf
        best = None
        for feature in features:
            result = _best_split_feature(
                X[:, feature], y_encoded, self._n_classes,
                self._impurity_fn, self.min_samples_leaf,
            )
            if result is not None and result[0] < best_score:
                best_score = result[0]
                best = (int(feature), result[1])
        return best

    def _build(self, X, y_encoded, depth, rng):
        """Recursively grow the tree and return the root ``_Node``.

        A node becomes a leaf when it is pure, ``depth == max_depth``,
        it has fewer than ``min_samples_split`` samples, it cannot give
        each child ``min_samples_leaf`` samples, or no valid split exists.
        """
        counts = np.bincount(y_encoded, minlength=self._n_classes)
        n_samples = y_encoded.shape[0]
        node = _Node(
            value=counts.astype(np.float64),
            n_samples=int(n_samples),
            impurity=float(self._impurity_fn(counts)),
        )

        if (np.count_nonzero(counts) <= 1
                or (self.max_depth is not None and depth >= self.max_depth)
                or n_samples < self.min_samples_split
                or n_samples < 2 * self.min_samples_leaf):
            return node

        split = self._best_split(X, y_encoded, rng)
        if split is None:
            return node

        feature, threshold = split
        goes_left = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[goes_left], y_encoded[goes_left],
                                depth + 1, rng)
        node.right = self._build(X[~goes_left], y_encoded[~goes_left],
                                 depth + 1, rng)
        return node

    def partial_fit(self, X_chunk, y_chunk):
        """Add a chunk to the buffer; retrain when the buffer is full.

        Once the buffer holds at least ``buffer_size`` samples, the tree is
        retrained from scratch on the buffered data and the buffer is
        cleared. Between retrains, ``predict`` uses the most recently
        trained tree. Empty chunks are accepted and ignored.

        Parameters
        ----------
        X_chunk : array-like, shape (n_chunk, n_features)
        y_chunk : array-like, shape (n_chunk,)

        Returns
        -------
        self
        """
        raise NotImplementedError

    def _clear_buffer(self):
        """Empty the streaming buffer."""
        self._buffer_X = []
        self._buffer_y = []
        self._buffer_count = 0

    @property
    def buffer_count_(self):
        """Number of samples currently waiting in the buffer."""
        return self._buffer_count

    # -- prediction ---------------------------------------------------------

    def _check_is_fitted(self):
        """Raise ``RuntimeError`` if no tree has been trained yet."""
        if self.root_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier is not trained yet. Call fit(), or "
                "keep calling partial_fit() until buffer_size="
                f"{self.buffer_size} samples are buffered "
                f"(currently {self._buffer_count})."
            )

    def _leaf_values(self, X):
        """Return the leaf class counts reached by each row of ``X``.

        Rows are routed down the tree in groups: at each internal node one
        boolean mask splits the current index set, so the work per node is
        vectorised instead of looping over samples.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
        """
        values = np.empty((X.shape[0], self._n_classes), dtype=np.float64)
        stack = [(self.root_, np.arange(X.shape[0]))]
        while stack:
            node, indices = stack.pop()
            if indices.size == 0:
                continue
            if node.is_leaf:
                values[indices] = node.value
                continue
            goes_left = X[indices, node.feature] <= node.threshold
            stack.append((node.left, indices[goes_left]))
            stack.append((node.right, indices[~goes_left]))
        return values

    def predict_proba(self, X):
        """Class probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            Fraction of training samples of each class in the reached leaf.
            Columns follow the order of ``classes_``.
        """
        X = self._validate_X_predict(X)
        counts = self._leaf_values(X)
        # Every leaf holds at least one sample, so totals are positive.
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Labels from ``classes_``. Ties in leaf probabilities go to the
            class that comes first in ``classes_``.
        """
        raise NotImplementedError

    def score(self, X, y):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        raise NotImplementedError

    # -- introspection ------------------------------------------------------

    def get_depth(self):
        """Depth of the trained tree (a single leaf has depth 0)."""
        raise NotImplementedError

    def get_n_leaves(self):
        """Number of leaves in the trained tree."""
        raise NotImplementedError
