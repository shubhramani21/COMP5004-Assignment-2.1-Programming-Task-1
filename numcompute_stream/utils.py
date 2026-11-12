"""Shared validation and numerical utilities."""

from numbers import Real
import numpy as np

def _as_float_array(x, *, name="x", ndim=None, finite=False):
    """Validate real numeric input and return an independent float64 array.

    Parameters
    ----------
    x : array-like
        Real numeric input. Boolean, complex, string and object arrays
        are not supported.
    name : str
        Input name used in error messages.
    ndim : int or None
        Required number of dimensions, or None for any shape.
    finite : bool
        If True, reject NaN and positive/negative infinity.

    Returns
    -------
    numpy.ndarray
        Independent float64 array preserving the input shape.

    Raises
    ------
    TypeError
        If the input contains unsupported types.
    ValueError
        If the shape is invalid or finite=True and values are non-finite.

    Notes
    -----
    Time: O(n). Additional space: O(n), where n is the element count.
    """

    try:
        array = np.asarray(x)
    except ValueError as exc:
        raise ValueError(
            f"{name} must have a regular rectangular shape."
        ) from exc

    # Check Python sequences before NumPy can silently convert mixed
    # values such as [True, 2] into an integer array.
    if isinstance(x, (list, tuple)):
        try:
            elements = np.asarray(x, dtype=object)
        except ValueError as exc:
            raise ValueError(
                f"{name} must have a regular rectangular shape."
            ) from exc

        for value in elements.flat:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Real
            ):
                raise TypeError(
                    f"{name} must contain only real numeric values."
                )


    # i: signed integer; u: unsigned integer; f: floating point.
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must contain real numbers, not boolean, "
            "complex, string or object data."
        )

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D; got {array.ndim}D."
        )

    result = array.astype(np.float64, copy=True)

    if finite and not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return result


def _validate_integer(value, *, name, minimum=None):
    """Validate a Python/NumPy integer and return a Python int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}; got {value}."
        )

    return value

def sigmoid(x):
    """Compute the sigmoid function without exponential overflow.

    Parameters
    ----------
    x : array-like
        Real numeric input of any shape, including a scalar.
        Boolean, complex, string and object data are rejected.

    Returns
    -------
    numpy.ndarray
        Float64 result with the same shape as the input.
        Scalar input produces a zero-dimensional array.
        Empty input produces an empty array.

    Raises
    ------
    TypeError
        If the input contains unsupported types.
    ValueError
        If the input has an irregular shape.

    Notes
    -----
    NaN propagates. Positive infinity returns 1, and negative
    infinity returns 0. The caller's input is not modified.

    Time: O(n). Additional space: O(n).
    """
    array = _as_float_array(x)

    result = np.empty_like(array)
    positive = array >= 0

    # Tiny exponentials may underflow to zero, which is acceptable.
    with np.errstate(under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))

        exp_x = np.exp(array[~positive])
        result[~positive] = exp_x / (1.0 + exp_x)

    return result

def _validate_axis(axis, ndim):
    """Validate one integer axis and return its non-negative index.

    Raise TypeError for unsupported types and ValueError for an
    out-of-range axis. Boolean values are rejected.

    Time: O(1). Additional space: O(1).
    """
    axis = _validate_integer(axis, name="axis")

    if not -ndim <= axis < ndim:
        raise ValueError(f"axis must satisfy {-ndim} <= axis < {ndim}; got {axis}.")

    return axis % ndim


def softmax(x, axis=-1):
    """Compute stable softmax probabilities along one axis.

    Parameters
    ----------
    x : array-like
        Real numeric input with at least one dimension.
        Converted to float64 without modifying the input.
        Boolean, complex, string and object data are rejected.
    axis : int, default=-1
        Axis along which probabilities are normalized.
        Negative axes are supported. Boolean values are rejected.
        The selected axis must have nonzero length.

    Returns
    -------
    numpy.ndarray
        Independent float64 array with the same shape as x.
        For slices containing finite values and no NaN or positive
        infinity, probabilities sum to 1 within floating-point
        precision. Negative infinity receives probability 0.

    Raises
    ------
    TypeError
        If x contains unsupported types or axis is not an integer.
    ValueError
        If x has an irregular shape, is scalar, axis is out of
        range, or the selected axis is empty.

    Notes
    -----
    Subtracts the slice maximum before exponentiation to avoid
    exponential overflow. Very small probabilities may underflow
    to zero.

    Special values are handled separately within each slice:
    - Any NaN makes the entire output slice NaN.
    - Positive infinities share probability equally; all other
      entries receive 0, unless NaN is present.
    - A slice containing only negative infinity returns all NaN.

    Zero-length dimensions outside the selected axis are allowed.
    The caller's input is not modified.

    Time complexity: O(N).
    Additional space complexity: O(N), including input conversion,
    output and temporary arrays, where N is the input element count.
    """
    array = _as_float_array(x)

    if array.ndim == 0:
        raise ValueError("x must have at least one dimension.")

    axis = _validate_axis(axis, array.ndim)

    if array.shape[axis] == 0:
        raise ValueError("The softmax axis must not be empty.")

    has_nan = np.any(np.isnan(array), axis=axis, keepdims=True)
    positive_inf = np.isposinf(array)
    inf_count = np.sum(positive_inf, axis=axis, keepdims=True)
    all_negative_inf = np.all(np.isneginf(array), axis=axis, keepdims=True)

    regular = (~has_nan & (inf_count == 0) & ~all_negative_inf)

    # Special slices use temporary zeros to avoid inf - inf.
    # Their final results are assigned separately below.
    safe_array = np.where(regular, array, 0.0)
    maximum = np.max(safe_array, axis=axis, keepdims=True)

    # Extremely separated finite values may subtract to -inf.
    # Their exponential is correctly represented as zero.
    with np.errstate(over="ignore"):
        shifted = safe_array - maximum

    with np.errstate(under="ignore"):
        exp_values = np.exp(shifted)
        result = exp_values / np.sum(exp_values, axis=axis, keepdims=True)

    # Use a denominator of at least 1 to avoid division by zero
    # in slices that contain no positive infinity.
    inf_probabilities = (positive_inf / np.maximum(inf_count, 1))
    result = np.where(inf_count > 0, inf_probabilities, result)

    # NaN takes precedence over positive infinity.
    result = np.where(has_nan | all_negative_inf, np.nan, result)

    return result

def logsumexp(x, axis=None, keepdims=False):
    """Compute log(sum(exp(x))) using a stable max-shift calculation.

    Parameters
    ----------
    x : array-like
        Real numeric input. Boolean, complex, string and object
        data are rejected.
    axis : int or None, default=None
        Reduce over all elements if None, otherwise over one axis.
        Negative axes are supported.
        Scalar input supports only axis=None.
    keepdims : bool, default=False
        If True, retain reduced dimensions with size 1.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Float64 result. Reduced axes are removed unless keepdims=True.
        Return a scalar when the output has no dimensions.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If the input is irregular or axis is out of range.

    Notes
    -----
    NaN propagates within each reduction slice.
    Positive infinity returns infinity unless NaN is present.
    All-negative-infinity and empty slices return negative infinity.
    The caller's input is not modified.

    Time: O(n + r). Additional space: O(n + r), where n is the
    input element count and r is the output element count.
    """
    array = _as_float_array(x)

    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be a boolean.")

    if axis is not None:
        axis = _validate_axis(axis, array.ndim)

    # initial allows empty slices to have a maximum of -inf.
    maximum = np.max(array, axis=axis, keepdims=True, initial=-np.inf,)

    regular = np.isfinite(maximum)
    base = np.where(regular, maximum, 0.0)

    # Avoid undefined operations such as inf - inf.
    # Special slices receive their final values separately.
    safe_array = np.where(regular, array, -np.inf)

    # A very large negative difference may become -inf.
    with np.errstate(over="ignore"):
        shifted = safe_array - base

    with np.errstate(under="ignore"):
        total = np.sum(np.exp(shifted), axis=axis, keepdims=True,)

    # Leave zero totals as -inf without evaluating log(0).
    log_total = np.full_like(total, -np.inf)
    np.log(total, out=log_total, where=total > 0)

    result = np.where(regular, base + log_total, maximum,)

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return result[()] if result.ndim == 0 else result

def euclidean_distances(X, Y=None):
    """Compute pairwise Euclidean distances between sample rows.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite real numeric input with at least one feature.
    Y : array-like of shape (m_samples, n_features) or None
        Finite real numeric input with the same feature count.
        If None, compare X with itself.

    Returns
    -------
    numpy.ndarray of shape (n_samples, m_samples)
        Float64 distance matrix. Zero samples are allowed.
        Distances beyond the float64 range return positive infinity.

    Raises
    ------
    TypeError
        If inputs contain unsupported data types.
    ValueError
        If inputs are not 2D, contain non-finite values,
        have zero features or have different feature counts.

    Notes
    -----
    Direct coordinate differences avoid cancellation from subtracting
    large squared norms. Hypot reduction avoids unnecessary overflow
    or underflow from explicitly squaring differences.

    Computation uses blocks of at most 64 rows from each input.
    Neither input is modified.

    For n and m samples with d features:
    Time: O((n + m) * d + n * m * d).
    Additional space, including copies and output:
    O((n + m) * d + n * m + b * b * d), where b = 64.
    """
    X_array = _as_float_array(X, name="X", ndim=2, finite=True)

    if Y is None:
        Y_array = X_array
    else:
        Y_array = _as_float_array(Y, name="Y", ndim=2, finite=True)

    if X_array.shape[1] == 0 or Y_array.shape[1] == 0:
        raise ValueError("X and Y must have at least one feature.")

    if X_array.shape[1] != Y_array.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    n_samples = X_array.shape[0]
    m_samples = Y_array.shape[0]

    result = np.empty((n_samples, m_samples), dtype=np.float64)

    block_size = 64

    for i in range(0, n_samples, block_size):
        X_block = X_array[i:i + block_size]

        for j in range(0, m_samples, block_size):
            Y_block = Y_array[j:j + block_size]

            # Unrepresentable distances become inf; tiny rounding underflow is permitted.
            with np.errstate(over="ignore", under="ignore"):
                difference = (X_block[:, None, :]- Y_block[None, :, :])

                result[i:i + block_size,j:j + block_size,] = np.hypot.reduce(difference, axis=2)

    return result

def batch_iter(X,batch_size=32,shuffle=False,random_state=None,drop_last=False,):
    """Yield independent array batches along the first dimension.

    Parameters
    ----------
    X : array-like
        Input with at least one dimension. Preserve its dtype.
    batch_size : int, default=32
        Positive batch size. Boolean values are rejected.
    shuffle : bool, default=False
        Shuffle sample indices using a local random generator.
    random_state : int or None, default=None
        Non-negative integer seed. Boolean values are rejected.
        Validated but unused when shuffle=False.
    drop_last : bool, default=False
        Discard the final incomplete batch if True.

    Yields
    ------
    numpy.ndarray
        Independent array copies with shape
        (batch_length, *X.shape[1:]).
        Empty input yields no batches.

    Raises
    ------
    TypeError
        If parameter types are unsupported.
    ValueError
        If X is scalar or irregular, batch_size is not positive,
        or random_state is negative.

    Notes
    -----
    Validation occurs when iteration starts.
    The input and global NumPy random state are not modified.
    Do not modify X while consuming the iterator.
    Object arrays copy references, not the nested Python objects.

    For an ndarray with n rows and d elements per row:
    Full iteration takes O(n + n*d) time.
    Additional space is O(b*d) without shuffling and O(n + b*d)
    with shuffling, where b = min(batch_size, n), assuming the
    consumer does not retain previous batches.
    Converting non-array input may require O(n*d) extra space.
    """
    batch_size = _validate_integer(batch_size, name="batch_size", minimum=1)

    for name, value in (("shuffle", shuffle),("drop_last", drop_last),):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean.")

    if random_state is not None:
        random_state = _validate_integer(random_state,name="random_state",minimum=0,)

    array = np.asarray(X)

    if array.ndim == 0:
        raise ValueError("X must have at least one dimension.")

    n_samples = array.shape[0]

    if drop_last:
        stop = n_samples - (n_samples % batch_size)
    else:
        stop = n_samples

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)

    for start in range(0, stop, batch_size):
        end = min(start + batch_size, stop)

        if shuffle:
            # Integer-array indexing creates an independent copy.
            yield array[indices[start:end]]
        else:
            # Slicing creates a view, so explicitly copy it.
            yield array[start:end].copy()