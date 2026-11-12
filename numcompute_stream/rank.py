"""One-based ranking and percentile calculation with explicit NaN policies."""
import numpy as np

from .sort_search import _as_numeric_array
from .stats import _validate_nan_policy, quantile
from .utils import _as_float_array


def rank(values, method="average", nan_policy="propagate"):
    """Assign one-based ascending ranks while preserving input positions.

    Parameters
    ----------
    values : array-like, shape (n,)
        Real integer or floating-point values. Infinity is allowed;
        NaN handling is controlled by nan_policy.
    method : {"average", "dense", "ordinal"}, default="average"
        "average" assigns each tied group the mean of its rank positions.
        "dense" assigns equal values the same rank, with no gaps between
        successive distinct values.
        "ordinal" assigns distinct ranks from 1 to the number of valid
        values, breaking ties by original input order.
    nan_policy : {"propagate", "omit", "raise"}, default="propagate"
        "propagate" returns an entirely NaN result if any input is NaN.
        "omit" ranks non-NaN values and leaves NaN positions unchanged.
        "raise" rejects any NaN input.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        Ranks aligned with the original input positions.
        Empty input produces an empty array. With nan_policy="omit",
        an all-NaN input produces an all-NaN result.

    Raises
    ------
    TypeError
        If values have an unsupported dtype or an option has an
        unsupported type.
    ValueError
        If values are not 1D, an option is unsupported, or NaN occurs
        when nan_policy="raise".

    Notes
    -----
    Uses stable sorting so ordinal ties follow input order.
    Time is O(n log n) and additional space is O(n).
    The input is not modified; the returned array is independent.

    Examples
    --------
    >>> rank([20, 10, 20]).tolist()
    [2.5, 1.0, 2.5]
    >>> rank([20, 10, 20], method="dense").tolist()
    [2.0, 1.0, 2.0]
    >>> rank([20, 10, 20], method="ordinal").tolist()
    [2.0, 1.0, 3.0]
    """
    array = _as_numeric_array(values)
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ("average", "dense", "ordinal"):
        raise ValueError("method must be 'average', 'dense', or 'ordinal'.")
    nan_policy = _validate_nan_policy(nan_policy)
    missing = np.isnan(array)
    if nan_policy == "raise" and np.any(missing):
        raise ValueError("values must not contain NaN when nan_policy='raise'.")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if (nan_policy == "propagate" and np.any(missing)) or array.size == 0:
        return result
    valid_indices = np.flatnonzero(~missing)
    data = array[valid_indices]
    if data.size == 0:
        return result
    order = np.argsort(data, kind="stable")
    ordered = data[order]
    if method == "ordinal":
        sorted_ranks = np.arange(1, data.size + 1, dtype=np.float64)
    else:
        starts = np.r_[True, ordered[1:] != ordered[:-1]]
        group = np.cumsum(starts) - 1
        if method == "dense":
            sorted_ranks = (group + 1).astype(np.float64)
        else:
            first = np.flatnonzero(starts)
            end = np.r_[first[1:], data.size]
            sorted_ranks = ((first + 1.0 + end) / 2.0)[group]
    result[valid_indices[order]] = sorted_ranks
    return result


def percentile(values, q, method="linear", axis=None,
               nan_policy="propagate", *, interpolation=None):
    """Compute percentiles using the project's quantile implementation.

    Parameters
    ----------
    values : array-like
        Input data accepted by numcompute.stats.quantile.
        Its validation and NaN-handling rules apply.
    q : real scalar or 1D array-like
        Finite percentile levels in the inclusive range [0, 100].
        Each level is divided by 100 before calling quantile.
    method : str, default="linear"
        Interpolation method accepted by numcompute.stats.quantile.
    axis : int or None, default=None
        Axis passed to quantile. None requests reduction over all values.
    nan_policy : {"propagate", "omit", "raise"}, default="propagate"
        NaN policy passed unchanged to quantile.
    interpolation : str or None, keyword-only, default=None
        Alias for method. When supplied, it replaces method="linear".
        If method is non-default, both arguments must agree.

    Returns
    -------
    scalar or numpy.ndarray
        Result returned by quantile for q / 100.
        Scalar q produces the reduced result; 1D q adds a leading
        percentile dimension, following quantile's shape conventions.

    Raises
    ------
    TypeError
        If q contains unsupported data, interpolation is not a string,
        or an argument fails quantile's type validation.
    ValueError
        If q is not scalar or 1D, is non-finite or outside [0, 100],
        method conflicts with interpolation, or an argument fails
        quantile's value validation.

    Notes
    -----
    This wrapper does not modify the input. Percentile validation and
    conversion add O(r) time and space for r requested levels.
    The remaining algorithm, complexity and numerical behavior are
    those of numcompute.stats.quantile.

    See Also
    --------
    numcompute.stats.quantile : Underlying quantile implementation.
    """
    if interpolation is not None:
        if not isinstance(interpolation, str):
            raise TypeError("interpolation must be a string.")
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        if method != "linear" and method != interpolation:
            raise ValueError("method and interpolation must agree.")
        method = interpolation
    levels = _as_float_array(q, name="q", finite=True)
    if levels.ndim > 1:
        raise ValueError("q must be a scalar or a 1D array.")
    if np.any((levels < 0) | (levels > 100)):
        raise ValueError("q must contain values between 0 and 100.")
    return quantile(values, levels / 100.0, axis=axis,
                    method=method, nan_policy=nan_policy)
