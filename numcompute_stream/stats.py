"""Descriptive statistics with explicit axis and NaN handling."""

import numpy as np

from .utils import _as_float_array, _validate_axis, _validate_integer


def _validate_nan_policy(nan_policy):
    """Validate and return the NaN handling policy."""
    if not isinstance(nan_policy, str):
        raise TypeError("nan_policy must be a string.")

    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError(
            "nan_policy must be 'propagate', 'omit', or 'raise'."
        )

    return nan_policy


def _prepare_reduction(X, axis, nan_policy):
    """Validate statistical input and normalize the reduction axis."""
    nan_policy = _validate_nan_policy(nan_policy)

    array = _as_float_array(X, name="X")

    if np.any(np.isinf(array)):
        raise ValueError("X must not contain positive or negative infinity.")

    if axis is not None:
        axis = _validate_axis(axis, array.ndim)

    nan_mask = np.isnan(array)

    if nan_policy == "raise" and np.any(nan_mask):
        raise ValueError(
            "X must not contain NaN when nan_policy='raise'."
        )

    return array, axis, nan_policy, nan_mask


def _scalar_or_array(result):
    """Return a NumPy scalar when the result has no dimensions."""
    result = np.asarray(result, dtype=np.float64)

    if result.ndim == 0:
        return result[()]

    return result


def mean(X, axis=None, nan_policy="propagate"):
    """Return the arithmetic mean over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input, converted to float64 without modifying X.
        Boolean, complex, string and object inputs are rejected.
        NaN is handled according to nan_policy; infinity is rejected.
    axis : int or None, default=None
        None reduces all elements. An integer reduces one axis.
        Negative axes are supported; tuple axes are not supported.
        Scalar input accepts only axis=None.
    nan_policy : str, default="propagate"
        "propagate": each slice containing NaN returns NaN.
        "omit": ignore NaN within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Float64 mean with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty slices and slices with no valid values return NaN,
        without deliberately emitting warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If the axis or policy is invalid, infinity is present,
        or NaN is present with nan_policy="raise".

    Notes
    -----
    Time complexity: O(N).
    Additional space complexity: O(N), including input conversion
    and temporary arrays, where N is the number of input elements.

    Computation uses float64. A sum outside its representable range
    may overflow even when all input values are finite.
    """
    array, axis, nan_policy, nan_mask = _prepare_reduction(
        X, axis, nan_policy
    )

    valid = ~nan_mask
    count = np.sum(valid, axis=axis)

    # Replacing NaN with zero makes summation safe for missing values.
    clean = np.where(valid, array, 0.0)
    total = np.sum(clean, axis=axis, dtype=np.float64)

    result = np.full(np.shape(total), np.nan, dtype=np.float64)

    # Empty or all-NaN slices remain NaN without division by zero.
    np.divide(total, count, out=result, where=count > 0)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)

def variance(X, axis=None, ddof=0, nan_policy="propagate"):
    """Return variance over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input converted to float64.
        Boolean, complex, string and object inputs are rejected.
        Infinity is rejected; NaN follows nan_policy.
    axis : int or None, default=None
        Reduce all elements or one integer axis.
        Negative axes are supported; tuple axes are rejected.
        Scalar input accepts only axis=None.
    ddof : int, default=0
        Non-negative integer subtracted from the valid sample count.
        Boolean values are rejected.
        Use 0 for population variance and 1 for sample variance.
    nan_policy : str, default="propagate"
        "propagate": return NaN for slices containing NaN.
        "omit": ignore NaN separately within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Float64 variance with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty slices, all-NaN slices and slices with valid_count <= ddof
        return NaN without deliberately emitting warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If ddof is negative, axis or policy is invalid,
        infinity is present, or nan_policy="raise" encounters NaN.

    Notes
    -----
    Uses a centered two-pass calculation instead of subtracting
    the squared mean from the mean of squared values.

    Time complexity: O(N).
    Additional space complexity: O(N), including input conversion
    and temporary arrays.

    Float64 range and precision still apply. Extreme finite input
    may overflow during summation, subtraction or squaring.
    """
    ddof = _validate_integer(ddof, name="ddof", minimum=0)

    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    valid = ~nan_mask

    # Retain reduced dimensions so the mean broadcasts against X.
    count_keepdims = np.sum(valid, axis=axis, keepdims=True)

    clean = np.where(valid, array, 0.0)
    total = np.sum(clean,axis=axis,dtype=np.float64,keepdims=True,)

    center = np.zeros_like(total, dtype=np.float64)
    np.divide(total,count_keepdims,out=center, where=count_keepdims > 0,)

    # Missing positions contribute zero to the squared deviations.
    deviations = np.zeros_like(array, dtype=np.float64)
    np.subtract(array, center, out=deviations, where=valid)

    squared_deviations = deviations * deviations
    squared_sum = np.sum(squared_deviations,axis=axis, dtype=np.float64,)

    count = np.sum(valid, axis=axis)
    enough_data = count > ddof

    # Use a harmless denominator for invalid slices.
    denominator = np.where(enough_data, count - ddof, 1)

    result = np.full(np.shape(squared_sum),np.nan,dtype=np.float64,)

    np.divide(squared_sum,denominator,out=result,where=enough_data,)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)


def std(X, axis=None, ddof=0, nan_policy="propagate"):
    """Return standard deviation over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input converted to float64; infinity is rejected.
        Boolean, complex, string and object inputs are rejected.
    axis : int or None, default=None
        Reduce all elements or one integer axis, including negative
        axes. Scalar input accepts only axis=None.
    ddof : int, default=0
        Non-negative integer; boolean values are rejected.
        The variance denominator is valid_count - ddof.
    nan_policy : str, default="propagate"
        One of "propagate", "omit" or "raise".

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Square root of variance, with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty slices, all-NaN slices and valid_count <= ddof
        return NaN without deliberately emitting warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If a parameter value is invalid, infinity is present,
        or nan_policy="raise" encounters NaN.

    Notes
    -----
    Reuses variance and its float64 numerical limits.
    If variance overflows, std also returns a non-finite result,
    even when the mathematical standard deviation is representable.

    Time complexity: O(N).
    Additional space complexity: O(N).
    """
    result = variance(X, axis=axis, ddof=ddof, nan_policy=nan_policy)

    return _scalar_or_array(np.sqrt(result))

def _extreme(X, axis, nan_policy, *, largest):
    """Compute a minimum or maximum after shared validation."""
    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    valid = ~nan_mask
    has_valid = np.any(valid, axis=axis)

    # The replacement cannot win against a valid finite value.
    fill = -np.inf if largest else np.inf
    clean = np.where(valid, array, fill)

    reducer = np.max if largest else np.min
    result = reducer(clean, axis=axis, initial=fill)

    # Empty and all-NaN slices return NaN instead of infinity.
    result = np.where(has_valid, result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)


def min(X, axis=None, nan_policy="propagate"):
    """Return the minimum over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input converted to float64 without modifying X.
        Boolean, complex, string, object and infinite input is rejected.
    axis : int or None, default=None
        Reduce all elements or one integer axis, including negative
        axes. Scalar input accepts only axis=None.
    nan_policy : str, default="propagate"
        "propagate": return NaN for slices containing NaN.
        "omit": ignore NaN within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Minimum with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty and all-NaN slices return NaN without warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If the axis or policy is invalid, infinity is present,
        or nan_policy="raise" encounters NaN.

    Notes
    -----
    Time complexity: O(N).
    Additional space complexity: O(N), including input conversion
    and temporary arrays.
    """
    return _extreme(X, axis, nan_policy, largest=False)


def max(X, axis=None, nan_policy="propagate"):
    """Return the maximum over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input converted to float64 without modifying X.
        Boolean, complex, string, object and infinite input is rejected.
    axis : int or None, default=None
        Reduce all elements or one integer axis, including negative
        axes. Scalar input accepts only axis=None.
    nan_policy : str, default="propagate"
        "propagate": return NaN for slices containing NaN.
        "omit": ignore NaN within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Maximum with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty and all-NaN slices return NaN without warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If the axis or policy is invalid, infinity is present,
        or nan_policy="raise" encounters NaN.

    Notes
    -----
    Time complexity: O(N).
    Additional space complexity: O(N), including input conversion
    and temporary arrays.
    """
    return _extreme(X, axis, nan_policy, largest=True)


def _safe_midpoint(lower, upper):
    """Average ordered finite endpoints without avoidable overflow."""
    lower, upper = np.broadcast_arrays(lower, upper)
    same_sign = np.signbit(lower) == np.signbit(upper)

    result = np.empty(lower.shape, dtype=np.float64)

    # Same-sign endpoints have a representable difference.
    result[same_sign] = (lower[same_sign] + (upper[same_sign] - lower[same_sign]) * 0.5)

    # Opposite-sign endpoints have a representable sum.
    opposite_sign = ~same_sign
    result[opposite_sign] = (lower[opposite_sign] + upper[opposite_sign]) * 0.5

    return result


def median(X, axis=None, nan_policy="propagate"):
    """Return the median over all values or one axis.

    Parameters
    ----------
    X : array-like
        Real numeric input converted to float64 without modifying X.
        Boolean, complex, string, object and infinite input is rejected.
    axis : int or None, default=None
        Reduce all elements or one integer axis, including negative
        axes. Scalar input accepts only axis=None.
    nan_policy : str, default="propagate"
        "propagate": return NaN for slices containing NaN.
        "omit": ignore NaN within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Median with the reduced axis removed.
        Return a scalar when no dimensions remain.
        Empty and all-NaN slices return NaN without warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If the axis or policy is invalid, infinity is present,
        or nan_policy="raise" encounters NaN.

    Notes
    -----
    Sorts each reduction slice. For even valid counts, averages
    the two middle values using an overflow-aware midpoint.

    Time complexity: O(N log L) for slice length L >= 2;
    O(N) for shorter slices.
    Additional space complexity: O(N), including input conversion,
    sorting and temporary arrays. Output storage is also required.
    """
    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    # Put the reduction dimension last for uniform indexing.
    if axis is None:
        values = array.reshape(-1)
    else:
        values = np.moveaxis(array, axis, -1)

    output_shape = values.shape[:-1]

    if values.shape[-1] == 0:
        return _scalar_or_array(np.full(output_shape, np.nan, dtype=np.float64))

    # NumPy places NaN after finite values in ascending order.
    ordered = np.sort(values, axis=-1, kind="mergesort")
    count = np.sum(~np.isnan(values), axis=-1)

    lower_index = np.maximum((count - 1) // 2, 0)
    upper_index = count // 2

    lower = np.take_along_axis(ordered, lower_index[..., None], axis=-1)[..., 0]

    upper = np.take_along_axis(ordered, upper_index[..., None], axis=-1)[..., 0]

    has_valid = count > 0

    # Use finite placeholders before restoring missing results.
    lower = np.where(has_valid, lower, 0.0)
    upper = np.where(has_valid, upper, 0.0)

    result = _safe_midpoint(lower, upper)
    result = np.where(has_valid, result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)

def _validate_quantile_parameters(q, method):
    """Validate quantile levels and interpolation method."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("linear", "lower", "higher", "midpoint"):
        raise ValueError("method must be 'linear', 'lower', 'higher', or 'midpoint'.")

    levels = _as_float_array(q, name="q", finite=True)

    if levels.ndim > 1:
        raise ValueError("q must be a scalar or a 1D array.")

    if np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("q must contain values between 0 and 1.")

    return levels, method


def _linear_interpolate(lower, upper, weight):
    """Interpolate ordered finite endpoints without avoidable overflow."""
    lower, upper, weight = np.broadcast_arrays(lower, upper, weight)

    same_sign = np.signbit(lower) == np.signbit(upper)
    result = np.empty(lower.shape, dtype=np.float64)

    # Same-sign endpoints have a representable difference.
    result[same_sign] = (lower[same_sign] + weight[same_sign] * (upper[same_sign] - lower[same_sign]))

    # Opposite-sign endpoints may have an overflowing difference.
    opposite_sign = ~same_sign
    result[opposite_sign] = ((1.0 - weight[opposite_sign]) * lower[opposite_sign] + weight[opposite_sign] * upper[opposite_sign])

    # Preserve exact endpoint values.
    result = np.where(weight == 0.0, lower, result)
    result = np.where(weight == 1.0, upper, result)

    return result


def quantile(values,q,axis=None, method="linear",nan_policy="propagate",):
    """Return quantiles over all values or one axis.

    Parameters
    ----------
    values : array-like
        Real numeric input converted to float64 without modification.
        Boolean, complex, string, object and infinite input is rejected.
    q : real scalar or 1D array-like
        Finite quantile levels in [0, 1].
        Boolean values are rejected. An empty 1D array is allowed.
    axis : int or None, default=None
        Reduce all elements or one integer axis, including negative
        axes. Scalar input accepts only axis=None.
    method : str, default="linear"
        One of "linear", "lower", "higher" or "midpoint".
    nan_policy : str, default="propagate"
        "propagate": return NaN for slices containing NaN.
        "omit": ignore NaN separately within each slice.
        "raise": reject any NaN before calculation.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Float64 quantiles.

        A scalar q adds no output dimension.
        A 1D q adds a leading dimension of length len(q).
        Remaining dimensions match the input with the reduced axis
        removed. With axis=None, no input dimensions remain.

        Return a scalar when no dimensions remain.
        Empty and all-NaN slices return NaN without warnings.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If q has invalid shape, range or non-finite values;
        method or axis is invalid; input contains infinity;
        or nan_policy="raise" encounters NaN.

    Notes
    -----
    Sort each slice and compute positions as q * (valid_count - 1).
    All requested quantiles share the same sorted data.

    Let N be the input size, L the reduction slice length,
    S the number of slices and Q the number of requested quantiles.
    Time complexity: O(N log L + Q*S) for L >= 2;
    O(N + Q*S) for shorter slices.
    Additional space complexity: O(N + Q*S + S + Q).

    Interpolation avoids avoidable overflow, but float64 rounding
    and underflow still limit numerical precision.
    """
    levels, method = _validate_quantile_parameters(q, method)

    array, axis, nan_policy, nan_mask = _prepare_reduction(values, axis, nan_policy)

    scalar_q = levels.ndim == 0
    levels = levels.reshape(-1)

    if axis is None:
        data = array.reshape(-1)
    else:
        data = np.moveaxis(array, axis, -1)

    output_shape = data.shape[:-1]
    slice_length = data.shape[-1]
    result_shape = (levels.size,) + output_shape

    if slice_length == 0:
        result = np.full(result_shape, np.nan, dtype=np.float64)
        if scalar_q:
            result = result[0]
        return _scalar_or_array(result)

    # Each row represents one reduction slice.
    rows = data.reshape(-1, slice_length)
    ordered = np.sort(rows, axis=1, kind="mergesort")
    count = np.sum(~np.isnan(rows), axis=1)

    # Empty slices temporarily use position zero.
    last_index = np.maximum(count - 1, 0)
    positions = levels[:, None] * last_index[None, :]

    lower_index = np.floor(positions).astype(np.intp)
    upper_index = np.ceil(positions).astype(np.intp)

    row_index = np.arange(rows.shape[0])[None, :]

    lower = ordered[row_index, lower_index]
    upper = ordered[row_index, upper_index]

    has_valid = count > 0

    # Avoid NaN arithmetic before restoring missing results.
    lower = np.where(has_valid[None, :], lower, 0.0)
    upper = np.where(has_valid[None, :], upper, 0.0)

    if method == "lower":
        result = lower
    elif method == "higher":
        result = upper
    elif method == "midpoint":
        result = _safe_midpoint(lower, upper)
    else:
        weight = positions - lower_index
        result = _linear_interpolate(lower, upper, weight)

    result = np.where(has_valid[None, :], result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis).reshape(-1)
        result = np.where(affected[None, :], np.nan, result)

    result = result.reshape(result_shape)

    if scalar_q:
        result = result[0]

    return _scalar_or_array(result)

def _histogram_edges(values, bins, range):
    """Validate or construct finite, strictly increasing bin edges."""
    if isinstance(bins, (int, np.integer, bool, np.bool_)):
        n_bins = _validate_integer(bins, name="bins", minimum=1)

        if range is not None:
            bounds = _as_float_array(range, name="range", finite=True)

            if bounds.shape != (2,):
                raise ValueError("range must contain exactly two finite values.")

            lower, upper = bounds

            if lower >= upper:
                raise ValueError("range must satisfy lower < upper.")

        elif values.size == 0:
            lower, upper = 0.0, 1.0

        else:
            lower = np.min(values)
            upper = np.max(values)

            if lower == upper:
                lower = lower - 0.5
                upper = upper + 0.5

        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError("Cannot construct distinct finite bin edges; "
                             "provide a suitable explicit range or edge array.")

        # Interpolate without forming a potentially overflowing
        # difference between opposite-sign extreme endpoints.
        weights = np.linspace(
            0.0, 1.0, n_bins + 1, dtype=np.float64
        )
        edges = _linear_interpolate(lower, upper, weights)

    else:
        edges = _as_float_array(bins, name="bins", finite=True)

        if edges.ndim == 0:
            raise TypeError("bins must be a positive integer or a 1D edge array.")

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Bin edges must be a 1D array with at least two values.")

        if range is not None:
            raise ValueError("range must be None when explicit bin edges are used.")

    # Compare neighbors directly rather than subtracting them.
    if (
        not np.all(np.isfinite(edges))
        or np.any(edges[1:] <= edges[:-1])
    ):
        raise ValueError("Bin edges must be finite and strictly increasing; "
                         "use fewer bins or a suitable explicit range or edge array.")

    return edges


def histogram(values, bins=10, range=None, nan_policy="omit"):
    """Count 1D numeric values in histogram bins.

    Parameters
    ----------
    values : array-like, shape (n,)
        Real numeric input converted to float64 without modification.
        Boolean, complex, string, object and infinite input is rejected.
    bins : int or 1D array-like, default=10
        Positive integer number of equal-width bins, or finite,
        strictly increasing edges with at least two entries.
        Boolean values are rejected.
    range : pair of finite real numbers or None, default=None
        Lower and upper bounds for integer bins, with lower < upper.
        Cannot be supplied with explicit bin edges.
    nan_policy : str, default="omit"
        "omit": ignore NaN values.
        "raise": reject any NaN.
        "propagate" is unsupported because counts are integers.

    Returns
    -------
    counts : numpy.ndarray, shape (n_bins,)
        Integer counts. Values outside the selected range are ignored.
    edges : numpy.ndarray, shape (n_bins + 1,)
        Independent float64 bin edges.

    Raises
    ------
    TypeError
        If input or parameter types are unsupported.
    ValueError
        If input is not 1D, parameters have invalid values or shapes,
        infinity is present, nan_policy="raise" encounters NaN,
        or distinct finite edges cannot be represented.

    Notes
    -----
    Each bin includes its left edge and excludes its right edge,
    except the final bin, which includes both edges.

    Empty data produces zero counts. With integer bins and no range,
    empty data uses (0, 1). Constant data tries (value-0.5, value+0.5);
    if that cannot produce distinct edges, an explicit range or edge
    array is required.

    Uses binary search to assign values to bins.
    For N values and B bins, time complexity is O(N log B + B)
    for B >= 2, and O(N + B) otherwise.
    Additional space complexity: O(N + B).
    """
    nan_policy = _validate_nan_policy(nan_policy)

    if nan_policy == "propagate":
        raise ValueError("histogram supports only nan_policy='omit' or 'raise'.")

    array, _, _, nan_mask = _prepare_reduction(values, None, nan_policy)

    if array.ndim != 1:
        raise ValueError("values must be a 1D array.")

    clean = array[~nan_mask]
    edges = _histogram_edges(clean, bins, range)
    n_bins = edges.size - 1

    # Include the final right endpoint; ignore out-of-range values.
    inside = ((clean >= edges[0]) & (clean <= edges[-1]))
    selected = clean[inside]

    indices = np.searchsorted(edges, selected, side="right") - 1

    # The final edge belongs to the last bin.
    indices = np.minimum(indices, n_bins - 1)

    counts = np.bincount(indices, minlength=n_bins)

    return counts, edges

class StreamingStats:
    """Accumulate statistics from independent 1D numeric batches.
    Stores count, mean and M2 without retaining previous batches.
    Uses the batch-merge equivalent of Welford's algorithm."""

    def __init__(self, nan_policy="propagate"):
        """Initialize an empty accumulator.

        Parameters
        ----------
        nan_policy : str, default="propagate"
            One of "propagate", "omit" or "raise".

        Raises
        ------
        TypeError
            If nan_policy is not a string.
        ValueError
            If nan_policy is unsupported.

        Complexity
        ----------
        O(1) time and space.
        """
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.reset()

    def reset(self):
        """Clear accumulated statistics and return self.

        Sets count=0, mean=NaN and M2=0.
        Preserves nan_policy and clears propagated NaN state.

        Complexity: O(1) time and space.
        """
        self.count = 0
        self.mean = np.float64(np.nan)
        self.M2 = np.float64(0.0)
        self._poisoned = False

        return self

    def update(self, values):
        """Merge a 1D numeric batch into the accumulated state.

        Parameters
        ----------
        values : array-like, shape (n,)
            Real numeric batch converted to float64 without modification.
            Boolean, complex, string, object and infinite input is rejected.
            Empty batches leave the state unchanged.

        Returns
        -------
        StreamingStats
            This accumulator, allowing chained updates.

        Raises
        ------
        TypeError
            If the input or nan_policy type is unsupported.
        ValueError
            If the batch is not 1D, nan_policy is invalid, infinity is
            present, NaN violates the "raise" policy, or finite-data
            calculations exceed supported float64 limits.

        Notes
        -----
        Validation and finite-data calculations complete before state
        is committed. Failed updates preserve the previous state.

        A propagated NaN remains active until reset, even if nan_policy
        is subsequently changed.

        For a batch of B elements:
        Time complexity: O(B).
        Temporary space complexity: O(B).
        Persistent accumulator space: O(1).
        """
        array, _, policy, nan_mask = _prepare_reduction(values, None, self.nan_policy)

        if array.ndim != 1:
            raise ValueError("values must be a 1D array.")

        if array.size == 0:
            return self

        clean = array[~nan_mask]
        batch_count = int(clean.size)
        new_count = self.count + batch_count

        propagate_nan = (policy == "propagate" and np.any(nan_mask))

        if self._poisoned or propagate_nan:
            self.count = new_count
            self.mean = np.float64(np.nan)
            self.M2 = np.float64(np.nan)
            self._poisoned = True
            return self

        # Under "omit", a batch containing only NaN changes nothing.
        if batch_count == 0:
            return self

        try:
            with np.errstate(over="raise",invalid="raise",divide="raise",):
                batch_mean = np.mean(clean, dtype=np.float64)
                deviations = clean - batch_mean
                batch_M2 = np.sum(deviations * deviations,dtype=np.float64,)

                if self.count == 0:
                    new_mean = batch_mean
                    new_M2 = batch_M2

                else:
                    delta = batch_mean - self.mean
                    batch_weight = batch_count / new_count

                    new_mean = (self.mean + delta * batch_weight)

                    # Equivalent to:
                    # delta**2 * old_count * batch_count / new_count
                    # Scale before squaring to reduce avoidable overflow.
                    merge_weight = self.count * batch_weight
                    scaled_delta = delta * np.sqrt(merge_weight)
                    correction = scaled_delta * scaled_delta

                    new_M2 = self.M2 + batch_M2 + correction

        except FloatingPointError as exc:
            raise ValueError("The batch statistics exceed supported float64 limits.") from exc

        if not np.isfinite(new_mean) or not np.isfinite(new_M2):
            raise ValueError("The updated statistics must remain finite.")

        # Commit only after the complete calculation succeeds.
        self.count = new_count
        self.mean = np.float64(new_mean)
        self.M2 = np.float64(new_M2)

        return self

    def variance(self, ddof=0):
        """Return accumulated variance as a numpy.float64 scalar.

        Parameters
        ----------
        ddof : int, default=0
            Non-negative integer; boolean values are rejected.
            The denominator is count - ddof.

        Returns
        -------
        numpy.float64
            M2 / (count - ddof).
            Return NaN if count <= ddof or NaN has propagated.

        Raises
        ------
        TypeError
            If ddof is not an integer or is boolean.
        ValueError
            If ddof is negative.

        Complexity
        ----------
        O(1) time and space.
        """
        ddof = _validate_integer(ddof, name="ddof", minimum=0)

        if self._poisoned or self.count <= ddof:
            return np.float64(np.nan)

        return np.float64(self.M2 / (self.count - ddof))

    def std(self, ddof=0):
        """Return accumulated standard deviation as a numpy.float64.

        Parameters
        ----------
        ddof : int, default=0
            Non-negative integer; boolean values are rejected.

        Returns
        -------
        numpy.float64
            Square root of accumulated variance.
            Return NaN if count <= ddof or NaN has propagated.

        Raises
        ------
        TypeError
            If ddof is not an integer or is boolean.
        ValueError
            If ddof is negative.

        Complexity
        ----------
        O(1) time and space.
        """
        return np.float64(np.sqrt(self.variance(ddof=ddof)))