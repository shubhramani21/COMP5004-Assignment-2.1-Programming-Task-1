"""Stable sorting, top-k selection, quickselect and binary search.

Numeric input dtypes are preserved after conversion to NumPy arrays.
Functions do not modify caller inputs.
"""
import numpy as np

from .utils import _as_float_array, _validate_integer


def _as_numeric_array(values, *, name="values", ndim=1):
    try:
        array = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must have a regular rectangular shape.") from exc
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, not boolean, "
                        "complex, string or object data.")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got {array.ndim}D.")
    if isinstance(values, (list, tuple)):
        _as_float_array(values, name=name, ndim=ndim)
    return array


def _validate_bool(value, *, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")


def _reject_nan(array, *, name="values"):
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN.")


def _stable_order(values, descending=False):
    if descending:
        return values.size - 1 - np.argsort(values[::-1], kind="stable")[::-1]
    return np.argsort(values, kind="stable")


def stable_sort(values, descending=False, return_indices=False):
    """Return a stable sorted copy of a 1D numeric array.

    Parameters
    ----------
    values : array-like, shape (n,)
        Real integer or floating-point values. NaN is rejected;
        positive and negative infinity are allowed.
    descending : bool, default=False
        Sort from largest to smallest when True.
    return_indices : bool, default=False
        Also return the original positions of the sorted values.

    Returns
    -------
    sorted_values : numpy.ndarray, shape (n,)
        Independent sorted array preserving the converted input dtype.
    indices : numpy.ndarray of integers, shape (n,)
        Returned only when return_indices=True, as
        (sorted_values, indices). Equal values retain their original
        relative order in both ascending and descending modes.

    Raises
    ------
    TypeError
        If values contain unsupported data types or either flag is
        not boolean.
    ValueError
        If values are not 1D or contain NaN.

    Notes
    -----
    Empty inputs are supported. The input is not modified.
    Time is O(n log n) as a general sorting upper bound; NumPy may use
    faster dtype-specific stable algorithms. Additional space is O(n).
    """
    array = _as_numeric_array(values)
    _validate_bool(descending, name="descending")
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array)
    if not descending and not return_indices:
        return np.sort(array, kind="stable")
    indices = _stable_order(array, descending)
    result = array[indices]
    return (result, indices) if return_indices else result


def multi_key_sort(X, columns, return_indices=False):
    """Stably sort rows by multiple columns in ascending order.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Real numeric table without NaN. Infinity is allowed.
    columns : 1D sequence of integers
        Non-empty, distinct column indices in decreasing priority:
        columns[0] is the primary key, columns[1] breaks its ties,
        and so on. Indices must satisfy 0 <= index < p.
    return_indices : bool, default=False
        Also return the original row indices.

    Returns
    -------
    sorted_rows : numpy.ndarray, shape (n, p)
        Independent array of sorted rows with the converted input dtype.
    indices : numpy.ndarray of integers, shape (n,)
        Returned only when return_indices=True, as
        (sorted_rows, indices). Rows equal on all selected keys retain
        their original relative order.

    Raises
    ------
    TypeError
        If X has an unsupported dtype, columns are not integers,
        or return_indices is not boolean.
    ValueError
        If X is not 2D or contains NaN, or columns are empty, not 1D,
        repeated, negative or out of bounds.

    Notes
    -----
    The input is not modified.
    For k keys, a general time upper bound is
    O(n*p + k*n*log(n) + k*log(k)), including table validation,
    sorting and duplicate-key validation.
    Additional space is O(n*p + n*k + n + k), including the output
    and selected-key arrays.
    """
    array = _as_numeric_array(X, name="X", ndim=2)
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array, name="X")
    keys = np.asarray(columns)
    if keys.ndim != 1 or keys.size == 0:
        raise ValueError("columns must be a non-empty 1D sequence.")
    if keys.dtype.kind not in "iu":
        raise TypeError("columns must contain integers, not booleans.")
    if isinstance(columns, (list, tuple)):
        for column in columns:
            _validate_integer(column, name="column", minimum=0)
    if np.any(keys < 0) or np.any(keys >= array.shape[1]):
        raise ValueError("columns must contain valid non-negative column indices.")
    if np.unique(keys).size != keys.size:
        raise ValueError("columns must not contain repeated indices.")
    indices = np.lexsort(array[:, keys[::-1]].T)
    result = array[indices]
    return (result, indices) if return_indices else result


def topk(values, k, largest=True, return_indices=True):
    """Return the k largest or smallest values in stable sorted order.

    Parameters
    ----------
    values : array-like, shape (n,)
        Real numeric values without NaN. Infinity is allowed.
    k : int
        Number of values to return, with 0 <= k <= n.
        Booleans are rejected.
    largest : bool, default=True
        Return the largest values in descending order when True;
        otherwise return the smallest values in ascending order.
    return_indices : bool, default=True
        Also return the original positions of the selected values.

    Returns
    -------
    selected_values : numpy.ndarray, shape (k,)
        Independent array preserving the converted input dtype.
    indices : numpy.ndarray of integers, shape (k,)
        Returned only when return_indices=True, as
        (selected_values, indices).

    Raises
    ------
    TypeError
        If values have an unsupported dtype, k is not an integer,
        or either flag is not boolean.
    ValueError
        If values are not 1D or contain NaN, or k is outside [0, n].

    Notes
    -----
    Ties retain original input order. If a tie crosses the selection
    boundary, earlier input positions are selected first.
    k=0 returns empty arrays, but input validation still runs.

    Partitioning identifies the cutoff; only the k selected candidates
    are sorted. Time is O(n + k log k), and additional space is O(n).
    The input is not modified.
    """
    array = _as_numeric_array(values)
    k = _validate_integer(k, name="k", minimum=0)
    _validate_bool(largest, name="largest")
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array)
    if k > array.size:
        raise ValueError("k must not exceed the number of values.")
    if k == 0:
        indices = np.empty(0, dtype=np.intp)
    else:
        position = array.size - k if largest else k - 1
        partition = np.argpartition(array, position)
        cutoff = array[partition[position]]
        better = np.flatnonzero(array > cutoff if largest else array < cutoff)
        tied = np.flatnonzero(array == cutoff)[:k - better.size]
        candidates = np.concatenate((better, tied))
        indices = candidates[_stable_order(array[candidates], largest)]
    result = array[indices]
    return (result, indices) if return_indices else result


def quickselect(values, k):
    """Return the value at zero-based position k in ascending order.

    Parameters
    ----------
    values : array-like, shape (n,)
        Non-empty real numeric values without NaN. Infinity is allowed.
    k : int
        Zero-based order statistic, with 0 <= k < n.
        Booleans are rejected. Duplicate values occupy separate positions.

    Returns
    -------
    numpy scalar
        The kth-smallest value, preserving the converted input dtype.

    Raises
    ------
    TypeError
        If values have an unsupported dtype or k is not an integer.
    ValueError
        If values are not 1D, are empty, contain NaN, or k is out of range.

    Notes
    -----
    Uses a median-of-three pivot and partitions into values below,
    equal to and above the pivot. Only the relevant partition continues.
    The input is copied and is not modified.

    Typical time is O(n); worst-case time is O(n**2) because the
    deterministic pivot does not guarantee balanced partitions.
    Peak additional space is O(n).
    """
    array = _as_numeric_array(values)
    k = _validate_integer(k, name="k", minimum=0)
    _reject_nan(array)
    if k >= array.size:
        raise ValueError("k must satisfy 0 <= k < the number of values.")
    work = array.copy()
    while work.size > 1:
        first, middle, last = work[0], work[work.size // 2], work[-1]
        pivot = max(min(first, middle), min(max(first, middle), last))
        smaller = work < pivot
        larger = work > pivot
        left_count = np.count_nonzero(smaller)
        equal_end = work.size - np.count_nonzero(larger)
        if k < left_count:
            work = work[smaller]
        elif k < equal_end:
            return pivot
        else:
            k -= equal_end
            work = work[larger]
    return work[0]


def binary_search(values, target):
    """Find the leftmost insertion position in an ascending array.

    Parameters
    ----------
    values : array-like, shape (n,)
        Real numeric values already sorted in ascending order and
        containing no NaN. These two preconditions are not checked.
        Infinity and duplicate values are allowed.
    target : real numeric scalar
        Value to find. NaN, booleans and complex values are rejected.

    Returns
    -------
    index : int
        Leftmost position at which target could be inserted while
        preserving ascending order, in the inclusive range [0, n].
    found : bool
        Whether target equals the value at index. For duplicates,
        index identifies the first occurrence.

    Raises
    ------
    TypeError
        If values or target have unsupported data types.
    ValueError
        If values are not 1D, target is not scalar, or target is NaN.

    Notes
    -----
    Empty input returns (0, False). The input is not modified.
    Results are not guaranteed for unsorted input or input containing NaN.

    For an existing numeric ndarray, time is O(log n) and additional
    space is O(1). Sequence conversion and validation may require
    O(n) time and space for list or tuple input.
    """
    array = _as_numeric_array(values)
    needle = _as_numeric_array(target, name="target", ndim=0).item()
    if np.isnan(needle):
        raise ValueError("target must not be NaN.")
    left, right = 0, array.size
    while left < right:
        middle = (left + right) // 2
        if array[middle].item() < needle:
            left = middle + 1
        else:
            right = middle
    return left, bool(left < array.size and array[left].item() == needle)
