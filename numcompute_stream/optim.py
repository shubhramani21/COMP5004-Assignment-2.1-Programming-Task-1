"""Finite-difference gradients and Jacobians with explicit validation."""

from numbers import Real
import numpy as np
from .utils import _as_float_array


def _validate_method(method):
    """Validate and return the finite-difference method name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'.")

    return method


def _validate_step(h):
    """Validate h and return it as a positive, finite Python float."""
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
        raise TypeError("h must be a real numeric scalar.")

    try:
        value = float(h)
    except (OverflowError, ValueError) as exc:
        raise ValueError("h must be representable as finite float64.") from exc

    if not np.isfinite(value):
        raise ValueError("h must be finite.")

    if value <= 0.0:
        raise ValueError(f"h must be positive; got {value}.")

    return value



def _prepare(x, method, h):
    """Validate inputs and return x, the method and the perturbed points.

    Return (array, method, upper, lower). For the central method, lower
    holds x - h; for the forward method, lower is the unperturbed x.
    """
    method = _validate_method(method)
    h = _validate_step(h)
    array = _as_float_array(x, name="x", ndim=1, finite=True)

    if array.size == 0:
        raise ValueError("x must contain at least one value.")

    # Overflow is reported below as a non-finite coordinate.
    with np.errstate(over="ignore"):
        upper = array + h
        lower = array - h if method == "central" else array.copy()

    for point in (upper, lower) if method == "central" else (upper,):
        if not np.all(np.isfinite(point)):
            raise ValueError("x + h and x - h must remain finite.")

        # A step below the spacing of float64 values near x has no effect.
        if np.any(point == array):
            raise ValueError(
                "h is too small to change every coordinate of x; "
                "use a larger h."
            )

    with np.errstate(over="ignore"):
        steps = upper - lower

    if not np.all(np.isfinite(steps)):
        raise ValueError("The step between perturbed points must be finite.")

    return array, method, upper, lower


def _perturbed(array, index, value):
    """Return an independent copy of array with one coordinate replaced."""
    point = array.copy()
    point[index] = value
    return point


def _scalar_output(value):
    """Validate one output of f for grad and return it as a float."""
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 0:
        raise ValueError(
            f"f(x) must return a scalar; got shape {result.shape}."
        )

    if not np.isfinite(result):
        raise ValueError("f(x) must return a finite value.")

    return float(result)


def _vector_output(value, n_outputs=None):
    """Validate one output of f for jacobian and return a float64 array."""
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 1:
        raise ValueError(
            f"f(x) must return a 1D array; got {result.ndim}D."
        )

    if result.size == 0:
        raise ValueError("f(x) must return at least one value.")

    if n_outputs is not None and result.size != n_outputs:
        raise ValueError(
            f"f(x) must return {n_outputs} values on every evaluation; "
            f"got {result.size}."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("f(x) must return only finite values.")

    return result


def _difference(f_upper, f_lower, step):
    """Return a finite difference quotient without avoidable overflow.

    Use direct subtraction normally. If subtraction overflows,
    divide each operand by step before subtracting instead.

    Raise ValueError if the resulting quotient is non-finite.
    Time and additional space are O(m) for m output elements.
    """
    upper = np.asarray(f_upper, dtype=np.float64)
    lower = np.asarray(f_lower, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore",):
        delta = upper - lower
        result = delta / step

        overflowed = ~np.isfinite(delta)
        if np.any(overflowed):
            fallback = upper / step - lower / step
            result = np.where(overflowed, fallback, result)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "The finite-difference result exceeds the supported numeric range."
        )

    return result


def grad(f, x, method="central", h=1e-5):
    """Estimate the gradient of a scalar function by finite differences.

    Parameters
    ----------
    f : callable
        Function taking a 1D float64 array of shape (n,) and returning a
        finite real scalar. Each call receives an independent array copy,
        so f may modify its argument without affecting x or later calls.
    x : array_like of real numbers, shape (n,)
        Non-empty, finite point at which the gradient is estimated.
        Integer input is converted to float64; slices are supported.
    method : {"central", "forward"}, default="central"
        "central" uses (f(x + h e_i) - f(x - h e_i)) / (2h), with error
        O(h**2). "forward" uses (f(x + h e_i) - f(x)) / h, with error O(h),
        and reuses a single baseline evaluation of f(x).
    h : real scalar, default=1e-5
        Positive, finite absolute step size. Booleans are rejected. The
        divisor is the step actually represented in float64, which may
        differ slightly from h after rounding.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        Estimated partial derivatives, one per coordinate of x.

    Raises
    ------
    TypeError
        If method is not a string, h is not a real scalar or is boolean,
        x is not real numeric data, or f returns non-numeric data.
    ValueError
        If method is unsupported; h is non-positive or non-finite; x is
        not 1D, is empty or contains NaN/infinity; x +/- h is non-finite
        or equal to x; f returns a non-scalar or non-finite value; or the
        difference quotient overflows.

    Notes
    -----
    Function evaluations: 2n for "central", n + 1 for "forward".
    Time: O(n * (C_f + n)), where C_f is the cost of one call to f and
    O(n) is the cost of copying x for each call.
    Additional space: O(n) besides the space used by f.
    The caller's x is never modified.
    """
    array, method, upper, lower = _prepare(x, method, h)

    if method == "forward":
        f_lower = _scalar_output(f(array.copy()))

    result = np.empty(array.size, dtype=np.float64)

    for i in range(array.size):
        f_upper = _scalar_output(f(_perturbed(array, i, upper[i])))

        if method == "central":
            f_lower = _scalar_output(f(_perturbed(array, i, lower[i])))

        result[i] = _difference(f_upper, f_lower, upper[i] - lower[i])

    return result


def jacobian(f, x, method="central", h=1e-5):
    """Estimate the Jacobian of a vector function by finite differences.

    Parameters
    ----------
    f : callable
        Function taking a 1D float64 array of shape (n,) and returning a
        non-empty, finite 1D real array of shape (m,). m must stay the
        same on every call. Each call receives an independent array copy.
    x : array_like of real numbers, shape (n,)
        Non-empty, finite point at which the Jacobian is estimated.
        Integer input is converted to float64; slices are supported.
    method : {"central", "forward"}, default="central"
        "central" uses (f(x + h e_j) - f(x - h e_j)) / (2h), with error
        O(h**2). "forward" uses (f(x + h e_j) - f(x)) / h, with error O(h),
        and reuses a single baseline evaluation of f(x).
    h : real scalar, default=1e-5
        Positive, finite absolute step size. Booleans are rejected. The
        divisor is the step actually represented in float64.

    Returns
    -------
    numpy.ndarray of float64, shape (m, n)
        Entry [i, j] estimates the derivative of output i with respect
        to input j.

    Raises
    ------
    TypeError
        If method is not a string, h is not a real scalar or is boolean,
        x is not real numeric data, or f returns non-numeric data.
    ValueError
        If method is unsupported; h is non-positive or non-finite; x is
        not 1D, is empty or contains NaN/infinity; x +/- h is non-finite
        or equal to x; f returns an output that is not 1D, is empty,
        changes length or is non-finite; or a difference quotient
        overflows.

    Notes
    -----
    Function evaluations: 2n for "central", n + 1 for "forward".
    Time: O(n * (C_f + n + m)), where C_f is the cost of one call to f.
    Additional space: O(m * n) for the result, plus the space used by f.
    The caller's x is never modified.
    """
    array, method, upper, lower = _prepare(x, method, h)

    n_outputs = None

    if method == "forward":
        f_lower = _vector_output(f(array.copy()))
        n_outputs = f_lower.size

    columns = []

    for j in range(array.size):
        f_upper = _vector_output(f(_perturbed(array, j, upper[j])), n_outputs)
        n_outputs = f_upper.size

        if method == "central":
            f_lower = _vector_output(
                f(_perturbed(array, j, lower[j])), n_outputs
            )

        columns.append(_difference(f_upper, f_lower, upper[j] - lower[j]))

    # Each column holds the derivatives with respect to one input.
    return np.stack(columns, axis=1)
