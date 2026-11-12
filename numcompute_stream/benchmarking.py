"""Timing utilities for performance comparisons."""

import time

from statistics import median

from .utils import _validate_integer

import platform
import numpy as np


def benchmark(func, *args, warmup=1, repeats=5, **kwargs):
    """Measure repeated function calls after untimed warmup.

    Parameters:
        func (callable): Function to measure; its result is discarded.
        *args, **kwargs: Arguments passed unchanged to func.
            Their accepted shapes depend on func.
        warmup (int): Untimed calls, default 1; must be non-negative.
        repeats (int): Measured calls, default 5; must be positive.
            Boolean values are rejected for both counts.

    Returns:
        dict: times_seconds is a list of length repeats.
            median_seconds and min_seconds are scalar floats.
            All durations are in seconds.

    Raises:
        TypeError: func is not callable or a count has an invalid type.
        ValueError: warmup is negative or repeats is less than one.
        Exceptions from func propagate unchanged.

    Complexity:
        Time O((w + r) * (C + 1) + r log r), where w is warmup,
        r is repeats and C is the cost of one function call.
        Additional space O(r), excluding func's own workspace.

    Notes:
        Internal validation, copying and allocation within func are timed.
        The same arguments are reused; input mutation depends on func.
    """

    if not callable(func): # check parameters, ensure it is a callable object
        raise TypeError("func must be callable.")
    """
    the times of warm up could be 0 but the formal test at least 1 times
    """
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(repeats): # record start time > excute functions > compute time gap > preserve the result
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        "times_seconds": times,
        "median_seconds": median(times),
        "min_seconds": min(times),
    }

def environment_info():
    """Collect runtime information for reproducible benchmarks.

    Parameters:
        None. No input arrays are required.

    Returns:
        dict: Scalar strings under python, numpy, os and cpu.
            CPU information is "unknown" when unavailable.
            No array output is produced.

    Raises:
        No function-specific validation errors.
        Exceptions from underlying platform queries propagate unchanged.

    Complexity:
        Time and additional space O(1) relative to benchmark data size.
        The actual system-query cost depends on the operating system.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }