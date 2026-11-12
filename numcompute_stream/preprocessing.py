"""Tabular preprocessing with explicit fitted state and independent outputs."""

from numbers import Real
import numpy as np
from .utils import _as_float_array


class SimpleImputer:
    """Replace numeric NaN cells with a finite constant.
    
        Parameters
        ----------
        strategy : str, default='constant'
            Only 'constant' is supported in the baseline.
        fill_value : real numeric scalar, default=0.0
            Must be finite and representable as float64. Booleans are rejected.
    
        Attributes
        ----------
        n_features_in_ : int
            Number of columns recorded by the last successful fit.
        fill_value_ : float
            Validated constant recorded by the last successful fit. Changes to
            constructor parameters take effect on the next fit.
    
        Notes
        -----
        Inputs must be 2D real numeric data, converted to independent float64
        arrays using the shared validator. NaN is the only missing value;
        other values, including infinities, are preserved. Scalers separately
        reject infinity. All-NaN columns are allowed. No training rows are stored.
        """

    def __init__(self, strategy='constant', fill_value=0.0):
        """Store and validate parameters; O(1) time and space.

        Raise TypeError for unsupported types and ValueError for an unsupported
        strategy or a constant not representable as finite float64.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self._validated_fill_value()

    def _validated_fill_value(self):
        if not isinstance(self.strategy, str):
            raise TypeError('strategy must be a string.')
        if self.strategy != 'constant':
            raise ValueError("Only strategy='constant' is supported.")
        if isinstance(self.fill_value, (bool, np.bool_)) or not isinstance(
                self.fill_value, Real):
            raise TypeError('fill_value must be a real numeric scalar.')
        try:
            with np.errstate(over='ignore'):
                value = float(self.fill_value)
        except (OverflowError, ValueError) as exc:
            raise ValueError('fill_value must be representable as finite float64.') from exc
        if not np.isfinite(value):
            raise ValueError('fill_value must be finite.')
        return value

    def fit(self, X, y=None):
        """Record feature count and fill constant; return self."""
        fill = self._validated_fill_value()
        array = _as_float_array(X, name='X', ndim=2)
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError('fit requires at least one sample and one feature.')
        self.n_features_in_ = array.shape[1]
        self.fill_value_ = fill
        return self

    def transform(self, X):
        """Return a new float64 array with NaN replaced by fitted fill_value_."""
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError('SimpleImputer must be fitted before transform.')
        array = _as_float_array(X, name='X', ndim=2)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X must have {self.n_features_in_} features; got {array.shape[1]}.')
        array[np.isnan(array)] = self.fill_value_
        return array

    def fit_transform(self, X, y=None):
        """Fit on X, then return its transformed independent float64 copy."""
        return self.fit(X, y=y).transform(X)

class StandardScaler:
    """Standardize each column using training mean and population std.
    Ignore NaN during fit and preserve NaN during transform.
    Reject infinity and all-NaN training columns.
    """

    def _validate_input(self, X, *, fitting=False):
        """Validate input and return an independent float64 array."""
        array = _as_float_array(X, name="X", ndim=2)

        if np.any(np.isinf(array)):
            raise ValueError("X must not contain infinity.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

            if np.any(np.all(np.isnan(array), axis=0)):
                raise ValueError("Each training column must contain a non-NaN value.")

        return array

    def fit(self, X, y=None):
        """Learn column means and population standard deviations.

        X: 2D real numeric data of shape (n_samples, n_features).
        y: ignored; included for the shared transformer interface.
        Return self. A successful refit replaces previous fitted state.

        Raise TypeError for unsupported input types.
        Raise ValueError for invalid shape, empty data, infinity,
        all-NaN columns or non-finite computed statistics.

        Time: O(n_samples * n_features).
        Additional space: O(n_samples * n_features).
        """
        array = self._validate_input(X, fitting=True)

        # Calculate each column's statistics, ignoring NaN.
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.nanmean(array, axis=0)
            scale = np.nanstd(array, axis=0, ddof=0)

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Training statistics exceed the supported numeric range.")

        # Avoid division by zero for constant training columns.
        scale = np.where(scale == 0.0, 1.0, scale)

        # Save state only after all validation succeeds.
        self.mean_ = mean
        self.scale_ = scale
        self.n_features_in_ = array.shape[1]

        return self

    def transform(self, X):
        """Transform data using the previously fitted statistics.

        X: 2D real numeric data with n_features_in_ columns.
        Return an independent float64 array with the same shape.
        Allow zero samples and preserve NaN. Do not refit or modify X.

        Raise RuntimeError before fit.
        Raise TypeError for unsupported input types.
        Raise ValueError for invalid shape, wrong feature count,
        infinity or arithmetic overflow.

        Time: O(n_samples * n_features).
        Additional space: O(n_samples * n_features).
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("StandardScaler must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features; " 
                             f"got {array.shape[1]}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            result = (array - self.mean_) / self.scale_

        # Input NaN is allowed, but finite input must not become inf/NaN.
        if np.any(np.isfinite(array) & ~np.isfinite(result)):
            raise ValueError("Transformed values exceed the supported numeric range.")

        return result

    def fit_transform(self, X, y=None):
        """Fit on X and return its standardized float64 copy."""
        return self.fit(X, y=y).transform(X)

class MinMaxScaler:
    """Scale columns using training minima and maxima.

    Ignore NaN during fit and preserve NaN during transform.
    New data is not clipped to feature_range.
    """

    def __init__(self, feature_range=(0, 1)):
        """Validate the target range; O(1) time and space."""
        self.feature_range = feature_range
        self._validate_range()

    def _validate_range(self):
        """Return validated float bounds and their span."""
        if not isinstance(self.feature_range, (tuple, list)):
            raise TypeError("feature_range must be a tuple or list.")

        if len(self.feature_range) != 2:
            raise ValueError("feature_range must contain two bounds.")

        bounds = []

        for value in self.feature_range:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("Each bound must be a real numeric scalar.")

            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Bounds must be representable as finite float64.") from exc

            if not np.isfinite(value):
                raise ValueError("Bounds must be finite.")

            bounds.append(value)

        lower, upper = bounds

        if lower >= upper:
            raise ValueError("The lower bound must be less than the upper.")

        span = upper - lower

        if not np.isfinite(span):
            raise ValueError("The target range exceeds the numeric limit.")

        return lower, upper, span

    def _validate_input(self, X, *, fitting=False):
        """Return a 2D float64 copy; allow NaN but reject infinity."""
        array = _as_float_array(X, name="X", ndim=2)

        if np.any(np.isinf(array)):
            raise ValueError("X must not contain infinity.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

            if np.any(np.all(np.isnan(array), axis=0)):
                raise ValueError("Each training column must contain a non-NaN value.")

        return array

    def fit(self, X, y=None):
        """Learn each column's minimum and maximum; return self.

        X: nonempty 2D real numeric data, shape (N, D).
        y: ignored. All-NaN columns and infinity are rejected.
        A successful refit replaces previous state; a failed fit preserves it.

        Raise TypeError for invalid types; ValueError for invalid shape,
        empty data, invalid bounds or an unrepresentable training range.

        Time: O(N * D). Additional space: O(N * D).
        """
        lower, upper, span = self._validate_range()
        array = self._validate_input(X, fitting=True)

        data_min = np.nanmin(array, axis=0)
        data_max = np.nanmax(array, axis=0)

        with np.errstate(over="ignore", invalid="ignore"):
            data_range = data_max - data_min

        if not np.all(np.isfinite(data_range)):
            raise ValueError("Training ranges exceed the supported numeric range.")

        # Constant columns use an effective denominator of 1.
        effective_range = np.where(data_range == 0.0, 1.0, data_range)

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.n_features_in_ = array.shape[1]

        self._effective_range = effective_range
        self._lower = lower
        self._span = span

        return self

    def transform(self, X):
        """Scale X using the previously fitted parameters.

        X: 2D real numeric data with n_features_in_ columns.
        Return an independent float64 array of the same shape.
        Allow zero samples, preserve NaN, and do not clip or refit.

        Raise RuntimeError before fit; TypeError for invalid input types;
        ValueError for invalid shape, wrong columns, infinity or overflow.

        Time: O(N * D). Additional space: O(N * D).
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("MinMaxScaler must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; "
                f"got {array.shape[1]}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            result = (array - self.data_min_) / self._effective_range
            result = result * self._span + self._lower

        if np.any(np.isfinite(array) & ~np.isfinite(result)):
            raise ValueError("Transformed values exceed the supported numeric range.")

        return result

    def fit_transform(self, X, y=None):
        """Fit on X and return its scaled float64 copy. """
        return self.fit(X, y=y).transform(X)

class OneHotEncoder:
    """Encode categories in first-appearance order.

    Each input column contains strings or real numeric categories.
    None and numeric NaN represent the same missing category.
    Output is a dense float64 array.
    """

    def __init__(self, handle_unknown="error"):
        """Validate unknown-category behavior; O(1) time and space."""
        self.handle_unknown = handle_unknown
        self._validate_option()

    def _validate_option(self):
        """Return the validated unknown-category option."""
        if not isinstance(self.handle_unknown, str):
            raise TypeError("handle_unknown must be a string.")

        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        return self.handle_unknown

    def _category_key(self, value):
        """Normalize missing values and validate one category."""
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(
                "Categories must be strings or real numbers, "
                "not booleans or other objects."
            )

        # NaN is unequal to itself; normalize it to the missing key.
        if value != value:
            return None

        if value == np.inf or value == -np.inf:
            raise ValueError("Categories must not contain infinity.")

        return value

    def _validate_input(self, X, *, fitting=False):
        """Return a normalized 2D object array without modifying X."""
        try:
            array = np.array(X, dtype=object, copy=True)
        except ValueError as exc:
            raise ValueError("X must have a regular rectangular shape.") from exc

        if array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

        for column in range(array.shape[1]):
            column_type = None

            for row in range(array.shape[0]):
                key = self._category_key(array[row, column])

                if key is not None:
                    current_type = ("string" if isinstance(key, str) else "numeric")

                    if column_type is None:
                        column_type = current_type
                    elif current_type != column_type:
                        raise TypeError(f"Column {column} must not mix, strings and numbers.")

                array[row, column] = key

        return array

    def fit(self, X, y=None):
        """Learn categories from nonempty 2D input X; return self.

        X has shape (N, D); y is ignored.
        Store categories_ and n_features_in_. Missing values are allowed.
        Successful refit replaces state; failed fit preserves previous state.

        Raise TypeError for unsupported or mixed category types.
        Raise ValueError for invalid shape, empty data, infinity or options.

        Expected time: O(N * D), treating category hashing as constant time.
        Additional space: O(N * D + K), where K is total learned categories.
        """
        option = self._validate_option()
        array = self._validate_input(X, fitting=True)

        categories = []
        mappings = []

        for column in range(array.shape[1]):
            mapping = {}
            values = []

            for key in array[:, column]:
                if key not in mapping:
                    mapping[key] = len(values)
                    values.append(key)

            categories.append(np.array(values, dtype=object))
            mappings.append(mapping)

        widths = [len(values) for values in categories]
        offsets = np.cumsum([0] + widths[:-1], dtype=np.intp)

        self.categories_ = categories
        self.n_features_in_ = array.shape[1]

        self._mappings = mappings
        self._offsets = offsets
        self._n_output_features = sum(widths)
        self._handle_unknown = option

        return self

    def transform(self, X):
        """Encode 2D input using previously learned categories.

        X has shape (N, n_features_in_); zero samples are allowed.
        Return an independent float64 array of shape (N, K).
        Unknown categories raise an error or produce an all-zero block
        for their input feature, depending on the fitted option.

        Raise RuntimeError before fit; TypeError for invalid categories;
        ValueError for invalid shape, feature count, infinity or unknowns.

        Expected time: O(N * D + N * K).
        Additional space: O(N * D + N * K), including dense output.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("OneHotEncoder must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; "
                f"got {array.shape[1]}."
            )

        result = np.zeros(
            (array.shape[0], self._n_output_features),
            dtype=np.float64,
        )

        for column, mapping in enumerate(self._mappings):
            offset = self._offsets[column]

            for row, key in enumerate(array[:, column]):
                index = mapping.get(key)

                if index is None:
                    if self._handle_unknown == "error":
                        raise ValueError(
                            f"Unknown category {key!r} "
                            f"in column {column}."
                        )
                    continue

                result[row, offset + index] = 1.0

        return result

    def fit_transform(self, X, y=None):
        """Fit on X and return its dense float64 encoding."""
        return self.fit(X, y=y).transform(X)

class ColumnSelector:
    """Select columns from a 2D table in the requested order.

    Parameters
    ----------
    columns : list, tuple or 1D numpy.ndarray of integers
        Non-empty, unique, non-negative column indices.
        Boolean indices are rejected.
    dtype : optional NumPy dtype specification
        None preserves the input dtype. Otherwise, selected values
        are converted using NumPy's astype conversion rules.

    Notes
    -----
    Outputs are independent arrays; object-array elements are not
    deep-copied. Non-contiguous inputs are supported.
    """

    def __init__(self, columns, dtype=None):
        self.columns = columns
        self.dtype = dtype
        self._validated_columns()
        self._validated_dtype()

    def _validated_columns(self):
        if not isinstance(self.columns, (list, tuple, np.ndarray)):
            raise TypeError("columns must be a list, tuple or 1D array.")

        # Object dtype preserves booleans for explicit validation.
        columns = np.asarray(self.columns, dtype=object)

        if columns.ndim != 1 or columns.size == 0:
            raise ValueError("columns must be a non-empty 1D sequence.")

        indices = []
        for column in columns:
            if isinstance(column, (bool, np.bool_)) or not isinstance(
                column, (int, np.integer)
            ):
                raise TypeError("columns must contain integers, not booleans.")

            column = int(column)
            if column < 0:
                raise ValueError("columns must contain non-negative indices.")

            indices.append(column)

        if len(set(indices)) != len(indices):
            raise ValueError("columns must not contain repeated indices.")

        return tuple(indices)

    def _validated_dtype(self):
        return None if self.dtype is None else np.dtype(self.dtype)

    @staticmethod
    def _as_table(X):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2D table.")
        return array

    @staticmethod
    def _select(array, columns, dtype):
        result = array[:, list(columns)]
        if dtype is not None:
            result = result.astype(dtype, copy=True)
        return result

    def fit(self, X, y=None):
        """Validate the table and store column-selection state.

        X has shape (n_samples, n_features); both dimensions must
        be non-zero. y is ignored. Return self.

        Raise TypeError for invalid column types and ValueError for
        invalid shape, indices or conversion. NumPy dtype errors
        propagate. A failed refit preserves previous fitted state.

        For n rows, p input columns and k selected columns, selection
        and conversion cost O(n*k) time and space. Converting a
        nested input sequence to an array additionally costs O(n*p).
        """
        array = self._as_table(X)
        columns = self._validated_columns()
        dtype = self._validated_dtype()

        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("fit requires at least one sample and feature.")

        if any(column >= array.shape[1] for column in columns):
            raise ValueError("columns contains an out-of-range index.")

        # Check conversion before committing the new fitted state.
        self._select(array, columns, dtype)

        self.columns_ = columns
        self.dtype_ = dtype
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        """Return selected columns with shape (n_samples, k).

        Require the fitted feature count; zero samples are allowed.
        Preserve column order and return an independent array.

        Raise RuntimeError before fit or if selection parameters
        changed after fit. Raise ValueError for invalid input shape,
        feature count or conversion. NumPy dtype errors propagate.

        Selection and conversion cost O(n*k) time and space.
        Nested input conversion additionally costs O(n*p), where
        p is the input feature count. Object copies are shallow.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("ColumnSelector must be fitted before transform.")

        columns = self._validated_columns()
        dtype = self._validated_dtype()

        if dtype is None or self.dtype_ is None:
            dtype_changed = dtype is not self.dtype_
        else:
            dtype_changed = dtype != self.dtype_

        if columns != self.columns_ or dtype_changed:
            raise RuntimeError("ColumnSelector parameters changed; fit again.")

        array = self._as_table(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features; "
                             f"got {array.shape[1]}.")

        return self._select(array, self.columns_, self.dtype_)

    def fit_transform(self, X, y=None):
        """Fit and return selected columns, equivalent to fit then transform.

        Accept the same inputs and raise the same errors as fit and
        transform. Time and space have the same asymptotic bounds.
        """
        return self.fit(X, y).transform(X)