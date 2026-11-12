"""Strict, unquoted UTF-8 CSV input using standard Python and NumPy."""

from decimal import Decimal, InvalidOperation
from io import StringIO
import numpy as np
from .utils import _validate_integer


def _options(delimiter, dtype, skip_header, usecols, missing, missing_values):
    if not isinstance(delimiter, str):
        raise TypeError("delimiter must be a string.")
    if len(delimiter) != 1 or delimiter in '\r\n"':
        raise ValueError("delimiter must be one character other than newline or quote.")
    if dtype is None:
        raise TypeError("dtype must explicitly specify real numeric or string data.")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError("Invalid dtype.") from exc
    if dtype.kind not in 'iufUS' or dtype.subdtype is not None:
        raise TypeError("dtype must be homogeneous real numeric or string data.")
    skip_header = _validate_integer(skip_header, name='skip_header', minimum=0)
    if not isinstance(missing, str):
        raise TypeError("missing must be a string.")
    if missing not in ('fill', 'skip'):
        raise ValueError("missing must be 'fill' or 'skip'.")
    if usecols is not None:
        if isinstance(usecols, (str, bytes)):
            raise TypeError("usecols must be a sequence of integers.")
        try:
            usecols = tuple(_validate_integer(c, name='usecols index', minimum=0) for c in usecols)
        except TypeError as exc:
            raise TypeError("usecols must contain integers, not booleans.") from exc
        if not usecols or len(set(usecols)) != len(usecols):
            raise ValueError("usecols must be nonempty and contain unique indices.")
    if isinstance(missing_values, (str, bytes)):
        raise TypeError("missing_values must be a collection of strings.")
    try:
        missing_values = tuple(missing_values)
    except TypeError as exc:
        raise TypeError("missing_values must be a collection of strings.") from exc
    if any(not isinstance(v, str) for v in missing_values):
        raise TypeError("missing_values must contain strings.")
    return dtype, skip_header, usecols, set(missing_values)


def _convert(rows, dtype):
    """Convert already validated fields without integer loss or string truncation."""
    if dtype.kind in 'iu':
        limits = np.iinfo(dtype)
        converted = []
        for row in rows:
            output = []
            for token in row:
                try:
                    value = Decimal(token)
                except InvalidOperation as exc:
                    raise ValueError(f"Invalid integer value: {token!r}.") from exc
                if (not value.is_finite() or value != value.to_integral_value()
                        or value < int(limits.min) or value > int(limits.max)):
                    raise ValueError(f"Value {token!r} is not representable as {dtype}.")
                output.append(int(value))
            converted.append(output)
        return np.asarray(converted, dtype=dtype)
    if dtype.kind in 'US':
        if dtype.kind == 'S':
            rows = [[v.encode('utf-8') for v in row] for row in rows]
            capacity = dtype.itemsize
        else:
            capacity = dtype.itemsize // np.dtype('U1').itemsize
        if capacity and any(len(v) > capacity for row in rows for v in row):
            raise ValueError(f"A string is too long for dtype {dtype}.")
        return np.asarray(rows, dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def _rows(path, delimiter, skip_header, usecols, missing, fill_value,
          missing_values, dtype):
    """Read, check and handle missing values line by line"""
    width = None
    with open(path, 'r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number <= skip_header or not line.strip():
                continue
            if '"' in line:
                raise ValueError(f"Line {line_number}: quoted fields are unsupported.")
            fields = line.rstrip('\r\n').split(delimiter)
            if width is None:
                width = len(fields)
                if usecols is not None and max(usecols) >= width:
                    raise ValueError("usecols contains an out-of-range column.")
            if len(fields) != width:
                raise ValueError(f"Line {line_number}: inconsistent field count.")
            # NumPy performs the delimited text read; prechecks enforce our subset.
            fields = np.loadtxt(StringIO(line), delimiter=delimiter, dtype=str,
                                comments=None, ndmin=2)[0].tolist()
            selected = fields if usecols is None else [fields[c] for c in usecols]
            selected = [v.strip() for v in selected]
            mask = [v in missing_values for v in selected]
            if any(mask):
                if missing == 'skip':
                    continue
                if fill_value is None:
                    if dtype.kind in 'iu':
                        raise ValueError("Integer missing values require an explicit fill_value.")
                    fill = 'nan' if dtype.kind == 'f' else ''
                else:
                    if not np.isscalar(fill_value):
                        raise TypeError("fill_value must be a scalar.")
                    fill = str(fill_value)
                selected = [fill if absent else v for v, absent in zip(selected, mask)]
            yield selected


def iter_csv_chunks(path, chunk_size=1024, delimiter=',', dtype=float,
                    skip_header=0, usecols=None, missing='fill', fill_value=None,
                    missing_values=('', 'NA', 'NaN', 'nan')):
    """Yield independent 2D arrays from an unquoted UTF-8 delimited file.
    
        Parameters
        ----------
        path : str or os.PathLike
            File to read. File-like objects are outside this baseline.
        chunk_size : positive int
            Usable rows per chunk; the last chunk may be smaller. Booleans rejected.
        delimiter : str
            One character (default comma); tab is supported.
        dtype : NumPy dtype specification
            Signed/unsigned integer, float, Unicode or UTF-8 byte string. Unsized
            strings infer width per chunk; concatenate preserves all their values.
        skip_header : non-negative int
            Number of physical initial lines to skip.
        usecols : sequence of unique non-negative ints or None
            Select columns in this order; None selects all columns.
        missing : {'fill', 'skip'}
            Replace missing selected cells or discard their entire row.
        fill_value : scalar or None
            Default: NaN for float, empty string for strings. Integer missing
            cells require an explicit representable integer. Used only when needed.
        missing_values : collection of str
            Exact tokens checked after stripping surrounding whitespace.
    
        Yields
        ------
        numpy.ndarray
            Shape (usable_rows, selected_columns), including single-row/column cases.
    
        Raises
        ------
        TypeError
            Unsupported parameter types or dtype.
        ValueError
            Invalid options, quotes, inconsistent widths, conversion loss, or no
            usable data. Nonintegral/out-of-range integers and string truncation
            are rejected. Blank lines are ignored; comments have no special meaning.
        FileNotFoundError
            Input path does not exist.
    
        Notes
        -----
        Validation occurs on iteration. Later errors may follow earlier yields.
        No embedded newlines or double-quoted fields are supported. Float NaN/inf
        values follow NumPy conversion semantics. Close a partially consumed
        generator to release its file promptly.
        Time O(B + N*C) for B input characters and N*C selected cells (bounded
        token lengths); additional space O(K*C*L + W), independent of total rows,
        for K=chunk_size, maximum token length L and maximum physical line size W.
        Integer conversion checks exact decimal values before casting.
        """
    chunk_size = _validate_integer(chunk_size, name='chunk_size', minimum=1)
    dtype, skip_header, usecols, missing_values = _options(
        delimiter, dtype, skip_header, usecols, missing, missing_values)
    rows = _rows(path, delimiter, skip_header, usecols, missing,
                 fill_value, missing_values, dtype)
    batch = []
    found = False
    try:
        for row in rows:
            batch.append(row)
            if len(batch) == chunk_size:
                result = _convert(batch, dtype)
                batch = []
                found = True
                yield result
        if batch:
            yield _convert(batch, dtype)
        elif not found:
            raise ValueError("No usable data rows remain.")
    finally:
        rows.close()


def load_csv(path, delimiter=',', dtype=float, skip_header=0, usecols=None,
             missing='fill', fill_value=None,
             missing_values=('', 'NA', 'NaN', 'nan')):
    """Read a complete unquoted UTF-8 CSV into a homogeneous 2D ndarray.
    
        Parameters and errors match iter_csv_chunks, except chunk_size is absent.
        path is str/PathLike; delimiter is one character; dtype is real numeric or
        string; skip_header is a non-negative physical-line count; usecols is None
        or unique non-negative indices. missing is 'fill' or 'skip'; fill_value is
        a scalar or None; missing_values is a collection of stripped-field tokens.
        Returns shape (N, C), never a squeezed vector. Empty, header-only and
        entirely skipped input raise ValueError. Bad types raise TypeError; missing
        paths raise FileNotFoundError; invalid data/options raise ValueError.
        Integer loss and string truncation are rejected. See iter_csv_chunks for
        detailed dtype, missing-value, quote and whitespace rules.
        Time O(B + N*C) at bounded token lengths; additional space O(N*C*L + W),
        including the full output and concatenation, where B is input characters,
        L maximum token length and W maximum physical line size.
        """
    return np.concatenate(list(iter_csv_chunks(
        path, delimiter=delimiter, dtype=dtype, skip_header=skip_header,
        usecols=usecols, missing=missing, fill_value=fill_value,
        missing_values=missing_values)), axis=0)