"""Read a whitespace-delimited data file using MATLAB fscanf semantics.

Replaces the MATLAB idiom ``fid = fopen(path); while ~feof(fid)
    data = fscanf(fid, fmt); end`` with a single call. The MATLAB format
specifier is mapped to a numpy dtype generally:

- %f / %e / %g -> np.float64
- %d / %i      -> np.int32
- %x           -> np.uint32  (hexadecimal)
- %o           -> np.uint32  (octal)
- %u           -> np.uint32  (unsigned decimal)
- %s / %c      -> str

A compound format string (e.g. ``'%f %f'``) is read as fixed-width
columns, so callers are not limited to a specific reshape size.
"""

import re

import numpy as np

_FORMAT_SPEC_RE = re.compile(r"%(\*)?(\d+)?(\.\d+)?([A-Za-z])")

_NUMERIC_CONVS = ("f", "e", "g", "d", "i", "x", "o", "u")

_DTYPE_MAP = {
    "f": np.float64,
    "e": np.float64,
    "g": np.float64,
    "d": np.int32,
    "i": np.int32,
    "x": np.uint32,
    "o": np.uint32,
    "u": np.uint32,
}


def _parse_token(token, conv):
    """Parse one whitespace-separated token according to a MATLAB
    conversion character."""
    if conv in ("f", "e", "g"):
        return float(token)
    if conv == "d":
        return int(token, 10)
    if conv == "x":
        return int(token, 16)
    if conv == "o":
        return int(token, 8)
    if conv == "u":
        return int(token, 10)
    if conv in ("s", "c"):
        return token
    raise ValueError("unsupported format specifier '%%%s'" % conv)


def _read_column(tokens, conv):
    if conv in _NUMERIC_CONVS:
        return np.array([_parse_token(t, conv) for t in tokens], dtype=_DTYPE_MAP[conv])
    if conv in ("s", "c"):
        return np.array(tokens, dtype=str)
    raise ValueError("unsupported format specifier '%%%s'" % conv)


def format_spec_to_columns(format_spec):
    """Return the list of MATLAB conversion characters in ``format_spec``.

    Args:
        format_spec (str): A MATLAB fscanf format string such as
            ``'%f'`` or ``'%d %x'``.

    Returns:
        list: Conversion characters, e.g. ``['f']`` or ``['d', 'x']``.
        Empty when the string contains no specifiers.
    """
    return [match[3] for match in _FORMAT_SPEC_RE.findall(format_spec)]


def read_matlab_scan_file(path, format_spec):
    """Read a file of whitespace-separated values like MATLAB's fscanf.

    Args:
        path (str): Path to the data file.
        format_spec (str): MATLAB fscanf format string, e.g. ``'%f'``,
            ``'%d'``, ``'%x'``, ``'%s'`` or a compound ``'%f %d %s'``.

    Returns:
        numpy.ndarray: With a single specifier, a 1-D array of the parsed
        values with the dtype implied by the specifier. With a compound
        format, a 2-D array with one column per specifier (object dtype
        when the columns mix numeric and string types).
    """
    with open(path, "r") as f:
        tokens = f.read().split()

    convs = format_spec_to_columns(format_spec)
    if not convs:
        raise ValueError("no format specifier found in %r" % format_spec)

    if len(convs) == 1:
        return _read_column(tokens, convs[0])

    width = len(convs)
    usable = tokens[: (len(tokens) // width) * width]
    rows = []
    for start in range(0, len(usable), width):
        row = usable[start:start + width]
        rows.append([_parse_token(tok, conv) for tok, conv in zip(row, convs)])

    if all(conv in _NUMERIC_CONVS for conv in convs):
        dtype = np.float64 if any(conv in ("f", "e", "g") for conv in convs) else np.int64
        return np.array(rows, dtype=dtype)
    return np.array(rows, dtype=object)
