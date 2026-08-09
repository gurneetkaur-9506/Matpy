"""Shared index shift between MATLAB and Python indexing.

MATLAB indexes from 1 and Python indexes from 0, so an index expression
must be shifted by one when moving between the two languages.
``shift_index`` is the single primitive both directions of translation
use: 'forward' (MATLAB -> Python) subtracts one, 'reverse' (Python ->
MATLAB) adds one.

The primitive recognizes three forms of index:

- a literal integer, which is shifted by one (5 -> 4 forward, 5 -> 6
  reverse);
- a single variable identifier, which is passed through unchanged -- a
  variable's value shifts together with the loop that drives it, so no
  numeric adjustment applies at index-translation time;
- an indexed access such as ``x(i)`` / ``x[i]``, whose parentheses or
  brackets are converted to the target language's syntax and whose inner
  index is shifted recursively (x(5) -> x[4] forward, x[i] -> x(i)
  reverse).

Computed index expressions (``i + 1``), end-keywords and ranges are
passed through unchanged; their adjustment belongs to the richer
indexing rules that call this primitive.
"""

import re

FORWARD = "forward"
REVERSE = "reverse"

_VALID_DIRECTIONS = (FORWARD, REVERSE)

_INTEGER_LITERAL = re.compile(r"^-?\d+$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDEXED_ACCESS = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[\(\[](.*)[\)\]]$")


def _split_args(text):
    """Split a parenthesized/bracketed index list on top-level commas."""
    parts = []
    depth = 0
    current = ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def shift_index(expr, direction):
    """Shift an index expression between MATLAB and Python indexing.

    Args:
        expr: The index expression as a string -- a literal ("5"), a
            single variable ("i"), an indexed access ("x(i)", "x[i]"), or
            a computed expression ("i + 1").
        direction: FORWARD ("forward") shifts MATLAB -> Python (subtract
            one, parentheses to brackets); REVERSE ("reverse") shifts
            Python -> MATLAB (add one, brackets to parentheses).

    Returns:
        The shifted expression as a string.  Single-variable indices and
        computed expressions are returned unchanged.

    Raises:
        ValueError: when ``direction`` is neither 'forward' nor 'reverse'.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError("direction must be %r or %r" % (FORWARD, REVERSE))
    expr = expr.strip()
    if _INTEGER_LITERAL.fullmatch(expr):
        value = int(expr)
        if direction == FORWARD:
            return str(value - 1)
        return str(value + 1)
    if _IDENTIFIER.fullmatch(expr):
        return expr
    match = _INDEXED_ACCESS.fullmatch(expr)
    if match:
        name, inner = match.group(1), match.group(2)
        args = _split_args(inner)
        shifted = ", ".join(shift_index(a, direction) for a in args)
        if direction == FORWARD:
            return "%s[%s]" % (name, shifted)
        return "%s(%s)" % (name, shifted)
    return expr
