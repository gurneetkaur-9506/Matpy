"""Shared literal-index shift between MATLAB and Python indexing.

MATLAB indexes from 1 and Python indexes from 0, so an index expression
must be shifted by one when moving between the two languages.
``shift_index`` is the single primitive both directions of translation
use: 'forward' (MATLAB -> Python) subtracts one, 'reverse' (Python ->
MATLAB) adds one.  Non-literal index expressions are passed through
unchanged -- handling end-keywords, ranges and computed offsets belongs
to the richer indexing rules that call this primitive.
"""

import re

FORWARD = "forward"
REVERSE = "reverse"

_VALID_DIRECTIONS = (FORWARD, REVERSE)

_INTEGER_LITERAL = re.compile(r"^-?\d+$")


def shift_index(expr, direction):
    """Shift a literal index expression between MATLAB and Python indexing.

    Args:
        expr: The index expression as a string (e.g. "1", "5", "i", "i + 1").
        direction: FORWARD ("forward") subtracts one (MATLAB -> Python);
            REVERSE ("reverse") adds one (Python -> MATLAB).

    Returns:
        The shifted expression as a string.  Anything that is not a plain
        integer literal (identifiers, computed expressions, end-keywords)
        is returned unchanged.

    Raises:
        ValueError: when ``direction`` is neither 'forward' nor 'reverse'.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError("direction must be %r or %r" % (FORWARD, REVERSE))
    expr = expr.strip()
    if not _INTEGER_LITERAL.fullmatch(expr):
        return expr
    value = int(expr)
    if direction == FORWARD:
        return str(value - 1)
    return str(value + 1)
