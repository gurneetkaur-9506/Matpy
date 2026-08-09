"""Shared index shift between MATLAB and Python indexing.

MATLAB indexes from 1 and Python indexes from 0, so an index expression
must be shifted by one when moving between the two languages.
``shift_index`` is the single primitive both directions of translation
use: 'forward' (MATLAB -> Python) subtracts one, 'reverse' (Python ->
MATLAB) adds one.

The primitive recognizes four forms of index:

- a literal integer, which is shifted by one (5 -> 4 forward, 5 -> 6
  reverse);
- a single variable identifier, which is passed through unchanged -- a
  variable's value shifts together with the loop that drives it, so no
  numeric adjustment applies at index-translation time;
- an indexed access such as ``x(i)`` / ``x[i]``, whose parentheses or
  brackets are converted to the target language's syntax and whose inner
  index is shifted recursively (x(5) -> x[4] forward, x[i] -> x(i)
  reverse);
- an arithmetic index expression such as ``i + 1``, shifted by folding
  the offset into its trailing constant term (i + 1 -> i forward,
  i - 1 -> i reverse); an expression without a foldable constant is
  wrapped instead (2*k -> (2*k) - 1 forward).

Colon-ranges and end-keywords pass through unchanged; their adjustment
belongs to the richer indexing rules that call this primitive.
"""

import re

FORWARD = "forward"
REVERSE = "reverse"

_VALID_DIRECTIONS = (FORWARD, REVERSE)

_INTEGER_LITERAL = re.compile(r"^-?\d+$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDEXED_ACCESS = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[\(\[](.*)[\)\]]$")
_TRAILING_CONSTANT = re.compile(r"^(.+?)\s*([+-])\s*(\d+)\s*$")


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


def _is_arithmetic(expr):
    """True when the expression has a top-level arithmetic operator.  A
    colon at the top level marks a range instead, which is not shifted
    here."""
    depth = 0
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0:
            if ch == ":":
                return False
            if ch in "+-*/" and i > 0:
                return True
    return False


def _strip_outer_parens(expr):
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for ch in expr[1:-1]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return expr
    return expr[1:-1] if depth == 0 else expr


def _fold_constant_shift(expr, direction):
    """Fold the +-1 shift into a trailing integer constant term of an
    arithmetic expression: ``i + 1`` -> ``i`` forward and ``i - 1`` ->
    ``i`` reverse.  Returns the folded expression, or None when there is
    no trailing constant term to fold."""
    match = _TRAILING_CONSTANT.fullmatch(expr)
    if match is None:
        return None
    lhs, sign, literal = match.group(1), match.group(2), int(match.group(3))
    if sign == "+":
        new_literal = literal - 1 if direction == FORWARD else literal + 1
    else:
        new_literal = literal + 1 if direction == FORWARD else literal - 1
    if new_literal == 0:
        return lhs
    return "%s %s %d" % (lhs, sign, abs(new_literal))


def shift_index(expr, direction):
    """Shift an index expression between MATLAB and Python indexing.

    Args:
        expr: The index expression as a string -- a literal ("5"), a
            single variable ("i"), an indexed access ("x(i)", "x[i]"), or
            an arithmetic expression ("i + 1").
        direction: FORWARD ("forward") shifts MATLAB -> Python (subtract
            one, parentheses to brackets); REVERSE ("reverse") shifts
            Python -> MATLAB (add one, brackets to parentheses).

    Returns:
        The shifted expression as a string.  Single-variable indices,
        colon-ranges and end-keywords are returned unchanged.

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
    stripped = _strip_outer_parens(expr)
    if _is_arithmetic(stripped):
        folded = _fold_constant_shift(stripped, direction)
        if folded is not None:
            return folded
        op = "- 1" if direction == FORWARD else "+ 1"
        return "(%s) %s" % (stripped, op)
    return expr
