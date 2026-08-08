"""MATLAB multi-output assignment ``[a, b] = func(...)`` decomposition.

MATLAB functions commonly return several outputs via a bracketed target
list. This module provides a general rule that looks the function name
up in a small registry and emits the matching multi-line (or tuple)
Python decomposition, instead of special-casing a single function:

- max/min  -> (value, index) as ``np.max`` / ``np.argmax``
- sort     -> (sorted_values, indices) as ``np.sort`` / ``np.argsort``
- size     -> (dim1, dim2, ...) via ``x.shape`` tuple unpacking
- find     -> (row, col) via ``np.where``

Nested reduction functions of any depth are resolved recursively: the
outermost call is the registered multi-output function, and everything
below it is peeled off one layer at a time (max/min/sum/mean wrapping
abs/transpose/other reductions) from the innermost operation outward:
``[v, i] = max(max(abs(X')))`` becomes ``np.max(np.max(np.abs(np.conj(X).T)))``
for the value and the same expression inside ``np.argmax`` for the index.
"""

import re

UNRESOLVED = "UNRESOLVED"

_TARGET_RE = re.compile(r"^\s*\[(.*)\]\s*$")
_CALL_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STRING_LITERAL_RE = re.compile(r"^'(?:[^']|'')*'$")

# Registry of known multi-output functions. Each entry declares how the
# outputs decompose into Python:
#   - "pair": first output is a value, second is an index/order from a
#     sibling numpy function (with an optional "options" table mapping
#     trailing string-literal modes to a suffix; None means string args
#     are not supported).
#   - "shape": outputs are the dimensions of the single array argument.
#   - "where": outputs are the index arrays of a single condition.
MULTI_OUTPUT_RULES = {
    "max": {
        "kind": "pair",
        "value": "np.max",
        "index": "np.argmax",
        "options": None,
    },
    "min": {
        "kind": "pair",
        "value": "np.min",
        "index": "np.argmin",
        "options": None,
    },
    "sort": {
        "kind": "pair",
        "value": "np.sort",
        "index": "np.argsort",
        "options": {"descend": "[::-1]", "ascend": ""},
    },
    "size": {"kind": "shape"},
    "find": {"kind": "where"},
}

# Single-output reduction functions that may wrap other reductions inside
# a multi-output assignment.  In MATLAB these reduce an array; the numpy
# equivalent is a plain ``np.<name>(...)`` call.  Peeling them outward
# lets the resolver compose arbitrary nesting depths.
_REDUCTION_CALLS = {
    "max": "np.max",
    "min": "np.min",
    "sum": "np.sum",
    "mean": "np.mean",
}


def _split_top_level(text, sep):
    """Split on ``sep`` at bracket depth 0, ignoring separators inside
    string literals (including MATLAB's doubled-quote escape)."""
    parts = []
    depth = 0
    in_string = False
    current = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            current += ch
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    current += "'"
                    i += 1
                else:
                    in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            current += ch
            i += 1
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def _split_call(expr):
    """Return (name, argtext) when ``expr`` is exactly ``name(args)``."""
    match = _CALL_RE.match(expr)
    if not match:
        return None
    start = expr.find("(", match.start())
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                if expr[i + 1:].strip() == "":
                    return match.group(1), expr[start + 1:i]
                return None
    return None


def _split_targets(text):
    """Split a bracketed target ``[a, b, ...]`` into plain identifiers."""
    match = _TARGET_RE.fullmatch(text)
    if not match:
        return None
    inner = match.group(1)
    if not inner.strip():
        return None
    targets = [t.strip() for t in _split_top_level(inner, ",") if t.strip()]
    if not targets or not all(_IDENTIFIER_RE.fullmatch(t) for t in targets):
        return None
    return targets


def _is_string_literal(text):
    return bool(_STRING_LITERAL_RE.fullmatch(text.strip()))


def _resolve_nested(expr, translate_arg):
    """Resolve a nested reduction expression outward, one layer at a time.

    The outermost call of a multi-output assignment (e.g. ``max``) is
    handled by the registry's own rule; everything below it may itself be
    a chain of reduction functions (max/min/sum/mean) wrapping abs,
    transpose, or other calls.  This resolver identifies the innermost
    operation and composes the numpy equivalent outward:

        max(max(abs(X')))  ->  np.max(np.abs(np.conj(X).T))

    ``translate_arg`` handles the innermost non-reduction operand (an
    identifier, an abs() call, a transpose, ...).
    """
    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name in _REDUCTION_CALLS:
            args = [a for a in _split_top_level(argtext, ",") if a.strip()]
            if len(args) == 1:
                inner = _resolve_nested(args[0], translate_arg)
                if inner == UNRESOLVED:
                    return UNRESOLVED
                return "%s(%s)" % (_REDUCTION_CALLS[name], inner)
    return translate_arg(expr)


def _dim_to_axis(dim):
    if dim.isdigit():
        return str(int(dim) - 1)
    return "(%s - 1)" % dim


def _translate_pair(targets, args, translate_arg, rule):
    suffix = ""
    real_args = []
    for a in args:
        if _is_string_literal(a):
            options = rule.get("options")
            if options is None or a.strip()[1:-1] not in options:
                return None
            suffix = options[a.strip()[1:-1]]
            continue
        real_args.append(a.strip())

    if len(real_args) == 1:
        arg = _resolve_nested(real_args[0], translate_arg)
        if arg == UNRESOLVED:
            return None
        call = "%s(%s)" % (rule["value"], arg)
        index_call = "%s(%s)" % (rule["index"], arg)
    elif len(real_args) == 3 and real_args[1] == "[]" and rule.get("options") is None:
        # max(x, [], dim) / min(x, [], dim): reduce along a dimension.
        base = _resolve_nested(real_args[0], translate_arg)
        if base == UNRESOLVED:
            return None
        axis = _dim_to_axis(real_args[2].strip())
        call = "%s(%s, axis=%s)" % (rule["value"], base, axis)
        index_call = "%s(%s, axis=%s)" % (rule["index"], base, axis)
    else:
        return None

    lines = ["%s = %s%s" % (targets[0], call, suffix)]
    if len(targets) >= 2:
        lines.append("%s = %s%s" % (targets[1], index_call, suffix))
    if len(targets) > 2:
        return None
    return lines


def _translate_shape(targets, args, translate_arg):
    if len(args) != 1 or _is_string_literal(args[0]):
        return None
    arg = translate_arg(args[0].strip())
    if arg == UNRESOLVED:
        return None
    return ["%s = %s.shape" % (", ".join(targets), arg)]


def _translate_where(targets, args, translate_arg):
    if len(args) != 1 or _is_string_literal(args[0]):
        return None
    arg = translate_arg(args[0].strip())
    if arg == UNRESOLVED:
        return None
    if len(targets) == 2:
        return ["%s = np.where(%s)" % (", ".join(targets), arg)]
    if len(targets) == 1:
        return ["%s = np.where(%s)[0]" % (targets[0], arg)]
    return None


def translate_multi_output_assignment(target_text, value_expr, translate_arg):
    """Return the Python decomposition of ``[a, b] = func(...)`` as a list
    of lines, or ``None`` when the pattern is not recognized.

    Args:
        target_text (str): The assignment target, e.g. ``[v, i]``.
        value_expr (str): The right-hand side, e.g. ``max(abs(x))``.
        translate_arg (callable): Maps a single argument expression from
            MATLAB to Python; must return UNRESOLVED when untranslatable.

    Returns:
        list or None: One line per Python statement, or None if ``func``
        is not in the registry, the target is not plain identifiers, or
        the argument count/signature is not supported.
    """
    targets = _split_targets(target_text)
    if targets is None:
        return None
    call = _split_call(value_expr)
    if call is None:
        return None
    name, argtext = call
    rule = MULTI_OUTPUT_RULES.get(name)
    if rule is None:
        return None
    args = [a for a in _split_top_level(argtext, ",") if a.strip()]
    kind = rule["kind"]
    if kind == "pair":
        return _translate_pair(targets, args, translate_arg, rule)
    if kind == "shape":
        return _translate_shape(targets, args, translate_arg)
    if kind == "where":
        return _translate_where(targets, args, translate_arg)
    return None
