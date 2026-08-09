import re

OPERATOR_RULES = {
    "matrix_multiply": {"matlab": "*", "python": "@"},
    "elementwise_multiply": {"matlab": ".*", "python": "*"},
    "matrix_right_divide": {
        "matlab": "/",
        "python": "np.linalg.solve(%s.T, %s.T).T",
    },
    "elementwise_divide": {"matlab": "./", "python": "/"},
}

# Simple operator mappings that CAN be mirrored from the forward rules:
# the forward 'python' value is itself a single-char operator, so we just
# swap it back to the 'matlab' operator.
REVERSE_OPERATOR_RULES = {}
for _rule in OPERATOR_RULES.values():
    _py = _rule["python"]
    if isinstance(_py, str) and len(_py) == 1 and _py in "*@/":
        REVERSE_OPERATOR_RULES[_py] = _rule["matlab"]

# matrix_right_divide is NOT in REVERSE_OPERATOR_RULES: forward translates
# MATLAB 'a / b' into 'np.linalg.solve(b.T, a.T).T', a template, not a
# single operator. Python '/' is always element-wise, so the matrix divide
# needs a separate reverse-only pattern rule (see apply_operator_rule_reverse).

_TWO_CHAR_OPS = {".*", "./"}

# Any token matching this pattern is a single atomic scientific-notation
# number literal (e.g. 10e-6, 1E+10, 2.5e-9). It must never be split by
# operator scanning, otherwise the '-'/'+' in the exponent or the '.' in the
# mantissa could be misread as operators in a larger expression.
_SCIENTIFIC_LITERAL = re.compile(r"\d+(?:\.\d+)?[eE][+-]?\d+")


def scientific_literals(expr):
    """Return every scientific-notation number literal found in ``expr``.

    The whole class of literals matching ``digits(.digits)?[eE][+-]?digits``
    is recognized here, so callers and tests can rely on a single tokenizer
    instead of value-specific special cases.
    """
    return _SCIENTIFIC_LITERAL.findall(expr)


def _protected_positions(expr):
    protected = set()
    for match in _SCIENTIFIC_LITERAL.finditer(expr):
        protected.update(range(match.start(), match.end()))
    return protected


_SCALAR_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
_IMAG_SCALAR_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?i$")
_SCALAR_CONSTANTS = {"pi", "e"}
# Calls that always yield a scalar: a count (length/numel/len), a rounded
# value (round), or a reduction over a whole array (max/min produce a scalar
# for the 1-D inputs used here).  Recognized so a later '/' with one of these
# as the divisor stays element-wise instead of becoming a matrix solve.
_SCALAR_FUNCTIONS = {"length", "numel", "len", "round", "max", "min"}


def _split_call_args(expr):
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
    if not match:
        return None
    start = match.end() - 1
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


def _strip_outer_parens(expr):
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        spanning = True
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    spanning = False
                    break
        if not spanning:
            break
        expr = expr[1:-1].strip()
    return expr


def is_scalar_like(expr):
    expr = _strip_outer_parens(expr)
    if not expr:
        return True
    if _SCALAR_RE.fullmatch(expr) or _IMAG_SCALAR_RE.fullmatch(expr):
        return True
    if expr in _SCALAR_CONSTANTS:
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        return False

    call = _split_call_args(expr)
    if call is not None:
        name, argtext = call
        args = [a.strip() for a in argtext.split(",") if a.strip()] if argtext else []
        if name in _SCALAR_FUNCTIONS:
            return True
        if not args:
            return False
        if any(a == ":" for a in args):
            return False
        return all(is_scalar_like(a) for a in args)

    depth = 0
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch in "+-*/":
            if ch == "-" and (i == 0 or expr[i - 1] in "+-*/"):
                continue
            return is_scalar_like(expr[:i]) and is_scalar_like(expr[i + 1:])
    return False


def _find_last_operator(expr):
    protected = _protected_positions(expr)

    def _find(chars):
        depth = 0
        i = len(expr) - 1
        while i >= 0:
            ch = expr[i]
            if i in protected:
                i -= 1
                continue
            if ch in ")]":
                depth += 1
            elif ch in "([":
                depth -= 1
            elif depth == 0:
                if ch in chars:
                    return i
            i -= 1
        return None

    # Lowest precedence first: relational (~=), then additive (+/-), then
    # multiplicative (* / .* ./). Splitting at the lowest-precedence operator
    # keeps a + b * c grouped as a + (b * c), so scalar/matrix decisions are
    # made on the true operands.
    idx = _find("~=")
    if idx is not None and idx > 0 and expr[idx - 1] == "~":
        return idx - 1, "~="

    idx = _find("+-")
    if idx is not None:
        j = idx - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j >= 0 and (expr[j].isalnum() or expr[j] in ")]_."):
            return idx, expr[idx]

    idx = _find("*/")
    if idx is not None:
        if (
            expr[idx] == "*"
            and idx > 0
            and expr[idx - 1] == "."
            and (idx - 1) not in protected
        ):
            return idx - 1, ".*"
        if (
            expr[idx] == "/"
            and idx > 0
            and expr[idx - 1] == "."
            and (idx - 1) not in protected
        ):
            return idx - 1, "./"
        return idx, expr[idx]

    # Element-wise power (.^) binds tighter than every multiplicative
    # operator, so it is looked for only after * / .* ./ have been ruled
    # out at the current depth.  A bare '^' (MATLAB matrix power) is not an
    # element-wise operator and is left for the caller to pass through.
    idx = _find("^")
    if idx is not None:
        if (
            idx > 0
            and expr[idx - 1] == "."
            and (idx - 1) not in protected
        ):
            return idx - 1, ".^"
        return None, None

    return None, None


def _split_transpose(expr):
    """Split a MATLAB postfix transpose operator off the end of ``expr``.

    Returns ``(base, kind)`` where ``kind`` is ``"conj"`` for ``expr'``
    (conjugate transpose) or ``"plain"`` for ``expr.'``, or None when
    ``expr`` does not end in a transpose operator.  The rule applies to any
    expression the operator follows: a plain variable, a function-call
    result, an indexed expression, or a parenthesized compound expression.
    A single-quoted string literal is not a transpose.
    """
    expr = expr.strip()
    if not expr or expr[-1] != "'":
        return None
    if _is_string_literal(expr):
        return None
    if len(expr) >= 2 and expr[-2] == ".":
        return expr[:-2].strip(), "plain"
    return expr[:-1].strip(), "conj"


def _is_string_literal(expr):
    """True when ``expr`` is a single-quoted MATLAB string literal (e.g.
    'abc' or 'it''s'), so its trailing quote is not a transpose operator."""
    expr = expr.strip()
    if not expr.startswith("'"):
        return False
    i = 1
    n = len(expr)
    while i < n:
        if expr[i] == "'":
            if i + 1 < n and expr[i + 1] == "'":
                i += 2
                continue
            return i == n - 1
        i += 1
    return False


def apply_transpose_rule(base, transpose_kind):
    """Map a transposed expression to its Python form.

    ``expr'``  (conjugate transpose) -> ``np.conj(expr).T``
    ``expr.'`` (plain transpose)     -> ``expr.T``
    """
    if transpose_kind == "plain":
        return "%s.T" % base
    return "np.conj(%s).T" % base


def apply_operator_rule(expr):
    expr = expr.strip()
    idx, op = _find_last_operator(expr)
    if op is None:
        transposed = _split_transpose(expr)
        if transposed is not None:
            base, transpose_kind = transposed
            return apply_transpose_rule(
                apply_operator_rule(base), transpose_kind
            )
        return expr

    left = apply_operator_rule(expr[:idx])
    right = apply_operator_rule(expr[idx + len(op):])

    if op == "*":
        if not is_scalar_like(expr[:idx]) and not is_scalar_like(
            expr[idx + len(op):]
        ):
            return "%s @ %s" % (left, right)
        return "%s * %s" % (left, right)
    if op == ".*":
        return "%s * %s" % (left, right)
    if op == "./":
        return "%s / %s" % (left, right)
    if op == ".^":
        return "%s ** %s" % (left, right)
    if op == "/":
        # MATLAB '/' is matrix right-division (solve a * x = b) ONLY when
        # both operands are arrays; dividing by a scalar (a count like
        # length(P2), a literal, or a scalar variable) is element-wise and
        # maps to plain Python '/'.
        if is_scalar_like(expr[:idx]) or is_scalar_like(expr[idx + len(op):]):
            return "%s / %s" % (left, right)
        template = OPERATOR_RULES["matrix_right_divide"]["python"]
        return template % (right, left)
    if op in ("+", "-"):
        # Rebuild so postfix transposes on the operands (e.g. ``A + B'``)
        # are preserved; '+'/'-' map to themselves.
        return "%s %s %s" % (left, op, right)
    return expr


def _find_last_reverse_operator(expr):
    protected = _protected_positions(expr)
    depth = 0
    i = len(expr) - 1
    while i >= 0:
        ch = expr[i]
        if i in protected:
            i -= 1
            continue
        if ch in ")]":
            depth += 1
        elif ch in "([":
            depth -= 1
        elif depth == 0:
            if ch in "*@/":
                return i, ch
        i -= 1
    return None, None


def apply_operator_rule_reverse(expr):
    expr = expr.strip()

    # Reverse-only rule for matrix right divide: forward turns MATLAB
    # 'a / b' into 'np.linalg.solve(b.T, a.T).T'. That template CANNOT be
    # mirrored back onto the '/' operator because Python '/' is always
    # element-wise (-> './'). So the matrix divide is matched by its own
    # pattern and rewritten to 'a / b', recursing over both operands.
    match = re.fullmatch(r"np\.linalg\.solve\((.*?)\.T, (.*?)\.T\)\.T", expr)
    if match:
        right = match.group(1)
        left = match.group(2)
        return "%s / %s" % (
            apply_operator_rule_reverse(left),
            apply_operator_rule_reverse(right),
        )

    idx, op = _find_last_reverse_operator(expr)
    if op is None:
        return expr

    left = apply_operator_rule_reverse(expr[:idx])
    right = apply_operator_rule_reverse(expr[idx + 1:])

    matlab_op = REVERSE_OPERATOR_RULES.get(op)
    if matlab_op is None:
        return expr
    return "%s %s %s" % (left, matlab_op, right)
