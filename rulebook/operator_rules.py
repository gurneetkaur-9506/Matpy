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

_SCALAR_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
_IMAG_SCALAR_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?i$")
_SCALAR_CONSTANTS = {"pi", "e"}
_SCALAR_FUNCTIONS = {"length", "numel", "len"}


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


def is_scalar_like(expr):
    expr = expr.strip()
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
    def _find(chars):
        depth = 0
        i = len(expr) - 1
        while i >= 0:
            ch = expr[i]
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
        if expr[idx] == "*" and idx > 0 and expr[idx - 1] == ".":
            return idx - 1, ".*"
        if expr[idx] == "/" and idx > 0 and expr[idx - 1] == ".":
            return idx - 1, "./"
        return idx, expr[idx]

    return None, None


def apply_operator_rule(expr):
    expr = expr.strip()
    idx, op = _find_last_operator(expr)
    if op is None:
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
    if op == "/":
        # MATLAB '/' is matrix right-division (solve a * x = b) ONLY when
        # both operands are arrays; dividing by a scalar (a count like
        # length(P2), a literal, or a scalar variable) is element-wise and
        # maps to plain Python '/'.
        if is_scalar_like(expr[:idx]) or is_scalar_like(expr[idx + len(op):]):
            return "%s / %s" % (left, right)
        template = OPERATOR_RULES["matrix_right_divide"]["python"]
        return template % (right, left)
    return expr


def _find_last_reverse_operator(expr):
    depth = 0
    i = len(expr) - 1
    while i >= 0:
        ch = expr[i]
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
