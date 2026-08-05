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


def _find_last_operator(expr):
    depth = 0
    i = len(expr) - 1
    while i >= 0:
        ch = expr[i]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            depth -= 1
        elif depth == 0:
            if ch in "*":
                if i > 0 and expr[i - 1] == ".":
                    return i - 1, ".*"
                return i, "*"
            if ch in "/":
                if i > 0 and expr[i - 1] == ".":
                    return i - 1, "./"
                return i, "/"
            if ch == "=" and i > 0 and expr[i - 1] == "~":
                return i - 1, "~="
            if ch in "+-":
                j = i - 1
                while j >= 0 and expr[j].isspace():
                    j -= 1
                if j >= 0 and (expr[j].isalnum() or expr[j] in ")]_."):
                    return i, ch
        i -= 1
    return None, None


def apply_operator_rule(expr):
    expr = expr.strip()
    idx, op = _find_last_operator(expr)
    if op is None:
        return expr

    left = apply_operator_rule(expr[:idx])
    right = apply_operator_rule(expr[idx + len(op):])

    if op == "*":
        return "%s @ %s" % (left, right)
    if op == ".*":
        return "%s * %s" % (left, right)
    if op == "./":
        return "%s / %s" % (left, right)
    if op == "/":
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
