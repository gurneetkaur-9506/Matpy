OPERATOR_RULES = {
    "matrix_multiply": {"matlab": "*", "python": "@"},
    "elementwise_multiply": {"matlab": ".*", "python": "*"},
    "matrix_right_divide": {
        "matlab": "/",
        "python": "np.linalg.solve(%s.T, %s.T).T",
    },
    "elementwise_divide": {"matlab": "./", "python": "/"},
}

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
