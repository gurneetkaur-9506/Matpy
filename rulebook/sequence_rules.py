import re

# General, table-driven reverse mapping for the numpy sequence-construction
# family.  Each member is one table entry; the family member name is used
# directly as the MATLAB output name (or "range" for the colon-range form
# of np.arange).  Argument-count handling (np.arange with 1/2/3 args,
# np.zeros/ones with a scalar vs a tuple shape) is shared below.
SEQUENCE_RULES = {
    "arange": "range",
    "linspace": "linspace",
    "zeros": "zeros",
    "ones": "ones",
}

_SEQUENCE_CALL = re.compile(r"(?:np|numpy)\.(arange|linspace|zeros|ones)\s*\((.*)\)\s*$", re.DOTALL)


def _split_args(s):
    args = []
    depth = 0
    current = ""
    for ch in s:
        if ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def apply_sequence_rule_reverse(expr):
    expr = expr.strip()
    match = _SEQUENCE_CALL.fullmatch(expr)
    if not match:
        return expr
    name = match.group(1)
    inner = match.group(2).strip()
    args = _split_args(inner) if inner else []

    if name == "arange":
        if len(args) == 3:
            return "%s:%s:%s" % (args[0], args[2], args[1])
        if len(args) == 2:
            return "%s:%s" % (args[0], args[1])
        if len(args) == 1:
            return "0:%s-1" % args[0]
        return expr

    matlab_name = SEQUENCE_RULES[name]
    if name in ("zeros", "ones"):
        if len(args) == 1:
            shape = args[0].strip()
            if shape.startswith("(") and shape.endswith(")"):
                dims = _split_args(shape[1:-1])
                return "%s(%s)" % (matlab_name, ", ".join(dims))
            return "%s(1, %s)" % (matlab_name, shape)
        return "%s(%s)" % (matlab_name, ", ".join(args))

    return "%s(%s)" % (matlab_name, ", ".join(args))
