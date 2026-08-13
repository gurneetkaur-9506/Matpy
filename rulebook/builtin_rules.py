import re

BUILTIN_RULES = {
    "abs": {"python": "np.abs", "arg_mode": "same"},
    "acos": {"python": "np.arccos", "arg_mode": "same"},
    "acosh": {"python": "np.arccosh", "arg_mode": "same"},
    "all": {"python": "np.all", "arg_mode": "same"},
    "angle": {"python": "np.angle", "arg_mode": "same"},
    "any": {"python": "np.any", "arg_mode": "same"},
    "asin": {"python": "np.arcsin", "arg_mode": "same"},
    "asinh": {"python": "np.arcsinh", "arg_mode": "same"},
    "atan": {"python": "np.arctan", "arg_mode": "same"},
    "atan2": {"python": "np.arctan2", "arg_mode": "same"},
    "atanh": {"python": "np.arctanh", "arg_mode": "same"},
    "awgn": {"python": "specialist_lib.awgn", "arg_mode": "same"},
    "ceil": {"python": "np.ceil", "arg_mode": "same"},
    "chirp": {"python": "specialist_lib.chirp", "arg_mode": "same"},
    "comm.AWGNChannel": {"python": "specialist_lib.awgn", "arg_mode": "same"},
    "conv": {"python": "specialist_lib.conv", "arg_mode": "same"},
    "cos": {"python": "np.cos", "arg_mode": "same"},
    "cosh": {"python": "np.cosh", "arg_mode": "same"},
    "cross": {"python": "np.cross", "arg_mode": "same"},
    "det": {"python": "np.linalg.det", "arg_mode": "same"},
    "diag": {"python": "np.diag", "arg_mode": "same"},
    "exp": {"python": "np.exp", "arg_mode": "same"},
    "expm1": {"python": "np.expm1", "arg_mode": "same"},
    "eye": {"python": "np.eye", "arg_mode": "same"},
    "fft": {"python": "np.fft.fft", "arg_mode": "same"},
    "fftshift": {"python": "np.fft.fftshift", "arg_mode": "same"},
    "fix": {"python": "np.trunc", "arg_mode": "same"},
    "flipud": {"python": "np.flipud", "arg_mode": "same"},
    "floor": {"python": "np.floor", "arg_mode": "same"},
    "hann": {"python": "scipy.signal.windows.hann", "arg_mode": "same"},
    "hypot": {"python": "np.hypot", "arg_mode": "same"},
    "ifft": {"python": "np.fft.ifft", "arg_mode": "same"},
    "ifftshift": {"python": "np.fft.ifftshift", "arg_mode": "same"},
    "imag": {"python": "np.imag", "arg_mode": "same"},
    "inv": {"python": "np.linalg.inv", "arg_mode": "same"},
    "kron": {"python": "np.kron", "arg_mode": "same"},
    "linspace": {"python": "np.linspace", "arg_mode": "same"},
    "log": {"python": "np.log", "arg_mode": "same"},
    "log1p": {"python": "np.log1p", "arg_mode": "same"},
    "log2": {"python": "np.log2", "arg_mode": "same"},
    "logspace": {"python": "np.logspace", "arg_mode": "same"},
    "mod": {"python": "np.mod", "arg_mode": "same"},
    "ones": {"python": "np.ones", "arg_mode": "tuple_dims"},
    "phased.ArrayResponse": {"python": "specialist_lib.array_factor", "arg_mode": "same"},
    "phased.Beamformer": {"python": "specialist_lib.beamform", "arg_mode": "same"},
    "pinv": {"python": "np.linalg.pinv", "arg_mode": "same"},
    "pow2": {"python": "np.exp2", "arg_mode": "same"},
    "rand": {"python": "np.random.rand", "arg_mode": "randn"},
    "randn": {"python": "np.random.randn", "arg_mode": "randn"},
    "real": {"python": "np.real", "arg_mode": "same"},
    "rem": {"python": "np.fmod", "arg_mode": "same"},
    "reshape": {"python": "np.reshape", "arg_mode": "tuple_dims"},
    "round": {"python": "np.round", "arg_mode": "same"},
    "sign": {"python": "np.sign", "arg_mode": "same"},
    "sin": {"python": "np.sin", "arg_mode": "same"},
    "sinh": {"python": "np.sinh", "arg_mode": "same"},
    "size": {"python": ".shape", "arg_mode": "size"},
    "squeeze": {"python": "np.squeeze", "arg_mode": "same"},
    "sqrt": {"python": "np.sqrt", "arg_mode": "same"},
    "steervec": {"python": "specialist_lib.steering_vector", "arg_mode": "same"},
    "tan": {"python": "np.tan", "arg_mode": "same"},
    "tanh": {"python": "np.tanh", "arg_mode": "same"},
    "trace": {"python": "np.trace", "arg_mode": "same"},
    "tril": {"python": "np.tril", "arg_mode": "same"},
    "triu": {"python": "np.triu", "arg_mode": "same"},
    "unique": {"python": "np.unique", "arg_mode": "same"},
    "zeros": {"python": "np.zeros", "arg_mode": "tuple_dims"},
}


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


def _size_dim(dim):
    if dim.isdigit():
        return str(int(dim) - 1)
    return "(%s - 1)" % dim


def _split_reverse_call(expr):
    """Split ``expr`` into ``(name, inner)`` only when it is exactly one
    well-formed call: a name (possibly dotted, e.g. ``np.sin``) followed by a
    balanced parenthesized argument list that ends at the string's end.

    A greedy regex like ``name(.*)`` is not enough here: it would read
    ``np.sin(theta) - np.sin(theta0)`` as the single call
    ``np.sin(theta) - np.sin(theta0)``.  Requiring balance means such an
    expression (with a leftover ``- np.sin(theta0)`` after the first closing
    paren) is not treated as one call.
    """
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\(", expr)
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


# Reverse mapping derived from the forward rules: every built-in whose
# python target is a plain 'np.<name>' call can be mirrored back to its
# MATLAB name. The 'size' rule (-> '.shape') and attribute access are not
# calls and are excluded.
_REVERSE_BUILTIN_MAP = {
    rule["python"]: name
    for name, rule in BUILTIN_RULES.items()
    if rule["python"].startswith("np.")
}


def _expand_tuple_arg(arg):
    arg = arg.strip()
    if arg.startswith("(") and arg.endswith(")"):
        return [a.strip() for a in _split_args(arg[1:-1]) if a.strip()]
    return [arg]


def apply_builtin_rule_reverse(call):
    call = call.strip()
    match = _split_reverse_call(call)
    if not match:
        return call
    name, inner = match
    if name not in _REVERSE_BUILTIN_MAP:
        return call

    inner = inner.strip()
    args = _split_args(inner) if inner else []
    args = [apply_builtin_rule_reverse(a) for a in args]

    matlab_name = _REVERSE_BUILTIN_MAP[name]
    rule = BUILTIN_RULES[matlab_name]

    if rule["arg_mode"] == "tuple_dims":
        if matlab_name == "reshape":
            if not args:
                return "%s()" % matlab_name
            flat = [args[0]]
            for a in args[1:]:
                flat.extend(_expand_tuple_arg(a))
            return "%s(%s)" % (matlab_name, ", ".join(flat))
        flat = []
        for a in args:
            flat.extend(_expand_tuple_arg(a))
        return "%s(%s)" % (matlab_name, ", ".join(flat))

    if rule["arg_mode"] == "randn":
        if len(args) == 1 and args[0].startswith("*") and args[0][1:].endswith(".shape"):
            return "%s(size(%s))" % (matlab_name, args[0][1:].rsplit(".shape", 1)[0])
        return "%s(%s)" % (matlab_name, ", ".join(args))

    return "%s(%s)" % (matlab_name, ", ".join(args))


def apply_builtin_rule(call):
    call = call.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)", call)
    if not match:
        return call
    name = match.group(1)
    if name not in BUILTIN_RULES:
        return call

    inner = match.group(2).strip()
    args = _split_args(inner) if inner else []
    args = [apply_builtin_rule(a) for a in args]

    rule = BUILTIN_RULES[name]
    py_name = rule["python"]
    mode = rule["arg_mode"]

    if mode == "same":
        return "%s(%s)" % (py_name, ", ".join(args))

    if mode == "tuple_dims":
        if name == "reshape":
            if len(args) < 2:
                return "%s(%s)" % (py_name, ", ".join(args))
            if len(args) == 2:
                return "%s(%s, %s)" % (py_name, args[0], args[1])
            return "%s(%s, (%s))" % (py_name, args[0], ", ".join(args[1:]))
        if len(args) == 1:
            return "%s(%s)" % (py_name, args[0])
        return "%s((%s))" % (py_name, ", ".join(args))

    if mode == "size":
        if len(args) == 1:
            return "%s.shape" % args[0]
        return "%s.shape[%s]" % (args[0], _size_dim(args[1]))

    if mode == "randn":
        if len(args) == 1 and args[0].endswith(".shape"):
            return "%s(*%s)" % (py_name, args[0])
        return "%s(%s)" % (py_name, ", ".join(args))

    return call
