import re

BUILTIN_RULES = {
    "abs": {"python": "np.abs", "arg_mode": "same"},
    "cos": {"python": "np.cos", "arg_mode": "same"},
    "exp": {"python": "np.exp", "arg_mode": "same"},
    "fft": {"python": "np.fft.fft", "arg_mode": "same"},
    "linspace": {"python": "np.linspace", "arg_mode": "same"},
    "log": {"python": "np.log", "arg_mode": "same"},
    "reshape": {"python": "np.reshape", "arg_mode": "tuple_dims"},
    "sin": {"python": "np.sin", "arg_mode": "same"},
    "size": {"python": ".shape", "arg_mode": "size"},
    "sqrt": {"python": "np.sqrt", "arg_mode": "same"},
    "tan": {"python": "np.tan", "arg_mode": "same"},
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


def apply_builtin_rule(call):
    call = call.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", call)
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

    return call
