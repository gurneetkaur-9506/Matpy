import re

BUILTIN_RULES = {
    "abs": {"python": "np.abs", "arg_mode": "same"},
    "ceil": {"python": "np.ceil", "arg_mode": "same"},
    "cos": {"python": "np.cos", "arg_mode": "same"},
    "exp": {"python": "np.exp", "arg_mode": "same"},
    "fft": {"python": "np.fft.fft", "arg_mode": "same"},
    "fix": {"python": "np.trunc", "arg_mode": "same"},
    "floor": {"python": "np.floor", "arg_mode": "same"},
    "linspace": {"python": "np.linspace", "arg_mode": "same"},
    "log": {"python": "np.log", "arg_mode": "same"},
    "randn": {"python": "np.random.randn", "arg_mode": "randn"},
    "reshape": {"python": "np.reshape", "arg_mode": "tuple_dims"},
    "round": {"python": "np.round", "arg_mode": "same"},
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
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)", call)
    if not match:
        return call
    name = match.group(1)
    if name not in _REVERSE_BUILTIN_MAP:
        return call

    inner = match.group(2).strip()
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

    if mode == "randn":
        if len(args) == 1 and args[0].endswith(".shape"):
            return "%s(*%s)" % (py_name, args[0])
        return "%s(%s)" % (py_name, ", ".join(args))

    return call
