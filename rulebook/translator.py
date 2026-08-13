import ast
import re

from reader.extract_structure import split_range

from .attribute_rules import apply_attribute_rule_reverse
from .builtin_rules import BUILTIN_RULES, apply_builtin_rule, apply_builtin_rule_reverse
from .complex_rules import apply_complex_rule
from .format_rules import convert_fprintf
from .indexing_rules import apply_indexing_rule, apply_indexing_rule_reverse
from .keyword_rules import (
    identifier_tokens,
    rename_comment,
    rename_for,
    rename_text,
    should_rename,
)
from .multi_output_rules import (
    _REDUCTION_CALLS,
    _dim_to_axis,
    translate_multi_output_assignment,
)
from .operator_rules import (
    _find_last_operator,
    _split_transpose,
    apply_operator_rule_reverse,
    apply_transpose_rule,
    is_known_scalar,
)
from .shape_inference import MATRIX, VECTOR, infer_shapes
from .scan_rules import (

    translate_feof_loop,
    translate_feof_statement,
    translate_fopen,
    translate_fscanf,
)
from .sequence_rules import apply_sequence_rule_reverse

UNRESOLVED = "UNRESOLVED"


class _ScalarScope(set):
    """A set of known-scalar names that also records the names the inference
    pass proved to be definite non-scalars (vector/matrix).  The extra
    attribute rides along the same ``scalars`` argument that every
    expression-translation call site already threads, so an operator rule can
    distinguish a proven array (matrix power) from an unknown operand
    (element-wise power) without a second threaded container."""

    def __init__(self, *args, non_scalars=None):
        super().__init__(*args)
        self.non_scalars = non_scalars or set()


def _non_scalar_names(shapes, key, renames):
    """Names the inference pass proved to be definite non-scalars (vector or
    matrix) in the given scope, in the renamed identifier space."""
    if shapes is None:
        return set()
    info = shapes.scope_for(key)
    if info is None:
        return set()
    return {
        renames.get(name, name)
        for name, shape in info.shapes.items()
        if shape in (VECTOR, MATRIX)
    }

_COMMANDS = {
    "clear": "",
    "close all": "",
    "clc": "",
    "figure": "plt.figure()",
}

_OTHER_BUILTINS = {
    "clc", "clear", "close", "cos", "exp", "log",
    "sin", "sqrt", "tan",
}

# General, table-driven mapping of the MATLAB plot-styling/config command
# class to its matplotlib equivalent.  Adding a new plotting command is a
# single table row; no per-command code.  A row's spec supports:
#   "func"   matplotlib callable; arguments are translated and passed
#            through (xlim([0 10]) -> plt.xlim([0, 10])).
#   "flags"  bare-command flag words (grid on, axis equal) -> the call
#            arguments to emit; a bare command with no flag emits a
#            no-argument call (colorbar -> plt.colorbar()).
#   "vector" the single argument is a limit vector flattened to a Python
#            list, for matplotlib functions that want [xmin, xmax].
#   "noop"   the command has no matplotlib equivalent; bare commands emit
#            an explanatory comment instead of a call.
PLOT_COMMANDS = {
    "axis": {"func": "plt.axis", "vector": True, "flags": {"auto": "'auto'", "equal": "'equal'", "image": "'scaled'", "normal": "'auto'", "off": "'off'", "on": "'on'", "square": "'square'", "tight": "'tight'"}},
    "axes": {"func": "plt.axes"},
    "caxis": {"func": "plt.clim", "spread": True},
    "cla": {"func": "plt.cla"},
    "clf": {"func": "plt.clf"},
    "colorbar": {"func": "plt.colorbar"},
    "figure": {"func": "plt.figure"},
    "gca": {"func": "plt.gca"},
    "gcf": {"func": "plt.gcf"},
    "grid": {"func": "plt.grid", "flags": {"off": "False", "on": "True", "minor": "True, which='minor'"}},
    "hold": {"noop": "matplotlib holds axes by default"},
    "legend": {"func": "plt.legend"},
    "plot": {"func": "plt.plot"},
    "shading": {"noop": "matplotlib handles shading via the colormap"},
    "subplot": {"func": "plt.subplot"},
    "title": {"func": "plt.title"},
    "xlabel": {"func": "plt.xlabel"},
    "xlim": {"func": "plt.xlim", "vector": True},
    "ylabel": {"func": "plt.ylabel"},
    "ylim": {"func": "plt.ylim", "vector": True},
    "zlabel": {"func": "ax.set_zlabel"},
}

# MATLAB functions that map 1:1 onto a numpy call but whose arguments need
# the full expression translator (indexing, operators, nested calls), so a
# call is never silently misread as an indexing expression.
_NUMPY_CALLS = {
    "conj": "np.conj",
    "fliplr": "np.fliplr",
    "log10": "np.log10",
}

# MATLAB inverse-trig functions in degrees: asind(x) = 180/pi * arcsin(x).
_TRIG_DEGREES = {
    "asind": ("np.arcsin", "np.degrees"),
    "acosd": ("np.arccos", "np.degrees"),
    "atand": ("np.arctan", "np.degrees"),
}

# MATLAB numeric constants matched as standalone identifiers anywhere in an
# expression (pi -> np.pi, eps -> np.finfo(float).eps).
_CONSTANTS = {
    "pi": "np.pi",
    "eps": "np.finfo(float).eps",
}


def _split_top_level(text, sep):
    parts = []
    depth = 0
    current = ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _split_call(expr, dotted=False):
    pattern = r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("
    if dotted:
        pattern = r"\s*([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*)\s*\("
    match = re.match(pattern, expr)
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


def _split_assignment(text):
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return text[:i].strip(), text[i + 1:].strip()
    return None, None


def _translate_matrix(expr, scalars=None, declared=None):
    inner = expr[1:-1].strip()
    rows = _split_top_level(inner, ";")
    arrays = []
    for row in rows:
        cells = [c for c in re.split(r"[,\s]+", row) if c]
        translated = [_translate_expr(c, scalars, declared) for c in cells]
        if any(t == UNRESOLVED for t in translated):
            return UNRESOLVED
        arrays.append("[%s]" % ", ".join(translated))
    return "np.array([%s])" % ", ".join(arrays)


def _is_index_like(arg):
    arg = arg.strip()
    if not arg:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?|end(?:-\d+)?|[A-Za-z_][A-Za-z0-9_]*|:", arg):
        return True
    return ":" in arg


def _valid_python(expr):
    try:
        ast.parse(expr)
        return True
    except SyntaxError:
        return False


def _translate_range_part(part, scalars=None, declared=None):
    part = part.strip()
    if not part:
        return UNRESOLVED
    translated = _translate_expr(part, scalars, declared)
    if translated != UNRESOLVED and "length(" not in translated and _valid_python(translated):
        return translated
    return apply_indexing_rule(part)


_LINSPACE_STEP = re.compile(
    r"^\s*\(?\s*1\s*/\s*\(\s*length\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*-\s*1\s*\)\s*\)?\s*$"
)

def _is_scalar(expr, scalars):
    """True when ``expr`` is a scalar: a literal, a scalar-producing call
    (length/numel/round/max/...), a variable previously assigned a scalar
    value (or a function parameter/loop index the Structure already tracks),
    or an operator expression built entirely from scalars.  Outer
    parentheses are stripped first so a divisor like ``(2*fs)`` is recognized
    as the scalar it is."""
    return is_known_scalar(expr, scalars)


def _inclusive_stop(expr):
    """Build the exclusive stop for np.arange() from an inclusive MATLAB
    endpoint: 'b' becomes 'b + 1', but a trailing '- 1' cancels it, so
    'len(P1) - 1' stays 'len(P1)' instead of 'len(P1) - 1 + 1'."""
    stripped = expr.strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    match = re.fullmatch(r"(.*) - 1", stripped)
    if match and match.group(1).strip():
        return match.group(1)
    return "%s + 1" % expr


def _outer_parens_wrap(expr):
    """True when ``expr`` is wrapped in one pair of matching parentheses
    that span the whole expression (so they can be dropped safely)."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(expr) - 1:
                return False
    return depth == 0


def _linspace_count(step_part):
    match = _LINSPACE_STEP.fullmatch(step_part)
    if match:
        return "len(%s)" % match.group(1)
    return None


def _translate_range(parts, scalars=None, declared=None):
    translated = [_translate_range_part(p, scalars, declared) for p in parts]
    if any(t == UNRESOLVED for t in translated):
        return UNRESOLVED
    if len(parts) == 2:
        return "np.arange(%s, %s)" % (translated[0], _inclusive_stop(translated[1]))
    if len(parts) == 3:
        count = _linspace_count(parts[1])
        if count is not None:
            return "np.linspace(%s, %s, %s)" % (translated[0], translated[2], count)
        return "np.arange(%s, %s, %s)" % (translated[0], translated[2], translated[1])
    return UNRESOLVED


def _translate_builtin_call(name, argtext, scalars=None, declared=None):
    """Translate a builtin call, first resolving every argument with the
    full expression translator so a known builtin never swallows an
    unconverted element-wise operator (``.*``/``./``), a nested call, a
    user ``name()`` indexing site, a range, or a transpose.  An argument
    the full translator cannot resolve is passed through verbatim rather
    than failing the whole call."""
    args = _split_top_level(argtext, ",") if argtext.strip() else []
    translated = []
    for a in args:
        stripped = a.strip()
        inner = _translate_expr(stripped, scalars, declared)
        if inner == UNRESOLVED:
            translated.append(a)
        else:
            translated.append(inner)
    rebuilt = "%s(%s)" % (name, ", ".join(translated))
    return apply_builtin_rule(rebuilt)


def _translate_reduction_call(name, argtext, scalars=None, declared=None):
    """Translate a single-output reduction (max/min/sum/mean) into its
    numpy equivalent.  A single argument reduces the whole array; the
    MATLAB dimension forms -- ``sum(x, dim)`` / ``mean(x, dim)`` and
    ``max(x, [], dim)`` / ``min(x, [], dim)`` -- become an explicit numpy
    axis.  Arguments go through the full expression translator, so a
    reduction composes naturally with anything wrapped inside it, most
    notably ``find(cond)`` -> ``np.where(cond)[0]``."""
    np_name = _REDUCTION_CALLS[name]
    args = [a.strip() for a in _split_top_level(argtext, ",") if a.strip()]
    if len(args) == 1:
        translated = _translate_expr(args[0], scalars, declared)
        if translated == UNRESOLVED:
            return UNRESOLVED
        return "%s(%s)" % (np_name, translated)
    if len(args) == 2 and name in ("sum", "mean"):
        axis = _dim_to_axis(args[1])
        if axis is None:
            return UNRESOLVED
        translated = _translate_expr(args[0], scalars, declared)
        if translated == UNRESOLVED:
            return UNRESOLVED
        return "%s(%s, axis=%s)" % (np_name, translated, axis)
    if len(args) == 3 and name in ("max", "min") and args[1] == "[]":
        axis = _dim_to_axis(args[2])
        if axis is None:
            return UNRESOLVED
        translated = _translate_expr(args[0], scalars, declared)
        if translated == UNRESOLVED:
            return UNRESOLVED
        return "%s(%s, axis=%s)" % (np_name, translated, axis)
    return UNRESOLVED


# Single-output builtins whose MATLAB dimension argument (1-based, so the
# second argument when the function takes one) must become an explicit numpy
# axis (0-based).  Mapping them through the plain "same" table would silently
# treat ``cumsum(x, 2)`` as ``np.cumsum(x, 2)`` (axis 2) instead of
# ``np.cumsum(x, axis=1)``.
_DIM_STAT_BUILTINS = {
    "cumprod": "np.cumprod",
    "cumsum": "np.cumsum",
    "median": "np.median",
    "prod": "np.prod",
}


def _translate_dim_builtin(name, argtext, scalars=None, declared=None):
    """Translate a single-output reduction that takes an optional trailing
    dimension argument.  ``prod(x)``/``cumsum(x)``/... reduce the whole
    array; ``prod(x, dim)``/``cumsum(x, dim)``/... reduce along the MATLAB
    dimension ``dim``, i.e. numpy axis ``dim - 1``."""
    np_name = _DIM_STAT_BUILTINS[name]
    args = [a.strip() for a in _split_top_level(argtext, ",") if a.strip()]
    if not args:
        return UNRESOLVED
    translated = _translate_expr(args[0], scalars, declared)
    if translated == UNRESOLVED:
        return UNRESOLVED
    if len(args) == 1:
        return "%s(%s)" % (np_name, translated)
    if len(args) == 2:
        axis = _dim_to_axis(args[1])
        if axis is None:
            return UNRESOLVED
        return "%s(%s, axis=%s)" % (np_name, translated, axis)
    return UNRESOLVED


def _translate_diff(argtext, scalars=None, declared=None):
    """Translate MATLAB diff(x[, n[, dim]]).  The third argument is the
    MATLAB dimension (1-based), so it becomes a numpy axis (0-based) rather
    than numpy's own ``n`` parameter."""
    args = [a.strip() for a in _split_top_level(argtext, ",") if a.strip()]
    if not args:
        return UNRESOLVED
    translated = _translate_expr(args[0], scalars, declared)
    if translated == UNRESOLVED:
        return UNRESOLVED
    if len(args) == 1:
        return "np.diff(%s)" % translated
    if len(args) == 2:
        n = _translate_expr(args[1], scalars, declared)
        if n == UNRESOLVED:
            return UNRESOLVED
        return "np.diff(%s, %s)" % (translated, n)
    if len(args) == 3:
        n = _translate_expr(args[1], scalars, declared)
        axis = _dim_to_axis(args[2])
        if n == UNRESOLVED or axis is None:
            return UNRESOLVED
        return "np.diff(%s, %s, axis=%s)" % (translated, n, axis)
    return UNRESOLVED


def _translate_var_std(name, argtext, scalars=None, declared=None):
    """Translate MATLAB var/std including their normalization argument.
    MATLAB's default (and w=0) normalizes by N-1, i.e. numpy ``ddof=1``;
    w=1 normalizes by N, i.e. numpy's default ``ddof=0``.  A weight vector
    (neither 0 nor 1) has no numpy scalar equivalent and stays unresolved."""
    np_name = "np.var" if name == "var" else "np.std"
    args = [a.strip() for a in _split_top_level(argtext, ",") if a.strip()]
    if not args:
        return UNRESOLVED
    translated = _translate_expr(args[0], scalars, declared)
    if translated == UNRESOLVED:
        return UNRESOLVED
    ddof = "1"
    axis = None
    rest = args[1:]
    if rest:
        if rest[0] not in ("0", "1"):
            return UNRESOLVED
        if rest[0] == "1":
            ddof = None
        rest = rest[1:]
    if rest:
        axis = _dim_to_axis(rest[0])
        if axis is None:
            return UNRESOLVED
    parts = [translated]
    if ddof == "1":
        parts.append("ddof=1")
    if axis is not None:
        parts.append("axis=%s" % axis)
    return "%s(%s)" % (np_name, ", ".join(parts))


_LIMIT_ARRAY = re.compile(r"^np\.array\(\[\[([^\[\]]*)\]\]\)$")


def _flatten_limit_vector(translated):
    """Flatten a single-row matrix argument into a Python list for the
    limit-style matplotlib functions (xlim/ylim/axis), which want
    ``[xmin, xmax]`` rather than a 2-D numpy array:
    ``xlim([0 10])`` -> ``plt.xlim([0, 10])``."""
    match = _LIMIT_ARRAY.match(translated)
    if match is None:
        return None
    return "[%s]" % match.group(1)


def _spread_limit_vector(translated):
    """Unpack a single-row matrix argument into comma-separated scalar args
    for matplotlib functions that take separate values (plt.clim):
    ``caxis([a b])`` -> ``a, b``."""
    match = _LIMIT_ARRAY.match(translated)
    if match is None:
        return None
    return match.group(1)


def _flatten_limit_vector_reverse(argtext):
    """Flatten a Python list argument into a MATLAB space-separated vector
    for the limit-style matplotlib functions (xlim/ylim/axis):
    ``plt.xlim([0, 10])`` -> ``xlim([0 10])``."""
    s = argtext.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    parts = _split_top_level(s[1:-1], ",")
    parts = [p.strip() for p in parts]
    if not parts or any(not p for p in parts):
        return None
    translated = [_translate_expr_reverse(p) for p in parts]
    if any(t == UNRESOLVED for t in translated):
        return None
    return "[%s]" % " ".join(translated)


def _translate_plot_command(text):
    """Translate a bare MATLAB plot-styling/config command statement (grid
    on, hold on, shading flat, axis equal, colorbar, ...) through the
    PLOT_COMMANDS table.  Returns the Python line, or None when the
    command (or its flag word) is not in the table."""
    parts = text.split()
    if not parts:
        return None
    spec = PLOT_COMMANDS.get(parts[0])
    if spec is None:
        return None
    if "noop" in spec:
        return "# %s: %s" % (text, spec["noop"])
    flag = parts[1] if len(parts) > 1 else None
    flags = spec.get("flags")
    if flag is not None:
        if flags is None or flag not in flags:
            return None
        call_args = flags[flag]
    else:
        call_args = ""
    return "%s(%s)" % (spec["func"], call_args)


# matplotlib keyword arguments whose value carries over to MATLAB as a
# name-value pair (keyword name -> MATLAB property name).
_PLOT_KWARGS_REVERSE = {
    "linewidth": "LineWidth",
    "color": "Color",
}

_KWARG_REVERSE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!\=)(.*)$")

# python call target (the PLOT_COMMANDS 'func' value) -> MATLAB command name.
_PLOT_COMMANDS_REVERSE = {
    spec["func"]: name for name, spec in PLOT_COMMANDS.items() if "func" in spec
}

# python call targets with no MATLAB equivalent -> reverse output emits a
# no-op comment instead of a translated command.
_PLOT_NOOP_REVERSE = {
    "plt.tight_layout": "no MATLAB equivalent; the figure layout adjusts automatically",
    "plt.show": "the figure is displayed automatically in MATLAB",
}

_PLOT_DOTTED_CALL = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", re.DOTALL
)


def _translate_plot_command_reverse(expr):
    """Translate a matplotlib plot command such as ``plt.subplot(2, 2, 1)``,
    ``plt.plot(x, y)`` or ``ax.set_zlabel('z')`` back to its MATLAB form.
    Only the PLOT_COMMANDS entries that map onto a plain call -- no spreads
    -- are handled; no-op calls such as ``plt.tight_layout()`` or
    ``plt.show()`` emit a comment instead.  Flag commands such as
    ``plt.grid(True)`` become MATLAB flag words (``grid on;``).  Limit
    commands such as ``plt.xlim([0, 10])`` become MATLAB space-separated
    vectors (``xlim([0 10]);``).  Keyword arguments such as ``linewidth=2``
    become MATLAB name-value pairs (``'LineWidth', 2``).  The trailing
    semicolon suppresses display of the returned handle, matching typical
    MATLAB script style."""
    match = _PLOT_DOTTED_CALL.fullmatch(expr)
    if match is None:
        return expr
    obj, method, argtext = match.group(1), match.group(2), match.group(3)
    target = "%s.%s" % (obj, method)
    noop_reason = _PLOT_NOOP_REVERSE.get(target)
    if noop_reason is not None:
        return "%% %s: %s" % (target, noop_reason)
    name = _PLOT_COMMANDS_REVERSE.get(target)
    if name is None:
        return expr
    spec = PLOT_COMMANDS[name]
    flags = spec.get("flags")
    if flags:
        rev_flags = {"".join(value.split()): flag for flag, value in flags.items()}
        flag_name = rev_flags.get("".join(argtext.split()))
        if flag_name is None:
            return UNRESOLVED
        return "%s %s;" % (name, flag_name)
    if spec.get("vector"):
        if len(_split_top_level(argtext, ",")) != 1:
            return UNRESOLVED
        flat = _flatten_limit_vector_reverse(argtext)
        if flat is None:
            return UNRESOLVED
        return "%s(%s);" % (name, flat)
    if "spread" in spec or "noop" in spec:
        return expr
    translated = []
    for a in _split_top_level(argtext, ","):
        a = a.strip()
        if not a:
            continue
        kwarg = _KWARG_REVERSE.fullmatch(a)
        if kwarg is not None:
            matlab_name = _PLOT_KWARGS_REVERSE.get(kwarg.group(1))
            if matlab_name is None:
                return UNRESOLVED
            value = _translate_expr_reverse(kwarg.group(2))
            if value == UNRESOLVED:
                return UNRESOLVED
            translated.append("'%s', %s" % (matlab_name, value))
        else:
            arg_py = _translate_expr_reverse(a)
            if arg_py == UNRESOLVED:
                return UNRESOLVED
            translated.append(arg_py)
    return "%s(%s);" % (name, ", ".join(translated))


def _translate_expr(expr, scalars=None, declared=None):
    expr = expr.strip()
    if not expr:
        return ""
    if expr.startswith("[") and expr.endswith("]"):
        return _translate_matrix(expr, scalars, declared)
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr

    # Drop redundant outer parentheses so parenthesized ranges like
    # (0:(length(P1)-1)) are translated as ranges, not passed through.
    # Re-wrap the result so operator precedence is preserved in context.
    if _outer_parens_wrap(expr):
        inner = _translate_expr(expr[1:-1], scalars, declared)
        return UNRESOLVED if inner == UNRESOLVED else "(%s)" % inner

    range_parts = split_range(expr)
    if len(range_parts) >= 2:
        return _translate_range(range_parts, scalars, declared)

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name == "disp":
            translated = [_translate_expr(a, scalars, declared) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "print(%s)" % ", ".join(translated)
        if name == "fprintf":
            converted = convert_fprintf(expr, lambda a: _translate_expr(a, scalars, declared))
            if converted is not None:
                return converted
            return UNRESOLVED
        if name == "fclose":
            args = _split_top_level(argtext, ",")
            if len(args) == 1 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args[0].strip()):
                return "%s.close()" % args[0].strip()
            return UNRESOLVED
        if name in _NUMPY_CALLS:
            translated = [
                _translate_expr(a, scalars, declared)
                for a in _split_top_level(argtext, ",")
            ]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "%s(%s)" % (_NUMPY_CALLS[name], ", ".join(translated))
        if name in _REDUCTION_CALLS:
            return _translate_reduction_call(name, argtext, scalars, declared)
        if name in _TRIG_DEGREES:
            translated = [
                _translate_expr(a, scalars, declared)
                for a in _split_top_level(argtext, ",")
            ]
            if len(translated) != 1 or any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "%s(%s(%s))" % (
                _TRIG_DEGREES[name][1],
                _TRIG_DEGREES[name][0],
                translated[0],
            )
        if name == "fft":
            args = [a.strip() for a in _split_top_level(argtext, ",") if a.strip()]
            if len(args) == 3 and args[1] == "[]":
                axis = _dim_to_axis(args[2])
                if axis is None:
                    return UNRESOLVED
                translated = _translate_expr(args[0], scalars, declared)
                if translated == UNRESOLVED:
                    return UNRESOLVED
                return "np.fft.fft(%s, axis=%s)" % (translated, axis)
            return _translate_builtin_call(name, argtext, scalars, declared)
        if name in _DIM_STAT_BUILTINS:
            return _translate_dim_builtin(name, argtext, scalars, declared)
        if name == "diff":
            return _translate_diff(argtext, scalars, declared)
        if name in ("var", "std"):
            return _translate_var_std(name, argtext, scalars, declared)
        if name in BUILTIN_RULES:
            return _translate_builtin_call(name, argtext, scalars, declared)
        if name in PLOT_COMMANDS:
            spec = PLOT_COMMANDS[name]
            if "func" not in spec:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars, declared) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            if spec.get("vector") and len(translated) == 1:
                flat = _flatten_limit_vector(translated[0])
                if flat is not None:
                    return "%s(%s)" % (spec["func"], flat)
            if spec.get("spread") and len(translated) == 1:
                spread = _spread_limit_vector(translated[0])
                if spread is not None:
                    return "%s(%s)" % (spec["func"], spread)
            return "%s(%s)" % (spec["func"], ", ".join(translated))
        if name in _OTHER_BUILTINS:
            return UNRESOLVED
        if name in ("length", "numel"):
            args = _split_top_level(argtext, ",")
            if len(args) != 1:
                return UNRESOLVED
            inner = _translate_expr(args[0], scalars, declared)
            if inner == UNRESOLVED:
                return UNRESOLVED
            return "len(%s)" % inner
        if name == "find":
            args = _split_top_level(argtext, ",")
            if len(args) != 1:
                return UNRESOLVED
            cond = _translate_expr(args[0], scalars, declared)
            if cond == UNRESOLVED:
                return UNRESOLVED
            return "np.where(%s)[0]" % cond
        if name == "interp1":
            args = _split_top_level(argtext, ",")
            if len(args) != 3:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars, declared) for a in args]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "np.interp(%s, %s, %s)" % (translated[2], translated[0], translated[1])
        if name == "imagesc":
            args = _split_top_level(argtext, ",")
            if len(args) != 3:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars, declared) for a in args]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            x, y, z = translated
            return (
                "plt.imshow(%s, extent=[%s[0],%s[-1],%s[0],%s[-1]], "
                "origin='lower', aspect='auto')"
                % (z, x, x, y, y)
            )
        if name == "surf":
            args = _split_top_level(argtext, ",")
            if len(args) < 3:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars, declared) for a in args[:3]]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return (
                "fig = plt.figure()\n"
                "ax = fig.add_subplot(projection='3d')\n"
                "ax.plot_surface(%s)"
                % ", ".join(translated)
            )
        if name == "view":
            args = _split_top_level(argtext, ",")
            if len(args) != 2:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars, declared) for a in args]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            az, el = translated
            return "ax.view_init(elev=%s, azim=%s)" % (el, az)
        args = _split_top_level(argtext, ",")
        if args and all(_is_index_like(a) for a in args):
            if declared is not None and name in declared:
                return apply_indexing_rule(expr)
            if declared is not None:
                translated = [_translate_expr(a, scalars, declared) for a in args]
                if any(t == UNRESOLVED for t in translated):
                    return UNRESOLVED
                return "%s(%s)" % (name, ", ".join(translated))
            return apply_indexing_rule(expr)
        return UNRESOLVED

    # A dotted MATLAB call such as a System-object constructor
    # (phased.ArrayResponse, comm.AWGNChannel).  Known specialist names
    # are routed through the same builtin table as bare calls; unknown
    # dotted calls keep their current pass-through behaviour.
    dotted = _split_call(expr, dotted=True)
    if dotted is not None:
        name, argtext = dotted
        if name in BUILTIN_RULES:
            return _translate_builtin_call(name, argtext, scalars, declared)
        return expr

    idx, op = _find_last_operator(expr)
    if op is None:
        # MATLAB logical NOT (``~flag``) binds looser than postfix transpose,
        # so it is resolved first: ``~flag'`` must read ``~(flag')``.
        not_match = re.match(r"^~\s*(.+)$", expr)
        if not_match:
            inner_py = _translate_expr(not_match.group(1), scalars, declared)
            if inner_py == UNRESOLVED:
                return UNRESOLVED
            return "np.logical_not(%s)" % inner_py
        # MATLAB postfix transpose (expr' / expr.') applies to any
        # expression -- a variable, a call result, an indexed expression,
        # or a parenthesized compound -- and binds more tightly than any
        # binary operator, so it is resolved only once no binary operator
        # remains at the top level (operands reach here through recursion).
        transposed = _split_transpose(expr)
        if transposed is not None:
            base, transpose_kind = transposed
            base_py = _translate_expr(base, scalars, declared)
            if base_py == UNRESOLVED:
                return UNRESOLVED
            return apply_transpose_rule(base_py, transpose_kind)
        if expr in _CONSTANTS:
            return _CONSTANTS[expr]
        return apply_complex_rule(expr)

    left_py = _translate_expr(expr[:idx], scalars, declared)
    right_py = _translate_expr(expr[idx + len(op):], scalars, declared)
    if left_py == UNRESOLVED or right_py == UNRESOLVED:
        return UNRESOLVED

    if op == "*":
        if not _is_scalar(expr[:idx], scalars) and not _is_scalar(
            expr[idx + len(op):], scalars
        ):
            return "%s @ %s" % (left_py, right_py)
        return "%s * %s" % (left_py, right_py)
    if op == ".*":
        return "%s * %s" % (left_py, right_py)
    if op == "./":
        return "%s / %s" % (left_py, right_py)
    if op == ".\\":
        # MATLAB element-wise left division: a .\ b is b ./ a element-wise.
        return "%s / %s" % (right_py, left_py)
    if op == ".^":
        return "%s ** %s" % (left_py, right_py)
    if op == "^":
        # MATLAB '^' is matrix power; scalar operands (2^3) are identical to
        # element-wise power and map to Python '**'.  A base the inference
        # pass proved to be a definite non-scalar (or a matrix literal)
        # becomes an explicit matrix power; an unknown operand keeps the
        # element-wise '**' because matrix_power fails at runtime on scalars.
        left_raw = expr[:idx].strip()
        non_scalars = getattr(scalars, "non_scalars", None)
        if non_scalars and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left_raw):
            if left_raw in non_scalars:
                return "np.linalg.matrix_power(%s, %s)" % (left_py, right_py)
        if left_raw.startswith("[") and left_raw.endswith("]"):
            return "np.linalg.matrix_power(%s, %s)" % (left_py, right_py)
        return "%s ** %s" % (left_py, right_py)
    if op == "/":
        # Matrix right-division only when both operands are arrays; a
        # scalar divisor (e.g. length(P2), fs) means element-wise division.
        if _is_scalar(expr[:idx], scalars) or _is_scalar(
            expr[idx + len(op):], scalars
        ):
            return "%s / %s" % (left_py, right_py)
        return "np.linalg.solve(%s.T, %s.T).T" % (right_py, left_py)
    if op == "\\":
        # MATLAB '\' is matrix left-division (solve a * x = b -> x = a \ b);
        # a scalar operand makes it element-wise (a \ b is b ./ a).
        if _is_scalar(expr[:idx], scalars) or _is_scalar(
            expr[idx + len(op):], scalars
        ):
            return "%s / %s" % (right_py, left_py)
        return "np.linalg.solve(%s, %s)" % (left_py, right_py)
    if op == "&":
        return "np.logical_and(%s, %s)" % (left_py, right_py)
    if op == "|":
        return "np.logical_or(%s, %s)" % (left_py, right_py)
    if op == "~=":
        return "%s != %s" % (left_py, right_py)
    if op in ("+", "-"):
        return "%s %s %s" % (left_py, op, right_py)
    return UNRESOLVED


def _note_for(stmt, python):
    src = stmt.text
    if stmt.kind == "command":
        return "re-initialize state (no-op here)"
    if stmt.kind == "assignment":
        if python and "np.array([[" in python:
            return "numpy array, ';' row separator -> list rows"
        if python and "np." in python:
            return "builtin mapped to numpy"
        if "(" in src.split("=")[0]:
            return "1-based index converted to 0-based"
        return "assignment"
    if stmt.kind == "function_call":
        if python and python.startswith("print("):
            arg = python[len("print("):-1].strip()
            if arg.startswith("'"):
                return "print string"
            if " @ " in arg:
                return "matrix multiplication '*' -> '@'"
            if ".*" in src and " * " in arg:
                return "element-wise multiplication '.*' -> '*'"
            if "[" in arg:
                return "1-based index converted to 0-based"
            return "print call"
        return "function call"
    return "statement"


def _comment_for(stmt, python):
    return "# MATLAB: %s; -> Python: %s" % (stmt.text, _note_for(stmt, python))


def _raw_block_source(node):
    """Return the raw source text that spans an entire block construct.

    For a ``Loop`` this is the full text from its opening keyword through
    the matching 'end'; for a ``Statement`` it is the statement's own text
    (which, for block constructs such as if/switch, already covers the whole
    construct including every clause and the closing 'end')."""
    if hasattr(node, "kind"):
        return node.text
    return _loop_source(node)


def _loop_source(loop):
    """Full source of a ``Loop`` block, opening keyword through 'end'.

    The Reader normally records this verbatim on ``loop.source``.  When a
    ``Loop`` is assembled programmatically without a source, a faithful
    reconstruction is built from the header and body statements so the
    atomic-block invariant still holds."""
    source = getattr(loop, "source", None)
    if source:
        return source
    body = "\n".join(
        "    " + line
        for s in loop.statements
        for line in _raw_block_source(s).splitlines()
    )
    return "%s %s\n%s\nend" % (loop.type, loop.header, body)


def _translate_loop(stmt, scalars=None, declared=None, io=None, renames=None, first_seen=None):
    feof_line = translate_feof_loop(stmt.header, stmt.statements, io)
    if feof_line is not None:
        feof_line = rename_text(feof_line, renames) if renames else feof_line
        return {
            "kind": "loop",
            "source": _loop_source(stmt),
            "python": feof_line,
            "comment": "# MATLAB: %s; -> Python: %s" % (stmt.header, feof_line),
            "body": [],
        }
    source = _loop_source(stmt)
    header_text = rename_text(stmt.header, renames) if renames else stmt.header
    header = "%s %s" % (stmt.type, header_text)
    result = {"kind": "loop", "source": source}
    _attach_renames(result, stmt.header, renames, first_seen)
    body_declared = set(declared) if declared else set()
    loop_var = _loop_variable(stmt)
    loop_var_renamed = renames.get(loop_var, loop_var) if renames else loop_var
    body_declared.add(loop_var_renamed)
    if scalars is not None:
        # A 'for n = 1:N' index takes one scalar element per iteration, so
        # it is scalar by construction and a '*' it participates in maps to
        # element-wise '*' rather than matrix '@'.
        scalars.add(loop_var_renamed)
    body = []
    for s in stmt.statements:
        body.append(
            _translate_statement(s, scalars, body_declared, io, renames, first_seen)
        )
        target = _declare_target(s.text) if hasattr(s, "kind") else None
        if target:
            body_declared.add(renames.get(target, target) if renames else target)
    if any(s["python"] == UNRESOLVED for s in body):
        return {"kind": "loop", "source": source, "python": UNRESOLVED}

    var, eq, expr = header_text.partition("=")
    if not eq or ":" not in expr:
        return {"kind": "loop", "source": source, "python": UNRESOLVED}
    converted = apply_indexing_rule(expr.strip())
    if ":" not in converted:
        return {"kind": "loop", "source": source, "python": UNRESOLVED}
    start, stop = converted.split(":", 1)
    if start == "0":
        loop_py = "for %s in range(%s):" % (var.strip(), stop)
    else:
        loop_py = "for %s in range(%s, %s):" % (var.strip(), start, stop)
    result["python"] = loop_py
    result["comment"] = "# MATLAB: %s; -> Python: %s" % (header, loop_py)
    result["body"] = body
    return result


def _record_scalar(stmt, scalars, renames=None):
    """Track variables assigned scalar values (e.g. fs = 1000) so a later
    '/' with that variable is recognized as scalar division.  Compound
    scalar expressions (``timeDelay = 2 * targetRange / c``) and
    multi-output scalar calls (``[peakValue,peakIndex] = max(...)``) are
    tracked too."""
    if scalars is None or not hasattr(stmt, "kind") or stmt.kind != "assignment":
        return
    text = rename_text(stmt.text, renames) if renames else stmt.text
    target, value = _split_assignment(text)
    if target is None or value is None:
        return
    if "(" in target:
        return
    if not _is_scalar(value, scalars):
        return
    if target.startswith("[") and target.endswith("]"):
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", target[1:-1]):
            scalars.add(name)
    else:
        scalars.add(target.strip())


def _translate_statement(stmt, scalars=None, declared=None, io=None, renames=None, first_seen=None):
    if not hasattr(stmt, "kind"):
        return _translate_loop(stmt, scalars, declared, io, renames, first_seen)
    if stmt.kind == "command":
        # Commands (clear, close all, grid on, ...) are bare MATLAB phrases
        # that never reference variables, so reserved-word renaming does
        # not apply to their text.
        if stmt.text in _COMMANDS:
            return {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": _COMMANDS[stmt.text],
                "comment": _comment_for(stmt, _COMMANDS[stmt.text]),
            }
        plot_line = _translate_plot_command(stmt.text)
        if plot_line is not None:
            return {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": plot_line,
                "comment": _comment_for(stmt, plot_line),
            }
        return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}

    text = rename_text(stmt.text, renames) if renames else stmt.text
    if stmt.kind == "while_statement":
        feof_line = translate_feof_statement(text, io)
        if feof_line is not None:
            result = {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": feof_line,
                "comment": _comment_for(stmt, feof_line),
            }
            _attach_renames(result, stmt.text, renames, first_seen)
            return result
        return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}

    if stmt.kind == "assignment":
        target, value = _split_assignment(text)
        if value is None:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        fopen_result = translate_fopen(target, value)
        if fopen_result is not None:
            python, record = fopen_result
            if io is not None:
                io[target] = record
            result = {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": python,
                "comment": _comment_for(stmt, python),
            }
            _attach_renames(result, stmt.text, renames, first_seen)
            return result
        fscanf_line = translate_fscanf(target, value, io)
        if fscanf_line is not None:
            result = {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": fscanf_line,
                "comment": _comment_for(stmt, fscanf_line),
            }
            _attach_renames(result, stmt.text, renames, first_seen)
            return result
        if target.startswith("[") and target.endswith("]"):
            lines = translate_multi_output_assignment(
                target, value, lambda a: _translate_expr(a, scalars, declared)
            )
            if lines is not None:
                python = "\n".join(lines)
                result = {
                    "kind": stmt.kind,
                    "source": stmt.text,
                    "python": python,
                    "comment": _comment_for(stmt, python),
                }
                _attach_renames(result, stmt.text, renames, first_seen)
                return result
            if "~" in target:
                return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        value_py = _translate_expr(value, scalars, declared)
        if value_py == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        target_py = apply_indexing_rule(target) if "(" in target else target
        python = "%s = %s" % (target_py, value_py)
        result = {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": python,
            "comment": _comment_for(stmt, python),
        }
        _attach_renames(result, stmt.text, renames, first_seen)
        return result

    if stmt.kind == "function_call":
        python = _translate_expr(text, scalars, declared)
        if python == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        result = {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": python,
            "comment": _comment_for(stmt, python),
        }
        _attach_renames(result, stmt.text, renames, first_seen)
        return result

    return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}


_PLAIN_ASSIGN_TARGET = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")
_INDEXED_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*=\s*(.*)$")
_INDEXED_REF = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)")


def _is_loop(stmt):
    return not hasattr(stmt, "kind")


def _loop_variable(loop):
    return loop.header.split("=", 1)[0].strip()


def _declare_target(text):
    match = _PLAIN_ASSIGN_TARGET.match(text)
    return match.group(1) if match else None


def _indexed_reference_matrix(loop, var, target, declared, renames=None):
    for body_stmt in loop.statements:
        if not hasattr(body_stmt, "kind"):
            continue
        text = rename_text(body_stmt.text, renames) if renames else body_stmt.text
        for match in _INDEXED_REF.finditer(text):
            name = match.group(1)
            if name == target:
                continue
            index_tokens = [t.strip() for t in match.group(2).split(",")]
            if var in index_tokens and name in declared:
                return name
    return None


def _build_preallocate(target, ref_matrix):
    if ref_matrix:
        return {
            "kind": "preallocate",
            "source": "%s(...) = ... (implicit array growth)" % target,
            "python": "%s = np.zeros_like(%s)" % (target, ref_matrix),
            "comment": (
                "# MATLAB: %s(...) = ... (implicit array growth) -> Python: "
                "preallocate once, same shape as %s" % (target, ref_matrix)
            ),
        }
    return {
        "kind": "preallocate",
        "source": "%s(...) = ... (implicit array growth)" % target,
        "python": "%s = np.zeros(0)" % target,
        "comment": (
            "# MATLAB: %s(...) = ... (implicit array growth) -> Python: "
            "preallocate (shape unresolved)" % target
        ),
    }


def _preallocations_for_loop(loop, declared, renames=None, first_seen=None):
    var = _loop_variable(loop)
    if not var:
        return []
    if renames:
        var = renames.get(var, var)
    preallocations = []
    for body_stmt in loop.statements:
        if not hasattr(body_stmt, "kind") or body_stmt.kind != "assignment":
            continue
        text = rename_text(body_stmt.text, renames) if renames else body_stmt.text
        match = _INDEXED_ASSIGN.match(text)
        if not match:
            continue
        target = match.group(1)
        index_tokens = [t.strip() for t in match.group(2).split(",")]
        if target in declared or var not in index_tokens:
            continue
        ref_matrix = _indexed_reference_matrix(
            loop, var, target, declared, renames
        )
        preallocate = _build_preallocate(target, ref_matrix)
        _attach_renames(preallocate, body_stmt.text, renames, first_seen)
        preallocations.append(preallocate)
    return preallocations


def _translation_key(stmt):
    """Return the output-field key ('python' or 'matlab') for a translated
    statement, whichever of the two the statement carries."""
    if "python" in stmt:
        return "python"
    return "matlab"


def _attach_origin_lower(imshow_line):
    """Return ``imshow_line`` with ``origin='lower'`` inserted before the
    final closing paren, unless it already carries an ``origin=`` argument."""
    if "origin=" in imshow_line:
        return imshow_line
    if imshow_line.startswith("plt.imshow(") and imshow_line.endswith(")"):
        return "%s, origin='lower')" % imshow_line[:-1]
    return imshow_line


def _resolve_axis_xy(statements):
    """Post-pass: a bare 'axis xy' command attaches ``origin='lower'`` to the
    nearest preceding imagesc/imshow statement instead of staying UNRESOLVED.

    MATLAB's ``axis xy`` flips the y-axis so image row 1 is at the bottom;
    matplotlib expresses the same thing with ``origin='lower'`` on the imshow
    call.  The pass walks the translated statements in order, remembers the
    most recent ``plt.imshow(...)`` line, and rewrites the axis-xy command to
    a resolved comment (attaching origin to that imshow) when one precedes it.
    Without a preceding imshow the command keeps its UNRESOLVED status.
    Recurses into block bodies so the invariant holds inside loops too.
    """
    cleaned = []
    last_imshow = None
    for stmt in statements:
        if not isinstance(stmt, dict):
            cleaned.append(stmt)
            continue
        if stmt.get("body"):
            stmt = dict(stmt)
            stmt["body"] = _resolve_axis_xy(stmt["body"])
        source = (stmt.get("source") or "").strip()
        if stmt.get("kind") == "command" and source == "axis xy":
            if last_imshow is not None:
                last_imshow["python"] = _attach_origin_lower(last_imshow["python"])
                stmt = dict(stmt)
                stmt["python"] = "# axis xy: image origin set to 'lower'"
            cleaned.append(stmt)
            continue
        cleaned.append(stmt)
        python = stmt.get("python")
        if isinstance(python, str) and python.startswith("plt.imshow("):
            last_imshow = stmt
    return cleaned


def collapse_unresolved_blocks(statements):
    """Enforce the atomic-block invariant across a list of translated
    statements, as a general post-pass applied after every translation
    attempt (never as per-construct-type logic).

    Invariant: every block construct is either fully resolved -- its
    ``body`` contains no UNRESOLVED descendant -- or reduced to a single
    atomic UNRESOLVED statement whose ``source`` spans the whole construct
    (opening keyword through matching 'end') and which carries no ``body``.
    A partial translation -- a translated header still holding an UNRESOLVED
    body, or any resolved statement with an UNRESOLVED descendant -- is
    collapsed here so it can never leak raw MATLAB syntax into the output.

    Returns a new list of statements with the invariant restored.
    """
    cleaned = []
    for stmt in statements:
        if not isinstance(stmt, dict):
            cleaned.append(stmt)
            continue
        key = _translation_key(stmt)
        body = stmt.get("body") or []
        if stmt.get(key) == UNRESOLVED:
            collapsed = dict(stmt)
            collapsed.pop("body", None)
            collapsed["source"] = stmt.get("source") or ""
            cleaned.append(collapsed)
            continue
        if body:
            body = collapse_unresolved_blocks(body)
            if any(b.get(key) == UNRESOLVED for b in body):
                collapsed = dict(stmt)
                collapsed[key] = UNRESOLVED
                collapsed.pop("body", None)
                collapsed["source"] = stmt.get("source") or ""
                cleaned.append(collapsed)
            else:
                collapsed = dict(stmt)
                collapsed["body"] = body
                cleaned.append(collapsed)
        else:
            cleaned.append(stmt)
    return cleaned


def _check_atomic_invariant(statements, path, violations):
    for stmt in statements:
        key = _translation_key(stmt)
        where = "%s[%r]" % (path, stmt.get("source", ""))
        if stmt.get(key) == UNRESOLVED:
            if stmt.get("body"):
                violations.append(
                    "UNRESOLVED block at %s still carries a body: %r"
                    % (where, list(stmt.get("body")))
                )
            source = stmt.get("source") or ""
            if not source.strip():
                violations.append(
                    "UNRESOLVED block at %s has no source text" % where
                )
            if stmt.get("kind") == "loop" and key == "python" and not re.match(
                r"^\s*(for|while)\b", source
            ):
                violations.append(
                    "UNRESOLVED loop at %s source does not open with "
                    "for/while: %r" % (where, source)
                )
            if stmt.get("kind") == "loop" and key == "python" and not source.rstrip().endswith(
                "end"
            ):
                violations.append(
                    "UNRESOLVED loop at %s source does not end with 'end': %r"
                    % (where, source)
                )
        elif stmt.get("body"):
            for child in stmt["body"]:
                if child.get(key) == UNRESOLVED:
                    violations.append(
                        "resolved statement %s contains an UNRESOLVED "
                        "descendant %r" % (where, child.get("source", ""))
                    )
            _check_atomic_invariant(stmt["body"], where, violations)


def assert_block_invariant(result):
    """Verify the atomic-block invariant across a translated result.

    After every translation attempt, for every block construct (while, for,
    if, switch, nested combinations) either the whole block was fully
    translated, or the entire block from its opening keyword to its matching
    'end' was captured as a single atomic UNRESOLVED statement with no
    partial body surviving.  Raises AssertionError listing every violation.
    """
    violations = []
    for func in result.get("functions", []):
        _check_atomic_invariant(
            func.get("statements", []), func.get("name", "?"), violations
        )
    _check_atomic_invariant(result.get("statements", []), "<top-level>", violations)
    if violations:
        raise AssertionError(
            "atomic-block invariant violated:\n- %s"
            % "\n- ".join(violations)
        )


def _collect_names(statements, names):
    """Collect every variable identifier used across a list of reader
    statements into ``names``, recursing into loop bodies so reserved-word
    collisions are detected for loop variables and nested bodies too.
    Command statements (``clear``, ``close all``, ``grid on``, ...) are
    bare MATLAB phrases, never variable expressions, so their words are
    not collected."""
    for stmt in statements:
        if hasattr(stmt, "kind"):
            if stmt.kind == "command":
                continue
            for _, _, ident in identifier_tokens(stmt.text):
                names.add(ident)
        else:
            for _, _, ident in identifier_tokens(stmt.header):
                names.add(ident)
            _collect_names(stmt.statements, names)


def _compute_renames(statements, params=(), outputs=(), name=None):
    """Build the per-file rename map for MATLAB identifiers that collide
    with Python reserved words.

    A MATLAB variable may legally be named ``lambda``, ``class``, ``type``
    or any other Python keyword/builtin.  Every such identifier appearing
    in the parameters, outputs, function name, loop headers, or statement
    text of this function/script is mapped to its ``name_`` form so the
    generated Python is valid and never silently shadows a builtin.
    """
    names = set(params)
    names.update(outputs)
    if name:
        names.add(name)
    _collect_names(statements, names)
    renames = {}
    for ident in names:
        if should_rename(ident):
            renames[ident] = rename_for(ident)
    return renames, set()


def _attach_renames(stmt, text, renames, first_seen):
    """Record a rename note comment on ``stmt`` the first time each renamed
    MATLAB identifier appears in ``text`` (a single set is threaded through
    the whole translation so the note is emitted exactly once per name)."""
    if not renames:
        return
    comments = []
    for _, _, ident in identifier_tokens(text):
        if ident in renames and ident not in first_seen:
            first_seen.add(ident)
            comments.append(rename_comment(ident))
    if comments:
        stmt["renamed"] = comments


def _translate_function(func, shapes=None):
    renames, first_seen = _compute_renames(
        func.statements, func.parameters, func.outputs, func.name
    )
    declared = {renames.get(p, p) for p in func.parameters}
    scalars = _ScalarScope(
        (renames.get(p, p) for p in func.parameters),
        non_scalars=_non_scalar_names(shapes, func.name, renames),
    )
    if shapes is not None:
        # Enrich the scalar set with names the inference pass proved to be
        # definite scalars (assigned exactly once), so '/' and '*' involving
        # them stay element-wise rather than a matrix solve / product.
        for name in shapes.scalar_names(func.name):
            scalars.add(renames.get(name, name))
    io = {}
    translated = []
    for stmt in func.statements:
        if _is_loop(stmt):
            translated.extend(
                _preallocations_for_loop(stmt, declared, renames, first_seen)
            )
        translated.append(
            _translate_statement(stmt, scalars, declared, io, renames, first_seen)
        )
        _record_scalar(stmt, scalars, renames)
        target = _declare_target(stmt.text) if hasattr(stmt, "kind") else None
        if target:
            declared.add(renames.get(target, target))
    return {
        "name": renames.get(func.name, func.name),
        "parameters": [renames.get(p, p) for p in func.parameters],
        "outputs": [renames.get(o, o) for o in func.outputs],
        "statements": translated,
    }


def translate_with_rulebook(structure, shapes=None):
    if shapes is None:
        shapes = infer_shapes(structure)
    result = {"functions": [], "statements": []}
    for func in structure.functions:
        result["functions"].append(_translate_function(func, shapes))
    renames, first_seen = _compute_renames(structure.statements)
    scalars = _ScalarScope(
        non_scalars=_non_scalar_names(shapes, "top", renames)
    )
    declared = set()
    io = {}
    if shapes is not None:
        for name in shapes.scalar_names("top"):
            scalars.add(renames.get(name, name))
    for stmt in structure.statements:
        result["statements"].append(
            _translate_statement(stmt, scalars, declared, io, renames, first_seen)
        )
        _record_scalar(stmt, scalars, renames)
        target = _declare_target(stmt.text) if hasattr(stmt, "kind") else None
        if target:
            declared.add(renames.get(target, target))
    result["statements"] = _resolve_axis_xy(result["statements"])
    result["statements"] = collapse_unresolved_blocks(result["statements"])
    for func in result["functions"]:
        func["statements"] = _resolve_axis_xy(func["statements"])
        func["statements"] = collapse_unresolved_blocks(func["statements"])
    assert_block_invariant(result)
    return result


def _translate_matrix_reverse(inner):
    inner = inner.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        return None
    body = inner[1:-1].strip()
    if not body:
        return "[]"
    elems = [e.strip() for e in _split_top_level(body, ",")]
    if not elems:
        return None
    if all(e.startswith("[") and e.endswith("]") for e in elems):
        rows = []
        for e in elems:
            cells = [c.strip() for c in _split_top_level(e[1:-1], ",")]
            rows.append(" ".join(cells))
        return "[%s]" % "; ".join(rows)
    return "[%s]" % " ".join(elems)


def _find_percent_format_op(text):
    """Return the index of the ``%`` operator in ``text`` when it is Python's
    printf-style format operation (a string literal immediately followed by
    ``%``), else None."""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == ch:
                    break
                j += 1
            k = j + 1
            while k < n and text[k] == " ":
                k += 1
            if k < n and text[k] == "%":
                return k
            i = j + 1
            continue
        i += 1
    return None


def _fstring_body(text):
    """Return the body of a single f-string literal (without the ``f`` prefix
    and quotes), or None if ``text`` is not such a literal."""
    match = re.match(r"\s*[fF](['\"])(.*)\1\s*$", text, re.DOTALL)
    if not match:
        return None
    body = match.group(2)
    if "{" not in body:
        return None
    return body


def _output_has_format_specifiers(argtext):
    """Single general rule: an output statement contains format specifiers when
    it is a printf-style ``%`` operation on a string literal or an f-string."""
    if _find_percent_format_op(argtext) is not None:
        return True
    return _fstring_body(argtext) is not None


def _py_literal_to_matlab(literal):
    """Convert a Python string literal (single or double quoted) to a MATLAB
    single-quoted literal suitable for ``fprintf``. ``%`` format specs and
    ``\\``-escapes pass through unchanged; embedded single quotes are doubled."""
    literal = literal.strip()
    if len(literal) < 2 or literal[0] not in "'\"" or literal[-1] != literal[0]:
        return None
    quote = literal[0]
    body = literal[1:-1]
    if quote == '"':
        body = body.replace('\\"', '"')
    else:
        body = body.replace("\\'", "'")
    return "'%s'" % body.replace("'", "''")


def _percent_print_to_fprintf(argtext):
    """Translate ``print("fmt" % args)`` to MATLAB ``fprintf('fmt', args)``."""
    op = _find_percent_format_op(argtext)
    if op is None:
        return None
    fmt = _py_literal_to_matlab(argtext[:op])
    if fmt is None:
        return None
    args_src = argtext[op + 1:].strip()
    if args_src.startswith("(") and args_src.endswith(")"):
        inner = args_src[1:-1].strip()
        args = _split_top_level(inner, ",") if inner else []
    else:
        args = [args_src]
    translated = [_translate_expr_reverse(a) for a in args]
    if any(t == UNRESOLVED for t in translated):
        return None
    if not translated:
        return "fprintf(%s)" % fmt
    return "fprintf(%s, %s)" % (fmt, ", ".join(translated))


def _fstring_to_fprintf(argtext):
    """Translate ``print(f"lit {expr} ...")`` to MATLAB ``fprintf``, replacing
    each ``{expr}`` placeholder with a ``%`` spec and passing ``expr`` as an
    argument."""
    body = _fstring_body(argtext)
    if body is None:
        return None
    fmt_parts = []
    arg_parts = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "{":
            if i + 1 < n and body[i + 1] == "{":
                fmt_parts.append("{")
                i += 2
                continue
            end = body.find("}", i)
            if end < 0:
                return None
            content = body[i + 1:end]
            i = end + 1
            if ":" in content:
                expr, _, spec = content.partition(":")
                placeholder = "%" + spec.strip()
            else:
                expr = content.partition("!")[0]
                placeholder = "%s"
            expr = expr.strip()
            if not expr:
                return None
            arg_parts.append(expr)
            fmt_parts.append(placeholder)
        elif ch == "}":
            if i + 1 < n and body[i + 1] == "}":
                fmt_parts.append("}")
                i += 2
                continue
            return None
        else:
            fmt_parts.append(ch)
            i += 1
    if not arg_parts:
        return None
    translated = [_translate_expr_reverse(a) for a in arg_parts]
    if any(t == UNRESOLVED for t in translated):
        return None
    fmt = "".join(fmt_parts).replace("'", "''")
    return "fprintf('%s', %s)" % (fmt, ", ".join(translated))


_PY_LEN_CALL = re.compile(r"\blen\s*\(")
_PY_SHAPE_ATTR = re.compile(r"\.shape\b")
_PY_SUM_CALL = re.compile(r"\.sum\s*\(")
_PY_POWER = re.compile(r"\*\*")
_PY_NP_NAME = re.compile(r"\bnp\.[A-Za-z_][A-Za-z0-9_.]*")

# np.-prefixed names that have a working reverse rule back to MATLAB.
# Any other np. name (e.g. np.pi, np.newaxis, np.mean) has no reverse rule
# and must never pass through as if it were valid MATLAB.
_PY_NP_ALLOWED_REVERSE = frozenset(
    {
        "np.abs",
        "np.array",
        "np.ceil",
        "np.cos",
        "np.exp",
        "np.fft.fft",
        "np.floor",
        "np.linalg.solve",
        "np.linspace",
        "np.log",
        "np.random.randn",
        "np.reshape",
        "np.round",
        "np.sin",
        "np.sqrt",
        "np.tan",
        "np.trunc",
        "np.zeros",
    }
)


def _strip_string_literals(text):
    """Blank out Python string literal contents so that Python-only
    construct detection never fires on printed text."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            quote = ch
            j = i + 1
            while j < n:
                if text[j] == quote:
                    if j + 1 < n and text[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def _contains_python_only_construct(expr):
    """True when expr contains a Python-only construct (len(), .shape,
    .sum(...), //, **, np.newaxis, modulo '%', or any np. name with no
    reverse rule) that must never be passed through as if it were valid
    MATLAB."""
    stripped = _strip_string_literals(expr)
    if _PY_LEN_CALL.search(stripped):
        return True
    if _PY_SHAPE_ATTR.search(stripped):
        return True
    if _PY_SUM_CALL.search(stripped):
        return True
    if _PY_POWER.search(stripped):
        return True
    if "%" in stripped and _find_percent_format_op(expr) is None:
        # A '%' outside string literals that is not a printf-style format
        # operation is Python modulo; it has no MATLAB mirror and a stray
        # '%' also starts a MATLAB comment.
        return True
    for match in _PY_NP_NAME.finditer(stripped):
        if match.group(0) not in _PY_NP_ALLOWED_REVERSE:
            return True
    return False


def _translate_expr_reverse(expr):
    expr = expr.strip()
    if not expr:
        return ""
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr
    if _contains_python_only_construct(expr):
        return UNRESOLVED

    match = re.fullmatch(r"np\.array\((.*)\)", expr, re.DOTALL)
    if match:
        matrix = _translate_matrix_reverse(match.group(1))
        return matrix if matrix is not None else UNRESOLVED

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name == "print":
            if _output_has_format_specifiers(argtext):
                fprintf = _percent_print_to_fprintf(argtext)
                if fprintf is None:
                    fprintf = _fstring_to_fprintf(argtext)
                if fprintf is not None:
                    return fprintf
                return UNRESOLVED
            translated = [
                _translate_expr_reverse(a) for a in _split_top_level(argtext, ",")
            ]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            # MATLAB disp() takes exactly one argument; 'print(a, b)' must
            # never become an invalid 'disp(a, b)'.
            if len(translated) != 1:
                return UNRESOLVED
            return "disp(%s)" % translated[0]
        args = _split_top_level(argtext, ",")
        if args and all(_is_index_like(a) for a in args):
            reversed_call = apply_indexing_rule_reverse(expr)
            if reversed_call != expr:
                return reversed_call
        return UNRESOLVED

    dotted = re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\s*\(.*\)", expr
    )
    if dotted:
        sequence = apply_sequence_rule_reverse(expr)
        if sequence != expr:
            return sequence
        reversed_call = apply_builtin_rule_reverse(expr)
        if reversed_call != expr:
            return reversed_call
        plot_reversed = _translate_plot_command_reverse(expr)
        if plot_reversed != expr:
            return plot_reversed
        return UNRESOLVED

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\[.*\]", expr):
        return apply_indexing_rule_reverse(expr)

    attribute = apply_attribute_rule_reverse(expr)
    if attribute != expr:
        return attribute

    reversed_expr = apply_operator_rule_reverse(expr)
    if reversed_expr != expr:
        return reversed_expr

    return expr


def _note_for_reverse(stmt, matlab):
    src = stmt.text
    if stmt.kind == "Import":
        return "no import; arrays built with [ ] brackets"
    if stmt.kind == "Assign":
        if "np.array([[" in src:
            return "nested lists -> rows separated by ';'"
        if "np." in src:
            return "numpy builtin mapped back to MATLAB"
        if "[" in src.split("=")[0]:
            return "0-based index converted to 1-based"
        return "assignment"
    if stmt.kind == "Expr":
        if matlab.startswith("fprintf("):
            return "print with format specifiers -> fprintf"
        if matlab.startswith("disp("):
            arg = matlab[len("disp("):-1].strip()
            if arg.startswith("'"):
                return "print string"
            if " @ " in src:
                return "'@' is matrix multiplication -> '*'"
            if " * " in src:
                return "'*' element-wise in numpy -> '.*' in MATLAB"
            if "[" in arg:
                return "0-based index converted to 1-based"
            return "print call"
        return "expression"
    return "statement"


def _comment_for_reverse(stmt, matlab):
    return "%% Python: %s -> MATLAB: %s" % (stmt.text, _note_for_reverse(stmt, matlab))


def _translate_statement_reverse(stmt):
    if not hasattr(stmt, "kind"):
        return {
            "kind": "loop",
            "source": _loop_source(stmt),
            "matlab": UNRESOLVED,
        }
    if stmt.kind in ("Import", "ImportFrom"):
        matlab = ""
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "matlab": matlab,
            "comment": _comment_for_reverse(stmt, matlab),
        }
    if stmt.kind == "Assign":
        target, value = _split_assignment(stmt.text)
        if value is None:
            return {"kind": stmt.kind, "source": stmt.text, "matlab": UNRESOLVED}
        value_matlab = _translate_expr_reverse(value)
        if value_matlab == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "matlab": UNRESOLVED}
        if _contains_python_only_construct(target):
            return {"kind": stmt.kind, "source": stmt.text, "matlab": UNRESOLVED}
        target_matlab = apply_indexing_rule_reverse(target) if "[" in target else target
        matlab = "%s = %s" % (target_matlab, value_matlab)
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "matlab": matlab,
            "comment": _comment_for_reverse(stmt, matlab),
        }
    if stmt.kind == "Expr":
        matlab = _translate_expr_reverse(stmt.text)
        if matlab == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "matlab": UNRESOLVED}
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "matlab": matlab,
            "comment": _comment_for_reverse(stmt, matlab),
        }
    return {"kind": stmt.kind, "source": stmt.text, "matlab": UNRESOLVED}


def translate_with_rulebook_reverse(structure):
    result = {"functions": [], "statements": []}
    for func in structure.functions:
        result["functions"].append(
            {
                "name": func.name,
                "parameters": list(func.parameters),
                "statements": collapse_unresolved_blocks(
                    [_translate_statement_reverse(s) for s in func.statements]
                ),
            }
        )
    result["statements"] = collapse_unresolved_blocks(
        [_translate_statement_reverse(s) for s in structure.statements]
    )
    assert_block_invariant(result)
    return result
