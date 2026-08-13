"""Lightweight offline type/shape inference over the Reader Structure.

The rulebook's translation decisions are heuristics over the statement text:
``is_known_scalar`` decides whether a ``*`` or ``/`` is element-wise or a
matrix operation from the small ``scalars`` set the reader records.  This
module adds a dedicated forward pass that walks every statement in order and
tracks a coarse shape (scalar/vector/matrix/unknown) per variable, using only
the text the Reader already captured -- no AI, no cloud, no extra parser.

The result feeds two consumers:

- the rulebook, which unions the inferred definite-scalar names into the
  ``scalars`` set it threads through translation, so ``/`` and ``*`` stay
  element-wise more often and a matrix solve is never emitted where scalar
  division was intended;
- the top-level pipeline, which reports the inferred shapes as a new
  ``sections["inference"]`` stage alongside reader/rulebook/checker.

Only names assigned exactly once (never reassigned or grown by indexing) are
treated as definite scalars, so the pass never overrides the rulebook's own
order-sensitive tracking on variables whose shape changes mid-stream.
"""

import re
from dataclasses import dataclass, field

from reader.extract_structure import split_range, split_top_level

from .operator_rules import _find_last_operator, _split_transpose

SCALAR = "scalar"
VECTOR = "vector"
MATRIX = "matrix"
UNKNOWN = "unknown"

SHAPES = (SCALAR, VECTOR, MATRIX, UNKNOWN)

# Calls that always produce a scalar: a count (length/numel), a rounded
# value (round/floor/ceil/fix), or a single-dimension size query.
_SCALAR_CALLS = {"length", "numel", "len", "round", "floor", "ceil", "fix"}
# Calls that preserve the shape of their single argument (element-wise
# math, the FFT family, and conjugation/transpose helpers).
_SHAPE_PRESERVING = {
    "abs", "acos", "asin", "atan", "conj", "cos", "exp", "fft", "ifft",
    "log", "log10", "sin", "sqrt", "tan", "fliplr", "flipud",
}
# Reductions; the result's shape depends on the argument's shape (a
# whole-array reduction of a vector is a scalar, of a matrix a row vector).
_REDUCTIONS = {"max", "min", "sum", "mean"}
# Array constructors; the MATLAB one-argument forms are square matrices.
_CONSTRUCTORS = {"zeros", "ones", "randn", "rand"}

_NUMBER = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
_IMAG = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?i$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STRING = re.compile(r"^'(?:[^']|'')*'$")
_INDEXED = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(")


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


def _split_call(expr, dotted=False):
    pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
    if dotted:
        pattern = r"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*)\s*\("
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


def _combine_elementwise(left, right):
    """Shape of an element-wise binary operation over two operand shapes."""
    if left == SCALAR and right == SCALAR:
        return SCALAR
    if left == UNKNOWN or right == UNKNOWN:
        return UNKNOWN
    if left == MATRIX or right == MATRIX:
        return MATRIX
    if left == VECTOR or right == VECTOR:
        return VECTOR
    return UNKNOWN


def _matrix_shape(expr, scope):
    inner = expr[1:-1].strip()
    rows = [r for r in split_top_level(inner, ";")]
    if not rows or not any(r.strip() for r in rows):
        return VECTOR
    if len(rows) > 1:
        return MATRIX
    cells = [c for c in re.split(r"[,\s]+", rows[0].strip()) if c]
    if len(cells) <= 1:
        return shape_of_expr(rows[0].strip(), scope)
    return VECTOR


def _index_shape(args, scope):
    colons = [a for a in args if a.strip() == ":"]
    has_slice = bool(colons)
    if not has_slice:
        has_slice = any(len(split_range(a)) >= 2 for a in args)
    if not has_slice:
        return SCALAR
    if len(args) >= 2 and len(colons) == len(args):
        return MATRIX
    return VECTOR


def _reduction_shape(args, scope):
    if not args:
        return UNKNOWN
    arg = shape_of_expr(args[0].strip(), scope)
    if len(args) == 1:
        if arg in (SCALAR, VECTOR):
            return SCALAR
        if arg == MATRIX:
            return VECTOR
        return UNKNOWN
    if arg == MATRIX:
        return VECTOR
    if arg in (SCALAR, VECTOR):
        return SCALAR
    return UNKNOWN


def _constructor_shape(args):
    if not args:
        return SCALAR
    if len(args) in (1, 2):
        return MATRIX
    return UNKNOWN


def _multi_output_shapes(value, scope, n_targets):
    """Shapes of a bracketed multi-output assignment's results, one per
    target.  Mirrors the MULTI_OUTPUT_RULES registry shapes so the two
    stay in step."""
    call = _split_call(value.strip())
    if call is None:
        return [UNKNOWN] * n_targets
    name, argtext = call
    args = [a for a in split_top_level(argtext, ",") if a.strip()]
    if name in ("max", "min", "sort"):
        arg = shape_of_expr(args[0].strip(), scope) if args else UNKNOWN
        out = (
            SCALAR
            if arg in (SCALAR, VECTOR)
            else (VECTOR if arg == MATRIX else UNKNOWN)
        )
        return [out] * n_targets
    if name == "size":
        return [SCALAR] * n_targets
    if name == "find":
        return [VECTOR] * n_targets
    if name == "meshgrid":
        return [MATRIX] * n_targets
    return [UNKNOWN] * n_targets


def shape_of_expr(expr, scope=None):
    """Infer the shape of a MATLAB expression string.

    ``scope`` maps variable names already assigned earlier in the enclosing
    scope to their inferred shape; unknown names stay ``UNKNOWN``.  The
    analysis is purely textual and order-aware: literals, matrix literals,
    colon ranges, known builtin calls, indexing, and operator expressions
    each produce a coarse scalar/vector/matrix/unknown verdict.
    """
    scope = scope or {}
    expr = _strip_outer_parens(expr)
    if not expr:
        return UNKNOWN
    if _NUMBER.fullmatch(expr) or _IMAG.fullmatch(expr):
        return SCALAR
    if expr in ("pi", "e", "eps"):
        return SCALAR
    if _STRING.fullmatch(expr):
        return SCALAR
    if expr.startswith("[") and expr.endswith("]"):
        return _matrix_shape(expr, scope)
    if len(split_range(expr)) >= 2:
        return VECTOR

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        args = [a for a in split_top_level(argtext, ",") if a.strip()]
        if name in _SCALAR_CALLS:
            return SCALAR
        if name in _SHAPE_PRESERVING:
            if len(args) != 1:
                return UNKNOWN
            return shape_of_expr(args[0], scope)
        if name in _REDUCTIONS:
            return _reduction_shape(args, scope)
        if name in _CONSTRUCTORS:
            return _constructor_shape(args)
        if name == "linspace":
            return VECTOR
        if name == "find":
            return VECTOR
        if name == "size":
            return SCALAR if len(args) == 2 else VECTOR
        if name == "sort":
            return shape_of_expr(args[0].strip(), scope) if len(args) == 1 else UNKNOWN
        if name == "meshgrid":
            return MATRIX
        if name in ("reshape", "interp1", "conv", "polyval"):
            return UNKNOWN
        if _IDENT.fullmatch(name) and name in scope:
            return _index_shape(args, scope)
        return UNKNOWN

    if _split_call(expr, dotted=True) is not None:
        return UNKNOWN

    transposed = _split_transpose(expr)
    if transposed is not None:
        base, _kind = transposed
        return shape_of_expr(base, scope)

    idx, op = _find_last_operator(expr)
    if op is None:
        if _IDENT.fullmatch(expr):
            return scope.get(expr, UNKNOWN)
        return UNKNOWN

    left = shape_of_expr(expr[:idx], scope)
    right = shape_of_expr(expr[idx + len(op):], scope)
    if op in ("+", "-", ".*", "./", ".^", "~=", "==", "<", ">", "<=", ">="):
        return _combine_elementwise(left, right)
    if op == "*":
        if left == SCALAR and right == SCALAR:
            return SCALAR
        if left == UNKNOWN or right == UNKNOWN:
            return UNKNOWN
        if left == SCALAR or right == SCALAR:
            # A scalar times an array is element-wise scaling in MATLAB.
            return _combine_elementwise(left, right)
        if left == MATRIX or right == MATRIX:
            return MATRIX
        return UNKNOWN
    if op == "/":
        if right == SCALAR:
            return _combine_elementwise(left, SCALAR)
        if left == UNKNOWN or right == UNKNOWN:
            return UNKNOWN
        return MATRIX
    if op == "^":
        return SCALAR if (left == SCALAR and right == SCALAR) else UNKNOWN
    return UNKNOWN


@dataclass
class ScopeInfo:
    """Inferred knowledge for one scope (the top level or one function)."""

    shapes: dict = field(default_factory=dict)
    assignments: dict = field(default_factory=dict)

    @property
    def scalars(self):
        """Names whose final shape is a definite scalar and which were
        assigned exactly once -- safe to treat as scalar throughout."""
        return {
            name
            for name, shape in self.shapes.items()
            if shape == SCALAR and self.assignments.get(name, 0) == 1
        }


@dataclass
class InferenceResult:
    top: ScopeInfo = field(default_factory=ScopeInfo)
    functions: dict = field(default_factory=dict)

    @property
    def counts(self):
        counts = {SCALAR: 0, VECTOR: 0, MATRIX: 0, UNKNOWN: 0}
        for scope in list(self.functions.values()) + [self.top]:
            for shape in scope.shapes.values():
                counts[shape] += 1
        return counts

    def scope_for(self, key):
        if key == "top":
            return self.top
        return self.functions.get(key)

    def scalar_names(self, key):
        info = self.scope_for(key)
        return set(info.scalars) if info is not None else set()


def _process_assignment(target, value, scope, assignment_count):
    target = target.strip()
    if target.startswith("[") and target.endswith("]"):
        names = [t.strip() for t in re.split(r"[,\s]+", target[1:-1]) if t.strip()]
        shapes = _multi_output_shapes(value, scope, len(names))
        for i, name in enumerate(names):
            if name == "~":
                continue
            scope[name] = shapes[i] if i < len(shapes) else UNKNOWN
            assignment_count[name] = assignment_count.get(name, 0) + 1
        return
    match = _INDEXED.match(target)
    if match:
        name = match.group(1)
        if name not in scope:
            scope[name] = UNKNOWN
        return
    scope[target] = shape_of_expr(value, scope)
    assignment_count[target] = assignment_count.get(target, 0) + 1


def _infer_scope(statements, scope, assignment_count):
    for stmt in statements:
        if not hasattr(stmt, "kind"):
            var = None
            if stmt.type == "for" and "=" in stmt.header:
                var = stmt.header.split("=", 1)[0].strip()
            if var:
                scope[var] = SCALAR
                assignment_count[var] = assignment_count.get(var, 0) + 1
            _infer_scope(stmt.statements, scope, assignment_count)
            if var:
                scope.pop(var, None)
            continue
        if stmt.kind != "assignment":
            continue
        target, value = _split_assignment(stmt.text)
        if value is not None:
            _process_assignment(target, value, scope, assignment_count)


def infer_shapes(structure):
    """Run the shape-inference pass over a Reader ``Structure``.

    Returns an :class:`InferenceResult` holding a per-scope shape map for the
    top-level statements and for each function, plus summary counts.  The
    pass is order-aware: later assignments overwrite earlier ones, and a
    variable's shape at any point reflects the statements seen so far.
    """
    top = ScopeInfo()
    _infer_scope(structure.statements, top.shapes, top.assignments)

    functions = {}
    for func in structure.functions:
        info = ScopeInfo()
        for param in func.parameters:
            info.shapes[param] = UNKNOWN
        _infer_scope(func.statements, info.shapes, info.assignments)
        functions[func.name] = info
    return InferenceResult(top=top, functions=functions)
