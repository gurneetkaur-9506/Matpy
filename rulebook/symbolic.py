"""Lightweight offline symbolic analysis of a translation result.

Runs after the rulebook has produced Python and before the Checker, as an
advisory stage alongside validation.  It is purely static and offline --
no AI, no cloud, no execution -- and never rejects or rewrites code.  It
only produces insights with a confidence level so a human can focus their
review:

    * constant_detection -- an RHS expression with no free variables folds
        to a fixed numeric value (``sin(0) == 0``, ``2 * pi``, ``sqrt(4)``).
    * simplification     -- an algebraic identity is visible in the code
        (``0 * x == 0``, ``x * 1 == x``, ``x + 0 == x``, ``x - x == 0``,
        ``x / 1 == x``, ``x ** 1 == x``, ``-(-x) == x``, ...).
    * math_reasoning     -- a property that holds for all real inputs
        (``abs(x) >= 0``, ``x ** 2 >= 0``, ``exp(x) > 0``, ``sin(x)``
        bounded in [-1, 1], ...).

Every insight carries ``confidence`` in {HIGH, MEDIUM, LOW} so valid code
is never treated as an error -- insights are advisory only.
"""

import ast
import math

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

_STAGE = "symbolic"

# ---------------------------------------------------------------------------
# Safe constant folding
# ---------------------------------------------------------------------------

_MATH_FUNCS = {
    "np.sin": math.sin,
    "np.cos": math.cos,
    "np.tan": math.tan,
    "np.arcsin": math.asin,
    "np.arccos": math.acos,
    "np.arctan": math.atan,
    "np.sinh": math.sinh,
    "np.cosh": math.cosh,
    "np.tanh": math.tanh,
    "np.exp": math.exp,
    "np.expm1": math.expm1,
    "np.log": math.log,
    "np.log1p": math.log1p,
    "np.log2": math.log2,
    "np.log10": math.log10,
    "np.sqrt": math.sqrt,
    "np.abs": abs,
    "np.floor": math.floor,
    "np.ceil": math.ceil,
    "np.trunc": math.trunc,
    "abs": abs,
}

_CONSTANTS = {
    "np.pi": math.pi,
    "np.e": math.e,
    "math.pi": math.pi,
    "math.e": math.e,
    "pi": math.pi,
    "e": math.e,
}

_BIN_OP_EVAL = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OP_EVAL = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _NotConstant(Exception):
    pass


def _constant_number(node):
    """Return a numeric constant for an AST node, or None when it is not a
    literal number."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    return None


def _eval_constant(node):
    """Evaluate an AST expression that contains no free variables.

    Only numeric literals, arithmetic, unary sign, math/numpy constants and
    a whitelist of one-argument math functions are folded.  Anything else
    raises :class:`_NotConstant`, which callers treat as "not a constant".
    """
    number = _constant_number(node)
    if number is not None:
        return number
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise _NotConstant()
    if isinstance(node, ast.Attribute):
        name = _fullname(node)
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        raise _NotConstant()
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OP_EVAL.get(type(node.op))
        if fn is None:
            raise _NotConstant()
        return fn(_eval_constant(node.operand))
    if isinstance(node, ast.BinOp):
        fn = _BIN_OP_EVAL.get(type(node.op))
        if fn is None:
            raise _NotConstant()
        return fn(_eval_constant(node.left), _eval_constant(node.right))
    if isinstance(node, ast.Call):
        name = _fullname(node.func)
        fn = _MATH_FUNCS.get(name)
        if fn is None or len(node.args) != 1 or node.keywords:
            raise _NotConstant()
        return fn(_eval_constant(node.args[0]))
    raise _NotConstant()


def _fullname(node):
    """Reconstruct a dotted name like ``np.sin`` from an AST name/attribute."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# ---------------------------------------------------------------------------
# Expression-level analysis
# ---------------------------------------------------------------------------


def _insight(line, source, category, confidence, message):
    return {
        "line": line,
        "source": source,
        "stage": _STAGE,
        "category": category,
        "confidence": confidence,
        "message": message,
    }


def _literal_text(node):
    return ast.unparse(node)


def _identities(expr, source_text):
    """Detect algebraic simplifications visible in ``expr``'s AST."""
    insights = []

    def add(category, confidence, message):
        insights.append(
            _insight(None, source_text, category, confidence, message)
        )

    for sub in ast.walk(expr):
        if isinstance(sub, ast.BinOp):
            lnum = _constant_number(sub.left)
            rnum = _constant_number(sub.right)
            op = sub.op
            if isinstance(op, ast.Mult):
                if lnum == 0 or rnum == 0:
                    add("simplification", HIGH,
                        "expression '%s' multiplies by 0; the product is 0"
                        % _literal_text(sub))
                elif lnum == 1 or rnum == 1:
                    add("simplification", LOW,
                        "expression '%s' multiplies by 1; the factor is a "
                        "no-op" % _literal_text(sub))
            elif isinstance(op, ast.Add):
                if lnum == 0 or rnum == 0:
                    add("simplification", LOW,
                        "expression '%s' adds 0; the term is a no-op"
                        % _literal_text(sub))
            elif isinstance(op, ast.Sub):
                if rnum == 0:
                    add("simplification", LOW,
                        "expression '%s' subtracts 0; the term is a no-op"
                        % _literal_text(sub))
                elif ast.dump(sub.left) == ast.dump(sub.right):
                    add("simplification", HIGH,
                        "expression '%s' subtracts a value from itself; the "
                        "result is 0" % _literal_text(sub))
            elif isinstance(op, ast.Div):
                if rnum == 1:
                    add("simplification", LOW,
                        "expression '%s' divides by 1; the divisor is a "
                        "no-op" % _literal_text(sub))
            elif isinstance(op, ast.Pow):
                if rnum == 1:
                    add("simplification", LOW,
                        "expression '%s' raises to the power 1; the exponent "
                        "is a no-op" % _literal_text(sub))
                elif rnum == 0:
                    add("simplification", MEDIUM,
                        "expression '%s' raises to the power 0; the result "
                        "is 1 (for nonzero bases)" % _literal_text(sub))
        elif (
            isinstance(sub, ast.UnaryOp)
            and isinstance(sub.op, ast.USub)
            and isinstance(sub.operand, ast.UnaryOp)
            and isinstance(sub.operand.op, ast.USub)
        ):
            add("simplification", HIGH,
                "expression '%s' negates a negation; the sign cancels"
                % _literal_text(sub))
        elif isinstance(sub, ast.Call):
            name = _fullname(sub.func)
            if name in _MATH_FUNCS and len(sub.args) == 1 and not sub.keywords:
                value = _constant_number(sub.args[0])
                if value is not None:
                    known = {
                        ("np.sin", 0.0): 0.0,
                        ("np.cos", 0.0): 1.0,
                        ("np.tan", 0.0): 0.0,
                        ("np.exp", 0.0): 1.0,
                        ("np.expm1", 0.0): 0.0,
                        ("np.sqrt", 0.0): 0.0,
                        ("np.log", 1.0): 0.0,
                        ("np.log1p", 0.0): 0.0,
                        ("np.abs", 0.0): 0.0,
                        ("np.arcsin", 0.0): 0.0,
                        ("np.arctan", 0.0): 0.0,
                        ("np.sinh", 0.0): 0.0,
                        ("np.tanh", 0.0): 0.0,
                    }.get((name, float(value)))
                    if known is not None:
                        add("simplification", HIGH,
                            "expression '%s' evaluates to %s at %s"
                            % (_literal_text(sub), known, value))
    return insights


def _constant_insight(expr, source_text):
    """Report when an expression has no free variables and folds to a value."""
    try:
        value = _eval_constant(expr)
    except _NotConstant:
        return []
    return [
        _insight(
            None, source_text, "constant_detection", HIGH,
            "expression '%s' has no free variables and folds to %s"
            % (_literal_text(expr), value),
        )
    ]


def _math_reasoning(expr, source_text):
    """Report guaranteed mathematical properties of subexpressions."""
    insights = []

    def add(confidence, message):
        insights.append(
            _insight(None, source_text, "math_reasoning", confidence, message)
        )

    for sub in ast.walk(expr):
        if isinstance(sub, ast.Call):
            name = _fullname(sub.func)
            if name in ("np.abs", "abs") and len(sub.args) == 1:
                add(HIGH,
                    "abs(%s) is always >= 0" % _literal_text(sub.args[0]))
            elif name in ("np.exp", "np.expm1") and len(sub.args) == 1:
                add(HIGH,
                    "%s is always > 0 (for exp) / > -1 (for expm1)"
                    % _literal_text(sub))
            elif name in ("np.sin", "np.cos") and len(sub.args) == 1:
                add(MEDIUM,
                    "%s is bounded in [-1, 1]" % _literal_text(sub))
            elif name in ("np.sqrt",) and len(sub.args) == 1:
                add(HIGH,
                    "%s is only defined for nonnegative inputs and is "
                    "always >= 0" % _literal_text(sub))
        elif isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            if _constant_number(sub.right) == 2:
                add(MEDIUM,
                    "%s is always >= 0 for real %s"
                    % (_literal_text(sub), _literal_text(sub.left)))
    return insights


def analyze_expression(expr_text, source_text=""):
    """Analyze a single Python expression string.

    Returns a list of advisory insight dicts.  Unparseable input produces
    an empty list (never raises).
    """
    try:
        expr = ast.parse(expr_text, mode="eval").body
    except (SyntaxError, ValueError):
        return []
    insights = []
    insights.extend(_constant_insight(expr, source_text))
    insights.extend(_identities(expr, source_text))
    insights.extend(_math_reasoning(expr, source_text))
    return insights


# ---------------------------------------------------------------------------
# Statement / translation-level analysis
# ---------------------------------------------------------------------------


def _split_assignment(line):
    """Split ``lhs = rhs`` at the first top-level ``=``."""
    depth = 0
    for i, ch in enumerate(line):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            if i > 0 and line[i - 1] in "<>~!=":
                continue
            return line[:i].strip(), line[i + 1:].strip()
    return None, None


def _line_numbers(result):
    """Map each statement's source to a 1-based line number."""
    source_lines = (result.get("source") or "").splitlines()
    mapping = {}

    def walk(statements):
        for stmt in statements:
            src = (stmt.get("source") or "").splitlines()
            first = next((l.strip() for l in src if l.strip()), None)
            if first:
                for index, line in enumerate(source_lines):
                    if first in line:
                        mapping.setdefault(stmt.get("source"), index + 1)
                        break
            if stmt.get("kind") == "loop":
                walk(stmt.get("body") or [])

    walk(result.get("statements") or [])
    for func in result.get("functions") or []:
        walk(func.get("statements") or [])
    return mapping


def analyze_translation(result):
    """Analyze a translation result and return the ``symbolic`` section.

    The section is advisory and always has status ``ok`` (or ``skipped`` in
    the reverse direction).  It never rejects or rewrites the translation.
    """
    insights = []
    line_map = _line_numbers(result)

    def collect(statements):
        for stmt in statements:
            python = stmt.get("python")
            if not python or python == "UNRESOLVED":
                continue
            source = stmt.get("source") or ""
            line = line_map.get(source)
            for sub in python.split("\n"):
                lhs, rhs = _split_assignment(sub)
                if rhs is None:
                    continue
                if "UNRESOLVED" in sub:
                    continue
                for insight in analyze_expression(rhs, source):
                    insight["line"] = line
                    insights.append(insight)

    collect(result.get("statements") or [])
    for func in result.get("functions") or []:
        collect(func.get("statements") or [])

    counts = {}
    for insight in insights:
        counts[insight["category"]] = counts.get(insight["category"], 0) + 1
    return {
        "status": "ok",
        "insights": insights,
        "counts": counts,
    }
