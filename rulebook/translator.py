import ast
import re

from .builtin_rules import BUILTIN_RULES, apply_builtin_rule, apply_builtin_rule_reverse
from .complex_rules import apply_complex_rule
from .format_rules import convert_fprintf
from .indexing_rules import apply_indexing_rule, apply_indexing_rule_reverse
from .multi_output_rules import translate_multi_output_assignment
from .operator_rules import (
    _find_last_operator,
    apply_operator_rule_reverse,
    is_scalar_like,
)

UNRESOLVED = "UNRESOLVED"

_COMMANDS = {
    "clear": "",
    "close all": "",
    "clc": "",
    "figure": "plt.figure()",
}

_OTHER_BUILTINS = {
    "clc", "clear", "close", "cos", "exp", "log", "max", "mean",
    "min", "sin", "sqrt", "sum", "tan",
}

_PLOT_BUILTINS = {
    "figure": "plt.figure",
    "grid": "plt.grid",
    "legend": "plt.legend",
    "plot": "plt.plot",
    "title": "plt.title",
    "xlabel": "plt.xlabel",
    "ylabel": "plt.ylabel",
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


def _split_call(expr):
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
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


def _translate_matrix(expr):
    inner = expr[1:-1].strip()
    rows = _split_top_level(inner, ";")
    arrays = []
    for row in rows:
        cells = [c for c in re.split(r"[,\s]+", row) if c]
        arrays.append("[%s]" % ", ".join(cells))
    return "np.array([%s])" % ", ".join(arrays)


def _is_index_like(arg):
    arg = arg.strip()
    if not arg:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?|end(?:-\d+)?|[A-Za-z_][A-Za-z0-9_]*|:", arg):
        return True
    return ":" in arg


def _split_range(expr):
    return [p for p in _split_top_level(expr, ":") if p.strip()]


def _valid_python(expr):
    try:
        ast.parse(expr)
        return True
    except SyntaxError:
        return False


def _translate_range_part(part, scalars=None):
    part = part.strip()
    if not part:
        return UNRESOLVED
    translated = _translate_expr(part, scalars)
    if translated != UNRESOLVED and "length(" not in translated and _valid_python(translated):
        return translated
    return apply_indexing_rule(part)


_LINSPACE_STEP = re.compile(
    r"^\s*\(?\s*1\s*/\s*\(\s*length\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*-\s*1\s*\)\s*\)?\s*$"
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_scalar(expr, scalars):
    """True when ``expr`` is a scalar: a literal, a scalar-producing call
    (length/numel/...), or a variable previously assigned a scalar value."""
    expr = expr.strip()
    if scalars and _IDENTIFIER.fullmatch(expr) and expr in scalars:
        return True
    return is_scalar_like(expr)


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


def _translate_range(parts, scalars=None):
    translated = [_translate_range_part(p, scalars) for p in parts]
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


def _translate_expr(expr, scalars=None):
    expr = expr.strip()
    if not expr:
        return ""
    if expr.startswith("[") and expr.endswith("]"):
        return _translate_matrix(expr)
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr

    # Drop redundant outer parentheses so parenthesized ranges like
    # (0:(length(P1)-1)) are translated as ranges, not passed through.
    # Re-wrap the result so operator precedence is preserved in context.
    if _outer_parens_wrap(expr):
        inner = _translate_expr(expr[1:-1], scalars)
        return UNRESOLVED if inner == UNRESOLVED else "(%s)" % inner

    range_parts = _split_range(expr)
    if len(range_parts) >= 2:
        return _translate_range(range_parts, scalars)

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name == "disp":
            translated = [_translate_expr(a, scalars) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "print(%s)" % ", ".join(translated)
        if name == "fprintf":
            converted = convert_fprintf(expr, lambda a: _translate_expr(a, scalars))
            if converted is not None:
                return converted
            return UNRESOLVED
        if name in BUILTIN_RULES:
            return apply_builtin_rule(expr)
        if name in _PLOT_BUILTINS:
            translated = [_translate_expr(a, scalars) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "%s(%s)" % (_PLOT_BUILTINS[name], ", ".join(translated))
        if name in _OTHER_BUILTINS:
            return UNRESOLVED
        if name == "find":
            args = _split_top_level(argtext, ",")
            if len(args) != 1:
                return UNRESOLVED
            cond = _translate_expr(args[0], scalars)
            if cond == UNRESOLVED:
                return UNRESOLVED
            return "np.where(%s)[0]" % cond
        if name == "interp1":
            args = _split_top_level(argtext, ",")
            if len(args) != 3:
                return UNRESOLVED
            translated = [_translate_expr(a, scalars) for a in args]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "np.interp(%s, %s, %s)" % (translated[2], translated[0], translated[1])
        args = _split_top_level(argtext, ",")
        if args and all(_is_index_like(a) for a in args):
            return apply_indexing_rule(expr)
        return UNRESOLVED

    idx, op = _find_last_operator(expr)
    if op is None:
        return apply_complex_rule(expr)

    left_py = _translate_expr(expr[:idx], scalars)
    right_py = _translate_expr(expr[idx + len(op):], scalars)
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
    if op == "/":
        # Matrix right-division only when both operands are arrays; a
        # scalar divisor (e.g. length(P2), fs) means element-wise division.
        if _is_scalar(expr[:idx], scalars) or _is_scalar(
            expr[idx + len(op):], scalars
        ):
            return "%s / %s" % (left_py, right_py)
        return "np.linalg.solve(%s.T, %s.T).T" % (right_py, left_py)
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


def _translate_loop(stmt, scalars=None):
    header = "%s %s" % (stmt.type, stmt.header)
    body = [_translate_statement(s, scalars) for s in stmt.statements]
    if any(s["python"] == UNRESOLVED for s in body):
        return {"kind": "loop", "source": header, "python": UNRESOLVED}

    var, eq, expr = stmt.header.partition("=")
    if not eq or ":" not in expr:
        return {"kind": "loop", "source": header, "python": UNRESOLVED}
    converted = apply_indexing_rule(expr.strip())
    if ":" not in converted:
        return {"kind": "loop", "source": header, "python": UNRESOLVED}
    start, stop = converted.split(":", 1)
    if start == "0":
        loop_py = "for %s in range(%s):" % (var.strip(), stop)
    else:
        loop_py = "for %s in range(%s, %s):" % (var.strip(), start, stop)
    return {
        "kind": "loop",
        "source": header,
        "python": loop_py,
        "comment": "# MATLAB: %s; -> Python: %s" % (header, loop_py),
        "body": body,
    }


def _record_scalar(stmt, scalars):
    """Track variables assigned scalar values (e.g. fs = 1000) so a later
    '/' with that variable is recognized as scalar division."""
    if scalars is None or not hasattr(stmt, "kind") or stmt.kind != "assignment":
        return
    target, value = _split_assignment(stmt.text)
    if target is None or value is None:
        return
    if "(" in target:
        return
    if _is_scalar(value, scalars):
        scalars.add(target.strip())


def _translate_statement(stmt, scalars=None):
    if not hasattr(stmt, "kind"):
        return _translate_loop(stmt, scalars)
    if stmt.kind == "command":
        if stmt.text in _COMMANDS:
            return {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": _COMMANDS[stmt.text],
                "comment": _comment_for(stmt, _COMMANDS[stmt.text]),
            }
        return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}

    if stmt.kind == "assignment":
        target, value = _split_assignment(stmt.text)
        if value is None:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        if target.startswith("[") and target.endswith("]"):
            lines = translate_multi_output_assignment(
                target, value, lambda a: _translate_expr(a, scalars)
            )
            if lines is not None:
                python = "\n".join(lines)
                return {
                    "kind": stmt.kind,
                    "source": stmt.text,
                    "python": python,
                    "comment": _comment_for(stmt, python),
                }
        value_py = _translate_expr(value, scalars)
        if value_py == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        target_py = apply_indexing_rule(target) if "(" in target else target
        python = "%s = %s" % (target_py, value_py)
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": python,
            "comment": _comment_for(stmt, python),
        }

    if stmt.kind == "function_call":
        python = _translate_expr(stmt.text, scalars)
        if python == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": python,
            "comment": _comment_for(stmt, python),
        }

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


def _indexed_reference_matrix(loop, var, target, declared):
    for body_stmt in loop.statements:
        if not hasattr(body_stmt, "kind"):
            continue
        for match in _INDEXED_REF.finditer(body_stmt.text):
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


def _preallocations_for_loop(loop, declared):
    var = _loop_variable(loop)
    if not var:
        return []
    preallocations = []
    for body_stmt in loop.statements:
        if not hasattr(body_stmt, "kind") or body_stmt.kind != "assignment":
            continue
        match = _INDEXED_ASSIGN.match(body_stmt.text)
        if not match:
            continue
        target = match.group(1)
        index_tokens = [t.strip() for t in match.group(2).split(",")]
        if target in declared or var not in index_tokens:
            continue
        ref_matrix = _indexed_reference_matrix(loop, var, target, declared)
        preallocations.append(_build_preallocate(target, ref_matrix))
    return preallocations


def _translate_function(func):
    declared = set(func.parameters)
    scalars = set()
    translated = []
    for stmt in func.statements:
        if _is_loop(stmt):
            translated.extend(_preallocations_for_loop(stmt, declared))
        translated.append(_translate_statement(stmt, scalars))
        _record_scalar(stmt, scalars)
        target = _declare_target(stmt.text) if hasattr(stmt, "kind") else None
        if target:
            declared.add(target)
    return {
        "name": func.name,
        "parameters": list(func.parameters),
        "outputs": list(func.outputs),
        "statements": translated,
    }


def translate_with_rulebook(structure):
    result = {"functions": [], "statements": []}
    for func in structure.functions:
        result["functions"].append(_translate_function(func))
    scalars = set()
    for stmt in structure.statements:
        result["statements"].append(_translate_statement(stmt, scalars))
        _record_scalar(stmt, scalars)
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


def _translate_expr_reverse(expr):
    expr = expr.strip()
    if not expr:
        return ""
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr

    match = re.fullmatch(r"np\.array\((.*)\)", expr, re.DOTALL)
    if match:
        matrix = _translate_matrix_reverse(match.group(1))
        return matrix if matrix is not None else UNRESOLVED

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name == "print":
            translated = [
                _translate_expr_reverse(a) for a in _split_top_level(argtext, ",")
            ]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "disp(%s)" % ", ".join(translated)
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
        reversed_call = apply_builtin_rule_reverse(expr)
        if reversed_call != expr:
            return reversed_call
        return UNRESOLVED

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\[.*\]", expr):
        return apply_indexing_rule_reverse(expr)

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
            "source": "%s %s" % (stmt.type, stmt.header),
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
                "statements": [
                    _translate_statement_reverse(s) for s in func.statements
                ],
            }
        )
    for stmt in structure.statements:
        result["statements"].append(_translate_statement_reverse(stmt))
    return result
