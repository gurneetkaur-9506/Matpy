import re

from .builtin_rules import BUILTIN_RULES, apply_builtin_rule, apply_builtin_rule_reverse
from .complex_rules import apply_complex_rule
from .indexing_rules import apply_indexing_rule, apply_indexing_rule_reverse
from .operator_rules import _find_last_operator, apply_operator_rule_reverse

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


def _translate_expr(expr):
    expr = expr.strip()
    if not expr:
        return ""
    if expr.startswith("[") and expr.endswith("]"):
        return _translate_matrix(expr)
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr

    call = _split_call(expr)
    if call is not None:
        name, argtext = call
        if name == "disp":
            translated = [_translate_expr(a) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "print(%s)" % ", ".join(translated)
        if name in BUILTIN_RULES:
            return apply_builtin_rule(expr)
        if name in _PLOT_BUILTINS:
            translated = [_translate_expr(a) for a in _split_top_level(argtext, ",")]
            if any(t == UNRESOLVED for t in translated):
                return UNRESOLVED
            return "%s(%s)" % (_PLOT_BUILTINS[name], ", ".join(translated))
        if name in _OTHER_BUILTINS:
            return UNRESOLVED
        if name == "find":
            args = _split_top_level(argtext, ",")
            if len(args) != 1:
                return UNRESOLVED
            cond = _translate_expr(args[0])
            if cond == UNRESOLVED:
                return UNRESOLVED
            return "np.where(%s)[0]" % cond
        if name == "interp1":
            args = _split_top_level(argtext, ",")
            if len(args) != 3:
                return UNRESOLVED
            translated = [_translate_expr(a) for a in args]
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

    left_py = _translate_expr(expr[:idx])
    right_py = _translate_expr(expr[idx + len(op):])
    if left_py == UNRESOLVED or right_py == UNRESOLVED:
        return UNRESOLVED

    if op == "*":
        return "%s @ %s" % (left_py, right_py)
    if op == ".*":
        return "%s * %s" % (left_py, right_py)
    if op == "./":
        return "%s / %s" % (left_py, right_py)
    if op == "/":
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


def _translate_loop(stmt):
    header = "%s %s" % (stmt.type, stmt.header)
    body = [_translate_statement(s) for s in stmt.statements]
    if any(s["python"] == UNRESOLVED for s in body):
        return {"kind": "loop", "source": header, "python": UNRESOLVED}

    var, eq, expr = stmt.header.partition("=")
    if not eq or ":" not in expr:
        return {"kind": "loop", "source": header, "python": UNRESOLVED}
    converted = apply_indexing_rule(expr.strip())
    if ":" not in converted:
        return {"kind": "loop", "source": header, "python": UNRESOLVED}
    start, stop = converted.split(":", 1)
    loop_py = "for %s in range(%s, %s):" % (var.strip(), start, stop)
    return {
        "kind": "loop",
        "source": header,
        "python": loop_py,
        "comment": "# MATLAB: %s; -> Python: %s" % (header, loop_py),
        "body": body,
    }


def _translate_statement(stmt):
    if not hasattr(stmt, "kind"):
        return _translate_loop(stmt)
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
        value_py = _translate_expr(value)
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
        python = _translate_expr(stmt.text)
        if python == UNRESOLVED:
            return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": python,
            "comment": _comment_for(stmt, python),
        }

    return {"kind": stmt.kind, "source": stmt.text, "python": UNRESOLVED}


def translate_with_rulebook(structure):
    result = {"functions": [], "statements": []}
    for func in structure.functions:
        result["functions"].append(
            {
                "name": func.name,
                "statements": [_translate_statement(s) for s in func.statements],
            }
        )
    for stmt in structure.statements:
        result["statements"].append(_translate_statement(stmt))
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
                "statements": [
                    _translate_statement_reverse(s) for s in func.statements
                ],
            }
        )
    for stmt in structure.statements:
        result["statements"].append(_translate_statement_reverse(stmt))
    return result
