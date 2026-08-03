import re

from .builtin_rules import BUILTIN_RULES, apply_builtin_rule
from .complex_rules import apply_complex_rule
from .indexing_rules import apply_indexing_rule
from .operator_rules import _find_last_operator

UNRESOLVED = "UNRESOLVED"

_COMMANDS = {
    "clear": "",
    "close all": "",
    "clc": "",
    "figure": "plt.figure()",
}

_OTHER_BUILTINS = {
    "clc", "clear", "close", "cos", "exp", "figure", "log", "max", "mean",
    "min", "plot", "sin", "sqrt", "sum", "tan", "title", "xlabel", "ylabel",
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
        if name in _OTHER_BUILTINS:
            return UNRESOLVED
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
    return UNRESOLVED


def _translate_statement(stmt):
    if stmt.kind == "command":
        if stmt.text in _COMMANDS:
            return {
                "kind": stmt.kind,
                "source": stmt.text,
                "python": _COMMANDS[stmt.text],
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
        return {
            "kind": stmt.kind,
            "source": stmt.text,
            "python": "%s = %s" % (target_py, value_py),
        }

    if stmt.kind == "function_call":
        return {"kind": stmt.kind, "source": stmt.text, "python": _translate_expr(stmt.text)}

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
