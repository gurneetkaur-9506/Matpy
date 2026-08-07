"""MATLAB fprintf -> Python print conversion.

MATLAB's fprintf(fmt, args...) and Python's ``%`` operator share the same
format-spec syntax (%.2f, %d, %s, %g, %e and \\n all map directly), so the
general conversion is: ``fprintf(fmt, a, b)`` -> ``print(fmt % (a, b))``.

The only rewriting needed is resolving MATLAB's single-quote escaping (a
single quote inside a literal is written as two, ``''``) and emitting the
literal as a Python string literal.
"""

import re

UNRESOLVED = "UNRESOLVED"

_CALL_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STRING_LITERAL_RE = re.compile(r"^'(?:[^']|'')*'$")
_FORMAT_SPEC_RE = re.compile(r"%[-+ #0]*(?:\d+)?(?:\.\d+)?[diouxXeEfFgGcs%]")


def _split_args(s):
    args = []
    depth = 0
    in_string = False
    current = ""
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_string:
            current += ch
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    current += "'"
                    i += 1
                else:
                    in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            current += ch
            i += 1
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        args.append(current.strip())
    return args


def _is_string_literal(text):
    return bool(_STRING_LITERAL_RE.fullmatch(text.strip()))


def matlab_string_literal_to_python(literal):
    """Convert a MATLAB single-quoted literal to a Python string literal.

    MATLAB escapes a single quote by doubling it (``''``); ``\\n`` and ``\\t``
    keep their usual meaning. The Python literal is emitted with double
    quotes so embedded single quotes need no escaping.
    """
    text = literal.strip()
    if not _is_string_literal(text):
        return text
    body = text[1:-1].replace("''", "'")
    return '"%s"' % body.replace('"', '\\"')


def format_spec_count(fmt):
    """Count format specifiers (like %.2f, %d) in a format string, ignoring
    the escaped literal percent ``%%``. Useful for arg-count sanity checks."""
    count = 0
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != "%":
            i += 1
            continue
        if i + 1 < n and fmt[i + 1] == "%":
            i += 2
            continue
        m = _FORMAT_SPEC_RE.match(fmt, i)
        if m:
            count += 1
            i = m.end()
        else:
            i += 1
    return count


def convert_fprintf(expr, translate_arg):
    """Translate a MATLAB ``fprintf(fmt, args...)`` call into a Python
    ``print(fmt % (args))`` expression.

    ``translate_arg`` maps each non-format argument expression from MATLAB to
    Python. Returns ``None`` when ``expr`` is not a well-formed ``fprintf``
    call so the caller can fall back to its default handling.
    """
    expr = expr.strip()
    match = _CALL_RE.match(expr)
    if not match:
        return None
    if match.group(1) != "fprintf":
        return None

    start = expr.find("(", match.start())
    depth = 0
    end = -1
    for i in range(start, len(expr)):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                if expr[i + 1:].strip() == "":
                    end = i
                break
    if end < 0:
        return None

    args = [a for a in _split_args(expr[start + 1:end]) if a]
    if not args:
        return None

    fmt = args[0]
    if _is_string_literal(fmt):
        fmt_py = matlab_string_literal_to_python(fmt)
    else:
        fmt_py = translate_arg(fmt)
    if fmt_py == UNRESOLVED:
        return None

    translated = [translate_arg(a) for a in args[1:]]
    if any(t == UNRESOLVED for t in translated):
        return None

    if not translated:
        return "print(%s)" % fmt_py
    return "print(%s %% (%s))" % (fmt_py, ", ".join(translated))
