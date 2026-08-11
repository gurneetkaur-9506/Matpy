"""MATLAB fopen / while-feof / fscanf file-reading idiom recognition.

MATLAB reads a whole whitespace-delimited data file with

    fid = fopen('file', 'r');
    while ~feof(fid)
        data = fscanf(fid, fmt);
    end

which this module recognizes generally and rewrites as a single call to
the specialist library helper ``read_matlab_scan_file(path, fmt)``. The
format string (any of %f / %d / %x / %s / ...) and the later reshape are
left general; only the file handle bookkeeping is specific.
"""

import re

_CALL_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_FEOF_RE = re.compile(r"^~\s*feof\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$")
_FEOF_STATEMENT_RE = re.compile(
    r"^while\s*~\s*feof\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\n(.*?)\s*\nend\s*$",
    re.DOTALL,
)
_STRING_LITERAL_RE = re.compile(r"^'(?:[^']|'')*'$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _split_top_level(text, sep):
    """Split on ``sep`` at bracket depth 0, ignoring separators inside
    string literals (including MATLAB's doubled-quote escape)."""
    parts = []
    depth = 0
    in_string = False
    current = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            current += ch
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
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
        if ch == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def _split_call(expr):
    """Return (name, argtext) when ``expr`` is exactly ``name(args)``."""
    match = _CALL_RE.match(expr)
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


def _matlab_string_to_python(literal):
    """Convert a MATLAB single-quoted literal to a double-quoted Python
    literal, or return None when ``literal`` is not a string literal."""
    text = literal.strip()
    if not _STRING_LITERAL_RE.fullmatch(text):
        return None
    body = text[1:-1].replace("''", "'")
    return '"%s"' % body.replace('"', '\\"')


def _split_assignment(text):
    depth = 0
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 1
                else:
                    in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            i += 1
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return text[:i].strip(), text[i + 1:].strip().rstrip(";").strip()
        i += 1
    return None, None


def translate_fopen(target, value):
    """Translate ``fid = fopen('file', 'r')``.

    Returns a ``(python_line, record)`` pair where ``record`` is the
    ``{"path": ..., "mode": ...}`` dictionary stored for the handle, or
    ``None`` when ``value`` is not an ``fopen`` call.
    """
    call = _split_call(value)
    if not call or call[0] != "fopen":
        return None
    args = [a for a in _split_top_level(call[1], ",") if a.strip()]
    if not args:
        return None

    path = _matlab_string_to_python(args[0])
    if path is None:
        return None

    mode = "r"
    if len(args) >= 2:
        mode_literal = _matlab_string_to_python(args[1])
        if mode_literal is None:
            return None
        mode = mode_literal[1:-1]

    line = "%s = open(%s, '%s')" % (target, path, mode)
    return line, {"path": path, "mode": mode}


def translate_fscanf(target, value, io):
    """Translate ``data = fscanf(fid, fmt)`` given the open-handle map.

    Returns the Python line calling ``read_matlab_scan_file``, or None
    when ``value`` is not a recognized ``fscanf`` over a known handle.
    """
    call = _split_call(value)
    if not call or call[0] != "fscanf":
        return None
    args = [a for a in _split_top_level(call[1], ",") if a.strip()]
    if len(args) < 2:
        return None

    fid = args[0].strip()
    if not _IDENTIFIER_RE.fullmatch(fid) or not io:
        return None
    record = io.get(fid)
    if record is None:
        return None

    fmt = _matlab_string_to_python(args[1])
    if fmt is None:
        return None

    return "%s = read_matlab_scan_file(%s, %s)" % (target, record["path"], fmt)


def translate_feof_loop(header, body, io):
    """Translate a ``while ~feof(fid)`` loop whose body is a single
    ``data = fscanf(fid, fmt)`` assignment into the scan helper call.

    ``header`` is the loop condition text (e.g. ``~feof(fid)``) and
    ``body`` the list of body statements. Returns the Python line, or
    None when the loop is not the recognized read-to-EOF idiom.
    """
    match = _FEOF_RE.fullmatch(header.strip())
    if not match or not io:
        return None
    record = io.get(match.group(1))
    if record is None:
        return None
    if len(body) != 1:
        return None
    stmt = body[0]
    if not hasattr(stmt, "kind") or stmt.kind != "assignment":
        return None
    target, value = _split_assignment(stmt.text)
    if not target or not value:
        return None
    return translate_fscanf(target, value, io)


def translate_feof_statement(text, io):
    """Translate a raw top-level ``while ~feof(fid) ... end`` statement.

    ``text`` is the full multi-line statement text. Returns the Python
    line calling the scan helper, or None when the statement is not the
    recognized read-to-EOF idiom.
    """
    match = _FEOF_STATEMENT_RE.match(text.strip())
    if not match or not io:
        return None
    record = io.get(match.group(1))
    if record is None:
        return None
    body = match.group(2).strip()
    target, value = _split_assignment(body)
    if not target or not value:
        return None
    return translate_fscanf(target, value, io)



