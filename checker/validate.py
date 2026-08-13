"""Lightweight static validation of a MATLAB-to-Python translation result.

Runs after the rulebook has produced Python and before the Checker.  It is
purely static and offline -- no AI, no cloud, no execution -- and never
rejects or rewrites code.  It only produces advisory warnings with a
confidence level so a human can focus their review:

    * undefined_variable     -- a name is used but never assigned in scope
                                 (HIGH for functions, MEDIUM for scripts).
    * unsupported_construct  -- a MATLAB feature the rulebook cannot
                                 translate: if/switch/try blocks, global/
                                 persistent, cell arrays, struct field
                                 access, anonymous functions, function
                                 handles, double-quoted strings, while
                                 loops.
    * suspicious_operator    -- a MATLAB-only operator (.*, ./, .\\, .^,
                                 ~=, &&, ||, ^) was left unconverted in the
                                 emitted Python.
    * unresolved_function    -- an unqualified call in the emitted Python
                                 that is not a Python builtin, a same-file
                                 function, or a defined variable.
    * unsafe_translation     -- an assigned name shadows a name the
                                 generated code (or Python) relies on,
                                 e.g. a variable named ``np``.

Every warning carries ``confidence`` in {HIGH, MEDIUM, LOW} so valid code is
never treated as an error -- warnings are advisory only.
"""

import builtins
import re

from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED
from rulebook.builtin_rules import BUILTIN_RULES
from rulebook.keyword_rules import PYTHON_BUILTINS, PYTHON_KEYWORDS
from rulebook.multi_output_rules import _REDUCTION_CALLS
from rulebook.translator import (
    PLOT_COMMANDS,
    _COMMANDS,
    _DIM_STAT_BUILTINS,
    _NUMPY_CALLS,
    _OTHER_BUILTINS,
    _TRIG_DEGREES,
)

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

_STAGE = "validation"

# ---------------------------------------------------------------------------
# Name tables
# ---------------------------------------------------------------------------

_MATLAB_CONSTANTS = {
    "pi", "i", "j", "inf", "Inf", "nan", "NaN", "eps",
    "true", "false", "end",
}

_MATLAB_KEYWORDS = {
    "if", "elseif", "else", "for", "while", "switch", "case", "otherwise",
    "function", "return", "global", "persistent", "try", "catch",
    "continue", "break", "parfor", "classdef", "properties", "methods",
    "events",
}

# Names the rulebook (or its specialist library) recognizes as callables.
# These are never undefined-variable candidates when they appear bare.
_HANDLED_CALLS = (
    set(BUILTIN_RULES)
    | set(_NUMPY_CALLS)
    | set(_TRIG_DEGREES)
    | set(_REDUCTION_CALLS)
    | set(_DIM_STAT_BUILTINS)
    | set(PLOT_COMMANDS)
    | set(_COMMANDS)
    | set(_OTHER_BUILTINS)
    | {
        "disp", "fprintf", "fclose", "fopen", "fscanf", "feof", "sprintf",
        "length", "numel", "find", "interp1",
    }
)

# Names the pipeline always qualifies as ``np.<name>`` in the output, so a
# variable with the same name cannot break the generated code.  They are
# excluded from the shadowing warnings to keep the noise down.
_HARMLESS_SHADOWS = {"abs", "sum", "min", "max", "round", "any", "all"}

# Python builtins the generated code itself calls unqualified, so a variable
# of the same name can change the meaning of the output at runtime.
_GENERATOR_BUILTINS = {"len", "range", "print", "open", "input"}

# MATLAB constructs the rulebook has no rule for, keyed by statement kind.
_UNSUPPORTED_KIND_MESSAGE = {
    "if_statement": (
        "if/elseif/else branches are not translated by the rulebook; the "
        "block is left as an UNRESOLVED comment for manual review"
    ),
    "switch_statement": (
        "switch/case is not supported; the block is left as an UNRESOLVED "
        "comment for manual review"
    ),
    "try_statement": (
        "try/catch is not supported; the block is left as an UNRESOLVED "
        "comment for manual review"
    ),
    "global_operator": (
        "the 'global' statement is not supported; global state does not "
        "translate to Python"
    ),
    "persistent_operator": (
        "the 'persistent' statement is not supported; persistent state does "
        "not translate to Python"
    ),
}

# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _strip_strings(text):
    """Replace MATLAB/Python string literals with spaces, preserving length.

    Handles single-quoted MATLAB char arrays with doubled-quote escapes
    (``'it''s'``) and double-quoted strings with backslash escapes.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(" " * (j - i + 1))
            i = j + 1
        elif ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(" " * (j - i + 1))
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _strip_comments(text):
    """Remove '%' line comments, keeping string literals intact."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
        elif ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
        elif ch == "%":
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


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


def _matching_bracket(text):
    """Index of the bracket that closes the first character, or -1."""
    if not text:
        return -1
    opener = text[0]
    closer = ")" if opener == "(" else "]"
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_assignment(text):
    """Split ``text`` at its first top-level assignment ``=``.

    Comparison operators (``==``, ``~=``, ``<=``, ``>=``) are not
    assignments, so their ``=`` is skipped.  Returns ``(lhs, rhs)`` or
    ``(None, None)`` when there is no assignment.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            if i > 0 and text[i - 1] in "<>~!=":
                continue
            return text[:i].strip(), text[i + 1:].strip()
    return None, None


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _assigned_names(lhs):
    """Names created by an assignment left-hand side.

    ``acc(n) = ...`` assigns ``acc``; ``[v, i] = max(x)`` assigns ``v`` and
    ``i``; ``s.field = 3`` creates struct ``s``.  Identifiers used only as
    indices or field names on the left side are not assigned names.
    """
    text = _strip_strings(lhs.strip())
    if not text:
        return set()
    if text.startswith("[") and _matching_bracket(text) == len(text) - 1:
        parts = _split_top_level(text[1:-1], ",")
        parts = [p for p in parts if p.strip()]
        names = set()
        for part in parts:
            names |= _assigned_names(part)
        return names
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if not match:
        return set()
    return {match.group(1)}


def _used_names(text):
    """Identifiers read from an expression, excluding call names, field
    names, keywords, and constants.

    A call name (identifier followed by ``(``) is handled by the
    unresolved-function check instead; a field name (after ``.``) is
    handled by the unsupported-construct check.
    """
    text = _strip_strings(text)
    names = set()
    for m in _IDENT.finditer(text):
        start = m.start()
        end = m.end()
        if start > 0 and text[start - 1] == ".":
            continue
        name = m.group()
        if name in _MATLAB_KEYWORDS or name in _MATLAB_CONSTANTS:
            continue
        j = end
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "(":
            continue
        names.add(name)
    return names


def _loop_header_parts(source):
    """Return ``(loop_var, rhs)`` for a for/while loop's first line."""
    first = (source or "").splitlines()[0].strip()
    for kw in ("for", "while"):
        if first.startswith(kw):
            first = first[len(kw):].strip()
            break
    return _split_assignment(first)


# ---------------------------------------------------------------------------
# Warning emission helpers
# ---------------------------------------------------------------------------


def _warn(line, source, category, confidence, message):
    return {
        "line": line,
        "source": source,
        "stage": _STAGE,
        "category": category,
        "confidence": confidence,
        "message": message,
    }


def _source_lines(result):
    source = result.get("source")
    if source:
        return source.splitlines()
    path = result.get("file")
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _locate(source, source_lines):
    if not source or not source_lines:
        return None
    first = next((line.strip() for line in source.splitlines() if line.strip()), None)
    if not first:
        return None
    for index, line in enumerate(source_lines):
        if first in line:
            return index + 1
    return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

_FIELD_ACCESS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")
_LEFT_OVER_OP = re.compile(r"\.\*|\./|\.\\|\.\^|~=|&&|\|\||\^")
_CALL_RE = re.compile(r"(?:^|[^.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ANON_FUNC = re.compile(r"@\s*\(")
_FUNC_HANDLE = re.compile(r"@\s*[A-Za-z_][A-Za-z0-9_]*")


def _check_unsupported(stmt, line, warnings):
    source = stmt.get("source") or ""
    kind = stmt.get("kind")
    if kind in _UNSUPPORTED_KIND_MESSAGE:
        warnings.append(
            _warn(
                line, source, "unsupported_construct", HIGH,
                _UNSUPPORTED_KIND_MESSAGE[kind],
            )
        )
    first = (source.splitlines()[0].strip() if source else "").lower()
    if kind == "loop" and first.startswith("while"):
        warnings.append(
            _warn(
                line, source, "unsupported_construct", HIGH,
                "while loops are not translated by the rulebook; the loop "
                "is left as an UNRESOLVED comment for manual review",
            )
        )

    text = _strip_strings(source)
    if "{" in text:
        warnings.append(
            _warn(
                line, source, "unsupported_construct", MEDIUM,
                "cell arrays ('{...}') are not supported; MATLAB cell "
                "semantics do not translate to Python",
            )
        )
    for m in _FIELD_ACCESS.finditer(text):
        j = m.end()
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "(":
            continue
        warnings.append(
            _warn(
                line, source, "unsupported_construct", MEDIUM,
                "struct field access ('%s') is not supported" % m.group(0),
            )
        )
        break
    if _ANON_FUNC.search(text):
        warnings.append(
            _warn(
                line, source, "unsupported_construct", HIGH,
                "anonymous functions ('@(...)') are not supported",
            )
        )
    elif _FUNC_HANDLE.search(text):
        warnings.append(
            _warn(
                line, source, "unsupported_construct", MEDIUM,
                "function handles ('@name') are not supported",
            )
        )
    if '"' in _strip_comments(source):
        warnings.append(
            _warn(
                line, source, "unsupported_construct", LOW,
                "double-quoted string arrays are not translated (only "
                "single-quoted char arrays are)",
            )
        )


def _check_undefined(name, source, line, scope, warnings, high=True):
    if name in scope:
        return
    warnings.append(
        _warn(
            line, source, "undefined_variable",
            HIGH if high else MEDIUM,
            "variable '%s' is used but never assigned in this scope" % name,
        )
    )


def _check_leftover_operators(python, source, line, warnings):
    text = _strip_strings(python)
    m = _LEFT_OVER_OP.search(text)
    if m is None:
        return
    warnings.append(
        _warn(
            line, source, "suspicious_operator", HIGH,
            "MATLAB operator '%s' was left unconverted in the translated "
            "output; the generated Python is likely wrong" % m.group(0),
        )
    )


def _check_unresolved_calls(python, source, line, warnings, scope, function_names):
    text = _strip_strings(python)
    python_builtins = set(dir(builtins))
    for m in _CALL_RE.finditer(text):
        name = m.group(1)
        if name in python_builtins or name in function_names or name in scope:
            continue
        warnings.append(
            _warn(
                line, source, "unresolved_function", HIGH,
                "function '%s' is not recognized by the rulebook and was "
                "passed through; it will raise NameError unless defined" % name,
            )
        )


def _check_shadows(assigned, first_assign, warnings):
    python_builtins = set(dir(builtins))
    for name in sorted(assigned):
        if name == "np":
            confidence, message = HIGH, (
                "variable 'np' shadows the NumPy module alias used by the "
                "generated code; every np.* call will break"
            )
        elif name in _GENERATOR_BUILTINS:
            confidence, message = MEDIUM, (
                "variable '%s' shadows the Python builtin '%s' which the "
                "generated code calls unqualified" % (name, name)
            )
        elif name in python_builtins and name not in _HARMLESS_SHADOWS:
            confidence, message = LOW, (
                "variable '%s' shadows a Python builtin; check it is not "
                "called where the builtin was intended" % name
            )
        else:
            continue
        line, source = first_assign[name]
        warnings.append(
            _warn(line, source, "unsafe_translation", confidence, message)
        )


# ---------------------------------------------------------------------------
# Scope walker
# ---------------------------------------------------------------------------


def _check_block(statements, initial_scope, function_names, warnings, script=False,
                 source_lines=None):
    # Function parameters arrive in their renamed Python form (``lambda_``),
    # so also seed the original MATLAB name (``lambda``) that the source
    # actually uses.
    scope = set(initial_scope)
    for name in initial_scope:
        base = name.rstrip("_")
        if base != name and base in (PYTHON_KEYWORDS | PYTHON_BUILTINS):
            scope.add(base)
    scope |= _MATLAB_CONSTANTS | _HANDLED_CALLS
    assigned = set()
    first_assign = {}

    def record_assign(name, line, source):
        assigned.add(name)
        if name not in first_assign:
            first_assign[name] = (line, source)

    def walk(stmts, source_lines):
        for stmt in stmts:
            source = stmt.get("source") or ""
            line = _locate(source, source_lines)
            kind = stmt.get("kind")
            unresolved = stmt.get("python") == UNRESOLVED

            _check_unsupported(stmt, line, warnings)

            if kind == "loop":
                if unresolved:
                    # The whole loop was left UNRESOLVED by the rulebook
                    # (e.g. a while loop); walking its body would only
                    # repeat the unsupported-construct warning in pieces.
                    continue
                var, rhs = _loop_header_parts(source)
                if rhs and line is not None:
                    for name in _used_names(rhs):
                        _check_undefined(name, source, line, scope, warnings, high=not script)
                if var:
                    scope.add(var)
                    record_assign(var, line, source)
                walk(stmt.get("body") or [], source_lines)
                continue

            if unresolved:
                continue

            # A function handle/anonymous-function line is already reported
            # as an unsupported construct; its parameters would otherwise
            # look like undefined variables.
            has_handle = "@" in _strip_strings(source)

            lhs, rhs = _split_assignment(source)
            if rhs is not None:
                if not has_handle and line is not None:
                    for name in _used_names(rhs):
                        _check_undefined(name, source, line, scope, warnings, high=not script)
                for name in _assigned_names(lhs):
                    scope.add(name)
                    record_assign(name, line, source)
            elif kind != "command" and not has_handle and line is not None:
                for name in _used_names(source):
                    _check_undefined(name, source, line, scope, warnings, high=not script)

            python = stmt.get("python") or ""
            _check_leftover_operators(python, source, line, warnings)
            _check_unresolved_calls(python, source, line, warnings, scope, function_names)

    walk(statements, source_lines)
    _check_shadows(assigned, first_assign, warnings)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_translation(result):
    """Return a list of advisory warnings for a translation result.

    Only the MATLAB-to-Python direction is validated (the rulebook stage is
    what produces the emitted Python the checks inspect).  Warnings are
    advisory: they carry a HIGH/MEDIUM/LOW confidence and never reject or
    rewrite the translation.
    """
    if result.get("direction") == PYTHON_TO_MATLAB:
        return []
    warnings = []
    source_lines = _source_lines(result)
    function_names = {f.get("name") for f in result.get("functions") or []}

    _check_block(
        result.get("statements") or [],
        set(),
        function_names,
        warnings,
        script=True,
        source_lines=source_lines,
    )
    for func in result.get("functions") or []:
        _check_block(
            func.get("statements") or [],
            set(func.get("parameters") or []),
            function_names,
            warnings,
            script=False,
            source_lines=source_lines,
        )
    warnings.sort(
        key=lambda w: (w["line"] if w["line"] is not None else float("inf"), w["category"])
    )
    return warnings
