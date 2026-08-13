"""Build a plain-language report of every issue in a translation result.

The Checker's job is to explain, not just to score.  build_translation_report()
flattens the three kinds of findings the pipeline produces into a single list
of dicts, one per issue:

    * rulebook lines that were left UNRESOLVED,
    * rulebook lines whose generated Python does not parse (syntax errors),
    * Checker verdicts of "failed", "review needed", or
      "inconclusive_no_matlab".

Each entry tells a human what went wrong: where it happened (line number in
the original source when it can be located), the original source text, what
the pipeline attempted, and a plain-language reason -- never a stack trace.
"""

import re

from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED

from .accuracy import syntax_error

_KIND_NORMALIZE = {
    "Assign": "assignment",
    "Expr": "function_call",
    "Return": "return",
    "Import": "command",
    "ImportFrom": "command",
}

_CALL_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _normalized_kind(stmt):
    kind = stmt.get("kind")
    return _KIND_NORMALIZE.get(kind, kind)


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
    """Return the 1-based line where ``source`` begins, or None."""
    if not source or not source_lines:
        return None
    first = next((line.strip() for line in source.splitlines() if line.strip()), None)
    if not first:
        return None
    for index, line in enumerate(source_lines):
        if first in line:
            return index + 1
    return None


def _walk(statements):
    for stmt in statements:
        yield stmt
        yield from _walk(stmt.get("body") or [])


def _all_statements(result):
    for stmt in _walk(result.get("statements") or []):
        yield stmt
    for func in result.get("functions") or []:
        for stmt in _walk(func.get("statements") or []):
            yield stmt


def _is_unresolved(stmt):
    return stmt.get("python") == UNRESOLVED or stmt.get("matlab") == UNRESOLVED


def _attempted_text(stmt):
    kind = _normalized_kind(stmt)
    if kind == "command":
        return "looked the command up in the rulebook's command table"
    if kind == "assignment":
        return (
            "tried to translate the right-hand expression with the operator "
            "and builtin rules"
        )
    if kind == "function_call":
        return "tried to map the call through the builtin, plot, and indexing rules"
    if kind == "loop":
        return "tried to convert the loop into a Python range() loop"
    if kind == "return":
        return "tried to reconstruct the return value with the reverse rules"
    return "tried to translate the statement with the rulebook"


def _unresolved_reason(stmt):
    kind = _normalized_kind(stmt)
    source = stmt.get("source") or ""
    if kind == "command":
        return (
            "MATLAB command %r has no direct Python equivalent in the "
            "rulebook." % source
        )
    if kind == "function_call":
        match = _CALL_NAME.match(source)
        if match:
            return (
                "The rulebook has no rule for the function %r, so the call "
                "was left for manual review." % match.group(1)
            )
        return "The call does not match any rulebook pattern."
    if kind == "assignment":
        target, sep, value = source.partition("=")
        expr = value.strip() or source
        return (
            "The expression %r could not be reduced by the rulebook's "
            "operator or builtin rules." % expr
        )
    if kind == "loop":
        return (
            "The loop %r does not fit the range pattern the rulebook "
            "translates." % source
        )
    if kind == "return":
        return (
            "The return value in %r could not be reconstructed by the "
            "reverse rules." % source
        )
    return "No rulebook rule matched this statement."


def _unresolved_entry(stmt, source_lines):
    source = stmt.get("source") or ""
    return {
        "line": _locate(source, source_lines),
        "source": source,
        "issue": "unresolved",
        "stage": "rulebook",
        "attempted": _attempted_text(stmt),
        "reason": _unresolved_reason(stmt),
    }


def _syntax_entries(result, source_lines):
    """Flag rulebook lines whose translated Python does not parse.

    A line that fails ast.parse cannot be trusted even if the rulebook
    matched it, so it is reported as a syntax error.  Only forward
    (MATLAB -> Python) output is validated; in the reverse direction the
    "python" key is the input, not generated code.
    """
    if result.get("direction") == PYTHON_TO_MATLAB:
        return []
    entries = []
    for stmt in _all_statements(result):
        if stmt.get("python") == UNRESOLVED:
            continue
        reason = syntax_error(stmt)
        if reason is None:
            continue
        source = stmt.get("source") or ""
        entries.append(
            {
                "line": _locate(source, source_lines),
                "source": source,
                "issue": "syntax error",
                "stage": "rulebook",
                "attempted": _attempted_text(stmt),
                "reason": reason,
            }
        )
    return entries


def _checker_entries(result):
    checker = (result.get("sections") or {}).get("checker") or {}
    status = checker.get("status")
    if status == "failed":
        reason = (
            "The checker compared the reference and translated outputs and "
            "they disagreed beyond the allowed tolerance."
        )
    elif status == "review needed":
        reason = (
            "The checker could not decide whether the outputs match: an "
            "execution failure, misaligned output names, shape mismatch, or "
            "non-finite values were involved."
        )
    elif status == "inconclusive_no_matlab":
        reason = (
            "The checker could not reach a conclusive verdict because the "
            "reference is only a seeded mock rather than real MATLAB "
            "output; the comparison ran but its result is inconclusive."
        )
    else:
        return []
    return [
        {
            "line": None,
            "source": str(result.get("file") or ""),
            "issue": status,
            "stage": "checker",
            "attempted": (
                "Compared the translated output against the reference "
                "numerically."
            ),
            "reason": reason,
        }
    ]


def build_translation_report(result):
    """Collect every issue in a translate_file() result into one list.

    Args:
        result: The result dict returned by translator.translate_file.

    Returns:
        A list of dicts, one per issue, in pipeline order (rulebook
        unresolved lines and syntax errors first, then the Checker verdict).
        Each dict has the keys:

            line:      1-based line number in the original source, or None
                       when it cannot be located (e.g. a whole-file verdict).
            source:    the original source text of the problematic line.
            issue:     "unresolved", "syntax error", "failed",
                       "review needed", or "inconclusive_no_matlab".
            stage:     the pipeline stage that reported it: "rulebook" or
                       "checker".
            attempted: what that stage tried to do.
            reason:    a plain-language explanation of why it could not be
                       resolved.  Never a stack trace.
    """
    source_lines = _source_lines(result)
    report = [
        _unresolved_entry(stmt, source_lines)
        for stmt in _all_statements(result)
        if _is_unresolved(stmt)
    ]
    report.extend(_syntax_entries(result, source_lines))
    report.extend(_checker_entries(result))
    return report
