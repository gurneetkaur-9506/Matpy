"""Accuracy scoring for translated output.

The score is based on real correctness, not merely on rule matching:

    1. Every generated line must be valid Python.  Each statement's output
       is validated with ast.parse (loop headers are parsed together with
       their indented bodies), and the whole generated module is parsed as
       one unit as well.  A line that does not parse counts as unresolved.

    2. Where possible, the score is driven by a numeric comparison against
       real output rather than by counting matched rules:
           "verified"  -> the translated output numerically matches the
                          reference, so every resolved line earns full
                          weight (1.0);
           "failed"    -> the numeric comparison found disagreement, so
                          every resolved line earns weight 0.0.
       Without a conclusive numeric verdict the score falls back to
       provenance weights:
           rulebook-resolved lines ......... 1.0  (fully resolved by rules)
           assistant-drafted lines ......... the Assistant's confidence
                                              (0.0-1.0)
           unresolved lines ................ 0.0  (needs human review)

The score is a single number in 0-100 percent.  The result also reports the
weighted line contribution of each source (``breakdown``) and the ``method``
that produced the score, so callers can see whether it came from a numeric
comparison or from rulebook matching.
"""

import ast

from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED

WEIGHTS = {
    "rulebook": 1.0,
    "verified": 1.0,
    "failed": 0.0,
    "unresolved": 0.0,
}

_METHODS = {
    "verified": "numeric comparison against real output passed",
    "failed": "numeric comparison against real output failed",
    "rulebook": "rulebook matching with per-line ast.parse validation",
}


def _syntax_code(stmt):
    """Reconstruct parseable Python for a statement.

    A loop's ``python`` is just the header (e.g. "for n in range(N):"),
    which does not parse on its own, so it is validated together with its
    indented body.
    """
    code = stmt.get("python") or ""
    if stmt.get("kind") == "loop":
        lines = [code]
        for body in stmt.get("body") or []:
            lines.extend("    " + line for line in _syntax_code(body).splitlines())
        return "\n".join(lines)
    return code


def syntax_error(stmt):
    """Return a plain-language reason when a statement's Python does not parse.

    Statements marked UNRESOLVED, or with no Python output, are skipped (they
    are already handled as unresolved).  Returns None when the code parses.
    """
    code = stmt.get("python")
    if not code or code == UNRESOLVED:
        return None
    try:
        ast.parse(_syntax_code(stmt))
    except SyntaxError as exc:
        return "The generated Python does not parse (%s)." % exc.msg
    return None


def module_syntax_error(result):
    """Return a plain-language reason when the whole generated module does not
    parse, or None when it does.

    Only forward (MATLAB -> Python) output is Python that ast.parse can
    validate; in the reverse direction the "python" key is the input.
    """
    if result.get("direction") == PYTHON_TO_MATLAB:
        return None
    code = result.get("python") or ""
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return "The generated Python module does not parse (%s)." % exc.msg
    return None


def _is_unresolved_or_invalid(stmt, check_syntax, module_invalid=False):
    if stmt.get("python") == UNRESOLVED or stmt.get("matlab") == UNRESOLVED:
        return True
    if module_invalid:
        return True
    if check_syntax and syntax_error(stmt) is not None:
        return True
    return False


def _line_weight(item):
    source = item["source"]
    if source in WEIGHTS:
        return WEIGHTS[source]
    try:
        return max(0.0, min(1.0, float(item.get("weight", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _method_for(items, fallback):
    for source in ("verified", "failed"):
        if any(item["source"] == source for item in items):
            return _METHODS[source]
    return fallback


def score_mix(items, method=None):
    """Score an explicit per-source line mix.

    Args:
        items: List of {"source": str, "lines": int, "weight": float}.
            "lines" is the number of translated lines with that provenance.
            "weight" is the confidence weight; only consulted for "assistant".
        method: Optional description of how the score was produced.  When
            omitted, it is derived from the sources present ("verified" or
            "failed" numeric verdicts, otherwise "rulebook").

    Returns:
        Dict with:
            score: 0-100 percent (rounded to two decimals)
            total_lines: total number of lines
            weighted_lines: sum of line_count * weight
            breakdown: weighted line contribution per source
            method: how the score was computed
    """
    total = sum(item["lines"] for item in items)
    if total <= 0:
        return {
            "score": 0.0,
            "total_lines": 0,
            "weighted_lines": 0.0,
            "breakdown": {},
            "method": method or "no lines to score",
        }
    weighted = sum(item["lines"] * _line_weight(item) for item in items)
    breakdown = {}
    for item in items:
        source = item["source"]
        breakdown[source] = breakdown.get(source, 0.0) + item["lines"] * _line_weight(
            item
        )
    return {
        "score": round(weighted / total * 100, 2),
        "total_lines": total,
        "weighted_lines": round(weighted, 4),
        "breakdown": {k: round(v, 4) for k, v in sorted(breakdown.items())},
        "method": method or _method_for(items, _METHODS["rulebook"]),
    }


def accuracy(result):
    """Compute the accuracy score of a translate_file() result.

    Args:
        result: The result dict returned by translator.translate_file.

    Returns:
        The same shape as score_mix.
    """
    sections = result.get("sections", {}) or {}
    rulebook = sections.get("rulebook", {}) or {}
    total = rulebook.get("total", 0)
    unresolved_total = rulebook.get("unresolved", 0)
    checker = sections.get("checker", {}) or {}
    verdict = checker.get("status")
    verified = verdict == "verified"
    failed = verdict == "failed"
    # Only forward (MATLAB -> Python) output is Python that ast.parse can
    # validate; in the reverse direction the "python" key is the input.
    check_syntax = result.get("direction") != PYTHON_TO_MATLAB
    module_invalid = check_syntax and module_syntax_error(result) is not None

    items = []
    func_lines_total = 0
    func_unresolved_total = 0
    for func in result.get("functions", []) or []:
        statements = func.get("statements") or []
        count = len(statements)
        if not count:
            continue
        func_lines_total += count
        unresolved = sum(
            1
            for s in statements
            if _is_unresolved_or_invalid(s, check_syntax, module_invalid)
        )
        func_unresolved_total += unresolved
        if verified:
            resolved = count - unresolved
            if resolved:
                items.append({"source": "verified", "lines": resolved, "weight": 1.0})
            if unresolved:
                items.append(
                    {"source": "unresolved", "lines": unresolved, "weight": 0.0}
                )
        elif failed:
            # The numeric comparison against real output disagreed, so even
            # rules that matched cannot be trusted.
            items.append({"source": "failed", "lines": count, "weight": 0.0})
        elif func.get("draft"):
            resolved = count - unresolved
            if resolved:
                items.append(
                    {
                        "source": "assistant",
                        "lines": resolved,
                        "weight": float(func["draft"].get("confidence", 0.0)),
                    }
                )
            if unresolved:
                items.append(
                    {"source": "unresolved", "lines": unresolved, "weight": 0.0}
                )
        else:
            resolved = count - unresolved
            if resolved:
                items.append({"source": "rulebook", "lines": resolved, "weight": 1.0})
            if unresolved:
                items.append(
                    {"source": "unresolved", "lines": unresolved, "weight": 0.0}
                )

    script_statements = result.get("statements") or []
    if script_statements:
        script_lines = len(script_statements)
        script_unresolved = sum(
            1
            for s in script_statements
            if _is_unresolved_or_invalid(s, check_syntax, module_invalid)
        )
    else:
        script_lines = max(total - func_lines_total, 0)
        script_unresolved = max(unresolved_total - func_unresolved_total, 0)
    script_resolved = max(script_lines - script_unresolved, 0)
    if script_lines and verified:
        if script_resolved:
            items.append({"source": "verified", "lines": script_resolved, "weight": 1.0})
        if script_unresolved:
            items.append(
                {"source": "unresolved", "lines": script_unresolved, "weight": 0.0}
            )
    elif script_lines and failed:
        items.append({"source": "failed", "lines": script_lines, "weight": 0.0})
    else:
        if script_resolved:
            items.append({"source": "rulebook", "lines": script_resolved, "weight": 1.0})
        if script_unresolved:
            items.append(
                {"source": "unresolved", "lines": script_unresolved, "weight": 0.0}
            )

    if verified:
        method = _METHODS["verified"]
    elif failed:
        method = _METHODS["failed"]
    elif check_syntax:
        method = _METHODS["rulebook"]
    else:
        method = "rulebook matching (reverse direction)"
    return score_mix(items, method=method)
