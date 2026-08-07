"""Accuracy scoring for translated output.

Every translated line is assigned a weight based on how trustworthy its
provenance is:

    score = sum(line_count * weight) / total_lines * 100

Weights by source:
    rulebook-resolved lines ......... 1.0  (fully resolved by rules)
    checker-verified lines .......... 1.0  (numeric cross-check passed)
    assistant-drafted lines ......... the Assistant's own confidence (0.0-1.0)
    unresolved lines ................ 0.0  (needs human review)

The score is a single number in 0-100 percent.  The result also reports the
weighted line contribution of each source so callers can see exactly where
the score comes from.
"""

import ast

from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED

WEIGHTS = {
    "rulebook": 1.0,
    "verified": 1.0,
    "unresolved": 0.0,
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


def _is_unresolved_or_invalid(stmt, check_syntax):
    if stmt.get("python") == UNRESOLVED or stmt.get("matlab") == UNRESOLVED:
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


def score_mix(items):
    """Score an explicit per-source line mix.

    Args:
        items: List of {"source": str, "lines": int, "weight": float}.
            "lines" is the number of translated lines with that provenance.
            "weight" is the confidence weight; only consulted for "assistant".

    Returns:
        Dict with:
            score: 0-100 percent (rounded to two decimals)
            total_lines: total number of lines
            weighted_lines: sum of line_count * weight
            breakdown: weighted line contribution per source
    """
    total = sum(item["lines"] for item in items)
    if total <= 0:
        return {
            "score": 0.0,
            "total_lines": 0,
            "weighted_lines": 0.0,
            "breakdown": {},
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
    verified = (sections.get("checker", {}) or {}).get("status") == "verified"
    # Only forward (MATLAB -> Python) output is Python that ast.parse can
    # validate; in the reverse direction the "python" key is the input.
    check_syntax = result.get("direction") != PYTHON_TO_MATLAB

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
            1 for s in statements if _is_unresolved_or_invalid(s, check_syntax)
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
        elif func.get("draft"):
            items.append(
                {
                    "source": "assistant",
                    "lines": count,
                    "weight": float(func["draft"].get("confidence", 0.0)),
                }
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
            1 for s in script_statements if _is_unresolved_or_invalid(s, check_syntax)
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
    else:
        if script_resolved:
            items.append({"source": "rulebook", "lines": script_resolved, "weight": 1.0})
        if script_unresolved:
            items.append(
                {"source": "unresolved", "lines": script_unresolved, "weight": 0.0}
            )

    return score_mix(items)
