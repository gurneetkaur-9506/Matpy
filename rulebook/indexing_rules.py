import re

from .index_shift import FORWARD, REVERSE, UNRESOLVED, shift_index

INDEXING_RULES = {
    "1_based_offset": 1,
    "0_based_offset": 0,
    "colon": ":",
    "end_keyword": "end",
    "patterns": {
        "integer": r"^([0-9]+)$",
        "colon": r"^:$",
        "end": r"^end$",
        "end_minus_int": r"^end\s*-\s*([0-9]+)$",
        "identifier": r"^[A-Za-z_][A-Za-z0-9_]*$",
        "length_call": r"^length\s*\(\s*(.*)\s*\)$",
        "range": r"^(.*):(.*)$",
        "index_expr": r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$",
    },
}


def _split_index_arithmetic(expr):
    """Return the index and operator of the last top-level arithmetic
    operator in ``expr``, or (None, None) when there is none."""
    depth = 0
    for i in range(len(expr) - 1, -1, -1):
        ch = expr[i]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            depth -= 1
        elif depth == 0 and i > 0 and ch in "+-*/":
            return i, ch
    return None, None


def _shift_compound_operand(part):
    """Shift a single operand of a compound index expression: indexed
    accesses are translated recursively, while non-indexed operands (value
    scalars, variable offsets) keep their literals and only get len()
    calls converted."""
    if "[" in part:
        return apply_indexing_rule_reverse(part)
    return part.strip().replace("len(", "length(")


def _convert_range_part_reverse(part, is_start):
    part = part.strip()
    if not part:
        return "1" if is_start else "end"
    patterns = INDEXING_RULES["patterns"]
    if re.fullmatch(patterns["integer"], part):
        return shift_index(part, REVERSE) if is_start else part
    negative = re.fullmatch(r"-([0-9]+)", part)
    if negative:
        return "%s-%s" % (INDEXING_RULES["end_keyword"], negative.group(1))
    return apply_indexing_rule_reverse(part)


def apply_indexing_rule_reverse(expr):
    """Translate a Python index expression into MATLAB.

    Pure index shifting (integer literals, variable pass-through) is
    delegated to the shared ``shift_index`` primitive; compound index
    expressions such as ``t[1] - t[0]`` are decomposed on their top-level
    operators and each indexed operand is shifted.  The Python-only
    conventions the primitive intentionally leaves alone -- len() calls,
    negative literals as end-keywords, range-stop collapsing -- are
    normalized here.
    """
    expr = expr.strip()
    patterns = INDEXING_RULES["patterns"]

    if re.fullmatch(patterns["integer"], expr):
        return shift_index(expr, REVERSE)

    if expr == INDEXING_RULES["colon"]:
        return INDEXING_RULES["colon"]

    negative = re.fullmatch(r"-([0-9]+)", expr)
    if negative:
        k = int(negative.group(1))
        if k == 1:
            return INDEXING_RULES["end_keyword"]
        return "%s-%d" % (INDEXING_RULES["end_keyword"], k - 1)

    if re.fullmatch(patterns["identifier"], expr):
        return shift_index(expr, REVERSE)

    match = re.fullmatch(r"^len\s*\(\s*(.*)\s*\)$", expr)
    if match:
        return "length(" + match.group(1) + ")"

    if "[" in expr:
        idx, op = _split_index_arithmetic(expr)
        if idx is not None:
            left = _shift_compound_operand(expr[:idx])
            right = _shift_compound_operand(expr[idx + 1:])
            return "%s %s %s" % (left, op, right)

    match = re.fullmatch(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]$", expr)
    if match:
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        converted = ", ".join(apply_indexing_rule_reverse(a) for a in args)
        return "%s(%s)" % (match.group(1), converted)

    match = re.fullmatch(patterns["range"], expr)
    if match:
        start = _convert_range_part_reverse(match.group(1), is_start=True)
        stop = _convert_range_part_reverse(match.group(2), is_start=False)
        return "%s:%s" % (start, stop)

    return expr.replace("len(", "length(")


def _convert_range_part(part, is_start):
    part = part.strip()
    if not part:
        return ""
    patterns = INDEXING_RULES["patterns"]
    if re.fullmatch(patterns["integer"], part):
        return shift_index(part, FORWARD) if is_start else part
    if part == INDEXING_RULES["end_keyword"]:
        return ""
    match = re.fullmatch(patterns["end_minus_int"], part)
    if match:
        return "-" + match.group(1)
    if is_start and not re.fullmatch(r"length\s*\(.*\)", part):
        # A slice start is a MATLAB index, so a compound expression such as
        # samplesDelay+1 must be shifted down by one (samplesDelay).  The
        # shared primitive folds the +-1 offset into the trailing constant
        # term; length() calls are normalized to len() afterwards.
        shifted = shift_index(part, FORWARD)
        if shifted != part and shifted != UNRESOLVED:
            return shifted.replace("length(", "len(")
    return apply_indexing_rule(part)


def apply_indexing_rule(expr):
    """Translate a MATLAB index expression into Python.

    Pure index shifting (integer literals, variable pass-through) is
    delegated to the shared ``shift_index`` primitive; the MATLAB-only
    conventions that primitive intentionally leaves alone -- end-keywords,
    length() calls, range-stop collapsing -- are normalized here.
    """
    expr = expr.strip()
    patterns = INDEXING_RULES["patterns"]

    if re.fullmatch(patterns["integer"], expr):
        return shift_index(expr, FORWARD)

    if expr == INDEXING_RULES["colon"]:
        return INDEXING_RULES["colon"]

    if expr == INDEXING_RULES["end_keyword"]:
        return "-1"

    match = re.fullmatch(patterns["end_minus_int"], expr)
    if match:
        return "-%d" % (int(match.group(1)) + 1)

    if re.fullmatch(patterns["identifier"], expr):
        return shift_index(expr, FORWARD)

    match = re.fullmatch(patterns["length_call"], expr)
    if match:
        return "len(" + match.group(1) + ")"

    match = re.fullmatch(patterns["index_expr"], expr)
    if match:
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        converted = ", ".join(apply_indexing_rule(a) for a in args)
        return "%s[%s]" % (match.group(1), converted)

    match = re.fullmatch(patterns["range"], expr)
    if match:
        start = _convert_range_part(match.group(1), is_start=True)
        stop = _convert_range_part(match.group(2), is_start=False)
        return "%s:%s" % (start, stop)

    return expr.replace("length(", "len(")
