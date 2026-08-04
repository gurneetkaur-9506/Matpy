import re

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


def _convert_range_part_reverse(part, is_start):
    part = part.strip()
    if not part:
        return "1" if is_start else "end"
    patterns = INDEXING_RULES["patterns"]
    if re.fullmatch(patterns["integer"], part):
        return str(int(part) + 1) if is_start else part
    negative = re.fullmatch(r"-([0-9]+)", part)
    if negative:
        return "%s-%s" % (INDEXING_RULES["end_keyword"], negative.group(1))
    return apply_indexing_rule_reverse(part)


def apply_indexing_rule_reverse(expr):
    expr = expr.strip()
    patterns = INDEXING_RULES["patterns"]

    if re.fullmatch(patterns["integer"], expr):
        return str(int(expr) + 1)

    if expr == INDEXING_RULES["colon"]:
        return INDEXING_RULES["colon"]

    negative = re.fullmatch(r"-([0-9]+)", expr)
    if negative:
        k = int(negative.group(1))
        if k == 1:
            return INDEXING_RULES["end_keyword"]
        return "%s-%d" % (INDEXING_RULES["end_keyword"], k - 1)

    if re.fullmatch(patterns["identifier"], expr):
        return expr

    match = re.fullmatch(r"^len\s*\(\s*(.*)\s*\)$", expr)
    if match:
        return "length(" + match.group(1) + ")"

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
        return str(int(part) - INDEXING_RULES["1_based_offset"]) if is_start else part
    if part == INDEXING_RULES["end_keyword"]:
        return ""
    match = re.fullmatch(patterns["end_minus_int"], part)
    if match:
        return "-" + match.group(1)
    return apply_indexing_rule(part)


def apply_indexing_rule(expr):
    expr = expr.strip()
    patterns = INDEXING_RULES["patterns"]

    if re.fullmatch(patterns["integer"], expr):
        return str(int(expr) - INDEXING_RULES["1_based_offset"])

    if expr == INDEXING_RULES["colon"]:
        return INDEXING_RULES["colon"]

    if expr == INDEXING_RULES["end_keyword"]:
        return "-1"

    match = re.fullmatch(patterns["end_minus_int"], expr)
    if match:
        return "-%d" % (int(match.group(1)) + 1)

    if re.fullmatch(patterns["identifier"], expr):
        return expr

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
