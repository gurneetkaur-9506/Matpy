import re

# General, table-driven reverse mapping from Python array/object attribute
# access (``obj.attr``) to the equivalent MATLAB function call.  Any
# attribute-access pattern is checked against this table before falling
# through to UNRESOLVED, so the rule is not tied to a specific variable
# name -- every property lookup uses the same table.
ATTRIBUTE_RULES_REVERSE = {
    "shape": "size",
    "size": "numel",
    "T": "transpose",
    "dtype": "class",
}

_ATTRIBUTE_ACCESS = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(shape|size|T|dtype)\s*"
)
_SHAPE_INDEX = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*shape\s*\[\s*([0-9]+)\s*\]"
)


def apply_attribute_rule_reverse(expr):
    expr = expr.strip()
    match = _SHAPE_INDEX.fullmatch(expr)
    if match:
        return "size(%s, %d)" % (match.group(1), int(match.group(2)) + 1)
    match = _ATTRIBUTE_ACCESS.fullmatch(expr)
    if not match:
        return expr
    obj, attr = match.group(1), match.group(2)
    return "%s(%s)" % (ATTRIBUTE_RULES_REVERSE[attr], obj)
