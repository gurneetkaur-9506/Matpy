import re

_IMAG_LITERAL = re.compile(r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)i\b")


def apply_complex_rule(expr):
    return _IMAG_LITERAL.sub(r"\1j", expr)
