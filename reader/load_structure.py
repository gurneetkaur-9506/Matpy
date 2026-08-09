import ast

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from .structure import build_structure

MATLAB_TO_PYTHON = "matlab_to_python"
PYTHON_TO_MATLAB = "python_to_matlab"
DIRECTIONS = (MATLAB_TO_PYTHON, PYTHON_TO_MATLAB)

_parser = None


def _matlab_parser():
    global _parser
    if _parser is None:
        _parser = Parser(Language(language()))
    return _parser


def _parse_matlab(source):
    return _matlab_parser().parse(source.encode("utf-8"))


def join_line_continuations(source):
    """Join any MATLAB line ending in '...' with the next line.

    MATLAB uses '...' as a line-continuation marker: everything from the
    marker to the end of the line is ignored and the statement continues on
    the following line.  Returning the source with such pairs merged into a
    single logical line lets the parser treat them as one statement.
    """
    lines = source.splitlines()
    if not lines:
        return source
    merged = []
    i = 0
    while i < len(lines):
        logical = lines[i]
        while logical.rstrip().endswith("...") and i + 1 < len(lines):
            logical = logical.rstrip()[: -len("...")].rstrip()
            i += 1
            logical = logical + " " + lines[i].lstrip()
        merged.append(logical)
        i += 1
    result = "\n".join(merged)
    if source.endswith("\n"):
        result += "\n"
    return result


def _parse_python(source):
    return ast.parse(source)


def load_structure_from_source(source, direction):
    if direction not in DIRECTIONS:
        raise ValueError("direction must be one of %s" % (DIRECTIONS,))
    if direction == MATLAB_TO_PYTHON:
        source = join_line_continuations(source)
    tree = _parse_python(source) if direction == PYTHON_TO_MATLAB else _parse_matlab(source)
    return build_structure(tree)


def load_structure(path, direction):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return load_structure_from_source(source, direction)
