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


def _parse_python(source):
    return ast.parse(source)


def load_structure_from_source(source, direction):
    if direction not in DIRECTIONS:
        raise ValueError("direction must be one of %s" % (DIRECTIONS,))
    tree = _parse_python(source) if direction == PYTHON_TO_MATLAB else _parse_matlab(source)
    return build_structure(tree)


def load_structure(path, direction):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return load_structure_from_source(source, direction)
