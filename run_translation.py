import inspect
import sys

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from assistant import draft_unresolved_functions
from reader import build_structure, load_matlab_file
from rulebook import UNRESOLVED, translate_with_rulebook
from specialist_lib import __all__ as SPECIALIST_NAMES

import specialist_lib

MATLAB_FILE = "/sample_matlab/indexing_ops.m"


def _specialist_lib_contents():
    return {
        name: inspect.getsource(getattr(specialist_lib, name))
        for name in SPECIALIST_NAMES
    }


def translate_to_python(path):
    parser = Parser(Language(language()))
    tree = parser.parse(load_matlab_file(path).encode("utf-8"))
    structure = build_structure(tree)
    result = translate_with_rulebook(structure)
    return draft_unresolved_functions(result, _specialist_lib_contents())


def _emit_block(statements, lines):
    dropped = []
    for stmt in statements:
        if stmt["kind"] == "command" and not stmt["python"]:
            dropped.append(stmt["source"])
            continue
        if dropped:
            lines.append("# MATLAB: %s -> Python: re-initialize state (no-op here)" % "; ".join(dropped))
            dropped = []
        if stmt["python"] == UNRESOLVED:
            lines.append("# UNRESOLVED: %s" % stmt["source"])
            continue
        lines.append(stmt["comment"])
        lines.append(stmt["python"])
    if dropped:
        lines.append("# MATLAB: %s -> Python: re-initialize state (no-op here)" % "; ".join(dropped))


def _emit_function(func, lines):
    lines.append("")
    lines.append("def %s(...):" % func["name"])
    _emit_block(func["statements"], lines)
    draft = func.get("draft")
    if draft:
        notes = "; ".join(draft["notes"]) if draft["notes"] else "none"
        lines.append(
            "# Assistant draft: confidence=%.2f notes=%s" % (draft["confidence"], notes)
        )
        if draft["code"]:
            lines.append(draft["code"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else MATLAB_FILE
    result = translate_to_python(path)
    lines = ["import numpy as np", ""]
    _emit_block(result["statements"], lines)
    for func in result["functions"]:
        _emit_function(func, lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
