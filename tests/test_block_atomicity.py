"""Atomic-block invariant tests for unresolvable MATLAB block constructs.

Whenever a block construct (while, for, if, switch, nested combinations)
cannot be fully translated, the Reader/Rulebook must capture the ENTIRE
block from its opening keyword through its matching 'end' as one atomic
unit and emit it as a single UNRESOLVED comment.  A partial translation
that leaks raw MATLAB syntax, a stray 'end', or half-converted lines must
never appear in the output.

These tests drive the full Reader -> Rulebook -> emitter path and assert,
for five different unresolvable block shapes, that the emitted output
always parses as valid Python, that the whole block survives as one
commented unit, and that no MATLAB keyword ever leaks as live code.
"""

import ast
import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure
from rulebook import UNRESOLVED, assert_block_invariant, translate_with_rulebook
from translator import code_for_result

# One unresolvable block shape per case, each with the raw source of the
# whole construct (opening keyword through the matching 'end').
BLOCK_SHAPES = {
    "bare_while": {
        "matlab": (
            "function y = foo(x)\n"
            "    while x > 0\n"
            "        x = x - 1;\n"
            "    end\n"
            "    y = x;\n"
            "end\n"
        ),
        "block": "while x > 0\n        x = x - 1;\n    end",
    },
    "for_with_if_else": {
        "matlab": (
            "function y = foo(x)\n"
            "    for i = 1:10\n"
            "        if mod(i, 2) == 0\n"
            "            y = y + i;\n"
            "        else\n"
            "            y = y - i;\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "for i = 1:10\n"
            "        if mod(i, 2) == 0\n"
            "            y = y + i;\n"
            "        else\n"
            "            y = y - i;\n"
            "        end\n"
            "    end"
        ),
    },
    "nested_for_for_if": {
        "matlab": (
            "function y = foo(A)\n"
            "    for i = 1:3\n"
            "        for j = 1:3\n"
            "            if A(i, j) > 0\n"
            "                y = y + 1;\n"
            "            end\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "for i = 1:3\n"
            "        for j = 1:3\n"
            "            if A(i, j) > 0\n"
            "                y = y + 1;\n"
            "            end\n"
            "        end\n"
            "    end"
        ),
    },
    "triple_nested_for_if_elseif_else": {
        "matlab": (
            "function y = foo(A)\n"
            "    for i = 1:3\n"
            "        for j = 1:3\n"
            "            for k = 1:3\n"
            "                if A(i, j, k) > 0\n"
            "                    y = y + 1;\n"
            "                elseif A(i, j, k) < 0\n"
            "                    y = y - 1;\n"
            "                else\n"
            "                    y = y;\n"
            "                end\n"
            "            end\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "for i = 1:3\n"
            "        for j = 1:3\n"
            "            for k = 1:3\n"
            "                if A(i, j, k) > 0\n"
            "                    y = y + 1;\n"
            "                elseif A(i, j, k) < 0\n"
            "                    y = y - 1;\n"
            "                else\n"
            "                    y = y;\n"
            "                end\n"
            "            end\n"
            "        end\n"
            "    end"
        ),
    },
    "triple_nested_for_mixed_ifs": {
        "matlab": (
            "function y = foo(A)\n"
            "    for i = 1:3\n"
            "        if i > 1\n"
            "            for j = 1:3\n"
            "                if j > 1\n"
            "                    for k = 1:3\n"
            "                        if A(i, j, k) > 0\n"
            "                            y = y + 1;\n"
            "                        elseif A(i, j, k) < 0\n"
            "                            y = y - 1;\n"
            "                        else\n"
            "                            y = y;\n"
            "                        end\n"
            "                    end\n"
            "                end\n"
            "            end\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "for i = 1:3\n"
            "        if i > 1\n"
            "            for j = 1:3\n"
            "                if j > 1\n"
            "                    for k = 1:3\n"
            "                        if A(i, j, k) > 0\n"
            "                            y = y + 1;\n"
            "                        elseif A(i, j, k) < 0\n"
            "                            y = y - 1;\n"
            "                        else\n"
            "                            y = y;\n"
            "                        end\n"
            "                    end\n"
            "                end\n"
            "            end\n"
            "        end\n"
            "    end"
        ),
    },
    "switch_case": {
        "matlab": (
            "function y = foo(x)\n"
            "    switch x\n"
            "        case 1\n"
            "            y = 10;\n"
            "        case 2\n"
            "            y = 20;\n"
            "        otherwise\n"
            "            y = 0;\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "switch x\n"
            "        case 1\n"
            "            y = 10;\n"
            "        case 2\n"
            "            y = 20;\n"
            "        otherwise\n"
            "            y = 0;\n"
            "    end"
        ),
    },
    "while_with_unknown_call": {
        "matlab": (
            "function y = foo(a, b)\n"
            "    k = 0;\n"
            "    while k < 5\n"
            "        k = k + 1;\n"
            "        y = blend(a + b);\n"
            "    end\n"
            "end\n"
        ),
        "block": (
            "while k < 5\n"
            "        k = k + 1;\n"
            "        y = blend(a + b);\n"
            "    end"
        ),
    },
}

_STRAY_KEYWORDS = {
    "end",
    "else",
    "elseif",
    "otherwise",
    "switch",
    "case",
    "while",
    "for",
}


def _translate_matlab(source):
    parser = Parser(Language(language()))
    tree = parser.parse(source.encode("utf-8"))
    return translate_with_rulebook(build_structure(tree))


def _unresolved_block_sources(result):
    """Raw source text of every block construct left UNRESOLVED."""
    sources = []

    def walk(statements):
        for stmt in statements:
            if stmt.get("python") == UNRESOLVED:
                sources.append(stmt.get("source", ""))
            walk(stmt.get("body") or [])

    for func in result.get("functions", []):
        walk(func.get("statements", []))
    walk(result.get("statements", []))
    return sources


class TestUnresolvableBlockAtomicity(unittest.TestCase):
    def _assert_safe_output(self, matlab_source, block_source):
        result = _translate_matlab(matlab_source)

        # The general invariant holds after this translation attempt.
        assert_block_invariant(result)

        block_sources = _unresolved_block_sources(result)
        self.assertIn(block_source, block_sources)
        # The whole block is captured as exactly one atomic UNRESOLVED unit:
        # no other statement carries a chunk of the same construct.
        self.assertEqual(
            [s for s in block_sources if s.strip() == block_source.strip()],
            [block_source],
        )

        code = code_for_result(result)
        # The emitted Python must always parse.
        ast.parse(code)
        return code

    def _assert_whole_block_comment(self, code, block_source):
        block_lines = [line.strip() for line in block_source.splitlines() if line.strip()]
        code_lines = code.splitlines()

        self.assertTrue(
            any(
                "# UNRESOLVED: %s" % block_lines[0] in line for line in code_lines
            ),
            "block opening not wrapped as a single UNRESOLVED comment",
        )

        for raw in block_lines:
            commented = [l for l in code_lines if raw in l]
            self.assertTrue(
                commented, "block line missing from output: %r" % raw
            )
            for line in commented:
                self.assertTrue(
                    line.lstrip().startswith("#"),
                    "raw MATLAB leaked uncommented: %r" % line,
                )

    def _assert_no_partial_mix(self, code, block_source):
        block_lines = {line.strip() for line in block_source.splitlines() if line.strip()}
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in _STRAY_KEYWORDS:
                self.fail("stray MATLAB keyword leaked as code: %r" % line)
            for raw in block_lines:
                if raw in line:
                    self.fail("raw MATLAB content leaked as code: %r" % line)

    def test_bare_while(self):
        shape = BLOCK_SHAPES["bare_while"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_for_containing_if_else(self):
        shape = BLOCK_SHAPES["for_with_if_else"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_nested_for_for_if(self):
        shape = BLOCK_SHAPES["nested_for_for_if"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_triple_nested_for_if_elseif_else(self):
        shape = BLOCK_SHAPES["triple_nested_for_if_elseif_else"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_triple_nested_for_mixed_ifs(self):
        shape = BLOCK_SHAPES["triple_nested_for_mixed_ifs"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_switch_case(self):
        shape = BLOCK_SHAPES["switch_case"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_while_containing_unknown_function_call(self):
        shape = BLOCK_SHAPES["while_with_unknown_call"]
        code = self._assert_safe_output(shape["matlab"], shape["block"])
        self._assert_whole_block_comment(code, shape["block"])
        self._assert_no_partial_mix(code, shape["block"])

    def test_unresolved_block_has_no_surviving_body(self):
        shape = BLOCK_SHAPES["nested_for_for_if"]
        result = _translate_matlab(shape["matlab"])
        for func in result["functions"]:
            for stmt in func["statements"]:
                if stmt.get("python") == UNRESOLVED:
                    self.assertNotIn("body", stmt)

    def test_deep_unresolved_block_has_no_surviving_body(self):
        shape = BLOCK_SHAPES["triple_nested_for_if_elseif_else"]
        result = _translate_matlab(shape["matlab"])
        for func in result["functions"]:
            for stmt in func["statements"]:
                if stmt.get("python") == UNRESOLVED:
                    self.assertNotIn("body", stmt)


class TestBlockInvariantFunction(unittest.TestCase):
    def test_assert_block_invariant_passes_on_resolved_loop(self):
        matlab = (
            "function total = sumup(N)\n"
            "    total = 0;\n"
            "    for n = 1:N\n"
            "        total = total + n;\n"
            "    end\n"
            "end\n"
        )
        result = _translate_matlab(matlab)
        assert_block_invariant(result)
        code = code_for_result(result)
        ast.parse(code)
        self.assertIn("for n in range(N):", code)

    def test_triple_nested_for_fully_translated(self):
        """The atomic-block invariant is depth-independent on the resolved
        side too: a 3-level nested for loop whose innermost body translates
        cleanly is emitted as fully nested Python, never collapsed."""
        matlab = (
            "function S = nested(N)\n"
            "    S = 0;\n"
            "    for i = 1:N\n"
            "        for j = 1:N\n"
            "            for k = 1:N\n"
            "                S = S + i + j + k;\n"
            "            end\n"
            "        end\n"
            "    end\n"
            "end\n"
        )
        result = _translate_matlab(matlab)
        assert_block_invariant(result)
        code = code_for_result(result)
        ast.parse(code)
        self.assertNotIn(UNRESOLVED, code)
        self.assertIn("for i in range(N):", code)
        self.assertIn("for j in range(N):", code)
        self.assertIn("for k in range(N):", code)
        self.assertIn("S = S + i + j + k", code)

    def test_assert_block_invariant_detects_partial_body(self):
        result = {
            "statements": [],
            "functions": [
                {
                    "name": "f",
                    "statements": [
                        {
                            "kind": "loop",
                            "source": "for i = 1:10\n    end",
                            "python": "for i in range(10):",
                            "body": [
                                {
                                    "kind": "if_statement",
                                    "source": "if x > 0\n    end",
                                    "python": "UNRESOLVED",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(AssertionError):
            assert_block_invariant(result)


if __name__ == "__main__":
    unittest.main()
