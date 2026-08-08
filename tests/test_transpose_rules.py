"""Unit tests for the general MATLAB postfix transpose rule.

MATLAB's postfix transpose operators apply to ANY expression they follow:

    expr'  (conjugate transpose) -> np.conj(expr).T
    expr.' (plain transpose)     -> expr.T

The rule is exercised on all four expression categories -- a plain
variable, a function-call result, an indexed expression, and a
parenthesized compound expression -- through the expression translator,
the standalone operator rule, and the full Reader -> Rulebook -> emitter
pipeline.
"""

import ast
import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure
from rulebook import apply_operator_rule, translate_with_rulebook
from rulebook.operator_rules import _split_transpose
from rulebook.translator import _translate_expr
from translator import code_for_result


class TestTranslateExprTranspose(unittest.TestCase):
    """Pipeline expression translator: ``_translate_expr``."""

    def _t(self, expr):
        return _translate_expr(expr, None)

    def test_plain_variable_conjugate_transpose(self):
        self.assertEqual(self._t("A'"), "np.conj(A).T")

    def test_plain_variable_transpose(self):
        self.assertEqual(self._t("A.'"), "A.T")

    def test_function_call_result_conjugate_transpose(self):
        self.assertEqual(self._t("abs(x)'"), "np.conj(np.abs(x)).T")

    def test_function_call_result_transpose(self):
        self.assertEqual(self._t("abs(x).'"), "np.abs(x).T")

    def test_indexed_expression_conjugate_transpose(self):
        self.assertEqual(self._t("M(1,:)'"), "np.conj(M[0, :]).T")

    def test_indexed_expression_transpose(self):
        self.assertEqual(self._t("M(1,:).'"), "M[0, :].T")

    def test_parenthesized_compound_conjugate_transpose(self):
        self.assertEqual(self._t("(A+B)'"), "np.conj((A + B)).T")

    def test_parenthesized_compound_transpose(self):
        self.assertEqual(self._t("(A+B).'"), "(A + B).T")

    def test_transpose_binds_tighter_than_matrix_multiply(self):
        self.assertEqual(self._t("A'*B"), "np.conj(A).T @ B")

    def test_transpose_binds_tighter_than_elementwise_multiply(self):
        self.assertEqual(self._t("A'.*B"), "np.conj(A).T * B")

    def test_transpose_binds_tighter_than_addition(self):
        self.assertEqual(self._t("A + B'"), "A + np.conj(B).T")

    def test_compound_transpose_of_matrix_product(self):
        self.assertEqual(self._t("(A*B)'"), "np.conj((A @ B)).T")

    def test_string_literal_is_not_a_transpose(self):
        self.assertEqual(self._t("'abc'"), "'abc'")

    def test_string_literal_with_escaped_quote_is_not_a_transpose(self):
        self.assertEqual(self._t("'it''s'"), "'it''s'")

    def test_nested_transpose_inside_call_argument(self):
        self.assertEqual(self._t("disp(A')"), "print(np.conj(A).T)")

    def test_double_transpose(self):
        self.assertEqual(self._t("x''"), "np.conj(np.conj(x).T).T")

    def test_matrix_literal_transpose(self):
        self.assertEqual(
            self._t("[1 2 3]'"), "np.conj(np.array([[1, 2, 3]])).T"
        )


class TestApplyOperatorRuleTranspose(unittest.TestCase):
    """Standalone operator rule: ``apply_operator_rule``."""

    def test_plain_variable_conjugate_transpose(self):
        self.assertEqual(apply_operator_rule("A'"), "np.conj(A).T")

    def test_plain_variable_transpose(self):
        self.assertEqual(apply_operator_rule("A.'"), "A.T")

    def test_function_call_result_conjugate_transpose(self):
        self.assertEqual(apply_operator_rule("abs(x)'"), "np.conj(abs(x)).T")

    def test_function_call_result_transpose(self):
        self.assertEqual(apply_operator_rule("abs(x).'"), "abs(x).T")

    def test_indexed_expression_conjugate_transpose(self):
        self.assertEqual(
            apply_operator_rule("M(1,:)'"), "np.conj(M(1,:)).T"
        )

    def test_indexed_expression_transpose(self):
        self.assertEqual(apply_operator_rule("M(1,:).'"), "M(1,:).T")

    def test_parenthesized_compound_conjugate_transpose(self):
        self.assertEqual(apply_operator_rule("(A+B)'"), "np.conj((A+B)).T")

    def test_parenthesized_compound_transpose(self):
        self.assertEqual(apply_operator_rule("(A+B).'"), "(A+B).T")

    def test_transpose_binds_tighter_than_matrix_multiply(self):
        self.assertEqual(apply_operator_rule("A'*B"), "np.conj(A).T @ B")

    def test_transpose_binds_tighter_than_addition(self):
        self.assertEqual(apply_operator_rule("A + B'"), "A + np.conj(B).T")

    def test_string_literal_is_not_a_transpose(self):
        self.assertEqual(apply_operator_rule("'abc'"), "'abc'")


class TestSplitTranspose(unittest.TestCase):
    def test_conjugate_transpose(self):
        self.assertEqual(_split_transpose("A'"), ("A", "conj"))
        self.assertEqual(_split_transpose("abs(x)'"), ("abs(x)", "conj"))
        self.assertEqual(_split_transpose("M(1,:)'"), ("M(1,:)", "conj"))
        self.assertEqual(_split_transpose("(A+B)'"), ("(A+B)", "conj"))

    def test_plain_transpose(self):
        self.assertEqual(_split_transpose("A.'"), ("A", "plain"))
        self.assertEqual(_split_transpose("abs(x).'"), ("abs(x)", "plain"))

    def test_string_literal_returns_none(self):
        self.assertIsNone(_split_transpose("'abc'"))

    def test_no_transpose_returns_none(self):
        self.assertIsNone(_split_transpose("A"))
        self.assertIsNone(_split_transpose("A+B"))
        self.assertIsNone(_split_transpose("A + B."))


class TestTransposePipeline(unittest.TestCase):
    """Full Reader -> Rulebook -> emitter path must produce parseable
    Python with the general transpose mapping applied."""

    @classmethod
    def setUpClass(cls):
        matlab = (
            "function y = go(A, x, M)\n"
            "    B = A';\n"
            "    C = A.';\n"
            "    D = abs(x)';\n"
            "    E = M(1,:)';\n"
            "    F = (A+B)';\n"
            "    G = A'*B;\n"
            "    H = A + B';\n"
            "end\n"
        )
        parser = Parser(Language(language()))
        tree = parser.parse(matlab.encode("utf-8"))
        cls.result = translate_with_rulebook(build_structure(tree))
        cls.code = code_for_result(cls.result)

    def test_output_parses(self):
        ast.parse(self.code)

    def test_variable_conjugate_transpose_emitted(self):
        self.assertIn("B = np.conj(A).T", self.code)

    def test_variable_plain_transpose_emitted(self):
        self.assertIn("C = A.T", self.code)

    def test_call_result_conjugate_transpose_emitted(self):
        self.assertIn("D = np.conj(np.abs(x)).T", self.code)

    def test_indexed_conjugate_transpose_emitted(self):
        self.assertIn("E = np.conj(M[0, :]).T", self.code)

    def test_parenthesized_compound_conjugate_transpose_emitted(self):
        self.assertIn("F = np.conj((A + B)).T", self.code)

    def test_transpose_of_operand_emitted(self):
        self.assertIn("G = np.conj(A).T @ B", self.code)
        self.assertIn("H = A + np.conj(B).T", self.code)


if __name__ == "__main__":
    unittest.main()
