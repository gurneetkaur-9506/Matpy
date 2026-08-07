import unittest

from rulebook import (
    apply_operator_rule,
    apply_operator_rule_reverse,
    scientific_literals,
)
from rulebook.translator import _translate_expr, _translate_expr_reverse


class TestScientificLiteralTokenizer(unittest.TestCase):
    """The tokenizer must recognize the whole class of scientific-notation
    literals matching digits(.digits)?[eE][+-]?digits."""

    def test_scientific_literals_found(self):
        self.assertEqual(scientific_literals("10e-6"), ["10e-6"])
        self.assertEqual(scientific_literals("1e-3"), ["1e-3"])
        self.assertEqual(scientific_literals("5e6"), ["5e6"])
        self.assertEqual(scientific_literals("2.5e-9"), ["2.5e-9"])
        self.assertEqual(scientific_literals("1E+10"), ["1E+10"])
        self.assertEqual(scientific_literals("3e0"), ["3e0"])

    def test_negative_mantissa_keeps_literal(self):
        self.assertEqual(scientific_literals("-1e-6"), ["1e-6"])

    def test_literals_in_larger_expression(self):
        self.assertEqual(scientific_literals("2*10e-6+1"), ["10e-6"])

    def test_no_scientific_literal(self):
        self.assertEqual(scientific_literals("a @ b"), [])
        self.assertEqual(scientific_literals("2.5 * x"), [])
        self.assertEqual(scientific_literals("A(1,:)"), [])


class TestScientificNotationAtomicForward(unittest.TestCase):
    """Operator splitting must treat scientific literals as atomic in the
    forward (MATLAB -> Python) direction."""

    def test_plain_literals_pass_through(self):
        for literal in ("10e-6", "1e-3", "5e6", "2.5e-9", "1E+10", "3e0"):
            self.assertEqual(apply_operator_rule(literal), literal)
            self.assertEqual(_translate_expr(literal), literal)

    def test_negative_mantissa_passthrough(self):
        self.assertEqual(apply_operator_rule("-1e-6"), "-1e-6")
        self.assertEqual(_translate_expr("-1e-6"), "-1e-6")

    def test_literal_embedded_in_expression_stays_atomic(self):
        result = apply_operator_rule("2*10e-6+1")
        self.assertIn("10e-6", result)
        self.assertNotIn("10e ", result)
        self.assertEqual(_translate_expr("2*10e-6+1"), "2 * 10e-6 + 1")

    def test_literal_after_multiply(self):
        result = apply_operator_rule("2*1e-3")
        self.assertEqual(result, "2 * 1e-3")
        self.assertEqual(_translate_expr("2*1e-3"), "2 * 1e-3")

    def test_literal_in_right_divide(self):
        result = apply_operator_rule("x/2.5e-9")
        self.assertEqual(result, "x / 2.5e-9")
        self.assertNotIn("2.5e - 9", result)

    def test_mantissa_dot_not_misread_as_elementwise(self):
        self.assertEqual(apply_operator_rule("2.5e-9 * x"), "2.5e-9 * x")
        self.assertEqual(apply_operator_rule("2.5e-9 .* x"), "2.5e-9 * x")
        self.assertEqual(_translate_expr("2.5e-9*x"), "2.5e-9 * x")

    def test_elementwise_between_literals(self):
        self.assertEqual(apply_operator_rule("1e-3 .* 2e-3"), "1e-3 * 2e-3")

    def test_chained_literals(self):
        self.assertEqual(apply_operator_rule("2*1e-3*5e6"), "2 * 1e-3 * 5e6")


class TestScientificNotationAtomicReverse(unittest.TestCase):
    """The same guarantee must hold in the reverse (Python -> MATLAB) rules."""

    def test_plain_literals_pass_through(self):
        for literal in ("10e-6", "1e-3", "5e6", "2.5e-9", "1E+10", "3e0"):
            self.assertEqual(apply_operator_rule_reverse(literal), literal)
            self.assertEqual(_translate_expr_reverse(literal), literal)

    def test_literal_embedded_in_expression_stays_atomic(self):
        result = apply_operator_rule_reverse("2*10e-6+1")
        self.assertEqual(result, "2 .* 10e-6+1")
        self.assertIn("10e-6", result)
        self.assertNotIn("10e .* -6", result)

    def test_literal_after_multiply(self):
        self.assertEqual(apply_operator_rule_reverse("2*1e-3"), "2 .* 1e-3")

    def test_literal_in_divide(self):
        result = apply_operator_rule_reverse("x/2.5e-9")
        self.assertEqual(result, "x ./ 2.5e-9")

    def test_mantissa_dot_not_misread(self):
        self.assertEqual(apply_operator_rule_reverse("2.5e-9 * x"), "2.5e-9 .* x")


if __name__ == "__main__":
    unittest.main()
