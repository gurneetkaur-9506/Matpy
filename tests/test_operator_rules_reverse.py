import unittest

from rulebook import (
    apply_operator_rule,
    apply_operator_rule_reverse,
)


class TestApplyOperatorRuleReverse(unittest.TestCase):
    def test_matrix_multiply(self):
        self.assertEqual(apply_operator_rule_reverse("a @ b"), "a * b")

    def test_elementwise_multiply(self):
        self.assertEqual(apply_operator_rule_reverse("a * b"), "a .* b")

    def test_elementwise_divide(self):
        self.assertEqual(apply_operator_rule_reverse("a / b"), "a ./ b")

    def test_matrix_right_divide_solve_pattern(self):
        self.assertEqual(
            apply_operator_rule_reverse("np.linalg.solve(b.T, a.T).T"),
            "a / b",
        )

    def test_matrix_right_divide_chained(self):
        self.assertEqual(
            apply_operator_rule_reverse(
                "np.linalg.solve(c.T, np.linalg.solve(b.T, a.T).T.T).T"
            ),
            "a / b / c",
        )

    def test_chained_elementwise(self):
        self.assertEqual(apply_operator_rule_reverse("x * y * z"), "x .* y .* z")

    def test_mixed_ops(self):
        self.assertEqual(apply_operator_rule_reverse("a * b @ c"), "a .* b * c")

    def test_operators_inside_indexing_ignored(self):
        self.assertEqual(
            apply_operator_rule_reverse("A(1,:) * B(2,:)"),
            "A(1,:) .* B(2,:)",
        )

    def test_no_operator_passthrough(self):
        self.assertEqual(apply_operator_rule_reverse("a + b"), "a + b")
        self.assertEqual(apply_operator_rule_reverse("sin(x)"), "sin(x)")
        self.assertEqual(apply_operator_rule_reverse("np.linalg.det(a)"), "np.linalg.det(a)")

    def test_round_trip_simple_operators(self):
        self.assertEqual(
            apply_operator_rule_reverse(apply_operator_rule("a * b")), "a * b"
        )
        self.assertEqual(
            apply_operator_rule_reverse(apply_operator_rule("a .* b")), "a .* b"
        )
        self.assertEqual(
            apply_operator_rule_reverse(apply_operator_rule("a ./ b")), "a ./ b"
        )
        self.assertEqual(
            apply_operator_rule_reverse(apply_operator_rule("a .* b * c")),
            "a .* b * c",
        )


if __name__ == "__main__":
    unittest.main()
