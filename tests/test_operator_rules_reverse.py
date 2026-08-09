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

    def test_scalar_floor_division(self):
        self.assertEqual(apply_operator_rule_reverse("7 // 2"), "floor(7 / 2)")
        self.assertEqual(
            apply_operator_rule_reverse("2.5 // 1.2"), "floor(2.5 / 1.2)"
        )
        self.assertEqual(apply_operator_rule_reverse("pi // 2"), "floor(pi / 2)")

    def test_array_floor_division(self):
        self.assertEqual(apply_operator_rule_reverse("a // b"), "floor(a ./ b)")
        self.assertEqual(apply_operator_rule_reverse("x // y"), "floor(x ./ y)")
        self.assertEqual(
            apply_operator_rule_reverse("(x + y) // z"), "floor((x + y) ./ z)"
        )

    def test_floor_division_never_double_slash(self):
        self.assertNotIn("//", apply_operator_rule_reverse("a // b"))
        self.assertNotIn("./ ./", apply_operator_rule_reverse("a // b"))
        self.assertNotIn("//", apply_operator_rule_reverse("7 // 2"))
        self.assertNotIn("./ ./", apply_operator_rule_reverse("7 // 2"))

    def test_no_operator_passthrough(self):
        self.assertEqual(apply_operator_rule_reverse("a + b"), "a + b")
        self.assertEqual(apply_operator_rule_reverse("sin(x)"), "sin(x)")
        self.assertEqual(apply_operator_rule_reverse("np.linalg.det(a)"), "np.linalg.det(a)")

    def test_floordiv_scalar_maps_to_floor_divide(self):
        self.assertEqual(apply_operator_rule_reverse("5 // 2"), "floor(5 / 2)")
        self.assertEqual(apply_operator_rule_reverse("7 // 3"), "floor(7 / 3)")

    def test_floordiv_array_maps_to_floor_elementwise(self):
        self.assertEqual(apply_operator_rule_reverse("x // y"), "floor(x ./ y)")
        self.assertEqual(apply_operator_rule_reverse("x // 2"), "floor(x ./ 2)")
        self.assertEqual(apply_operator_rule_reverse("2 // y"), "floor(2 ./ y)")

    def test_floordiv_respects_additive_precedence(self):
        self.assertEqual(
            apply_operator_rule_reverse("x + a // b"), "x + floor(a ./ b)"
        )
        self.assertEqual(
            apply_operator_rule_reverse("a // b + c"), "floor(a ./ b) + c"
        )

    def test_floordiv_chains_left_to_right(self):
        self.assertEqual(
            apply_operator_rule_reverse("a // b // c"), "floor(floor(a ./ b) ./ c)"
        )

    def test_floordiv_never_malformed_double_operator(self):
        self.assertNotIn("./ ./", apply_operator_rule_reverse("a // b"))
        self.assertNotIn("./ ./", apply_operator_rule_reverse("a / b // c"))

    def test_power_not_misparsed_as_elementwise_multiply(self):
        self.assertEqual(apply_operator_rule_reverse("a ** b"), "a ** b")

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
