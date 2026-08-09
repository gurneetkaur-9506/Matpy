import unittest

from rulebook import apply_operator_rule
from rulebook.operator_rules import _find_last_operator


class TestApplyOperatorRule(unittest.TestCase):
    def test_matrix_multiply(self):
        self.assertEqual(apply_operator_rule("a * b"), "a @ b")

    def test_elementwise_multiply(self):
        self.assertEqual(apply_operator_rule("a .* b"), "a * b")

    def test_elementwise_divide(self):
        self.assertEqual(apply_operator_rule("a ./ b"), "a / b")

    def test_elementwise_power(self):
        self.assertEqual(apply_operator_rule("X.^2"), "X ** 2")

    def test_elementwise_power_spaced(self):
        self.assertEqual(apply_operator_rule("X .^ 2"), "X ** 2")

    def test_elementwise_power_binds_tighter_than_multiply(self):
        self.assertEqual(apply_operator_rule("a .* b .^ c"), "a * b ** c")

    def test_elementwise_power_found_by_operator_split(self):
        self.assertEqual(_find_last_operator("X.^2"), (1, ".^"))
        self.assertEqual(_find_last_operator("a * b .^ c"), (2, "*"))

    def test_matrix_right_divide(self):
        self.assertEqual(
            apply_operator_rule("a / b"),
            "np.linalg.solve(b.T, a.T).T",
        )

    def test_elementwise_ops_win_over_single_char(self):
        self.assertEqual(apply_operator_rule("a .* b"), "a * b")
        self.assertEqual(apply_operator_rule("a ./ b"), "a / b")

    def test_chained_elementwise(self):
        self.assertEqual(apply_operator_rule("x .* y .* z"), "x * y * z")

    def test_mixed_ops(self):
        self.assertEqual(apply_operator_rule("a .* b * c"), "a * b @ c")

    def test_operators_inside_indexing_ignored(self):
        self.assertEqual(apply_operator_rule("A(1,:) .* B(2,:)"), "A(1,:) * B(2,:)")

    def test_decimal_number_not_elementwise(self):
        self.assertEqual(apply_operator_rule("1.5 * x"), "1.5 * x")
        self.assertEqual(apply_operator_rule("a .* b"), "a * b")

    def test_scalar_multiply_uses_plain_star(self):
        self.assertEqual(apply_operator_rule("2 * xAxisStep"), "2 * xAxisStep")
        self.assertEqual(apply_operator_rule("pi * x"), "pi * x")

    def test_lowest_precedence_operator_split_for_scalar_check(self):
        self.assertEqual(
            _find_last_operator("sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t)"),
            (15, "+"),
        )
        self.assertEqual(
            _find_last_operator("0.5*sin(2*pi*f2*t)"), (3, "*")
        )
        self.assertEqual(
            apply_operator_rule("0.5*sin(2*pi*f2*t)"),
            "0.5 * sin(2*pi*f2*t)",
        )

    def test_scalar_indexed_multiply_uses_plain_star(self):
        self.assertEqual(
            apply_operator_rule("2 * nonZeroXAxis(1) * xAxisStep"),
            "2 * nonZeroXAxis(1) * xAxisStep",
        )

    def test_matrix_multiply_still_uses_at(self):
        self.assertEqual(apply_operator_rule("a * b"), "a @ b")
        self.assertEqual(apply_operator_rule("A * B * C"), "A @ B @ C")

    def test_no_operator_passthrough(self):
        self.assertEqual(apply_operator_rule("a + b"), "a + b")
        self.assertEqual(apply_operator_rule("sin(x)"), "sin(x)")

    def test_matrix_divide_chained(self):
        self.assertEqual(
            apply_operator_rule("a / b / c"),
            "np.linalg.solve(c.T, np.linalg.solve(b.T, a.T).T.T).T",
        )

    def test_scalar_power_maps_to_double_star(self):
        self.assertEqual(apply_operator_rule("2^3"), "2 ** 3")

    def test_elementwise_power_maps_to_double_star(self):
        self.assertEqual(apply_operator_rule("x.^2"), "x ** 2")

    def test_scalar_power_in_expression_maps_to_double_star(self):
        self.assertEqual(apply_operator_rule("2^3 * xAxisStep"), "2 ** 3 * xAxisStep")


if __name__ == "__main__":
    unittest.main()
