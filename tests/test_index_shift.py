import unittest

import pytest

from rulebook.index_shift import FORWARD, REVERSE, UNRESOLVED, shift_index


class TestShiftIndexForward(unittest.TestCase):
    """MATLAB -> Python: a literal index is shifted down by one."""

    def test_zero_becomes_negative_one(self):
        self.assertEqual(shift_index("0", FORWARD), "-1")

    def test_one_becomes_zero(self):
        self.assertEqual(shift_index("1", FORWARD), "0")

    def test_five_becomes_four(self):
        self.assertEqual(shift_index("5", FORWARD), "4")

    def test_negative_literal_is_unresolved(self):
        self.assertEqual(shift_index("-1", FORWARD), UNRESOLVED)

    def test_whitespace_is_tolerated(self):
        self.assertEqual(shift_index(" 5 ", FORWARD), "4")


class TestShiftIndexReverse(unittest.TestCase):
    """Python -> MATLAB: a literal index is shifted up by one."""

    def test_zero_becomes_one(self):
        self.assertEqual(shift_index("0", REVERSE), "1")

    def test_one_becomes_two(self):
        self.assertEqual(shift_index("1", REVERSE), "2")

    def test_five_becomes_six(self):
        self.assertEqual(shift_index("5", REVERSE), "6")

    def test_negative_literal_is_unresolved(self):
        self.assertEqual(shift_index("-2", REVERSE), UNRESOLVED)

    def test_negative_indexed_access_is_unresolved(self):
        self.assertEqual(shift_index("x[-1]", REVERSE), UNRESOLVED)
        self.assertEqual(shift_index("x[-1]", FORWARD), UNRESOLVED)
        self.assertEqual(shift_index("x[1:-1]", REVERSE), UNRESOLVED)


class TestShiftIndexPassThrough(unittest.TestCase):
    """End-keywords and the bare colon are left to the richer indexing
    rules."""

    def test_identifier_unchanged(self):
        self.assertEqual(shift_index("i", FORWARD), "i")
        self.assertEqual(shift_index("i", REVERSE), "i")

    def test_bare_colon_unchanged(self):
        self.assertEqual(shift_index(":", FORWARD), ":")
        self.assertEqual(shift_index(":", REVERSE), ":")

    def test_end_keyword_unchanged(self):
        self.assertEqual(shift_index("end", FORWARD), "end")
        self.assertEqual(shift_index("end", REVERSE), "end")


class TestShiftIndexSlice(unittest.TestCase):
    """A slice/range shifts its start bound by the direction offset while
    its stop bound stays unchanged: MATLAB's inclusive stop maps onto
    Python's exclusive stop with no offset."""

    def test_forward_literal_bounds(self):
        self.assertEqual(shift_index("2:5", FORWARD), "1:5")
        self.assertEqual(shift_index("1:3", FORWARD), "0:3")

    def test_reverse_literal_bounds(self):
        self.assertEqual(shift_index("2:5", REVERSE), "3:5")
        self.assertEqual(shift_index("0:3", REVERSE), "1:3")

    def test_forward_slice_in_indexed_access(self):
        self.assertEqual(shift_index("x(2:5)", FORWARD), "x[1:5]")
        self.assertEqual(shift_index("x(1:3)", FORWARD), "x[0:3]")

    def test_reverse_slice_in_indexed_access(self):
        self.assertEqual(shift_index("x[2:5]", REVERSE), "x(3:5)")
        self.assertEqual(shift_index("x[0:3]", REVERSE), "x(1:3)")

    def test_variable_bounds_unchanged(self):
        self.assertEqual(shift_index("i:j", FORWARD), "i:j")
        self.assertEqual(shift_index("i:j", REVERSE), "i:j")

    def test_two_dimension_with_slice(self):
        self.assertEqual(shift_index("A(1:2, :)", FORWARD), "A[0:2, :]")
        self.assertEqual(shift_index("A[1:2, :]", REVERSE), "A(2:2, :)")

    def test_arithmetic_start_bound_folds(self):
        self.assertEqual(shift_index("x(i+1:j)", FORWARD), "x[i:j]")

    def test_open_ended_slice(self):
        self.assertEqual(shift_index("2:", FORWARD), "1:")
        self.assertEqual(shift_index(":5", FORWARD), ":5")

    def test_end_as_stop_unchanged(self):
        self.assertEqual(shift_index("2:end", FORWARD), "1:end")


class TestShiftIndexArithmetic(unittest.TestCase):
    """An arithmetic index expression is shifted by folding the offset
    into its constant term."""

    def test_forward_plus_one_folds_to_zero(self):
        self.assertEqual(shift_index("i + 1", FORWARD), "i")

    def test_forward_minus_one_becomes_minus_two(self):
        self.assertEqual(shift_index("i - 1", FORWARD), "i - 2")

    def test_forward_plus_five_becomes_plus_four(self):
        self.assertEqual(shift_index("i + 5", FORWARD), "i + 4")

    def test_forward_minus_five_becomes_minus_six(self):
        self.assertEqual(shift_index("i - 5", FORWARD), "i - 6")

    def test_reverse_plus_one_becomes_plus_two(self):
        self.assertEqual(shift_index("i + 1", REVERSE), "i + 2")

    def test_reverse_minus_one_folds_to_zero(self):
        self.assertEqual(shift_index("i - 1", REVERSE), "i")

    def test_reverse_plus_five_becomes_plus_six(self):
        self.assertEqual(shift_index("i + 5", REVERSE), "i + 6")

    def test_reverse_minus_five_becomes_minus_four(self):
        self.assertEqual(shift_index("i - 5", REVERSE), "i - 4")

    def test_expression_without_constant_is_wrapped(self):
        self.assertEqual(shift_index("2 * k", FORWARD), "(2 * k) - 1")
        self.assertEqual(shift_index("2 * k", REVERSE), "(2 * k) + 1")

    def test_constant_on_the_left_is_wrapped(self):
        self.assertEqual(shift_index("1 + i", FORWARD), "(1 + i) - 1")

    def test_end_minus_constant_shifts(self):
        self.assertEqual(shift_index("end - 1", REVERSE), "end")


class TestShiftIndexVariables(unittest.TestCase):
    """A single variable as the index needs no numeric shift."""

    def test_single_variable_forward(self):
        self.assertEqual(shift_index("i", FORWARD), "i")

    def test_single_variable_reverse(self):
        self.assertEqual(shift_index("i", REVERSE), "i")

    def test_longer_variable_name_unchanged(self):
        self.assertEqual(shift_index("row_index", FORWARD), "row_index")
        self.assertEqual(shift_index("row_index", REVERSE), "row_index")


class TestShiftIndexedAccess(unittest.TestCase):
    """Indexed access converts the target syntax and shifts the inner
    index: a literal shifts numerically, a variable passes through."""

    def test_variable_index_forward_paren_to_bracket(self):
        self.assertEqual(shift_index("x(i)", FORWARD), "x[i]")

    def test_variable_index_reverse_bracket_to_paren(self):
        self.assertEqual(shift_index("x[i]", REVERSE), "x(i)")

    def test_variable_index_forward_already_bracketed(self):
        self.assertEqual(shift_index("x[i]", FORWARD), "x[i]")

    def test_literal_index_forward_paren_to_bracket(self):
        self.assertEqual(shift_index("x(5)", FORWARD), "x[4]")

    def test_literal_index_reverse_bracket_to_paren(self):
        self.assertEqual(shift_index("x[5]", REVERSE), "x(6)")

    def test_two_dimension_variable_index(self):
        self.assertEqual(shift_index("x(i, j)", FORWARD), "x[i, j]")
        self.assertEqual(shift_index("x[i, j]", REVERSE), "x(i, j)")

    def test_two_dimension_literal_index(self):
        self.assertEqual(shift_index("x(1, 2)", FORWARD), "x[0, 1]")
        self.assertEqual(shift_index("x[1, 2]", REVERSE), "x(2, 3)")

    def test_arithmetic_index_inside_shifts(self):
        self.assertEqual(shift_index("x(i + 1)", FORWARD), "x[i]")
        self.assertEqual(shift_index("x[i + 1]", REVERSE), "x(i + 2)")
        self.assertEqual(shift_index("x(i - 1)", FORWARD), "x[i - 2]")
        self.assertEqual(shift_index("x[i - 1]", REVERSE), "x(i)")


class TestShiftIndexValidation(unittest.TestCase):
    def test_invalid_direction_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_index("1", "sideways")

    def test_direction_is_required(self):
        with self.assertRaises(TypeError):
            shift_index("1")


class TestShiftIndexDivision(unittest.TestCase):
    """A division-based index is emitted as floor division so a Python
    slice bound stays integer-valued, matching MATLAB's integer colon
    endpoints."""

    def test_forward_length_half_plus_one(self):
        self.assertEqual(shift_index("length(x)/2+1", FORWARD), "length(x)//2")

    def test_forward_n_half_minus_one(self):
        self.assertEqual(shift_index("n/2-1", FORWARD), "n//2 - 2")

    def test_forward_length_third(self):
        self.assertEqual(shift_index("length(x)/3", FORWARD), "(length(x)//3) - 1")

    def test_forward_bare_half(self):
        self.assertEqual(shift_index("n/2", FORWARD), "(n//2) - 1")

    def test_forward_range_stop_division(self):
        self.assertEqual(shift_index("2:n/2", FORWARD), "1:n//2")

    def test_reverse_n_half_minus_one(self):
        self.assertEqual(shift_index("n//2-1", REVERSE), "n//2")

    def test_reverse_length_half(self):
        self.assertEqual(shift_index("length(x)//2", REVERSE), "(length(x)//2) + 1")

    def test_reverse_range_stop_division(self):
        self.assertEqual(shift_index("2:n//2", REVERSE), "3:n//2")

    def test_fft_basic_slice_bound_is_integer_safe(self):
        from rulebook import apply_indexing_rule

        self.assertEqual(
            apply_indexing_rule("P2(1:length(P2)/2+1)"), "P2[0:len(P2)//2+1]"
        )


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("n/2:5", "(n//2) - 1:5"),
        ("n/2:n", "(n//2) - 1:n"),
        ("1:n/2+1", "0:n//2+1"),
        ("1:(n+1)/2", "0:(n+1)//2"),
        ("2:k/3", "1:k//3"),
        ("x(2:n/3)", "x[1:n//3]"),
        ("A(1:n/2, :)", "A[0:n//2, :]"),
    ],
)
def test_forward_division_range_patterns(expr, expected):
    """A division-derived range shifts its start and floor-divides its stop."""
    assert shift_index(expr, FORWARD) == expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("n//2:5", "(n//2) + 1:5"),
        ("1:n//2+1", "2:n//2+1"),
        ("2:k//3", "3:k//3"),
        ("x[2:n//3]", "x(3:n//3)"),
        ("A[1:n//2, :]", "A(2:n//2, :)"),
    ],
)
def test_reverse_division_range_patterns(expr, expected):
    """The mirror image of the forward division-range cases."""
    assert shift_index(expr, REVERSE) == expected


@pytest.mark.parametrize(
    "expr,direction,expected",
    [
        ("x(n/2)", FORWARD, "x[(n//2) - 1]"),
        ("x[n//2]", REVERSE, "x((n//2) + 1)"),
        ("(n+1)/2", FORWARD, "((n+1)//2) - 1"),
        ("(n+1)//2", REVERSE, "((n+1)//2) + 1"),
        ("end/2", FORWARD, "(end//2) - 1"),
        ("end/2", REVERSE, "(end//2) + 1"),
        ("n/2/3", FORWARD, "(n//2//3) - 1"),
    ],
)
def test_division_index_forms(expr, direction, expected):
    """Division indices outside a range still floor-divide and shift."""
    assert shift_index(expr, direction) == expected


if __name__ == "__main__":
    unittest.main()
