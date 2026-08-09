import unittest

from rulebook.index_shift import FORWARD, REVERSE, shift_index


class TestShiftIndexForward(unittest.TestCase):
    """MATLAB -> Python: a literal index is shifted down by one."""

    def test_zero_becomes_negative_one(self):
        self.assertEqual(shift_index("0", FORWARD), "-1")

    def test_one_becomes_zero(self):
        self.assertEqual(shift_index("1", FORWARD), "0")

    def test_five_becomes_four(self):
        self.assertEqual(shift_index("5", FORWARD), "4")

    def test_negative_literal_shifts_down(self):
        self.assertEqual(shift_index("-1", FORWARD), "-2")

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

    def test_negative_literal_shifts_up(self):
        self.assertEqual(shift_index("-2", REVERSE), "-1")


class TestShiftIndexPassThrough(unittest.TestCase):
    """Non-literal index expressions are returned unchanged."""

    def test_identifier_unchanged(self):
        self.assertEqual(shift_index("i", FORWARD), "i")
        self.assertEqual(shift_index("i", REVERSE), "i")

    def test_computed_expression_unchanged(self):
        self.assertEqual(shift_index("i + 1", FORWARD), "i + 1")
        self.assertEqual(shift_index("2 * k", REVERSE), "2 * k")

    def test_end_keyword_unchanged(self):
        self.assertEqual(shift_index("end", FORWARD), "end")
        self.assertEqual(shift_index("end - 1", REVERSE), "end - 1")


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

    def test_computed_index_inside_unchanged(self):
        self.assertEqual(shift_index("x(i + 1)", FORWARD), "x[i + 1]")


class TestShiftIndexValidation(unittest.TestCase):
    def test_invalid_direction_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_index("1", "sideways")

    def test_direction_is_required(self):
        with self.assertRaises(TypeError):
            shift_index("1")


if __name__ == "__main__":
    unittest.main()
