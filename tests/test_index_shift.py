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


class TestShiftIndexValidation(unittest.TestCase):
    def test_invalid_direction_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_index("1", "sideways")

    def test_direction_is_required(self):
        with self.assertRaises(TypeError):
            shift_index("1")


if __name__ == "__main__":
    unittest.main()
