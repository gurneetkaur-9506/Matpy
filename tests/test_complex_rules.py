import unittest

from rulebook import apply_complex_rule


class TestApplyComplexRule(unittest.TestCase):
    def test_imaginary_unit(self):
        self.assertEqual(apply_complex_rule("1i"), "1j")
        self.assertEqual(apply_complex_rule("3i"), "3j")
        self.assertEqual(apply_complex_rule("0.5i"), "0.5j")

    def test_already_python(self):
        self.assertEqual(apply_complex_rule("2j"), "2j")

    def test_scientific_notation(self):
        self.assertEqual(apply_complex_rule("1.5e3i"), "1.5e3j")

    def test_in_expressions(self):
        self.assertEqual(apply_complex_rule("1i * (n - 1) * phase"), "1j * (n - 1) * phase")
        self.assertEqual(apply_complex_rule("exp(1i * theta)"), "exp(1j * theta)")

    def test_bare_i_untouched(self):
        self.assertEqual(apply_complex_rule("i"), "i")
        self.assertEqual(apply_complex_rule("for i = 1:N"), "for i = 1:N")


if __name__ == "__main__":
    unittest.main()
