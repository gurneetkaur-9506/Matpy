import unittest

from rulebook import (
    convert_fprintf,
    format_spec_count,
    matlab_string_literal_to_python,
)
from rulebook.translator import _translate_expr


class TestMatlabStringToPython(unittest.TestCase):
    def test_plain_literal(self):
        self.assertEqual(
            matlab_string_literal_to_python("'x = %.2f\\n'"), '"x = %.2f\\n"'
        )

    def test_escaped_single_quotes(self):
        self.assertEqual(
            matlab_string_literal_to_python("'it''s %.2f%%\\n'"), '"it\'s %.2f%%\\n"'
        )

    def test_non_literal_passthrough(self):
        self.assertEqual(matlab_string_literal_to_python("fmt"), "fmt")


class TestFormatSpecCount(unittest.TestCase):
    def test_specs_counted(self):
        self.assertEqual(format_spec_count("a=%.2f b=%d c=%s"), 3)

    def test_escaped_percent_ignored(self):
        self.assertEqual(format_spec_count("100%% %.1f"), 1)

    def test_no_specs(self):
        self.assertEqual(format_spec_count("hello\\n"), 0)


class TestConvertFprintf(unittest.TestCase):
    def _convert(self, expr):
        return convert_fprintf(expr, lambda a: a)

    def test_single_argument(self):
        self.assertEqual(
            self._convert("fprintf('x = %.2f\\n', x)"),
            'print("x = %.2f\\n" % (x))',
        )

    def test_multiple_arguments(self):
        self.assertEqual(
            self._convert("fprintf('a=%.2f, b=%d\\n', a, b)"),
            'print("a=%.2f, b=%d\\n" % (a, b))',
        )

    def test_multiple_specifiers_combined(self):
        self.assertEqual(
            self._convert("fprintf('%.2f %d %s %.3f\\n', a, b, c, d)"),
            'print("%.2f %d %s %.3f\\n" % (a, b, c, d))',
        )

    def test_no_format_args(self):
        self.assertEqual(self._convert("fprintf('hello\\n')"), 'print("hello\\n")')

    def test_escaped_quotes_and_percent(self):
        self.assertEqual(
            self._convert("fprintf('it''s 100%% %.2f\\n', x)"),
            'print("it\'s 100%% %.2f\\n" % (x))',
        )

    def test_variable_format(self):
        self.assertEqual(self._convert("fprintf(fmt, a)"), "print(fmt % (a))")

    def test_not_fprintf_returns_none(self):
        self.assertIsNone(self._convert("disp(x)"))
        self.assertIsNone(self._convert("fprintf"))
        self.assertIsNone(self._convert("x + 1"))


class TestFprintfIntegration(unittest.TestCase):
    def test_translate_expr_multiple_specifiers(self):
        self.assertEqual(
            _translate_expr("fprintf('a=%.2f, b=%d, c=%s\\n', a, b, c)"),
            'print("a=%.2f, b=%d, c=%s\\n" % (a, b, c))',
        )

    def test_translate_expr_with_expression_argument(self):
        self.assertEqual(
            _translate_expr("fprintf('peak = %.2f dB\\n', abs(x))"),
            'print("peak = %.2f dB\\n" % (np.abs(x)))',
        )

    def test_translate_expr_no_args(self):
        self.assertEqual(
            _translate_expr("fprintf('all done\\n')"), 'print("all done\\n")'
        )


if __name__ == "__main__":
    unittest.main()
