import ast
import unittest

from reader import PYTHON_TO_MATLAB, load_structure
from reader.structure import Function, Statement, Structure
from rulebook import UNRESOLVED, translate_with_rulebook_reverse
from tests.paths import sample_python


class TestTranslateWithRulebookReverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = load_structure(
            sample_python("indexing_ops_py.py"), PYTHON_TO_MATLAB
        )
        cls.result = translate_with_rulebook_reverse(cls.structure)

    def _translations(self):
        return [s["matlab"] for s in self.result["statements"]]

    def test_fully_resolved(self):
        translations = self._translations()
        self.assertNotIn(UNRESOLVED, translations)

    def test_expected_lines(self):
        translations = [t for t in self._translations() if t]
        expected = [
            "A = [1 2 3; 4 5 6]",
            "B = [7 8; 9 10; 11 12]",
            "disp('A(1,1) (first row, first col):')",
            "disp(A(1, 1))",
            "disp('A(2,3) (second row, third col):')",
            "disp(A(2, 3))",
            "disp('Matrix multiplication A * B:')",
            "disp(A * B)",
            "disp('Element-wise multiplication A .* A:')",
            "disp(A .* A)",
            "disp('First row of A via 1-based indexing:')",
            "disp(A(1, :))",
        ]
        self.assertEqual(translations, expected)

    def test_import_becomes_noop(self):
        import_stmt = self.result["statements"][0]
        self.assertEqual(import_stmt["kind"], "Import")
        self.assertEqual(import_stmt["matlab"], "")

    def test_comments_present(self):
        for s in self.result["statements"]:
            self.assertIn("comment", s)
            self.assertTrue(s["comment"].startswith("% Python:"))

    def test_loop_reported_unresolved(self):
        structure = Structure(statements=[Statement("For", "for i in range(3):")])
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], UNRESOLVED)

    def test_unknown_call_reported_unresolved(self):
        structure = Structure(statements=[Statement("Expr", "np.mean(x)")])
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], UNRESOLVED)

    def test_builtin_reverse_in_expr(self):
        structure = Structure(
            statements=[
                Statement("Expr", "print(np.abs(np.fft.fft(x)))"),
                Statement("Assign", "y = np.zeros((2, 5))"),
            ]
        )
        result = translate_with_rulebook_reverse(structure)
        matlab = [s["matlab"] for s in result["statements"]]
        self.assertEqual(matlab, ["disp(abs(fft(x)))", "y = zeros(2, 5)"])

    def test_functions_handled(self):
        structure = Structure(functions=[Function("f", [Statement("Expr", "print(x)")])])
        result = translate_with_rulebook_reverse(structure)
        statements = result["functions"][0]["statements"]
        self.assertEqual(statements[0]["matlab"], "disp(x)")

    def _translate_expr_stmt(self, source):
        structure = Structure(statements=[Statement("Expr", source)])
        result = translate_with_rulebook_reverse(structure)
        return result["statements"][0]["matlab"]

    def test_percent_format_print_maps_to_fprintf(self):
        self.assertEqual(
            self._translate_expr_stmt('print("x = %d" % 5)'),
            "fprintf('x = %d', 5)",
        )

    def test_percent_format_print_with_tuple_maps_to_fprintf(self):
        self.assertEqual(
            self._translate_expr_stmt('print("a=%.2f and %s" % (x, y))'),
            "fprintf('a=%.2f and %s', x, y)",
        )

    def test_percent_format_print_single_quote_escaped(self):
        self.assertEqual(
            self._translate_expr_stmt('print("it\'s %d" % x)'),
            "fprintf('it''s %d', x)",
        )

    def test_fstring_print_maps_to_fprintf(self):
        self.assertEqual(
            self._translate_expr_stmt('print(f"x = {y}")'),
            "fprintf('x = %s', y)",
        )

    def test_fstring_print_with_expression_maps_to_fprintf(self):
        self.assertEqual(
            self._translate_expr_stmt('print(f"sum = {x + y}")'),
            "fprintf('sum = %s', x + y)",
        )

    def test_fstring_print_with_format_spec_maps_to_fprintf(self):
        self.assertEqual(
            self._translate_expr_stmt('print(f"val = {y:.2f}")'),
            "fprintf('val = %.2f', y)",
        )

    def test_plain_print_stays_disp(self):
        self.assertEqual(
            self._translate_expr_stmt('print("hello")'),
            'disp("hello")',
        )

    def test_print_with_percent_in_string_but_no_format_op_stays_disp(self):
        self.assertEqual(
            self._translate_expr_stmt('print("50% done")'),
            'disp("50% done")',
        )


    def test_plt_subplot_maps_to_matlab_subplot(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.subplot(2, 2, 1)"),
            "subplot(2, 2, 1);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.subplot(1, 2, 2)"),
            "subplot(1, 2, 2);",
        )


    def test_plt_plot_maps_to_matlab_plot(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(x, y)"),
            "plot(x, y);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(t, x)"),
            "plot(t, x);",
        )


    def test_plt_plot_with_linewidth_kwarg(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(x, y, linewidth=2)"),
            "plot(x, y, 'LineWidth', 2);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(t, x, linewidth=1.5)"),
            "plot(t, x, 'LineWidth', 1.5);",
        )


    def test_plt_plot_with_color(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(x, y, color='r')"),
            "plot(x, y, 'Color', 'r');",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(x, y, 'r')"),
            "plot(x, y, 'r');",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.plot(x, y, 'r--')"),
            "plot(x, y, 'r--');",
        )


    def test_plt_title_maps_to_matlab_title(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.title('t')"),
            "title('t');",
        )
        self.assertEqual(
            self._translate_expr_stmt('plt.title("Frequency Spectrum")'),
            'title("Frequency Spectrum");',
        )


    def test_plt_xlabel_maps_to_matlab_xlabel(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.xlabel('Time (s)')"),
            "xlabel('Time (s)');",
        )
        self.assertEqual(
            self._translate_expr_stmt('plt.xlabel("Frequency (Hz)")'),
            'xlabel("Frequency (Hz)");',
        )

    def test_plt_ylabel_maps_to_matlab_ylabel(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.ylabel('Amplitude')"),
            "ylabel('Amplitude');",
        )
        self.assertEqual(
            self._translate_expr_stmt('plt.ylabel("Power (dB)")'),
            'ylabel("Power (dB)");',
        )


if __name__ == "__main__":
    unittest.main()
