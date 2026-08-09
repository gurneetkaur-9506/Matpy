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


    def test_ax_set_zlabel_maps_to_matlab_zlabel(self):
        self.assertEqual(
            self._translate_expr_stmt("ax.set_zlabel('z')"),
            "zlabel('z');",
        )
        self.assertEqual(
            self._translate_expr_stmt('ax.set_zlabel("Range (m)")'),
            'zlabel("Range (m)");',
        )


    def test_plt_grid_true_maps_to_matlab_grid_on(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.grid(True)"),
            "grid on;",
        )

    def test_plt_grid_false_maps_to_matlab_grid_off(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.grid(False)"),
            "grid off;",
        )


    def test_plt_legend_maps_to_matlab_legend(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.legend(['first', 'second'])"),
            "legend(['first', 'second']);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.legend()"),
            "legend();",
        )


    def test_plt_xlim_maps_to_matlab_xlim_vector(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.xlim([a, b])"),
            "xlim([a b]);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.xlim([0, 10])"),
            "xlim([0 10]);",
        )

    def test_plt_ylim_maps_to_matlab_ylim_vector(self):
        self.assertEqual(
            self._translate_expr_stmt("plt.ylim([a, b])"),
            "ylim([a b]);",
        )
        self.assertEqual(
            self._translate_expr_stmt("plt.ylim([-1.5, 1.5])"),
            "ylim([-1.5 1.5]);",
        )


    def test_plt_tight_layout_is_noop_comment(self):
        result = self._translate_expr_stmt("plt.tight_layout()")
        self.assertNotEqual(result, UNRESOLVED)
        self.assertTrue(result.startswith("%"))

    def test_plt_show_is_noop_comment(self):
        result = self._translate_expr_stmt("plt.show()")
        self.assertNotEqual(result, UNRESOLVED)
        self.assertTrue(result.startswith("%"))
    def test_python_len_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = len(x)"),
            Statement("Assign", "z = samplesDelay + len(txPulse) + 100"),
            Statement("Assign", "w = len(x) ./ 2"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn("len(", s["matlab"])

    def test_python_shape_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = x.shape"),
            Statement("Assign", "z = x.shape[0]"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn(".shape", s["matlab"])

    def test_python_sum_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = x.sum(axis=0)"),
            Statement("Assign", "z = x.sum()"),
            Statement("Assign", "w = x.sum(axis=0) + 1"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn(".sum", s["matlab"])

    def test_python_newaxis_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = x[:, np.newaxis]"),
            Statement("Expr", "print(x[:, np.newaxis])"),
            Statement("Assign", "z = np.newaxis"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn("newaxis", s["matlab"])

    def test_python_only_construct_in_target_flagged(self):
        structure = Structure(statements=[Statement("Assign", "x[:, np.newaxis] = 5")])
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], UNRESOLVED)

    def test_string_contents_do_not_trigger_guard(self):
        structure = Structure(
            statements=[Statement("Expr", "print('x.shape and len(x) are just text')")]
        )
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], "disp('x.shape and len(x) are just text')")

    def test_multi_arg_print_flagged_not_emitted_as_disp(self):
        cases = [
            Statement("Expr", "print('first 3 values:', A(1:3))"),
            Statement("Expr", "print(a, b)"),
            Statement("Expr", "print()"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn("disp(a, b)", s["matlab"])

    def test_single_arg_print_still_emits_disp(self):
        structure = Structure(statements=[Statement("Expr", "print(A[0, 3])")])
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], "disp(A(1, 4))")

    def test_python_modulo_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = a % b"),
            Statement("Assign", "z = (x % 2) + 1"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn(" % ", s["matlab"])

    def test_percent_format_print_maps_to_fprintf(self):
        structure = Structure(
            statements=[Statement("Expr", "print('Estimated Range: %.2f m' % r)")]
        )
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(
            result["statements"][0]["matlab"], "fprintf('Estimated Range: %.2f m', r)"
        )

    def test_whitelisted_np_names_still_reverse(self):
        structure = Structure(
            statements=[
                Statement("Assign", "y = np.linspace(0, 100, 91)"),
                Statement("Assign", "y = np.zeros((2, 5))"),
            ]
        )
        result = translate_with_rulebook_reverse(structure)
        matlab = [s["matlab"] for s in result["statements"]]
        self.assertEqual(matlab, ["y = linspace(0, 100, 91)", "y = zeros(2, 5)"])

    def test_unwhitelisted_np_name_flags_whole_expr(self):
        structure = Structure(
            statements=[Statement("Assign", "y = np.linspace(0, np.pi, 91)")]
        )
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["statements"][0]["matlab"], UNRESOLVED)

    def test_python_floordiv_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "half = len(magnitude) // 2 + 1"),
            Statement("Assign", "y = a // b"),
            Statement("Assign", "z = (a // b) + 1"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn("./ ./", s["matlab"])

    def test_python_power_flagged_not_passed_through(self):
        cases = [
            Statement("Assign", "y = x ** 2"),
            Statement("Assign", "z = x ** (1 / 2)"),
        ]
        result = translate_with_rulebook_reverse(Structure(statements=cases))
        for s in result["statements"]:
            self.assertEqual(s["matlab"], UNRESOLVED)
            self.assertNotIn(".* .*", s["matlab"])


if __name__ == "__main__":
    unittest.main()
