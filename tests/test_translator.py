import ast
import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure, load_matlab_file
from reader.structure import Loop, Statement, Structure
from rulebook import UNRESOLVED, translate_with_rulebook
from rulebook.translator import _translate_expr, _translate_loop
from tests.paths import sample_matlab, sample_matlab_real


class TestTranslateWithRulebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("indexing_ops.m")).encode("utf-8")
        )
        cls.structure = build_structure(tree)

    def _translations(self):
        result = translate_with_rulebook(self.structure)
        return [s["python"] for s in result["statements"]]

    def test_fully_resolved(self):
        translations = self._translations()
        self.assertNotIn(UNRESOLVED, translations)

    def test_expected_lines(self):
        translations = [t for t in self._translations() if t]
        expected = [
            "A = np.array([[1, 2, 3], [4, 5, 6]])",
            "B = np.array([[7, 8], [9, 10], [11, 12]])",
            "print('A(1,1) (first row, first col):')",
            "print(A[0, 0])",
            "print('A(2,3) (second row, third col):')",
            "print(A[1, 2])",
            "print('Matrix multiplication A * B:')",
            "print(A @ B)",
            "print('Element-wise multiplication A .* A:')",
            "print(A * A)",
            "print('First row of A via 1-based indexing:')",
            "print(A[0, :])",
        ]
        self.assertEqual(translations, expected)


class TestPlotBuiltins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("fft_basic.m")).encode("utf-8")
        )
        cls.result = translate_with_rulebook(build_structure(tree))

    def test_plot_calls_resolved(self):
        translations = [s["python"] for s in self.result["statements"]]
        self.assertIn("plt.plot(f, P1)", translations)
        self.assertIn("plt.title('Single-Sided Amplitude Spectrum')", translations)
        self.assertIn("plt.xlabel('Frequency (Hz)')", translations)
        self.assertIn("plt.ylabel('Magnitude')", translations)

    def test_no_unresolved(self):
        translations = [s["python"] for s in self.result["statements"]]
        self.assertNotIn(UNRESOLVED, translations)

    def test_scalar_multiply_in_signal_mix_stays_plain_star(self):
        translations = [s["python"] for s in self.result["statements"]]
        self.assertIn(
            "x = np.sin(2*pi*f1*t) + 0.5 * np.sin(2*pi*f2*t)", translations
        )


class TestLoopHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("beamform_basic.m")).encode("utf-8")
        )
        cls.result = translate_with_rulebook(build_structure(tree))

    def test_loop_body_translated(self):
        statements = self.result["functions"][0]["statements"]
        self.assertNotIn(UNRESOLVED, [s["python"] for s in statements])
        loop = [s for s in statements if s["kind"] == "loop"]
        self.assertEqual(len(loop), 1)
        self.assertEqual(loop[0]["python"], "for n in range(N):")
        self.assertEqual(
            [b["python"] for b in loop[0]["body"]],
            ["af = af + np.exp(1i * (n - 1) * phase)"],
        )

    def test_loop_with_unresolved_body_marked_unresolved(self):
        loop = Loop(
            type="for",
            header="n = 1:N",
            statements=[Statement("command", "fprintf(1, '%d', n)")],
        )
        self.assertEqual(_translate_loop(loop)["python"], UNRESOLVED)


class TestFindAndInterp1Rules(unittest.TestCase):
    def _translate(self, text):
        structure = Structure(statements=[Statement("assignment", text)])
        return translate_with_rulebook(structure)["statements"][0]["python"]

    def test_find_condition_maps_to_np_where(self):
        self.assertEqual(
            self._translate("nonZeroIndex = find(dataInRow ~= 0)"),
            "nonZeroIndex = np.where(dataInRow != 0)[0]",
        )

    def test_find_with_multiple_args_stays_unresolved(self):
        self.assertEqual(self._translate("i = find(a, b)"), UNRESOLVED)


class TestFindReductionComposition(unittest.TestCase):
    """find(cond) is a first-class expression mapping to
    np.where(cond)[0], so any single-output reduction wrapping it composes
    naturally: sum/mean/max/min/length/numel all reduce the index array."""

    def _translate(self, text):
        structure = Structure(statements=[Statement("assignment", text)])
        return translate_with_rulebook(structure)["statements"][0]["python"]

    def test_sum_of_find(self):
        self.assertEqual(
            self._translate("n = sum(find(x > 0))"),
            "n = np.sum(np.where(x > 0)[0])",
        )

    def test_mean_of_find(self):
        self.assertEqual(
            self._translate("m = mean(find(x > 0))"),
            "m = np.mean(np.where(x > 0)[0])",
        )

    def test_max_of_find(self):
        self.assertEqual(
            self._translate("mx = max(find(x > 0))"),
            "mx = np.max(np.where(x > 0)[0])",
        )

    def test_min_of_find(self):
        self.assertEqual(
            self._translate("mn = min(find(x > 0))"),
            "mn = np.min(np.where(x > 0)[0])",
        )

    def test_length_of_find(self):
        self.assertEqual(
            self._translate("nz = length(find(z ~= 0))"),
            "nz = len(np.where(z != 0)[0])",
        )

    def test_numel_of_find(self):
        self.assertEqual(
            self._translate("nt = numel(find(z ~= 0))"),
            "nt = len(np.where(z != 0)[0])",
        )

    def test_reduction_of_find_with_nested_condition(self):
        self.assertEqual(
            self._translate("m = mean(find(abs(x) > 1))"),
            "m = np.mean(np.where(abs(x) > 1)[0])",
        )

    def test_reduction_of_find_composes_with_operators(self):
        self.assertEqual(
            self._translate("s = sum(find(x > 0) + 1)"),
            "s = np.sum(np.where(x > 0)[0] + 1)",
        )

    def test_reduction_axis_forms_still_translate(self):
        self.assertEqual(self._translate("s = sum(x, 1)"), "s = np.sum(x, axis=0)")
        self.assertEqual(
            self._translate("m = max(x, [], 2)"), "m = np.max(x, axis=1)"
        )

    def test_interp1_reorders_args_to_np_interp(self):
        self.assertEqual(
            self._translate(
                "atlasRow(rowNo,:) = interp1(nonZeroXAxis,tempE,interpPoints)"
            ),
            "atlasRow[rowNo, :] = np.interp(interpPoints, nonZeroXAxis, tempE)",
        )

    def test_interp1_wrong_arity_stays_unresolved(self):
        self.assertEqual(self._translate("y = interp1(x, v)"), UNRESOLVED)


class TestProblemLineCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("fft_basic.m")).encode("utf-8")
        )
        cls.result = translate_with_rulebook(build_structure(tree))

    def test_fully_resolved_file_has_no_problems(self):
        from translator import code_for_result

        problems = []
        code_for_result(self.result, problems=problems)
        self.assertEqual(problems, [])

    def test_unresolved_lines_recorded(self):
        from translator import code_for_result

        structure = Structure(statements=[Statement("assignment", "x = find(a, b)")])
        problems = []
        code = code_for_result(translate_with_rulebook(structure), problems=problems)
        self.assertIn("# UNRESOLVED:", code)
        unresolved_index = next(
            i for i, line in enumerate(code.splitlines()) if "UNRESOLVED" in line
        )
        self.assertIn(unresolved_index, problems)


class TestAtlasDisplayResolves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab_real("atlasDisplay.m")).encode("utf-8")
        )
        cls.func = translate_with_rulebook(build_structure(tree))["functions"][0]

    def test_parameters_preserved_in_signature(self):
        self.assertEqual(self.func["parameters"], ["array_factor", "xAxis", "yAxis"])

    def test_emitted_def_line_keeps_parameter_names(self):
        from translator import code_for_result

        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab_real("atlasDisplay.m")).encode("utf-8")
        )
        result = translate_with_rulebook(build_structure(tree))
        code = code_for_result(result)
        self.assertIn("def atlasDisplay(array_factor, xAxis, yAxis):", code)
        self.assertNotIn("*args, **kwargs", code)

    def test_zero_unresolved_including_loop_body(self):
        all_py = [s["python"] for s in self.func["statements"]]
        for s in self.func["statements"]:
            if s["kind"] == "loop":
                all_py.extend(b["python"] for b in s["body"])
        self.assertNotIn(UNRESOLVED, all_py)

    def test_colon_range_step_expression_translated(self):
        stmt = next(
            s for s in self.func["statements"] if s["source"].startswith("xAxisStep =")
        )
        self.assertEqual(
            stmt["python"],
            "xAxisStep = np.linspace(-0.5, 0.5, len(xAxis))",
        )
        compile(stmt["python"], "<test>", "exec")

    def test_implicit_growth_preallocated_before_loop(self):
        statements = self.func["statements"]
        prealloc = next(
            s for s in statements
            if s["kind"] == "preallocate" and s["python"].startswith("atlasRow =")
        )
        self.assertEqual(prealloc["python"], "atlasRow = np.zeros_like(array_factor)")
        self.assertIn(
            prealloc["python"],
            [s["python"] for s in statements if s["kind"] == "preallocate"],
        )
        loop_index = next(
            i for i, s in enumerate(statements) if s["kind"] == "loop"
        )
        self.assertLess(statements.index(prealloc), loop_index)

    def test_loop_assignment_uses_preallocated_row(self):
        loop = next(s for s in self.func["statements"] if s["kind"] == "loop")
        body_py = [b["python"] for b in loop["body"]]
        self.assertIn(
            "atlasRow[rowNo, :] = np.interp(interpPoints, nonZeroXAxis, tempE)",
            body_py,
        )

    def test_scalar_multiply_not_matrix_multiply(self):
        loop = next(s for s in self.func["statements"] if s["kind"] == "loop")
        body_py = [b["python"] for b in loop["body"]]
        self.assertIn("interpPoints = 2 * nonZeroXAxis[0] * xAxisStep", body_py)
        self.assertNotIn("2 @ nonZeroXAxis[0]", body_py)

    def test_output_returned(self):
        from translator import code_for_result

        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab_real("atlasDisplay.m")).encode("utf-8")
        )
        code = code_for_result(translate_with_rulebook(build_structure(tree)))
        self.assertTrue(code.rstrip().endswith("return Eatlas"))
        self.assertIn("    return Eatlas", code)

    def test_loop_header_translated(self):
        loop = next(s for s in self.func["statements"] if s["kind"] == "loop")
        self.assertEqual(loop["python"], "for rowNo in range(len(yAxis)):")

    def test_find_in_loop_body(self):
        loop = next(s for s in self.func["statements"] if s["kind"] == "loop")
        body_py = [b["python"] for b in loop["body"]]
        self.assertIn("nonZeroIndex = np.where(dataInRow != 0)[0]", body_py)

    def test_interp1_in_loop_body(self):
        loop = next(s for s in self.func["statements"] if s["kind"] == "loop")
        body_py = [b["python"] for b in loop["body"]]
        self.assertIn(
            "atlasRow[rowNo, :] = np.interp(interpPoints, nonZeroXAxis, tempE)",
            body_py,
        )


class TestFFTDivisionAndRange(unittest.TestCase):
    """The exact fs=1000 FFT frequency-axis line: fs*(0:(length(P1)-1))/length(P2).

    '/' dividing by the scalar length(P2) must become plain Python division,
    and the parenthesized colon range must become np.arange(0, len(P1)).
    """

    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("fft_basic.m")).encode("utf-8")
        )
        cls.result = translate_with_rulebook(build_structure(tree))

    def _python_of(self, source):
        for s in self.result["statements"]:
            if s["source"] == source:
                return s["python"]
        return None

    def test_fft_frequency_line_produces_valid_correct_python(self):
        py = self._python_of("f = fs*(0:(length(P1)-1))/length(P2)")
        self.assertIsNotNone(py)
        compile(py, "<test>", "exec")
        self.assertIn("/ len(P2)", py)
        self.assertIn("np.arange(0, len(P1))", py)
        self.assertNotIn("np.linalg.solve", py)
        self.assertNotIn("length(", py)
        self.assertNotIn(":", py)

    def test_scalar_division_in_time_axis(self):
        py = self._python_of("t = 0:1/fs:1-1/fs")
        self.assertIsNotNone(py)
        compile(py, "<test>", "exec")
        self.assertIn("1 / fs", py)
        self.assertNotIn("np.linalg.solve", py)

    def test_parenthesized_range_translates_to_arange(self):
        self.assertEqual(
            _translate_expr("(0:(length(P1)-1))"),
            "(np.arange(0, len(P1)))",
        )

    def test_division_by_scalar_count_uses_plain_slash(self):
        self.assertEqual(_translate_expr("P2/length(P2)"), "P2 / len(P2)")
        self.assertEqual(_translate_expr("1/fs"), "1 / fs")
        self.assertEqual(_translate_expr("a / fs", {"fs"}), "a / fs")

    def test_division_by_array_keeps_matrix_solve(self):
        self.assertEqual(
            _translate_expr("a / b"),
            "np.linalg.solve(b.T, a.T).T",
        )

    def test_known_scalar_multiply_uses_plain_star(self):
        self.assertEqual(
            _translate_expr("fs*(0:(length(P1)-1))/length(P2)", {"fs"}),
            "fs * (np.arange(0, len(P1))) / len(P2)",
        )


class TestRangesInAnyContext(unittest.TestCase):
    """Colon ranges translate in ANY syntactic context, not just as a
    direct index.  Range detection is context-independent in the Reader's
    structure extraction, so a range inside a [ ] matrix literal, a bare
    function-call argument, or nested two levels deep inside another
    expression all go through the same code path."""

    def _translate(self, text):
        structure = Structure(statements=[Statement("assignment", text)])
        return translate_with_rulebook(structure)["statements"][0]["python"]

    def test_range_inside_square_brackets(self):
        self.assertEqual(
            self._translate("y = [1:5]"),
            "y = np.array([[np.arange(1, 5 + 1)]])",
        )

    def test_step_range_inside_square_brackets(self):
        self.assertEqual(
            self._translate("y = [0:0.5:2]"),
            "y = np.array([[np.arange(0, 2, 0.5)]])",
        )

    def test_range_rows_inside_square_brackets(self):
        self.assertEqual(
            self._translate("y = [1:3; 4:6]"),
            "y = np.array([[np.arange(1, 3 + 1)], [np.arange(4, 6 + 1)]])",
        )

    def test_range_as_bare_function_argument(self):
        self.assertEqual(
            self._translate("y = foo(1:5)"),
            "y = foo(np.arange(1, 5 + 1))",
        )

    def test_range_as_builtin_argument(self):
        self.assertEqual(
            self._translate("y = abs(1:5)"),
            "y = np.abs(np.arange(1, 5 + 1))",
        )

    def test_range_nested_two_levels_deep(self):
        self.assertEqual(
            self._translate("y = foo(2 * (1:5))"),
            "y = foo(2 * (np.arange(1, 5 + 1)))",
        )

    def test_translated_outputs_are_valid_python(self):
        for source in (
            "y = [1:5]",
            "y = [0:0.5:2]",
            "y = [1:3; 4:6]",
            "y = foo(1:5)",
            "y = foo(2 * (1:5))",
        ):
            py = self._translate(source)
            compile(py.split(" = ", 1)[1], "<test>", "eval")


if __name__ == "__main__":
    unittest.main()
