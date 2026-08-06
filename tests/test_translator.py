import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure, load_matlab_file
from reader.structure import Loop, Statement, Structure
from rulebook import UNRESOLVED, translate_with_rulebook
from rulebook.translator import _translate_loop
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

    def test_interp1_reorders_args_to_np_interp(self):
        self.assertEqual(
            self._translate(
                "atlasRow(rowNo,:) = interp1(nonZeroXAxis,tempE,interpPoints)"
            ),
            "atlasRow[rowNo, :] = np.interp(interpPoints, nonZeroXAxis, tempE)",
        )

    def test_interp1_wrong_arity_stays_unresolved(self):
        self.assertEqual(self._translate("y = interp1(x, v)"), UNRESOLVED)


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


if __name__ == "__main__":
    unittest.main()
