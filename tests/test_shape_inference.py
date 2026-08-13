import unittest
from unittest import mock

from reader import MATLAB_TO_PYTHON, load_structure_from_source
from rulebook import translate_with_rulebook
from rulebook.shape_inference import (
    MATRIX,
    SCALAR,
    UNKNOWN,
    VECTOR,
    infer_shapes,
    shape_of_expr,
)
from tests.paths import sample_matlab
from translator import translate_file, translate_source


def _infer(source):
    return infer_shapes(load_structure_from_source(source, MATLAB_TO_PYTHON))


class TestShapeOfExpr(unittest.TestCase):
    """shape_of_expr is a pure function: given an expression string and a
    scope of already-inferred names, it returns a coarse shape verdict."""

    def test_numeric_literals_are_scalar(self):
        for expr in ("5", "-3", "2.5", "10e-6", "1E+10", "3i"):
            self.assertEqual(shape_of_expr(expr), SCALAR, expr)

    def test_constants_are_scalar(self):
        for expr in ("pi", "e", "eps"):
            self.assertEqual(shape_of_expr(expr), SCALAR, expr)

    def test_string_literal_is_scalar(self):
        self.assertEqual(shape_of_expr("'hello'"), SCALAR)

    def test_row_vector_literal(self):
        self.assertEqual(shape_of_expr("[1 2 3]"), VECTOR)
        self.assertEqual(shape_of_expr("[a b c]", {"a": VECTOR}), VECTOR)

    def test_matrix_literal(self):
        self.assertEqual(shape_of_expr("[1 2; 3 4]"), MATRIX)
        self.assertEqual(shape_of_expr("[1:3; 4:6]"), MATRIX)

    def test_single_cell_bracket_delegates_to_inner(self):
        self.assertEqual(shape_of_expr("[5]"), SCALAR)
        self.assertEqual(shape_of_expr("[x]", {"x": VECTOR}), VECTOR)

    def test_empty_matrix_literal_is_vector(self):
        self.assertEqual(shape_of_expr("[]"), VECTOR)

    def test_colon_ranges_are_vectors(self):
        self.assertEqual(shape_of_expr("1:10"), VECTOR)
        self.assertEqual(shape_of_expr("0:0.5:2"), VECTOR)

    def test_scalar_calls(self):
        self.assertEqual(shape_of_expr("length(x)"), SCALAR)
        self.assertEqual(shape_of_expr("numel(x)"), SCALAR)
        self.assertEqual(shape_of_expr("round(x)"), SCALAR)
        self.assertEqual(shape_of_expr("size(x, 1)"), SCALAR)

    def test_shape_preserving_calls(self):
        self.assertEqual(shape_of_expr("fft(x)", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("abs(x)", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("sin(x)", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("fft(X)", {"X": MATRIX}), MATRIX)

    def test_constructors(self):
        self.assertEqual(shape_of_expr("zeros(3)"), MATRIX)
        self.assertEqual(shape_of_expr("zeros(1, N)"), MATRIX)
        self.assertEqual(shape_of_expr("randn(2, 2)"), MATRIX)

    def test_linspace_find_meshgrid(self):
        self.assertEqual(shape_of_expr("linspace(0, 1, 10)"), VECTOR)
        self.assertEqual(shape_of_expr("find(x > 0)"), VECTOR)
        self.assertEqual(shape_of_expr("meshgrid(x, y)"), MATRIX)

    def test_reductions_depend_on_argument(self):
        self.assertEqual(shape_of_expr("sum(x)", {"x": VECTOR}), SCALAR)
        self.assertEqual(shape_of_expr("sum(X)", {"X": MATRIX}), VECTOR)
        self.assertEqual(shape_of_expr("mean(x)", {"x": SCALAR}), SCALAR)
        self.assertEqual(shape_of_expr("max(x)", {"x": UNKNOWN}), UNKNOWN)

    def test_indexing(self):
        self.assertEqual(shape_of_expr("A(1, 2)", {"A": MATRIX}), SCALAR)
        self.assertEqual(shape_of_expr("A(:, 2)", {"A": MATRIX}), VECTOR)
        self.assertEqual(shape_of_expr("A(1, :)", {"A": MATRIX}), VECTOR)
        self.assertEqual(shape_of_expr("A(:, :)", {"A": MATRIX}), MATRIX)
        self.assertEqual(shape_of_expr("x(1:3)", {"x": VECTOR}), VECTOR)

    def test_operators(self):
        self.assertEqual(shape_of_expr("2*fs", {"fs": SCALAR}), SCALAR)
        self.assertEqual(
            shape_of_expr("2*pi*f1*t", {"f1": SCALAR, "t": VECTOR}), VECTOR
        )
        self.assertEqual(
            shape_of_expr("gain * x", {"gain": SCALAR, "x": VECTOR}), VECTOR
        )
        self.assertEqual(shape_of_expr("A * B", {"A": MATRIX, "B": MATRIX}), MATRIX)
        self.assertEqual(shape_of_expr("x .* y", {"x": VECTOR, "y": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("a / fs", {"a": VECTOR, "fs": SCALAR}), VECTOR)
        self.assertEqual(shape_of_expr("A / B", {"A": MATRIX, "B": MATRIX}), MATRIX)
        self.assertEqual(shape_of_expr("x + 1", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("x.^2", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("x^2", {"x": SCALAR}), SCALAR)
        self.assertEqual(shape_of_expr("a ~= 0", {"a": VECTOR}), VECTOR)

    def test_transpose_preserves_shape(self):
        self.assertEqual(shape_of_expr("x'", {"x": VECTOR}), VECTOR)
        self.assertEqual(shape_of_expr("X.'", {"X": MATRIX}), MATRIX)

    def test_unknown_identifier(self):
        self.assertEqual(shape_of_expr("mystery"), UNKNOWN)
        self.assertEqual(shape_of_expr("someFunc(x)"), UNKNOWN)


class TestInferShapes(unittest.TestCase):
    """infer_shapes walks a whole Structure in order, tracking per-scope
    shapes and the definite-scalar set the rulebook consumes."""

    def test_order_sensitive_last_assignment_wins(self):
        result = _infer("x = 5;\nx = [1 2 3];\n")
        self.assertEqual(result.top.shapes["x"], VECTOR)
        self.assertNotIn("x", result.top.scalars)

    def test_plain_assignments_mark_definite_scalars(self):
        result = _infer("fs = 1000;\nt = 0:1/fs:1-1/fs;\ngain = 0.5;\n")
        self.assertEqual(result.top.shapes["fs"], SCALAR)
        self.assertEqual(result.top.shapes["t"], VECTOR)
        self.assertEqual(result.top.shapes["gain"], SCALAR)
        self.assertEqual(result.top.scalars, {"fs", "gain"})

    def test_compound_chain_infers_through_calls(self):
        result = _infer(
            "fs = 1000;\nt = 0:1/fs:1-1/fs;\n"
            "x = sin(2*pi*50*t);\nY = fft(x);\nP2 = abs(Y);\n"
            "s = sum(x);\n"
        )
        self.assertEqual(result.top.shapes["x"], VECTOR)
        self.assertEqual(result.top.shapes["Y"], VECTOR)
        self.assertEqual(result.top.shapes["P2"], VECTOR)
        self.assertEqual(result.top.shapes["s"], SCALAR)

    def test_indexed_assignment_keeps_container_shape(self):
        result = _infer("acc = zeros(1, 10);\nacc(3) = 5;\n")
        self.assertEqual(result.top.shapes["acc"], MATRIX)
        self.assertEqual(result.top.shapes["acc"], MATRIX)

    def test_loop_variable_is_scalar_and_does_not_leak(self):
        result = _infer(
            "function y = f()\n"
            "    for n = 1:10\n"
            "        x = n;\n"
            "    end\n"
            "    y = x;\n"
            "end\n"
        )
        info = result.functions["f"]
        self.assertEqual(info.shapes["x"], SCALAR)
        self.assertNotIn("n", info.shapes)
        self.assertEqual(info.shapes["y"], SCALAR)

    def test_multi_output_assignments(self):
        result = _infer("x = [1 2 3];\n[v, i] = max(x);\n[r, c] = size(x);\n")
        self.assertEqual(result.top.shapes["v"], SCALAR)
        self.assertEqual(result.top.shapes["i"], SCALAR)
        self.assertEqual(result.top.shapes["r"], SCALAR)
        self.assertEqual(result.top.shapes["c"], SCALAR)

    def test_find_and_meshgrid_outputs(self):
        result = _infer("idx = find(x ~= 0);\n[FX, FY] = meshgrid(x, y);\n")
        self.assertEqual(result.top.shapes["idx"], VECTOR)
        self.assertEqual(result.top.shapes["FX"], MATRIX)
        self.assertEqual(result.top.shapes["FY"], MATRIX)

    def test_function_scope_is_independent(self):
        result = _infer(
            "function y = f(a, b)\n    y = a + b;\nend\n"
        )
        self.assertEqual(result.functions["f"].shapes["a"], UNKNOWN)
        self.assertEqual(result.functions["f"].shapes["y"], UNKNOWN)

    def test_function_body_scalars(self):
        result = _infer(
            "function y = f(fs)\n    gain = 0.5;\n    y = gain * fs;\nend\n"
        )
        self.assertEqual(result.functions["f"].shapes["gain"], SCALAR)
        self.assertEqual(result.functions["f"].scalars, {"gain"})

    def test_counts_summarize_union_of_scopes(self):
        result = _infer("a = 1;\nv = [1 2];\nM = [1 2; 3 4];\n")
        self.assertEqual(result.counts[SCALAR], 1)
        self.assertEqual(result.counts[VECTOR], 1)
        self.assertEqual(result.counts[MATRIX], 1)
        self.assertEqual(result.counts[UNKNOWN], 0)

    def test_sample_file_shapes(self):
        structure = load_structure_from_source(
            sample_matlab("shape_inference.m").read_text(), MATLAB_TO_PYTHON
        )
        result = infer_shapes(structure)
        info = result.functions["shape_inference"]
        self.assertEqual(info.shapes["f1"], SCALAR)
        self.assertEqual(info.shapes["t"], VECTOR)
        self.assertEqual(info.shapes["A"], MATRIX)
        self.assertEqual(info.shapes["v"], VECTOR)
        self.assertEqual(info.shapes["s"], SCALAR)
        self.assertEqual(info.shapes["scale"], SCALAR)


class TestInferenceSection(unittest.TestCase):
    def test_translate_source_reports_inference_section(self):
        result = translate_source("fs = 1000;\nt = 0:1/fs:1-1/fs;\n")
        inference = result["sections"]["inference"]
        self.assertEqual(inference["status"], "ok")
        self.assertEqual(inference["counts"][SCALAR], 1)
        self.assertEqual(inference["counts"][VECTOR], 1)
        self.assertIn("fs", inference["scalars"])

    def test_translate_file_reports_inference_for_sample(self):
        result = translate_file(sample_matlab("shape_inference.m"))
        self.assertEqual(result["status"], "ok")
        inference = result["sections"]["inference"]
        self.assertEqual(inference["status"], "ok")
        self.assertGreater(inference["counts"][SCALAR], 0)
        self.assertGreater(inference["counts"][VECTOR], 0)
        self.assertGreater(inference["counts"][MATRIX], 0)

    def test_existing_sections_untouched(self):
        result = translate_file(sample_matlab("indexing_ops.m"))
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertEqual(result["sections"]["inference"]["status"], "ok")

    def test_no_inference_section_in_reverse_direction(self):
        result = translate_source(
            "A = np.array([[1, 2], [3, 4]])\nprint(A[0, 0])\n",
            direction="python_to_matlab",
        )
        self.assertNotIn("inference", result["sections"])

    @mock.patch(
        "rulebook.shape_inference.infer_shapes", side_effect=RuntimeError("boom")
    )
    def test_inference_failure_does_not_break_translation(self, _mock_infer):
        result = translate_source("fs = 1000;\nt = 0:1/fs:1-1/fs;\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sections"]["inference"]["status"], "error")
        self.assertIn("boom", result["sections"]["inference"]["detail"])
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertIn("import numpy as np", result["python"])


def _translate_source_lines(source):
    structure = load_structure_from_source(source, MATLAB_TO_PYTHON)
    shapes = infer_shapes(structure)
    result = translate_with_rulebook(structure, shapes=shapes)
    return [s["python"] for s in result["statements"]]


class TestInferenceFeedsRulebook(unittest.TestCase):
    """The inferred scalar set is unioned into the rulebook's ``scalars``,
    so '*' and '/' involving a proven scalar stay element-wise."""

    def test_inferred_scalar_keeps_plain_division(self):
        lines = _translate_source_lines(
            "scale = length(P2);\nf = 1000*(0:(length(P1)-1))/scale;\n"
        )
        self.assertIn("/ scale", lines[1])
        self.assertNotIn("np.linalg.solve", lines[1])

    def test_single_cell_bracket_scalar_stays_elementwise_multiply(self):
        # q = [5] is a 1x1 matrix; inference proves it scalar, so 'q * v'
        # is element-wise ('*'), which broadcasts correctly, instead of the
        # conservative matrix '@' (which numpy would reject for (1,1)@(n,)).
        lines = _translate_source_lines("q = [5];\nz = q * v;\n")
        self.assertEqual(lines[0], "q = np.array([[5]])")
        self.assertIn("q * v", lines[1])
        self.assertNotIn("q @ v", lines[1])

    def test_matrix_multiply_stays_matrix(self):
        lines = _translate_source_lines(
            "A = [1 2; 3 4];\nB = [5 6; 7 8];\nC = A * B;\n"
        )
        self.assertIn("A @ B", lines[2])

    def test_default_call_computes_inference_too(self):
        structure = load_structure_from_source(
            "q = [5];\nz = q * v;\n", MATLAB_TO_PYTHON
        )
        lines = [s["python"] for s in translate_with_rulebook(structure)["statements"]]
        self.assertIn("q * v", lines[1])
        self.assertNotIn("q @ v", lines[1])


if __name__ == "__main__":
    unittest.main()
