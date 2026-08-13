import unittest

from rulebook import apply_indexing_rule
from rulebook.operator_rules import _find_last_operator, apply_operator_rule
from rulebook.translator import _translate_expr

from reader.load_structure import load_structure_from_source
from rulebook import translate_with_rulebook


class TestReverseStepRanges(unittest.TestCase):
    """MATLAB ranges with a negative step (``end:-1:1``) map onto Python
    slices whose bounds both shift down by one; a literal stop of 1 is
    omitted so index 0 is still included."""

    def test_end_to_one(self):
        self.assertEqual(apply_indexing_rule("x(end:-1:1)"), "x[-1::-1]")
        self.assertEqual(apply_indexing_rule("x(end-1:-1:1)"), "x[-2::-1]")
        self.assertEqual(apply_indexing_rule("x(end-2:-1:1)"), "x[-3::-1]")

    def test_literal_bounds(self):
        self.assertEqual(apply_indexing_rule("x(2:-1:1)"), "x[1::-1]")
        self.assertEqual(apply_indexing_rule("x(5:-1:2)"), "x[4:1:-1]")
        self.assertEqual(apply_indexing_rule("x(3:-2:1)"), "x[2::-2]")

    def test_end_start_with_other_stop(self):
        self.assertEqual(apply_indexing_rule("x(end:-2:2)"), "x[-1:1:-2]")
        self.assertEqual(apply_indexing_rule("x(end:-1:2)"), "x[-1:1:-1]")

    def test_identifier_start_passes_through(self):
        self.assertEqual(apply_indexing_rule("x(i:-1:1)"), "x[i::-1]")

    def test_positive_step_ranges_unchanged(self):
        self.assertEqual(apply_indexing_rule("x(1:2:end)"), "x[0:2:]")
        self.assertEqual(apply_indexing_rule("x(2:end)"), "x[1:]")
        self.assertEqual(apply_indexing_rule("x(end-2:end)"), "x[-2:]")
        self.assertEqual(apply_indexing_rule("x(1:end-1)"), "x[0:-1]")


class TestMatrixPower(unittest.TestCase):
    """MATLAB ``^`` is matrix power; a proven non-scalar base becomes an
    explicit matrix power while scalar/unknown bases keep element-wise
    ``**``."""

    def test_literal_scalar_power(self):
        self.assertEqual(apply_operator_rule("2^3"), "2 ** 3")
        self.assertEqual(_translate_expr("2^3"), "2 ** 3")

    def test_matrix_literal_base(self):
        self.assertEqual(
            apply_operator_rule("[1 2; 3 4]^2"),
            "np.linalg.matrix_power([1 2; 3 4], 2)",
        )

    def test_unknown_base_keeps_elementwise(self):
        self.assertEqual(_translate_expr("x^2"), "x ** 2")

    def test_pipeline_proven_matrix_uses_matrix_power(self):
        src = "A = [1 2; 3 4];\nC = A^2;\nD = fs^2;\nfs = 1000;\n"
        structure = load_structure_from_source(src, "matlab_to_python")
        result = translate_with_rulebook(structure)
        python = [s["python"] for s in result["statements"]]
        self.assertIn("C = np.linalg.matrix_power(A, 2)", python)
        self.assertIn("D = fs ** 2", python)


class TestBackslashLeftDivide(unittest.TestCase):
    """MATLAB ``\\`` is matrix left-division (solve a * x = b); a scalar
    operand makes it element-wise (a \\ b is b ./ a)."""

    def test_matrix_left_divide(self):
        self.assertEqual(
            apply_operator_rule("A \\ B"), "np.linalg.solve(A, B)"
        )
        self.assertEqual(
            _translate_expr("A \\ B"), "np.linalg.solve(A, B)"
        )

    def test_scalar_left_divide_is_elementwise(self):
        self.assertEqual(apply_operator_rule("2 \\ x"), "x / 2")
        self.assertEqual(_translate_expr("2 \\ x"), "x / 2")

    def test_elementwise_left_divide(self):
        self.assertEqual(apply_operator_rule("A .\\ B"), "B / A")
        self.assertEqual(_translate_expr("A .\\ B"), "B / A")

    def test_operator_split_finds_backslash(self):
        self.assertEqual(_find_last_operator("A \\ B"), (2, "\\"))
        self.assertEqual(_find_last_operator("A .\\ B"), (2, ".\\"))


class TestLogicalOperators(unittest.TestCase):
    """MATLAB ``&``/``|`` are element-wise logical ops and unary ``~`` is
    logical NOT; all map to explicit numpy calls."""

    def test_logical_and_or(self):
        self.assertEqual(
            apply_operator_rule("flag & other"), "np.logical_and(flag, other)"
        )
        self.assertEqual(
            apply_operator_rule("flag | other"), "np.logical_or(flag, other)"
        )

    def test_unary_not(self):
        self.assertEqual(apply_operator_rule("~flag"), "np.logical_not(flag)")
        self.assertEqual(_translate_expr("~flag"), "np.logical_not(flag)")
        self.assertEqual(
            _translate_expr("~(x > 0)"), "np.logical_not((x > 0))"
        )

    def test_not_binds_looser_than_transpose(self):
        self.assertEqual(
            apply_operator_rule("~flag'"), "np.logical_not(np.conj(flag).T)"
        )

    def test_not_of_logical_and(self):
        self.assertEqual(
            _translate_expr("~(flag & other)"),
            "np.logical_not((np.logical_and(flag, other)))",
        )

    def test_inequality_unaffected(self):
        self.assertEqual(_translate_expr("x ~= 0"), "x != 0")


class TestTransposeInExpressions(unittest.TestCase):
    """A binary '+'/'-' whose left operand ends in a transpose quote is
    still recognized as a binary operator."""

    def test_both_operands_transposed(self):
        self.assertEqual(
            _translate_expr("x' + y'"),
            "np.conj(x).T + np.conj(y).T",
        )
        self.assertEqual(
            apply_operator_rule("x' + y'"),
            "np.conj(x).T + np.conj(y).T",
        )

    def test_transpose_plus_matrix_product(self):
        self.assertEqual(
            _translate_expr("x' + A'*B"),
            "np.conj(x).T + np.conj(A).T @ B",
        )


if __name__ == "__main__":
    unittest.main()
