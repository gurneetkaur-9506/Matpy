import unittest

from reader import MATLAB_TO_PYTHON, load_structure_from_source
from reader.structure import Statement
from rulebook import MULTI_OUTPUT_RULES, translate_multi_output_assignment, translate_with_rulebook
from rulebook.translator import _translate_statement, _translate_expr


def _translate(expr):
    return lambda a: _translate_expr(a)


class TestRegistryContents(unittest.TestCase):
    def test_registered_functions(self):
        self.assertIn("max", MULTI_OUTPUT_RULES)
        self.assertIn("min", MULTI_OUTPUT_RULES)
        self.assertIn("sort", MULTI_OUTPUT_RULES)
        self.assertIn("size", MULTI_OUTPUT_RULES)
        self.assertIn("find", MULTI_OUTPUT_RULES)

    def test_max_and_min_are_pair_rules(self):
        for name in ("max", "min"):
            self.assertEqual(MULTI_OUTPUT_RULES[name]["kind"], "pair")
            self.assertIn("value", MULTI_OUTPUT_RULES[name])
            self.assertIn("index", MULTI_OUTPUT_RULES[name])

    def test_size_is_shape_rule(self):
        self.assertEqual(MULTI_OUTPUT_RULES["size"]["kind"], "shape")

    def test_find_is_where_rule(self):
        self.assertEqual(MULTI_OUTPUT_RULES["find"]["kind"], "where")


class TestTranslateMultiOutput(unittest.TestCase):
    def test_max(self):
        self.assertEqual(
            translate_multi_output_assignment("[v, i]", "max(x)", _translate("x")),
            ["v = np.max(x)", "i = np.argmax(x)"],
        )

    def test_min(self):
        self.assertEqual(
            translate_multi_output_assignment("[v, i]", "min(x)", _translate("x")),
            ["v = np.min(x)", "i = np.argmin(x)"],
        )

    def test_sort(self):
        self.assertEqual(
            translate_multi_output_assignment("[s, si]", "sort(x)", _translate("x")),
            ["s = np.sort(x)", "si = np.argsort(x)"],
        )

    def test_size(self):
        self.assertEqual(
            translate_multi_output_assignment("[nr, nc]", "size(A)", _translate("A")),
            ["nr, nc = A.shape"],
        )

    def test_size_three_outputs(self):
        self.assertEqual(
            translate_multi_output_assignment("[a, b, c]", "size(M)", _translate("M")),
            ["a, b, c = M.shape"],
        )

    def test_find(self):
        self.assertEqual(
            translate_multi_output_assignment("[r, c]", "find(B)", _translate("B")),
            ["r, c = np.where(B)"],
        )

    def test_single_target_max(self):
        self.assertEqual(
            translate_multi_output_assignment("[z]", "max(x)", _translate("x")),
            ["z = np.max(x)"],
        )

    def test_single_target_find(self):
        self.assertEqual(
            translate_multi_output_assignment("[r]", "find(B)", _translate("B")),
            ["r = np.where(B)[0]"],
        )

    def test_too_many_targets_returns_none(self):
        self.assertIsNone(
            translate_multi_output_assignment("[a, b, c]", "max(x)", _translate("x"))
        )
        self.assertIsNone(
            translate_multi_output_assignment("[a, b, c]", "find(B)", _translate("B"))
        )

    def test_unknown_function_returns_none(self):
        self.assertIsNone(
            translate_multi_output_assignment(
                "[a, b]", "myfunc(x)", _translate("x")
            )
        )

    def test_non_call_value_returns_none(self):
        self.assertIsNone(
            translate_multi_output_assignment("[a, b]", "x + 1", _translate("x"))
        )

    def test_indexed_target_returns_none(self):
        self.assertIsNone(
            translate_multi_output_assignment(
                "[v, i(1)]", "max(x)", _translate("x")
            )
        )

    def test_wrong_arg_count_returns_none(self):
        self.assertIsNone(
            translate_multi_output_assignment(
                "[v, i]", "max(x, y)", _translate("x")
            )
        )
        self.assertIsNone(
            translate_multi_output_assignment(
                "[nr, nc]", "size(A, 1)", _translate("A")
            )
        )
        self.assertIsNone(
            translate_multi_output_assignment("[r, c]", "find(B, 1)", _translate("B"))
        )


class TestTranslateStatementIntegration(unittest.TestCase):
    def _python(self, text):
        return _translate_statement(Statement("assignment", text))["python"]

    def test_max_statement(self):
        self.assertEqual(
            self._python("[v, i] = max(x)"), "v = np.max(x)\ni = np.argmax(x)"
        )

    def test_min_statement(self):
        self.assertEqual(
            self._python("[v, i] = min(x)"), "v = np.min(x)\ni = np.argmin(x)"
        )

    def test_sort_statement(self):
        self.assertEqual(
            self._python("[s, si] = sort(x)"),
            "s = np.sort(x)\nsi = np.argsort(x)",
        )

    def test_size_statement(self):
        self.assertEqual(self._python("[nr, nc] = size(A)"), "nr, nc = A.shape")

    def test_find_statement(self):
        self.assertEqual(self._python("[r, c] = find(B)"), "r, c = np.where(B)")

    def test_nested_max_abs(self):
        self.assertEqual(
            self._python("[v2, i2] = max(abs(x))"),
            "v2 = np.max(np.abs(x))\ni2 = np.argmax(np.abs(x))",
        )

    def test_max_with_dim(self):
        self.assertEqual(
            self._python("[v, i] = max(x, [], 2)"),
            "v = np.max(x, axis=1)\ni = np.argmax(x, axis=1)",
        )

    def test_sort_descend(self):
        self.assertEqual(
            self._python("[s2, si2] = sort(x, 'descend')"),
            "s2 = np.sort(x)[::-1]\nsi2 = np.argsort(x)[::-1]",
        )

    def test_sort_ascend(self):
        self.assertEqual(
            self._python("[s3, si3] = sort(x, 'ascend')"),
            "s3 = np.sort(x)\nsi3 = np.argsort(x)",
        )


class TestMultiOutputFilePipeline(unittest.TestCase):
    def _translate_file(self, source):
        structure = load_structure_from_source(source, MATLAB_TO_PYTHON)
        result = translate_with_rulebook(structure)
        return "\n".join(s["python"] for s in result["statements"])

    def test_pipeline_emits_decomposition_lines(self):
        source = (
            "[v, i] = max(abs(x));\n"
            "[s, si] = sort(x);\n"
            "[nr, nc] = size(A);\n"
            "[r, c] = find(x > 0);\n"
        )
        python = self._translate_file(source)
        self.assertIn("v = np.max(np.abs(x))", python)
        self.assertIn("i = np.argmax(np.abs(x))", python)
        self.assertIn("s = np.sort(x)", python)
        self.assertIn("si = np.argsort(x)", python)
        self.assertIn("nr, nc = A.shape", python)
        self.assertIn("r, c = np.where(x > 0)", python)


if __name__ == "__main__":
    unittest.main()
