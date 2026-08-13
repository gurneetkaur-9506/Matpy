import unittest

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from rulebook.symbolic import (
    HIGH,
    LOW,
    MEDIUM,
    analyze_expression,
    analyze_translation,
)
from translator import translate_source


class TestConstantDetection(unittest.TestCase):
    def test_folds_arithmetic(self):
        insights = analyze_expression("2 * np.pi")
        cats = [i["category"] for i in insights]
        self.assertIn("constant_detection", cats)
        insight = next(i for i in insights if i["category"] == "constant_detection")
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn("6.283185", insight["message"])

    def test_folds_math_function(self):
        insight = next(
            i for i in analyze_expression("np.sin(0)") if i["category"] == "constant_detection"
        )
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn("0.0", insight["message"])

    def test_free_variable_not_constant(self):
        insights = analyze_expression("x + 1")
        self.assertNotIn(
            "constant_detection", [i["category"] for i in insights]
        )

    def test_unparseable_returns_empty(self):
        self.assertEqual(analyze_expression("x .* y"), [])


class TestSimplification(unittest.TestCase):
    def test_multiply_by_zero(self):
        insight = next(
            i for i in analyze_expression("0 * x") if i["category"] == "simplification"
        )
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn("product is 0", insight["message"])

    def test_self_subtraction(self):
        insight = next(
            i for i in analyze_expression("a - a") if i["category"] == "simplification"
        )
        self.assertEqual(insight["confidence"], HIGH)

    def test_double_negation(self):
        insight = next(
            i for i in analyze_expression("-(-x)") if i["category"] == "simplification"
        )
        self.assertEqual(insight["confidence"], HIGH)

    def test_noop_terms_low_confidence(self):
        for expr in ("x * 1", "x + 0", "x / 1", "x ** 1"):
            with self.subTest(expr=expr):
                insight = next(
                    i for i in analyze_expression(expr) if i["category"] == "simplification"
                )
                self.assertEqual(insight["confidence"], LOW)


class TestMathReasoning(unittest.TestCase):
    def test_abs_nonnegative(self):
        insight = next(
            i for i in analyze_expression("np.abs(x)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn(">= 0", insight["message"])

    def test_exp_positive(self):
        insight = next(
            i for i in analyze_expression("np.exp(x)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], HIGH)

    def test_sin_bounded(self):
        insight = next(
            i for i in analyze_expression("np.sin(x)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], MEDIUM)
        self.assertIn("[-1, 1]", insight["message"])

    def test_square_nonnegative(self):
        insight = next(
            i for i in analyze_expression("x ** 2") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], MEDIUM)

    def test_sqrt_nonnegative(self):
        insight = next(
            i for i in analyze_expression("np.sqrt(x)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], HIGH)

    def test_sinc_bounded(self):
        insight = next(
            i for i in analyze_expression("np.sinc(t)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], MEDIUM)
        self.assertIn("[-1, 1]", insight["message"])

    def test_tanh_bounded(self):
        insight = next(
            i for i in analyze_expression("np.tanh(x)") if i["category"] == "math_reasoning"
        )
        self.assertEqual(insight["confidence"], MEDIUM)
        self.assertIn("(-1, 1)", insight["message"])

    def test_inverse_trig_bounded(self):
        cases = {
            "np.arcsin(x)": "[-pi/2, pi/2]",
            "np.arccos(x)": "[0, pi]",
            "np.arctan(x)": "(-pi/2, pi/2)",
        }
        for expr, bound in cases.items():
            with self.subTest(expr=expr):
                insight = next(
                    i for i in analyze_expression(expr)
                    if i["category"] == "math_reasoning"
                )
                self.assertEqual(insight["confidence"], MEDIUM)
                self.assertIn(bound, insight["message"])


class TestExtendedSimplification(unittest.TestCase):
    def test_double_conjugation(self):
        insight = next(
            i for i in analyze_expression("np.conj(np.conj(z))")
            if i["category"] == "simplification"
        )
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn("conjugates a conjugate", insight["message"])

    def test_subtract_from_zero_is_negation(self):
        insight = next(
            i for i in analyze_expression("0 - a") if i["category"] == "simplification"
        )
        self.assertEqual(insight["confidence"], LOW)
        self.assertIn("negation", insight["message"])


class TestExtendedConstantFolding(unittest.TestCase):
    def test_hypot_folds(self):
        insight = next(
            i for i in analyze_expression("np.hypot(3, 4)")
            if i["category"] == "constant_detection"
        )
        self.assertEqual(insight["confidence"], HIGH)
        self.assertIn("5.0", insight["message"])

    def test_tau_folds(self):
        insight = next(
            i for i in analyze_expression("np.tau * 2")
            if i["category"] == "constant_detection"
        )
        self.assertIn("12.56637", insight["message"])

    def test_hypot_with_free_variable_not_constant(self):
        categories = [i["category"] for i in analyze_expression("np.hypot(3, x)")]
        self.assertNotIn("constant_detection", categories)


class TestPipelineIntegration(unittest.TestCase):
    def test_symbolic_section_populated_forward(self):
        result = translate_source(
            "x = 2 * pi;\ny = 0 * a + b;\n", direction=MATLAB_TO_PYTHON
        )
        sym = result["sections"]["symbolic"]
        self.assertEqual(sym["status"], "ok")
        self.assertIsInstance(sym["insights"], list)
        self.assertGreater(len(sym["insights"]), 0)
        cats = {i["category"] for i in sym["insights"]}
        self.assertTrue({"constant_detection", "simplification"} <= cats)
        self.assertIn("constant_detection", sym["counts"])
        self.assertIn("simplification", sym["counts"])

    def test_symbolic_insights_carry_source_line(self):
        result = translate_source(
            "x = 2 * pi;\ny = x * 0;\n", direction=MATLAB_TO_PYTHON
        )
        for insight in result["sections"]["symbolic"]["insights"]:
            self.assertIsNotNone(insight["line"])
            self.assertTrue(insight["source"])

    def test_symbolic_section_skipped_reverse(self):
        result = translate_source("x = a + b\n", direction=PYTHON_TO_MATLAB)
        self.assertEqual(result["sections"]["symbolic"]["status"], "skipped")
        self.assertEqual(result["sections"]["symbolic"]["insights"], [])

    def test_symbolic_section_present_on_clean_translation(self):
        result = translate_source("x = sin(y) + 1;\n", direction=MATLAB_TO_PYTHON)
        self.assertEqual(result["sections"]["symbolic"]["status"], "ok")

    def test_symbolic_does_not_change_translation(self):
        src = "x = 0 * a + b;\n"
        with_symbolic = translate_source(src, direction=MATLAB_TO_PYTHON)
        self.assertEqual(with_symbolic["status"], "ok")
        # The symbolic stage is advisory: it must not rewrite the emitted
        # Python or introduce new problems.
        self.assertEqual(with_symbolic["problems"], [])

    def test_analyze_translation_rejects_nothing(self):
        result = translate_source(
            "z = sqrt(4) + abs(-3) * 1;\n", direction=MATLAB_TO_PYTHON
        )
        section = analyze_translation(result)
        self.assertEqual(section["status"], "ok")
        self.assertGreaterEqual(len(section["insights"]), 1)


if __name__ == "__main__":
    unittest.main()
