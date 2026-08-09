import unittest
from unittest import mock

from checker import accuracy, score_mix
from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED
from tests.paths import sample_matlab, sample_python
from translator import translate_file


def _func(name, count, unresolved=0, confidence=None):
    statements = [
        {"kind": "assignment", "source": "x%d = %d;" % (i, i), "python": "x%d = %d" % (i, i)}
        for i in range(count)
    ]
    for i in range(unresolved):
        statements[i]["python"] = UNRESOLVED
    func = {"name": name, "parameters": [], "outputs": [], "statements": statements}
    if confidence is not None:
        func["draft"] = {"code": "draft code", "confidence": confidence, "notes": []}
    return func


def _result(total, unresolved=0, checker="skipped", functions=None, statements=None):
    return {
        "sections": {
            "rulebook": {
                "status": "unresolved" if unresolved else "ok",
                "unresolved": unresolved,
                "total": total,
            },
            "assistant": {"status": "none", "drafted": []},
            "checker": {"status": checker},
        },
        "functions": functions or [],
        "statements": statements or [],
    }


class TestScoreMix(unittest.TestCase):
    def test_all_rulebook_scores_100(self):
        result = score_mix(
            [{"source": "rulebook", "lines": 10, "weight": 1.0}]
        )
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["total_lines"], 10)
        self.assertEqual(result["breakdown"], {"rulebook": 10.0})

    def test_mixed_rulebook_and_unresolved(self):
        result = score_mix(
            [
                {"source": "rulebook", "lines": 8, "weight": 1.0},
                {"source": "unresolved", "lines": 2, "weight": 0.0},
            ]
        )
        self.assertEqual(result["total_lines"], 10)
        self.assertAlmostEqual(result["score"], 80.0)

    def test_assistant_weighted_at_confidence(self):
        result = score_mix(
            [
                {"source": "rulebook", "lines": 8, "weight": 1.0},
                {"source": "assistant", "lines": 4, "weight": 0.5},
            ]
        )
        self.assertEqual(result["total_lines"], 12)
        self.assertAlmostEqual(result["weighted_lines"], 10.0)
        self.assertAlmostEqual(result["score"], 83.33, places=2)
        self.assertAlmostEqual(result["breakdown"]["rulebook"], 8.0)
        self.assertAlmostEqual(result["breakdown"]["assistant"], 2.0)

    def test_verified_weights_full(self):
        result = score_mix(
            [
                {"source": "assistant", "lines": 6, "weight": 0.3},
                {"source": "verified", "lines": 6, "weight": 1.0},
            ]
        )
        self.assertAlmostEqual(result["score"], 65.0)

    def test_empty_mix_scores_zero(self):
        result = score_mix([])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["total_lines"], 0)
        self.assertEqual(result["breakdown"], {})

    def test_weight_clamped_to_unit_interval(self):
        result = score_mix(
            [{"source": "assistant", "lines": 4, "weight": 2.5}]
        )
        self.assertAlmostEqual(result["score"], 100.0)
        result = score_mix(
            [{"source": "assistant", "lines": 4, "weight": -1.0}]
        )
        self.assertEqual(result["score"], 0.0)


class TestAccuracy(unittest.TestCase):
    def test_fully_resolved_scores_100(self):
        result = accuracy(_result(total=10, functions=[_func("f", 5), _func("g", 5)]))
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["breakdown"], {"rulebook": 10.0})

    def test_script_unresolved_reduce_score(self):
        result = accuracy(_result(total=10, unresolved=2))
        self.assertEqual(result["total_lines"], 10)
        self.assertAlmostEqual(result["score"], 80.0)
        self.assertEqual(result["breakdown"]["rulebook"], 8.0)
        self.assertEqual(result["breakdown"]["unresolved"], 0.0)

    def test_drafted_function_weighted_by_confidence(self):
        result = accuracy(
            _result(
                total=12,
                functions=[_func("f", 8), _func("g", 4, confidence=0.5)],
            )
        )
        self.assertEqual(result["total_lines"], 12)
        self.assertAlmostEqual(result["score"], 83.33, places=2)
        self.assertAlmostEqual(result["breakdown"]["rulebook"], 8.0)
        self.assertAlmostEqual(result["breakdown"]["assistant"], 2.0)

    def test_drafted_with_low_confidence_scores_low(self):
        result = accuracy(_result(total=4, functions=[_func("g", 4, confidence=0.25)]))
        self.assertEqual(result["score"], 25.0)

    def test_verified_overrides_drafted_weight(self):
        result = accuracy(
            _result(
                total=6,
                checker="verified",
                functions=[_func("g", 6, confidence=0.4)],
            )
        )
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["breakdown"], {"verified": 6.0})

    def test_missing_sections_scores_zero(self):
        result = accuracy({})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["total_lines"], 0)

    def test_syntax_invalid_statement_downgraded(self):
        stmts = [
            {
                "kind": "assignment",
                "source": "k = 2 * pi / lambda;",
                "python": "k = 2 * pi / lambda",
            },
            {"kind": "assignment", "source": "y = x;", "python": "y = x"},
        ]
        func = {"name": "f", "parameters": [], "outputs": [], "statements": stmts}
        scored = accuracy(_result(total=2, functions=[func]))
        self.assertEqual(scored["total_lines"], 2)
        self.assertEqual(scored["score"], 50.0)
        self.assertEqual(scored["breakdown"]["rulebook"], 1.0)
        self.assertEqual(scored["breakdown"]["unresolved"], 0.0)

    def test_broken_line_in_clean_file_always_unresolved(self):
        clean = [
            {
                "kind": "assignment",
                "source": "a = 1;",
                "python": "a = 1",
            },
            {
                "kind": "assignment",
                "source": "b = 2;",
                "python": "b = 2",
            },
        ]
        broken = [
            {
                "kind": "assignment",
                "source": "c = (;",
                "python": "c = (",
            }
        ]
        func = {
            "name": "f",
            "parameters": [],
            "outputs": [],
            "statements": clean + broken,
        }
        scored = accuracy(_result(total=3, functions=[func]))
        self.assertEqual(scored["total_lines"], 3)
        self.assertEqual(scored["score"], 66.67)
        self.assertEqual(scored["breakdown"]["rulebook"], 2.0)
        self.assertEqual(scored["breakdown"]["unresolved"], 0.0)

    def test_broken_line_in_drafted_function_still_unresolved(self):
        statements = [
            {"kind": "assignment", "source": "a = 1;", "python": "a = 1"},
            {"kind": "assignment", "source": "c = (;", "python": "c = ("},
        ]
        func = {
            "name": "f",
            "parameters": [],
            "outputs": [],
            "statements": statements,
            "draft": {"code": "draft", "confidence": 1.0, "notes": []},
        }
        scored = accuracy(_result(total=2, functions=[func]))
        self.assertEqual(scored["score"], 50.0)
        self.assertEqual(scored["breakdown"]["assistant"], 1.0)
        self.assertIn("unresolved", scored["breakdown"])
        self.assertEqual(scored["breakdown"]["unresolved"], 0.0)

    def test_loop_with_invalid_body_downgraded(self):
        body = [
            {
                "kind": "assignment",
                "source": "af = af + exp(1i * (n - 1) * phase);",
                "python": "af = af + np.exp(1i * (n - 1) * phase)",
            }
        ]
        loop = {
            "kind": "loop",
            "source": "for n = 1:N",
            "python": "for n in range(N):",
            "body": body,
        }
        good = {"kind": "assignment", "source": "y = x;", "python": "y = x"}
        func = {"name": "f", "parameters": [], "outputs": [], "statements": [good, loop]}
        scored = accuracy(_result(total=2, functions=[func]))
        self.assertEqual(scored["score"], 50.0)

    def test_reverse_direction_not_syntax_checked(self):
        stmts = [{"kind": "Return", "source": "return af", "matlab": "return af"}]
        func = {"name": "f", "parameters": [], "outputs": [], "statements": stmts}
        result = _result(total=1, functions=[func])
        result["direction"] = PYTHON_TO_MATLAB
        scored = accuracy(result)
        self.assertEqual(scored["score"], 100.0)
        self.assertEqual(scored["breakdown"], {"rulebook": 1.0})

    def test_failed_numeric_verdict_zeroes_all_resolved_lines(self):
        func = _func("f", 4)
        result = _result(total=4, checker="failed", functions=[func])
        scored = accuracy(result)
        self.assertEqual(scored["score"], 0.0)
        self.assertEqual(scored["breakdown"], {"failed": 0.0})
        self.assertEqual(
            scored["method"], "numeric comparison against real output failed"
        )

    def test_failed_numeric_verdict_on_script(self):
        stmts = [
            {"kind": "assignment", "source": "a = 1;", "python": "a = 1"},
            {"kind": "assignment", "source": "b = 2;", "python": "b = 2"},
        ]
        result = _result(total=2, checker="failed", statements=stmts)
        scored = accuracy(result)
        self.assertEqual(scored["score"], 0.0)
        self.assertEqual(scored["breakdown"], {"failed": 0.0})

    def test_verified_numeric_verdict_reports_method(self):
        func = _func("f", 3, confidence=0.4)
        result = _result(total=3, checker="verified", functions=[func])
        scored = accuracy(result)
        self.assertEqual(scored["score"], 100.0)
        self.assertEqual(scored["breakdown"], {"verified": 3.0})
        self.assertEqual(
            scored["method"], "numeric comparison against real output passed"
        )

    def test_module_that_does_not_parse_downgrades_every_line(self):
        stmts = [
            {"kind": "assignment", "source": "a = 1;", "python": "a = 1"},
            {"kind": "assignment", "source": "b = 2;", "python": "b = 2"},
        ]
        result = _result(total=2, functions=[_func("f", 2)])
        result["python"] = "def f():\n    a = 1\n    b = 2\n      bork\n"
        scored = accuracy(result)
        self.assertEqual(scored["score"], 0.0)
        self.assertEqual(scored["breakdown"]["unresolved"], 0.0)

    def test_whole_module_parses_keeps_full_score(self):
        stmts = [
            {"kind": "assignment", "source": "a = 1;", "python": "a = 1"},
            {"kind": "assignment", "source": "b = 2;", "python": "b = 2"},
        ]
        result = _result(total=2, functions=[_func("f", 2)])
        result["python"] = "a = 1\nb = 2\n"
        scored = accuracy(result)
        self.assertEqual(scored["score"], 100.0)
        self.assertEqual(scored["breakdown"], {"rulebook": 2.0})

    def test_rulebook_method_reported_when_no_numeric_verdict(self):
        func = _func("f", 3)
        result = _result(total=3, checker="inconclusive_no_matlab", functions=[func])
        scored = accuracy(result)
        self.assertEqual(scored["score"], 100.0)
        self.assertIn("rulebook matching", scored["method"])


class TestAccuracyIntegration(unittest.TestCase):
    def test_fft_basic_fully_resolved_scores_100(self):
        result = translate_file(sample_matlab("fft_basic.m"))
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertEqual(accuracy(result)["score"], 100.0)

    @mock.patch("assistant.draft_translation._call_ollama", return_value="""CODE
def beamform_basic(N, d, lamb, theta, theta0):
    return np.zeros_like(theta)
END CODE
CONFIDENCE
0.5
END CONFIDENCE
UNSURE
none
END UNSURE
""")
    def test_reverse_drafted_function_weighted(self, mock_call):
        result = translate_file(
            sample_python("beamform_basic_py.py"), direction=PYTHON_TO_MATLAB
        )
        scored = accuracy(result)
        self.assertGreater(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertIn("assistant", scored["breakdown"])
        self.assertLess(scored["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
