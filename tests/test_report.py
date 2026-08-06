import unittest
from unittest import mock

import numpy as np

from checker import build_translation_report
from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED
from tests.paths import sample_matlab, sample_python
from translator import translate_file

BEAMFORM_MATLAB = sample_matlab("beamform_basic.m")
BEAMFORM_PYTHON = sample_python("beamform_basic_py.py")

BEAMFORM_INPUTS = {
    "N": 3,
    "d": 0.5,
    "lamb": 1.0,
    "theta": np.linspace(0, np.pi, 3),
    "theta0": 0.0,
}

LOW_CONFIDENCE_RESPONSE = """CODE
function af = beamform_basic(N, d, lamb, theta, theta0)
    af = zeros(size(theta));
end
END CODE
CONFIDENCE
0.4
END CONFIDENCE
UNSURE
- assumed phase expansion broadcasts
END UNSURE
"""


def _stmt(kind, source, unresolved=True, body=None):
    stmt = {"kind": kind, "source": source}
    if body is not None:
        stmt["body"] = body
    stmt["python"] = UNRESOLVED if unresolved else "%s -> ok" % source
    return stmt


def _func(name, statements, confidence=None, notes=None):
    func = {"name": name, "parameters": [], "outputs": [], "statements": statements}
    if confidence is not None:
        func["draft"] = {
            "code": "draft code",
            "confidence": confidence,
            "notes": notes or [],
        }
    return func


def _result(functions=None, statements=None, checker="skipped", file=None):
    return {
        "file": file,
        "direction": "matlab_to_python",
        "status": "ok",
        "python": "",
        "functions": functions or [],
        "statements": statements or [],
        "sections": {"checker": {"status": checker}},
    }


class TestBuildTranslationReport(unittest.TestCase):
    def test_clean_result_returns_empty(self):
        result = _result(
            functions=[
                _func("f", [_stmt("assignment", "x = 1", unresolved=False)])
            ]
        )
        self.assertEqual(build_translation_report(result), [])

    def test_empty_result_returns_empty(self):
        self.assertEqual(build_translation_report({}), [])

    def test_collects_unresolved_script_statement(self):
        result = _result(statements=[_stmt("assignment", "x = fft(y)")])
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["issue"], "unresolved")
        self.assertEqual(entry["stage"], "rulebook")
        self.assertEqual(entry["source"], "x = fft(y)")
        self.assertIsNone(entry["line"])

    def test_unresolved_entry_uses_plain_language(self):
        result = _result(statements=[_stmt("function_call", "fft(y)")])
        entry = build_translation_report(result)[0]
        self.assertTrue(entry["attempted"])
        self.assertTrue(entry["reason"])
        self.assertNotIn("Traceback", entry["reason"])
        self.assertNotIn("raise ", entry["reason"])

    def test_reason_names_unknown_function(self):
        result = _result(statements=[_stmt("function_call", "fft(y)")])
        self.assertIn("fft", build_translation_report(result)[0]["reason"])

    def test_reason_for_unknown_command(self):
        result = _result(statements=[_stmt("command", "hold on")])
        self.assertIn("hold on", build_translation_report(result)[0]["reason"])

    def test_reason_for_untranslatable_assignment(self):
        result = _result(statements=[_stmt("assignment", "x = interp1(a, b)")])
        self.assertIn("interp1(a, b)", build_translation_report(result)[0]["reason"])

    def test_nested_loop_body_unresolved_collected(self):
        body = [_stmt("assignment", "y = conv(x, h)")]
        loop = _stmt("loop", "for i = 1:N", unresolved=False, body=body)
        result = _result(functions=[_func("f", [loop])])
        report = build_translation_report(result)
        self.assertEqual([e["issue"] for e in report], ["unresolved"])
        self.assertEqual(report[0]["source"], "y = conv(x, h)")

    def test_unresolved_loop_itself_collected(self):
        loop = _stmt("loop", "for i = 1:N")
        result = _result(functions=[_func("f", [loop])])
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        self.assertIn("for i = 1:N", report[0]["reason"])

    def test_reverse_unresolved_matlab_key_detected(self):
        stmt = {"kind": "Assign", "source": "n = np.arange(N)", "matlab": UNRESOLVED}
        result = {
            "file": None,
            "direction": "python_to_matlab",
            "functions": [_func("beamform_basic", [stmt])],
            "statements": [],
            "sections": {"checker": {"status": "skipped"}},
        }
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["issue"], "unresolved")
        self.assertEqual(report[0]["source"], "n = np.arange(N)")


class TestLineNumbers(unittest.TestCase):
    def test_line_number_located_in_original_source(self):
        stmt = _stmt("loop", "for n = 1:N")
        result = _result(statements=[stmt], file=BEAMFORM_MATLAB)
        self.assertEqual(build_translation_report(result)[0]["line"], 5)

    def test_beamform_basic_lines_are_located(self):
        statements = [
            _stmt("assignment", "k = 2 * pi / lambda"),
            _stmt("function_call", "fft(theta)"),
        ]
        result = _result(
            functions=[_func("beamform_basic", statements)], file=BEAMFORM_MATLAB
        )
        report = build_translation_report(result)
        self.assertEqual([e["line"] for e in report], [2, None])

    def test_line_none_when_source_not_in_file(self):
        stmt = _stmt("assignment", "z = nowhere(v)")
        result = _result(statements=[stmt], file=BEAMFORM_MATLAB)
        self.assertIsNone(build_translation_report(result)[0]["line"])

    def test_line_none_without_source_file(self):
        result = _result(statements=[_stmt("assignment", "x = fft(y)")])
        self.assertIsNone(build_translation_report(result)[0]["line"])


class TestAssistantFlags(unittest.TestCase):
    def test_low_confidence_draft_flagged_with_notes(self):
        func = _func(
            "beamform_basic",
            [],
            confidence=0.4,
            notes=["uncertainty flagged: assumed phase expansion broadcasts"],
        )
        result = _result(functions=[func], file=BEAMFORM_PYTHON)
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["issue"], "low confidence")
        self.assertEqual(entry["stage"], "assistant")
        self.assertEqual(entry["source"], "beamform_basic")
        self.assertEqual(entry["line"], 4)
        self.assertIn("assumed phase expansion broadcasts", entry["reason"])
        self.assertIn("beamform_basic", entry["attempted"])

    def test_low_confidence_without_notes_uses_fallback(self):
        func = _func("f", [], confidence=0.2, notes=[])
        result = _result(functions=[func])
        entry = build_translation_report(result)[0]
        self.assertIn("confidence", entry["reason"])

    def test_high_confidence_draft_not_flagged(self):
        func = _func("f", [], confidence=0.9, notes=[])
        result = _result(functions=[func])
        self.assertEqual(build_translation_report(result), [])

    def test_confidence_just_below_threshold_flagged(self):
        func = _func("f", [], confidence=0.49, notes=[])
        result = _result(functions=[func])
        self.assertEqual(len(build_translation_report(result)), 1)

    def test_confidence_at_threshold_not_flagged(self):
        func = _func("f", [], confidence=0.5, notes=[])
        result = _result(functions=[func])
        self.assertEqual(build_translation_report(result), [])


class TestCheckerVerdicts(unittest.TestCase):
    def test_failed_verdict_reported(self):
        result = _result(checker="failed")
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["issue"], "failed")
        self.assertEqual(entry["stage"], "checker")
        self.assertIsNone(entry["line"])
        self.assertIn("disagreed", entry["reason"])

    def test_review_needed_verdict_reported(self):
        result = _result(checker="review needed")
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["issue"], "review needed")
        self.assertIn("could not decide", report[0]["reason"])

    def test_skipped_and_verified_not_reported(self):
        for status in ("skipped", "verified"):
            result = _result(checker=status)
            self.assertEqual(build_translation_report(result), [])


class TestReportWithBeamform(unittest.TestCase):
    def test_beamform_basic_forward_is_fully_resolved(self):
        result = translate_file(BEAMFORM_MATLAB)
        self.assertEqual(build_translation_report(result), [])

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        return_value=LOW_CONFIDENCE_RESPONSE,
    )
    def test_beamform_reverse_report_has_all_issue_kinds(self, mock_call):
        result = translate_file(
            BEAMFORM_PYTHON,
            direction=PYTHON_TO_MATLAB,
            inputs=BEAMFORM_INPUTS,
        )
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertIn("unresolved", issues)
        self.assertIn("low confidence", issues)
        self.assertIn("failed", issues)

        by_issue = {e["issue"]: e for e in report}
        low = by_issue["low confidence"]
        self.assertEqual(low["line"], 4)
        self.assertEqual(low["source"], "beamform_basic")
        self.assertIn("assumed phase expansion broadcasts", low["reason"])

        unresolved = [e for e in report if e["issue"] == "unresolved"]
        sources = {e["source"] for e in unresolved}
        self.assertIn("n = np.arange(N)", sources)
        self.assertIn("return af", sources)
        lines = {e["line"] for e in unresolved}
        self.assertIn(7, lines)
        self.assertIn(9, lines)

        self.assertIsNone(by_issue["failed"]["line"])


if __name__ == "__main__":
    unittest.main()
