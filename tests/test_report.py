import unittest

import ast
import numpy as np
import pytest

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


def _stmt(kind, source, unresolved=True, body=None):
    stmt = {"kind": kind, "source": source}
    if body is not None:
        stmt["body"] = body
    if unresolved:
        stmt["python"] = UNRESOLVED
    elif kind == "loop":
        stmt["python"] = "for _ in range(1):"
    else:
        stmt["python"] = "x = 1"
    return stmt


def _func(name, statements):
    func = {"name": name, "parameters": [], "outputs": [], "statements": statements}
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
        result = _result(statements=[_stmt("command", "widgetx on")])
        self.assertIn("widgetx on", build_translation_report(result)[0]["reason"])

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


class TestSyntaxErrorEntries(unittest.TestCase):
    def test_syntax_invalid_statement_reported(self):
        stmt = {
            "kind": "assignment",
            "source": "k = 2 * pi / lambda;",
            "python": "k = 2 * pi / lambda",
        }
        report = build_translation_report(_result(statements=[stmt]))
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["issue"], "syntax error")
        self.assertEqual(entry["stage"], "rulebook")
        self.assertEqual(entry["source"], "k = 2 * pi / lambda;")
        self.assertIn("does not parse", entry["reason"])
        self.assertNotIn("Traceback", entry["reason"])

    def test_loop_with_invalid_body_reported_for_loop_and_line(self):
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
        result = _result(functions=[_func("f", [loop])])
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertEqual(issues, ["syntax error", "syntax error"])
        sources = {e["source"] for e in report}
        self.assertEqual(sources, {"for n = 1:N", "af = af + exp(1i * (n - 1) * phase);"})

    def test_valid_statement_not_reported_as_syntax_error(self):
        stmt = {
            "kind": "assignment",
            "source": "y = x;",
            "python": "y = x",
        }
        self.assertEqual(build_translation_report(_result(statements=[stmt])), [])

    def test_syntax_check_skipped_in_reverse_direction(self):
        stmt = {
            "kind": "Assign",
            "source": "return af",
            "python": "return af",
            "matlab": "return af;",
        }
        result = {
            "file": None,
            "direction": "python_to_matlab",
            "functions": [],
            "statements": [stmt],
            "sections": {"checker": {"status": "skipped"}},
        }
        self.assertEqual(build_translation_report(result), [])


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

    def test_inconclusive_no_matlab_verdict_reported(self):
        result = _result(checker="inconclusive_no_matlab")
        report = build_translation_report(result)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["issue"], "inconclusive_no_matlab")
        self.assertEqual(entry["stage"], "checker")
        self.assertIsNone(entry["line"])
        self.assertIn("seeded mock", entry["reason"])

    def test_skipped_and_verified_not_reported(self):
        for status in ("skipped", "verified"):
            result = _result(checker=status)
            self.assertEqual(build_translation_report(result), [])


class TestReportWithBeamform(unittest.TestCase):
    def test_beamform_basic_forward_flags_no_syntax_errors(self):
        result = translate_file(BEAMFORM_MATLAB)
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertEqual(issues, ["inconclusive_no_matlab"])
        self.assertNotIn("syntax error", issues)

    def test_beamform_basic_output_parses(self):
        result = translate_file(BEAMFORM_MATLAB)
        ast.parse(result["python"])
        self.assertIn("lambda_", result["python"])
        code = "\n".join(
            line
            for line in result["python"].splitlines()
            if not line.strip().startswith("#")
        )
        self.assertNotIn("lambda", code.replace("lambda_", ""))

    def test_beamform_reserved_keyword_lambda_renamed(self):
        result = translate_file(BEAMFORM_MATLAB)
        self.assertIn(
            "def beamform_basic(N, d, lambda_, theta, theta0):", result["python"]
        )
        self.assertIn(
            "# renamed: MATLAB 'lambda' -> Python 'lambda_' (reserved keyword)",
            result["python"],
        )
        self.assertIn("k = 2 * np.pi / lambda_", result["python"])
        self.assertNotIn("k = 2 * pi / lambda\n", result["python"])
        syntax_sources = {e["source"] for e in build_translation_report(result)}
        self.assertNotIn("k = 2 * pi / lambda", syntax_sources)

    def test_beamform_reverse_report_has_unresolved_and_checker_verdict(self):
        result = translate_file(
            BEAMFORM_PYTHON,
            direction=PYTHON_TO_MATLAB,
            inputs=BEAMFORM_INPUTS,
        )
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertIn("unresolved", issues)
        self.assertIn("inconclusive_no_matlab", issues)

        by_issue = {e["issue"]: e for e in report}

        unresolved = [e for e in report if e["issue"] == "unresolved"]
        sources = {e["source"] for e in unresolved}
        self.assertIn("af = beamform_basic(N=8, d=0.5, lamb=1.0, theta=theta, theta0=0.0)", sources)
        self.assertIn("return af", sources)
        lines = {e["line"] for e in unresolved}
        self.assertIn(13, lines)
        self.assertIn(9, lines)

        self.assertIsNone(by_issue["inconclusive_no_matlab"]["line"])


@pytest.mark.parametrize(
    "kind,source,attempted,reason",
    [
        (
            "command",
            "widgetx on",
            "looked the command up in the rulebook's command table",
            "MATLAB command 'widgetx on' has no direct Python equivalent in the rulebook.",
        ),
        (
            "function_call",
            "fft(y)",
            "tried to map the call through the builtin, plot, and indexing rules",
            "The rulebook has no rule for the function 'fft', so the call was left for manual review.",
        ),
        (
            "assignment",
            "x = interp1(a, b)",
            "tried to translate the right-hand expression with the operator and builtin rules",
            "The expression 'interp1(a, b)' could not be reduced by the rulebook's operator or builtin rules.",
        ),
        (
            "loop",
            "for i = 1:N",
            "tried to convert the loop into a Python range() loop",
            "The loop 'for i = 1:N' does not fit the range pattern the rulebook translates.",
        ),
        (
            "return",
            "return af",
            "tried to reconstruct the return value with the reverse rules",
            "The return value in 'return af' could not be reconstructed by the reverse rules.",
        ),
    ],
)
def test_unresolved_plain_language_wording(kind, source, attempted, reason):
    """The exact plain-language 'attempted' and 'reason' texts are the
    contract for each unresolved statement kind."""
    entry = build_translation_report(_result(statements=[_stmt(kind, source)]))[0]
    assert entry["issue"] == "unresolved"
    assert entry["attempted"] == attempted
    assert entry["reason"] == reason


def test_function_call_without_name_reason_wording():
    entry = build_translation_report(
        _result(statements=[_stmt("function_call", "2 + 2")])
    )[0]
    assert entry["reason"] == "The call does not match any rulebook pattern."


@pytest.mark.parametrize(
    "status,reason",
    [
        (
            "failed",
            "The checker compared the reference and translated outputs and they disagreed beyond the allowed tolerance.",
        ),
        (
            "review needed",
            "The checker could not decide whether the outputs match: an execution failure, misaligned output names, shape mismatch, or non-finite values were involved.",
        ),
        (
            "inconclusive_no_matlab",
            "The checker could not reach a conclusive verdict because the reference is only a seeded mock rather than real MATLAB output; the comparison ran but its result is inconclusive.",
        ),
    ],
)
def test_checker_verdict_plain_language_wording(status, reason):
    """Checker verdicts carry a fixed plain-language explanation."""
    entry = build_translation_report(_result(checker=status))[0]
    assert entry["attempted"] == (
        "Compared the translated output against the reference numerically."
    )
    assert entry["reason"] == reason


if __name__ == "__main__":
    unittest.main()
