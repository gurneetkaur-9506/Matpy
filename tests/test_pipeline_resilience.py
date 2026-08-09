import contextlib
import os
import tempfile
import unittest
from unittest import mock

from checker import build_translation_report
from reader import PYTHON_TO_MATLAB
from tests.paths import sample_python
from translator import translate_file

TWO_FUNCTION_MATLAB = """\
function out = broken_func(x)
    out = fftshift(sum(x));
end

function total = add_one(x)
    total = x + 1;
end
"""

TWO_UNRESOLVED_MATLAB = """\
function out = broken_func(x)
    out = fftshift(sum(x));
end

function y = shaky_func(a, b)
    y = blend(a + b);
end
"""

HIGH_CONFIDENCE_RESPONSE = """CODE
import numpy as np

def f(x):
    return np.asarray(x)
END CODE
CONFIDENCE
0.9
END CONFIDENCE
UNSURE
none
END UNSURE
"""


@contextlib.contextmanager
def _matlab_file(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pipeline_test.m")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        yield path


class TestRulebookResilience(unittest.TestCase):
    @mock.patch(
        "assistant.draft_translation._call_ollama",
        return_value=HIGH_CONFIDENCE_RESPONSE,
    )
    def test_unresolvable_line_is_logged_and_pipeline_completes(self, mock_call):
        with _matlab_file(TWO_FUNCTION_MATLAB) as path:
            result = translate_file(path)
            report = build_translation_report(result)
        self.assertNotEqual(result["status"], "error")
        self.assertIn("def add_one", result["python"])
        self.assertIn("def broken_func", result["python"])

        issues = [e["issue"] for e in report]
        self.assertIn("unresolved", issues)
        unresolved = next(e for e in report if e["issue"] == "unresolved")
        self.assertEqual(unresolved["source"], "out = fftshift(sum(x))")
        self.assertEqual(unresolved["line"], 2)


class TestAssistantResilience(unittest.TestCase):
    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=ConnectionError("assistant service unavailable"),
    )
    def test_assistant_error_on_one_function_does_not_halt_rest(self, mock_call):
        with _matlab_file(TWO_FUNCTION_MATLAB) as path:
            result = translate_file(path)
            report = build_translation_report(result)
        self.assertNotEqual(result["status"], "error")
        self.assertIn("def add_one", result["python"])
        self.assertIn("def broken_func", result["python"])

        functions = {f["name"]: f for f in result["functions"]}
        self.assertIn("draft_error", functions["broken_func"])
        self.assertNotIn("draft_error", functions["add_one"])
        self.assertEqual(result["sections"]["assistant"]["errors"], ["broken_func"])

        by_issue = {e["issue"]: e for e in report}
        self.assertIn("assistant error", by_issue)
        error_entry = by_issue["assistant error"]
        self.assertEqual(error_entry["source"], "broken_func")
        self.assertEqual(error_entry["line"], 1)
        self.assertEqual(error_entry["stage"], "assistant")
        self.assertIn("unresolved", by_issue)

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=RuntimeError("assistant service unavailable"),
    )
    def test_assistant_error_reason_is_plain_language(self, mock_call):
        with _matlab_file(TWO_FUNCTION_MATLAB) as path:
            result = translate_file(path)
            report = build_translation_report(result)
        reason = next(e["reason"] for e in report if e["issue"] == "assistant error")
        self.assertEqual(reason, "assistant service unavailable")
        self.assertNotIn("Traceback", reason)
        self.assertNotIn("raise ", reason)

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=ConnectionError("connection refused"),
    )
    def test_connection_error_is_labeled_ollama_unavailable(self, mock_call):
        with _matlab_file(TWO_FUNCTION_MATLAB) as path:
            result = translate_file(path)
            report = build_translation_report(result)
        reason = next(e["reason"] for e in report if e["issue"] == "assistant error")
        self.assertIn("Ollama not running", reason)
        self.assertIn("not a missing rule", reason)
        functions = {f["name"]: f for f in result["functions"]}
        self.assertIn("Ollama not running", functions["broken_func"]["draft_error"])

    def test_assistant_error_on_one_function_still_drafts_another(self):
        side_effect = [ConnectionError("first call failed"), HIGH_CONFIDENCE_RESPONSE]
        with _matlab_file(TWO_UNRESOLVED_MATLAB) as path:
            with mock.patch(
                "assistant.draft_translation._call_ollama", side_effect=side_effect
            ):
                result = translate_file(path)
                report = build_translation_report(result)
        self.assertNotEqual(result["status"], "error")
        functions = {f["name"]: f for f in result["functions"]}
        self.assertIn("draft_error", functions["broken_func"])
        self.assertIn("draft", functions["shaky_func"])
        self.assertIn("def shaky_func", result["python"])

        assistant_errors = {
            e["source"] for e in report if e["issue"] == "assistant error"
        }
        self.assertEqual(assistant_errors, {"broken_func"})


class TestReverseDirectionResilience(unittest.TestCase):
    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=RuntimeError("model timed out"),
    )
    def test_reverse_pipeline_completes_when_assistant_errors(self, mock_call):
        result = translate_file(
            sample_python("beamform_basic_py.py"),
            direction=PYTHON_TO_MATLAB,
        )
        self.assertNotEqual(result["status"], "error")
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertIn("assistant error", issues)
        reason = next(
            e["reason"] for e in report if e["issue"] == "assistant error"
        )
        self.assertEqual(reason, "model timed out")


if __name__ == "__main__":
    unittest.main()
