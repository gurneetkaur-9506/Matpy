import contextlib
import os
import tempfile
import unittest

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


@contextlib.contextmanager
def _matlab_file(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pipeline_test.m")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        yield path


class TestRulebookResilience(unittest.TestCase):
    def test_unresolvable_line_is_logged_and_pipeline_completes(self):
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


class TestReverseDirectionResilience(unittest.TestCase):
    def test_reverse_pipeline_completes_with_unresolved_lines(self):
        result = translate_file(
            sample_python("beamform_basic_py.py"),
            direction=PYTHON_TO_MATLAB,
        )
        self.assertNotEqual(result["status"], "error")
        report = build_translation_report(result)
        issues = [e["issue"] for e in report]
        self.assertIn("unresolved", issues)


if __name__ == "__main__":
    unittest.main()
