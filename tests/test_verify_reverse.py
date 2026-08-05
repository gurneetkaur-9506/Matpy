import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from checker import compare_outputs, verify
from reader import PYTHON_TO_MATLAB, load_structure
from rulebook import translate_with_rulebook_reverse
from tests.paths import sample_python

PY_SOURCE = sample_python("beamform_basic_py.py")

INPUTS = {
    "N": 8,
    "d": 0.5,
    "lamb": 1.0,
    "theta": np.linspace(0, np.pi, 5),
    "theta0": 0.0,
}


class TestVerifyReverseTranslation(unittest.TestCase):
    """The Checker is identical in either direction: verify() consumes a
    .m/.py pair regardless of which side was translated, so it must work
    unchanged on Python->MATLAB translation output."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _reverse_translated_pair(self):
        structure = load_structure(PY_SOURCE, PYTHON_TO_MATLAB)
        result = translate_with_rulebook_reverse(structure)
        func = result["functions"][0]
        self.assertEqual(func["name"], "beamform_basic")

        lines = ["function af = %s(N, d, lamb, theta, theta0)" % func["name"]]
        for stmt in func["statements"]:
            matlab = stmt.get("matlab")
            if matlab not in (None, "", "UNRESOLVED"):
                lines.append("    %s;" % matlab)
        lines.append("end")

        with open(PY_SOURCE, "r", encoding="utf-8") as f:
            py_source = f.read()

        m_path = self._write("beamform_basic_py.m", "\n".join(lines) + "\n")
        py_path = self._write("beamform_basic_py.py", py_source)
        return m_path, py_path

    def test_verify_runs_on_reverse_translation_output(self):
        m_path, py_path = self._reverse_translated_pair()
        verdict = verify(m_path, py_path, INPUTS)
        # Mock MATLAB produces deterministic-but-fake values, so the real
        # computed Python output disagrees numerically -> "failed", the
        # same structured outcome the forward direction produces.
        self.assertEqual(verdict, "failed")

    def test_compare_outputs_used_unchanged_on_reverse_output(self):
        m_path, py_path = self._reverse_translated_pair()
        with mock.patch(
            "checker.verify.compare_outputs", wraps=compare_outputs
        ) as spy:
            verdict = verify(m_path, py_path, INPUTS)

        spy.assert_called_once()
        args = spy.call_args[0]
        matlab_result, python_result = args[0], args[1]
        self.assertTrue(matlab_result["success"])
        self.assertTrue(python_result["success"])
        self.assertEqual(set(matlab_result["outputs"]), set(python_result["outputs"]))
        self.assertEqual(matlab_result["outputs"]["af"].shape, (5,))
        self.assertEqual(python_result["outputs"]["af"].shape, (5,))
        self.assertEqual(verdict, "failed")


if __name__ == "__main__":
    unittest.main()
