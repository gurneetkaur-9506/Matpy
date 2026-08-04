import os
import tempfile
import unittest

import numpy as np

from checker import run_matlab_mock


class TestRunMatlabMock(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_function_with_multiple_outputs(self):
        path = self._write("foo.m", "function [a, b] = foo(x, y)\nend\n")
        result = run_matlab_mock(path, {"x": np.ones((2, 3)), "y": 1.0})
        self.assertTrue(result["success"])
        self.assertEqual(result["function"], "foo")
        self.assertIn("a", result["outputs"])
        self.assertIn("b", result["outputs"])
        self.assertEqual(result["outputs"]["a"].shape, (2, 3))

    def test_single_output(self):
        path = self._write("bar.m", "function out = bar(n)\nend\n")
        result = run_matlab_mock(path, {"n": 5})
        self.assertEqual(result["outputs"]["out"].ndim, 0)

    def test_script_no_function(self):
        path = self._write("script.m", "x = 1;\n")
        result = run_matlab_mock(path, {})
        self.assertEqual(result["function"], None)
        self.assertIn("result", result["outputs"])

    def test_deterministic(self):
        path = self._write("baz.m", "function z = baz(a)\nend\n")
        inputs = {"a": np.arange(4.0)}
        r1 = run_matlab_mock(path, inputs)
        r2 = run_matlab_mock(path, inputs)
        np.testing.assert_array_equal(r1["outputs"]["z"], r2["outputs"]["z"])

    def test_values_are_numeric(self):
        path = self._write("num.m", "function v = num(q)\nend\n")
        result = run_matlab_mock(path, {"q": np.zeros(3)})
        self.assertTrue(np.issubdtype(result["outputs"]["v"].dtype, np.number))

    def test_complex_input_gives_complex_output(self):
        path = self._write("cpx.m", "function w = cpx(z)\nend\n")
        result = run_matlab_mock(path, {"z": np.array([1 + 2j, 3 - 1j])})
        self.assertTrue(np.iscomplexobj(result["outputs"]["w"]))

    def test_missing_input_noted(self):
        path = self._write("req.m", "function o = req(a, b)\nend\n")
        result = run_matlab_mock(path, {"a": 1.0})
        self.assertIn("missing input 'b'", result["notes"])

    def test_beamform_basic_structured_output(self):
        result = run_matlab_mock(
            "/sample_matlab/beamform_basic.m",
            {"N": 8, "d": 0.5, "lambda": 1.0, "theta": np.linspace(0, np.pi, 91), "theta0": 0.0},
        )
        self.assertEqual(result["function"], "beamform_basic")
        self.assertEqual(result["outputs"]["af"].shape, (91,))


if __name__ == "__main__":
    unittest.main()
