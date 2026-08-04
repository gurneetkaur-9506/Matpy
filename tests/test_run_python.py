import os
import tempfile
import unittest

import numpy as np

from checker import run_python


class TestRunPython(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_executes_function_and_captures_output(self):
        path = self._write(
            "add.py",
            "import numpy as np\n"
            "def add(a, b):\n"
            "    return np.asarray(a) + np.asarray(b)\n",
        )
        result = run_python(path, {"a": [1.0, 2.0], "b": [3.0, 4.0]})
        self.assertTrue(result["success"])
        self.assertEqual(result["function"], "add")
        np.testing.assert_array_equal(result["outputs"]["add"], [4.0, 6.0])

    def test_matches_function_by_file_stem(self):
        path = self._write(
            "compute.py",
            "import numpy as np\n"
            "def compute(x):\n"
            "    return np.sum(x)\n",
        )
        result = run_python(path, {"x": np.arange(5)})
        self.assertEqual(result["function"], "compute")
        self.assertAlmostEqual(result["outputs"]["compute"], 10.0)

    def test_missing_required_arg_fails(self):
        path = self._write("need.py", "def need(a, b):\n    return a + b\n")
        result = run_python(path, {"a": 1})
        self.assertFalse(result["success"])
        self.assertIn("missing input 'b'", result["notes"])

    def test_output_names_mapping(self):
        path = self._write("multi.py", "def multi():\n    return 1, 2\n")
        result = run_python(path, {}, output_names=["a", "b"])
        self.assertTrue(result["success"])
        self.assertEqual(result["outputs"]["a"], 1)
        self.assertEqual(result["outputs"]["b"], 2)

    def test_module_load_failure(self):
        path = self._write("bad.py", "raise ValueError('boom')\n")
        result = run_python(path, {})
        self.assertFalse(result["success"])
        self.assertIn("failed to load module", result["notes"][0])

    def test_no_function_in_module(self):
        path = self._write("data.py", "x = 1\n")
        result = run_python(path, {})
        self.assertFalse(result["success"])
        self.assertIn("no function found in module", result["notes"][0])

    def test_beamform_basic_reference(self):
        result = run_python(
            "reference_set/beamform_basic.py",
            {
                "N": 8,
                "d": 0.5,
                "lamb": 1.0,
                "theta": np.linspace(0, np.pi, 5),
                "theta0": 0.0,
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["function"], "beamform_basic")
        self.assertEqual(result["outputs"]["beamform_basic"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
