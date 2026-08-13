import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from checker import verify
from tests.paths import reference_set, sample_matlab


class TestVerify(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_indexing_ops_script_not_verifiable(self):
        verdict = verify(
            sample_matlab("indexing_ops.m"),
            reference_set("indexing_ops.py"),
            {},
        )
        self.assertEqual(verdict, "review needed")

    def test_function_pair_failed_on_fake_values(self):
        matlab = self._write("mul.m", "function y = mul(a, b)\nend\n")
        python = self._write(
            "mul.py",
            "import numpy as np\n"
            "def mul(a, b):\n"
            "    return np.asarray(a) * np.asarray(b)\n",
        )
        verdict = verify(matlab, python, {"a": [2.0, 3.0], "b": [4.0, 5.0]})
        self.assertEqual(verdict, "failed")

    @mock.patch("checker.verify.run_matlab_mock")
    @mock.patch("checker.verify.run_python")
    def test_verified_verdict_propagates(self, mock_python, mock_matlab):
        m_path = self._write("a.m", "function y = f(a, b)\nend\n")
        py_path = self._write(
            "a.py", "def f(a, b):\n    return np.asarray(a) + np.asarray(b)\n"
        )
        mock_matlab.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0, 3.0])},
        }
        mock_python.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0, 3.0])},
        }
        verdict = verify(m_path, py_path, {"a": [1.0, 2.0], "b": [1.0, 1.0]})
        self.assertEqual(verdict, "verified")
        mock_python.assert_called_once()
        args, kwargs = mock_python.call_args
        self.assertEqual(kwargs["output_names"], ["y"])

    @mock.patch("checker.verify.run_matlab_mock")
    @mock.patch("checker.verify.run_python")
    def test_python_failure_gives_review_needed(self, mock_python, mock_matlab):
        m_path = self._write("a.m", "function y = f(a, b)\nend\n")
        py_path = self._write(
            "a.py", "def f(a, b):\n    return np.asarray(a) + np.asarray(b)\n"
        )
        mock_matlab.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0])},
        }
        mock_python.return_value = {"success": False, "outputs": {}}
        verdict = verify(m_path, py_path, {"a": [1.0], "b": [1.0]})
        self.assertEqual(verdict, "review needed")

    @mock.patch("checker.verify.run_matlab_mock")
    @mock.patch("checker.verify.run_python")
    def test_mock_path_binds_renamed_parameter_positionally(
        self, mock_python, mock_matlab
    ):
        # Regression test: a MATLAB keyword parameter renamed in the
        # translated Python (lambda -> lambda_) must reach run_python keyed
        # by its Python name, otherwise it reports a false "missing input".
        mock_matlab.return_value = {
            "success": True,
            "outputs": {"y": np.array([1.0, 1.0])},
        }
        mock_python.return_value = {
            "success": True,
            "outputs": {"y": np.array([1.0, 1.0])},
        }
        m_path = self._write("g.m", "function y = g(lambda, x)\nend\n")
        py_path = self._write(
            "g.py",
            "import numpy as np\n"
            "def g(lambda_, x):\n"
            "    return np.asarray(lambda_) + np.asarray(x)\n",
        )
        verdict = verify(m_path, py_path, {"lambda": [1.0], "x": [2.0]})
        self.assertEqual(verdict, "verified")

        args, _kwargs = mock_matlab.call_args
        matlab_inputs = args[1]
        self.assertIn("lambda", matlab_inputs)
        self.assertIn("x", matlab_inputs)
        self.assertNotIn("lambda_", matlab_inputs)

        args, _kwargs = mock_python.call_args
        python_inputs = args[1]
        self.assertIn("lambda_", python_inputs)
        self.assertIn("x", python_inputs)
        self.assertNotIn("lambda", python_inputs)

    def test_renamed_parameter_inputs_keyed_by_matlab_name(self):
        matlab = self._write(
            "scale.m",
            "function y = scale(lambda, x)\n"
            "    y = lambda .* x;\n"
            "end\n",
        )
        python = self._write(
            "scale.py",
            "import numpy as np\n"
            "def scale(lambda_, x):\n"
            "    return np.asarray(lambda_) * np.asarray(x)\n",
        )
        verdict = verify(
            matlab,
            python,
            {"lambda": np.array([2.0, 3.0]), "x": np.array([4.0, 5.0])},
        )
        # The renamed parameter must bind (no false "review needed"), so the
        # real computed output disagrees with the seeded mock -> "failed".
        self.assertEqual(verdict, "failed")

    def test_renamed_parameter_inputs_keyed_by_python_name(self):
        matlab = self._write(
            "scale2.m",
            "function y = scale2(lambda, x)\n"
            "    y = lambda .* x;\n"
            "end\n",
        )
        python = self._write(
            "scale2.py",
            "import numpy as np\n"
            "def scale2(lambda_, x):\n"
            "    return np.asarray(lambda_) * np.asarray(x)\n",
        )
        verdict = verify(
            matlab,
            python,
            {"lambda_": np.array([2.0, 3.0]), "x": np.array([4.0, 5.0])},
        )
        self.assertEqual(verdict, "failed")

    def test_normal_parameter_names_verify_positionally(self):
        matlab = self._write(
            "add.m", "function y = add(a, b)\n    y = a + b;\nend\n"
        )
        python = self._write(
            "add.py",
            "import numpy as np\n"
            "def add(a, b):\n"
            "    return np.asarray(a) + np.asarray(b)\n",
        )
        verdict = verify(matlab, python, {"a": [1.0, 2.0], "b": [3.0, 4.0]})
        self.assertEqual(verdict, "failed")


if __name__ == "__main__":
    unittest.main()
