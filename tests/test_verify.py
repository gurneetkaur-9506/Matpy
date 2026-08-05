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
        mock_matlab.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0, 3.0])},
        }
        mock_python.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0, 3.0])},
        }
        verdict = verify("a.m", "a.py", {})
        self.assertEqual(verdict, "verified")
        mock_python.assert_called_once()
        args, kwargs = mock_python.call_args
        self.assertEqual(kwargs["output_names"], ["y"])

    @mock.patch("checker.verify.run_matlab_mock")
    @mock.patch("checker.verify.run_python")
    def test_python_failure_gives_review_needed(self, mock_python, mock_matlab):
        mock_matlab.return_value = {
            "success": True,
            "outputs": {"y": np.array([2.0])},
        }
        mock_python.return_value = {"success": False, "outputs": {}}
        verdict = verify("a.m", "a.py", {})
        self.assertEqual(verdict, "review needed")


if __name__ == "__main__":
    unittest.main()
