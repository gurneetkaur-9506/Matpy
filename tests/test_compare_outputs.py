import unittest

import numpy as np

from checker import compare_outputs


def _result(success, outputs):
    return {"success": success, "outputs": outputs}


class TestCompareOutputs(unittest.TestCase):
    def test_verified_when_close(self):
        matlab = _result(True, {"af": np.array([1.0, 2.0, 3.0])})
        python = _result(True, {"af": np.array([1.0, 2.0, 3.0])})
        self.assertEqual(compare_outputs(matlab, python), "verified")

    def test_verified_within_tolerance(self):
        matlab = _result(True, {"x": np.array([1.0, 2.0])})
        python = _result(True, {"x": np.array([1.0 + 1e-10, 2.0 - 1e-10])})
        self.assertEqual(compare_outputs(matlab, python), "verified")

    def test_failed_beyond_tolerance(self):
        matlab = _result(True, {"x": np.array([1.0, 2.0])})
        python = _result(True, {"x": np.array([1.0, 3.0])})
        self.assertEqual(compare_outputs(matlab, python), "failed")

    def test_failed_complex_values(self):
        matlab = _result(True, {"y": np.array([1 + 2j, 3 + 4j])})
        python = _result(True, {"y": np.array([1 + 2j, 9 + 4j])})
        self.assertEqual(compare_outputs(matlab, python), "failed")

    def test_verified_complex_values(self):
        matlab = _result(True, {"y": np.array([1 + 2j, 3 + 4j])})
        python = _result(True, {"y": np.array([1 + 2j, 3 + 4j])})
        self.assertEqual(compare_outputs(matlab, python), "verified")

    def test_review_needed_when_python_failed(self):
        matlab = _result(True, {"x": np.array([1.0])})
        python = _result(False, {})
        self.assertEqual(compare_outputs(matlab, python), "review needed")

    def test_review_needed_when_matlab_failed(self):
        matlab = _result(False, {})
        python = _result(True, {"x": np.array([1.0])})
        self.assertEqual(compare_outputs(matlab, python), "review needed")

    def test_review_needed_when_output_names_mismatch(self):
        matlab = _result(True, {"af": np.array([1.0])})
        python = _result(True, {"beamform_basic": np.array([1.0])})
        self.assertEqual(compare_outputs(matlab, python), "review needed")

    def test_review_needed_when_shape_mismatch(self):
        matlab = _result(True, {"x": np.array([1.0, 2.0])})
        python = _result(True, {"x": np.array([[1.0], [2.0]])})
        self.assertEqual(compare_outputs(matlab, python), "review needed")

    def test_review_needed_when_non_finite(self):
        matlab = _result(True, {"x": np.array([1.0, np.nan])})
        python = _result(True, {"x": np.array([1.0, np.nan])})
        self.assertEqual(compare_outputs(matlab, python), "review needed")

    def test_review_needed_when_no_outputs(self):
        matlab = _result(True, {})
        python = _result(True, {})
        self.assertEqual(compare_outputs(matlab, python), "review needed")


if __name__ == "__main__":
    unittest.main()
