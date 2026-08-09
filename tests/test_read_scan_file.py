import os
import tempfile
import unittest

import numpy as np

from specialist_lib import format_spec_to_columns, read_matlab_scan_file


class _TempFileMixin:
    def _write(self, content, name="data.txt"):
        self._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def tearDown(self):
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()


class TestReadMatlabScanFileDtypes(_TempFileMixin, unittest.TestCase):
    def test_float_specifier(self):
        path = self._write("1.5\n2.5\n3.0\n")
        result = read_matlab_scan_file(path, "%f")
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, np.array([1.5, 2.5, 3.0]))

    def test_decimal_specifier(self):
        path = self._write("10\n-3\n42\n")
        result = read_matlab_scan_file(path, "%d")
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, np.array([10, -3, 42], dtype=np.int32))

    def test_hex_specifier(self):
        path = self._write("FF\n10\nA0\n")
        result = read_matlab_scan_file(path, "%x")
        self.assertEqual(result.dtype, np.uint32)
        np.testing.assert_array_equal(result, np.array([255, 16, 160], dtype=np.uint32))

    def test_string_specifier(self):
        path = self._write("alpha beta gamma\n")
        result = read_matlab_scan_file(path, "%s")
        self.assertEqual(result.dtype.kind, "U")
        np.testing.assert_array_equal(result, np.array(["alpha", "beta", "gamma"]))

    def test_float_scientific_notation(self):
        path = self._write("1e-3\n2.5e3\n-4E-2\n")
        result = read_matlab_scan_file(path, "%f")
        np.testing.assert_allclose(result, np.array([1e-3, 2.5e3, -4e-2]))


class TestReadMatlabScanFileCompound(_TempFileMixin, unittest.TestCase):
    def test_compound_numeric(self):
        path = self._write("1 2\n3 4\n5 6\n")
        result = read_matlab_scan_file(path, "%d %d")
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_compound_mixed_types(self):
        path = self._write("1 2.5\n3 4.5\n")
        result = read_matlab_scan_file(path, "%d %f")
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result[0][0], 1)
        self.assertAlmostEqual(result[0][1], 2.5)

    def test_format_spec_to_columns(self):
        self.assertEqual(format_spec_to_columns("%f"), ["f"])
        self.assertEqual(format_spec_to_columns("%d %x %s"), ["d", "x", "s"])
        self.assertEqual(format_spec_to_columns("no specifiers"), [])


class TestReadMatlabScanFileErrors(_TempFileMixin, unittest.TestCase):
    def test_no_specifier_raises(self):
        path = self._write("1 2 3\n")
        with self.assertRaises(ValueError):
            read_matlab_scan_file(path, "plain text")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_matlab_scan_file("/nonexistent/path/data.txt", "%f")


if __name__ == "__main__":
    unittest.main()
