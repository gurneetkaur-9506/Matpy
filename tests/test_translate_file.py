import unittest
from unittest import mock

import numpy as np

from reader import PYTHON_TO_MATLAB
from tests.paths import sample_matlab, sample_python
from translator import translate_file, translate_source

FFT_MATLAB = sample_matlab("fft_basic.m")
INDEXING_PYTHON = sample_python("indexing_ops_py.py")


class TestTranslateFile(unittest.TestCase):
    def test_indexing_ops_sections(self):
        result = translate_file(sample_matlab("indexing_ops.m"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertEqual(
            result["sections"]["checker"]["status"], "inconclusive_no_matlab"
        )
        self.assertIn("import numpy as np", result["python"])

    def test_beamform_sections(self):
        result = translate_file(sample_matlab("beamform_basic.m"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["reader"]["functions"], ["beamform_basic"])
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertEqual(
            result["sections"]["checker"]["status"], "inconclusive_no_matlab"
        )

    def test_reader_error(self):
        result = translate_file("/no/such/file.m")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sections"]["reader"]["status"], "error")

    def test_checker_runs_when_inputs_provided(self):
        result = translate_file(
            sample_matlab("beamform_basic.m"),
            inputs={
                "N": 3,
                "d": 0.5,
                "lamb": 1.0,
                "theta": np.linspace(0, np.pi, 3),
                "theta0": 0.0,
            },
        )
        checker = result["sections"]["checker"]
        self.assertNotEqual(checker["status"], "skipped")
        self.assertIn(
            checker["status"],
            {"verified", "failed", "review needed", "inconclusive_no_matlab"},
        )

    @mock.patch("translator.matlab_engine_available", return_value=True)
    def test_checker_skipped_when_engine_available_no_inputs(self, mock_engine):
        result = translate_file(sample_matlab("indexing_ops.m"))
        self.assertEqual(result["sections"]["checker"]["status"], "skipped")
    @mock.patch("translator.verify", return_value="failed")
    @mock.patch("translator.matlab_engine_available", return_value=True)
    def test_checker_failed_preserved_when_engine_available(
        self, mock_engine, mock_verify
    ):
        result = translate_file(
            sample_matlab("beamform_basic.m"),
            inputs={
                "N": 3,
                "d": 0.5,
                "lamb": 1.0,
                "theta": np.linspace(0, np.pi, 3),
                "theta0": 0.0,
            },
        )
        self.assertEqual(result["sections"]["checker"]["status"], "failed")

    @mock.patch("translator.verify", return_value="failed")
    def test_checker_failed_maps_to_inconclusive_without_engine(self, mock_verify):
        result = translate_file(
            sample_matlab("beamform_basic.m"),
            inputs={
                "N": 3,
                "d": 0.5,
                "lamb": 1.0,
                "theta": np.linspace(0, np.pi, 3),
                "theta0": 0.0,
            },
        )
        self.assertEqual(
            result["sections"]["checker"]["status"], "inconclusive_no_matlab"
        )
        self.assertIn(
            "MATLAB engine", result["sections"]["checker"]["detail"]
        )

    def test_default_direction_is_matlab_to_python(self):
        result = translate_file(FFT_MATLAB)
        self.assertEqual(result["direction"], "matlab_to_python")
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertIn("import numpy as np", result["python"])

    def test_reverse_direction_uses_python_parser(self):
        result = translate_file(INDEXING_PYTHON, direction=PYTHON_TO_MATLAB)
        self.assertEqual(result["direction"], "python_to_matlab")
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertIn("A = [1 2 3; 4 5 6];", result["python"])
        self.assertIn("disp(A(1, 1));", result["python"])
        self.assertNotIn("import numpy as np", result["python"])

    def test_reverse_reader_errors_on_matlab_source(self):
        result = translate_file(FFT_MATLAB, direction=PYTHON_TO_MATLAB)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sections"]["reader"]["status"], "error")

    def test_reverse_outputs_matlab_from_beamform(self):
        result = translate_file(
            sample_python("beamform_basic_py.py"),
            direction=PYTHON_TO_MATLAB,
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["sections"]["rulebook"]["status"], "unresolved")
        self.assertGreater(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertIn("% UNRESOLVED: return af", result["python"])


    def test_forward_problem_lines_empty(self):
        result = translate_file(sample_matlab("indexing_ops.m"))
        self.assertEqual(result["problems"], [])

    def test_reverse_problem_lines_include_unresolved(self):
        result = translate_file(
            sample_python("beamform_basic_py.py"),
            direction=PYTHON_TO_MATLAB,
        )
        self.assertIn(
            "% UNRESOLVED: return af", result["python"]
        )
        unresolved_index = next(
            i
            for i, line in enumerate(result["python"].splitlines())
            if "UNRESOLVED: return af" in line
        )
        self.assertIn(unresolved_index, result["problems"])


class TestTranslateSource(unittest.TestCase):
    def test_forward_from_matlab_text(self):
        result = translate_source("fs = 1000;\nP1 = fft(x);\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["direction"], "matlab_to_python")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertIn("import numpy as np", result["python"])
        self.assertEqual(result["source"], "fs = 1000;\nP1 = fft(x);\n")

    def test_reverse_from_python_text(self):
        result = translate_source(
            "A = np.array([[1, 2], [3, 4]])\nprint(A[0, 0])\n",
            direction=PYTHON_TO_MATLAB,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["direction"], "python_to_matlab")
        self.assertIn("A = [1 2; 3 4];", result["python"])
        self.assertIn("disp(A(1, 1));", result["python"])

    def test_reader_error_on_invalid_source(self):
        result = translate_source("this is not matlab @@@", direction=PYTHON_TO_MATLAB)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sections"]["reader"]["status"], "error")

    def test_result_carries_source_and_file_label(self):
        result = translate_source("x = 1;", name="custom.m")
        self.assertEqual(result["file"], "custom.m")
        self.assertEqual(result["source"], "x = 1;")

    def test_default_file_label_for_python_direction(self):
        result = translate_source("x = 1", direction=PYTHON_TO_MATLAB)
        self.assertEqual(result["file"], "input.py")
        result = translate_source("x = 1;")
        self.assertEqual(result["file"], "input.m")

    def test_translate_file_delegates_with_source(self):
        result = translate_file(FFT_MATLAB)
        self.assertEqual(result["file"], FFT_MATLAB)
        self.assertIn("fs = 1000;", result["source"])


if __name__ == "__main__":
    unittest.main()
