import unittest
from unittest import mock

import numpy as np

from reader import PYTHON_TO_MATLAB
from tests.paths import sample_matlab, sample_python
from translator import translate_file

FAKE_RESPONSE = """CODE
import numpy as np

def f(x):
    return np.sum(x)
END CODE
CONFIDENCE
0.6
END CONFIDENCE
UNSURE
- assumed x is 1-D
END UNSURE
"""

FFT_MATLAB = sample_matlab("fft_basic.m")
INDEXING_PYTHON = sample_python("indexing_ops_py.py")


class TestTranslateFile(unittest.TestCase):
    def test_indexing_ops_sections(self):
        result = translate_file(sample_matlab("indexing_ops.m"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["status"], "ok")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertEqual(result["sections"]["assistant"]["status"], "none")
        self.assertEqual(result["sections"]["checker"]["status"], "skipped")
        self.assertIn("import numpy as np", result["python"])

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_beamform_sections(self, mock_call):
        result = translate_file(sample_matlab("beamform_basic.m"))
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["sections"]["reader"]["status"], "ok")
        self.assertEqual(result["sections"]["reader"]["functions"], ["beamform_basic"])
        self.assertEqual(result["sections"]["rulebook"]["status"], "unresolved")
        self.assertEqual(result["sections"]["rulebook"]["unresolved"], 1)
        self.assertEqual(result["sections"]["assistant"]["status"], "drafted")
        self.assertEqual(result["sections"]["assistant"]["drafted"], ["beamform_basic"])
        self.assertEqual(result["sections"]["checker"]["status"], "skipped")

    def test_reader_error(self):
        result = translate_file("/no/such/file.m")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sections"]["reader"]["status"], "error")

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_checker_runs_when_inputs_provided(self, mock_call):
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
        self.assertIn(checker["status"], {"verified", "failed", "review needed"})

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

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_reverse_outputs_matlab_from_beamform(self, mock_call):
        result = translate_file(
            sample_python("beamform_basic_py.py"),
            direction=PYTHON_TO_MATLAB,
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["sections"]["rulebook"]["status"], "unresolved")
        self.assertGreater(result["sections"]["rulebook"]["unresolved"], 0)
        self.assertIn("% UNRESOLVED: n = np.arange(N)", result["python"])


if __name__ == "__main__":
    unittest.main()
