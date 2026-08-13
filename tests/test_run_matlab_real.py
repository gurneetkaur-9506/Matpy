import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from checker.run_matlab_real import (
    close_matlab_engine,
    matlab_engine_available,
    run_matlab_engine,
)
from checker.verify import verify


try:
    import matlab.engine  # noqa: F401

    HAS_REAL_MATLAB = True
except ImportError:
    HAS_REAL_MATLAB = False


class FakeDouble:
    """Minimal stand-in for matlab.double mlarrays."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)
        if self._data.ndim == 1:
            self._data = self._data.reshape(1, -1)

    @property
    def size(self):
        return self._data.shape

    def tolist(self):
        return self._data.tolist()

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._data, dtype=dtype)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class FakeComplex:
    """Minimal stand-in for matlab.complex mlarrays."""

    def __init__(self, real, imag):
        real_arr = np.asarray(real, dtype=float)
        imag_arr = np.asarray(imag, dtype=float)
        if real_arr.ndim == 1:
            real_arr = real_arr.reshape(1, -1)
        if imag_arr.ndim == 1:
            imag_arr = imag_arr.reshape(1, -1)
        self.real = FakeDouble(real_arr)
        self.imag = FakeDouble(imag_arr)

    @property
    def size(self):
        return self.real.size

    def tolist(self):
        return (self.real._data + 1j * self.imag._data).tolist()

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.real._data + 1j * self.imag._data, dtype=dtype)


class FakeEngine:
    def __init__(self):
        self.functions = {}
        self.quit_called = False

    def __getattr__(self, name):
        if name in self.functions:
            return self.functions[name]
        raise AttributeError(name)

    def quit(self):
        self.quit_called = True


def install_fake_matlab(engine):
    matlab = types.ModuleType("matlab")
    matlab.double = FakeDouble
    matlab.complex = FakeComplex
    sys.modules["matlab"] = matlab

    matlab_engine = types.ModuleType("matlab.engine")
    matlab_engine.start_matlab = lambda: engine
    sys.modules["matlab.engine"] = matlab_engine
    matlab.engine = matlab_engine


def _restore_modules(saved):
    close_matlab_engine()
    for name in ("matlab", "matlab.engine"):
        sys.modules.pop(name, None)
        if name in saved:
            sys.modules[name] = saved[name]


def _beamform(N, d, lamb, theta, theta0):
    theta_arr = np.asarray(theta, dtype=float)
    k = 2 * np.pi / lamb
    phase = k * d * (np.sin(theta_arr) - np.sin(theta0))
    total = np.zeros_like(theta_arr, dtype=complex)
    for n in range(1, int(N) + 1):
        total = total + np.exp(1j * (n - 1) * phase)
    return FakeComplex(total.real.tolist(), total.imag.tolist())


BEAMFORM_MATLAB = (
    "function af = beamform_basic(N, d, lambda, theta, theta0)\n"
    "    k = 2 * pi / lambda;\n"
    "    phase = k * d * (sin(theta) - sin(theta0));\n"
    "    af = zeros(size(theta));\n"
    "    for n = 1:N\n"
    "        af = af + exp(1i * (n - 1) * phase);\n"
    "    end\n"
    "end\n"
)

BEAMFORM_PYTHON = (
    "import numpy as np\n"
    "def beamform_basic(N, d, lamb, theta, theta0):\n"
    "    k = 2 * np.pi / lamb\n"
    "    phase = k * d * (np.sin(theta) - np.sin(theta0))\n"
    "    total = np.zeros_like(np.asarray(theta), dtype=complex)\n"
    "    for n in range(N):\n"
    "        total = total + np.exp(1j * n * phase)\n"
    "    return total\n"
)

BEAMFORM_INPUTS = {
    "N": 3,
    "d": 0.5,
    "lambda": 1.0,
    "theta": [0.0, 1.0, 2.0],
    "theta0": 0.0,
}


class TestRunMatlabEngine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self._saved = {
            name: sys.modules[name]
            for name in ("matlab", "matlab.engine")
            if name in sys.modules
        }
        self.addCleanup(_restore_modules, self._saved)

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _beamform_engine(self):
        engine = FakeEngine()
        engine.functions["beamform_basic"] = (
            lambda N, d, lamb, theta, theta0, nargout=1: _beamform(
                N, d, lamb, theta, theta0
            )
        )
        install_fake_matlab(engine)
        return engine

    def test_engine_available_tracks_fake_module(self):
        install_fake_matlab(FakeEngine())
        self.assertTrue(matlab_engine_available())
        close_matlab_engine()
        sys.modules.pop("matlab", None)
        sys.modules.pop("matlab.engine", None)
        self.assertFalse(matlab_engine_available())

    def test_verify_falls_back_to_mock_without_engine(self):
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        py_path = self._write("beamform_basic.py", BEAMFORM_PYTHON)
        python_keyed = {
            "N": BEAMFORM_INPUTS["N"],
            "d": BEAMFORM_INPUTS["d"],
            "lamb": BEAMFORM_INPUTS["lambda"],
            "theta": BEAMFORM_INPUTS["theta"],
            "theta0": BEAMFORM_INPUTS["theta0"],
        }
        with mock.patch("checker.verify.run_matlab_mock") as mock_run:
            mock_run.return_value = {
                "success": True,
                "outputs": {"af": np.array([1.0, 2.0, 3.0])},
            }
            verdict = verify(m_path, py_path, python_keyed, use_real_matlab=False)
        mock_run.assert_called_once()
        self.assertEqual(verdict, "failed")

    def test_run_matlab_engine_matching_outputs(self):
        engine = self._beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        result = run_matlab_engine(m_path, BEAMFORM_INPUTS)
        self.assertTrue(result["success"])
        self.assertEqual(result["source"], "matlab")
        self.assertIn("af", result["outputs"])
        af = result["outputs"]["af"]
        theta = np.asarray(BEAMFORM_INPUTS["theta"], dtype=float)
        phase = 2 * np.pi * 0.5 * np.sin(theta)
        expected = sum(np.exp(1j * n * phase) for n in range(3))
        self.assertEqual(af.shape, (3,))
        self.assertTrue(np.allclose(af, expected))
        self.assertTrue(engine.quit_called)

    def test_run_matlab_engine_missing_input_failure(self):
        self._beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        inputs = dict(BEAMFORM_INPUTS)
        del inputs["theta0"]
        result = run_matlab_engine(m_path, inputs)
        self.assertFalse(result["success"])
        self.assertTrue(
            any("missing input 'theta0'" in note for note in result["notes"])
        )

    def test_run_matlab_engine_execution_error(self):
        engine = FakeEngine()

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated MATLAB failure")

        engine.functions["beamform_basic"] = _boom
        install_fake_matlab(engine)
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        result = run_matlab_engine(m_path, BEAMFORM_INPUTS)
        self.assertFalse(result["success"])
        self.assertTrue(
            any("execution failed" in note for note in result["notes"])
        )
        self.assertTrue(engine.quit_called)

    def test_run_matlab_engine_unknown_function(self):
        engine = FakeEngine()
        install_fake_matlab(engine)
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        result = run_matlab_engine(m_path, BEAMFORM_INPUTS)
        self.assertFalse(result["success"])
        self.assertTrue(
            any("execution failed" in note for note in result["notes"])
        )

    def test_run_matlab_engine_parse_error(self):
        install_fake_matlab(FakeEngine())
        m_path = self._write("script.m", "x = 1;\ny = x + 2;\n")
        result = run_matlab_engine(m_path, {})
        self.assertFalse(result["success"])
        self.assertTrue(
            any("could not parse" in note for note in result["notes"])
        )


class TestVerifyWithRealEngine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self._saved = {
            name: sys.modules[name]
            for name in ("matlab", "matlab.engine")
            if name in sys.modules
        }
        self.addCleanup(_restore_modules, self._saved)

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _install_beamform_engine(self):
        engine = FakeEngine()
        engine.functions["beamform_basic"] = (
            lambda N, d, lamb, theta, theta0, nargout=1: _beamform(
                N, d, lamb, theta, theta0
            )
        )
        install_fake_matlab(engine)
        return engine

    def test_verify_real_engine_verified(self):
        self._install_beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        py_path = self._write("beamform_basic.py", BEAMFORM_PYTHON)
        with mock.patch("checker.verify.run_matlab_mock") as mock_run:
            verdict = verify(m_path, py_path, BEAMFORM_INPUTS)
        self.assertEqual(verdict, "verified")
        mock_run.assert_not_called()

    def test_verify_real_engine_mismatch_failed(self):
        self._install_beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        wrong_python = (
            "import numpy as np\n"
            "def beamform_basic(N, d, lamb, theta, theta0):\n"
            "    return np.asarray(theta, dtype=float) * 0.0\n"
        )
        py_path = self._write("beamform_basic.py", wrong_python)
        verdict = verify(m_path, py_path, BEAMFORM_INPUTS)
        self.assertEqual(verdict, "failed")

    def test_verify_real_engine_positional_remap(self):
        """Inputs keyed by Python names still reach the MATLAB function."""
        self._install_beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        py_path = self._write("beamform_basic.py", BEAMFORM_PYTHON)
        python_keyed = {
            "N": BEAMFORM_INPUTS["N"],
            "d": BEAMFORM_INPUTS["d"],
            "lamb": BEAMFORM_INPUTS["lambda"],
            "theta": BEAMFORM_INPUTS["theta"],
            "theta0": BEAMFORM_INPUTS["theta0"],
        }
        verdict = verify(m_path, py_path, python_keyed)
        self.assertEqual(verdict, "verified")

    def test_verify_real_engine_python_failure_review_needed(self):
        self._install_beamform_engine()
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        bad_python = (
            "def beamform_basic(N, d, lamb, theta, theta0):\n"
            "    raise RuntimeError('boom')\n"
        )
        py_path = self._write("beamform_basic.py", bad_python)
        verdict = verify(m_path, py_path, BEAMFORM_INPUTS)
        self.assertEqual(verdict, "review needed")


@unittest.skipUnless(HAS_REAL_MATLAB, "MATLAB Engine for Python is not installed")
class TestGenuineMatlabIntegration(unittest.TestCase):
    """Only runs on a machine with a real MATLAB Engine for Python.

    The translated Python produced by the rulebook uses the ``@`` operator
    for scalar multiplication (a pre-existing quirk), so this test proves the
    live engine runs and agrees with a *correct* Python implementation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_verify_beamform_against_real_matlab(self):
        m_path = self._write("beamform_basic.m", BEAMFORM_MATLAB)
        py_path = self._write("beamform_basic.py", BEAMFORM_PYTHON)
        verdict = verify(
            m_path,
            py_path,
            {
                "N": 3,
                "d": 0.5,
                "lambda": 1.0,
                "theta": [0.0, 1.0, 2.0],
                "theta0": 0.0,
            },
            use_real_matlab=True,
        )
        self.assertEqual(verdict, "verified")


if __name__ == "__main__":
    unittest.main()
