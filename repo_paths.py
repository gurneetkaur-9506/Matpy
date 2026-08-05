from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_MATLAB_DIR = REPO_ROOT / "sample_matlab"
SAMPLE_MATLAB_REAL_DIR = REPO_ROOT / "sample_matlab_real"
SAMPLE_PYTHON_DIR = REPO_ROOT / "sample_python"
REFERENCE_SET_DIR = REPO_ROOT / "reference_set"


def sample_matlab(name):
    return SAMPLE_MATLAB_DIR / name


def sample_matlab_real(name):
    return SAMPLE_MATLAB_REAL_DIR / name


def sample_python(name):
    return SAMPLE_PYTHON_DIR / name


def reference_set(name):
    return REFERENCE_SET_DIR / name
