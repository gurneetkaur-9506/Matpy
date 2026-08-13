"""Wire tools/check_offline.py into the normal test suite.

The offline-import check must run with pytest, not only as a separate CI
step.  These tests drive the check function directly and exercise the
CLI through a subprocess, using only the stdlib plus the built-in
``tmp_path`` fixture.
"""

import subprocess
import sys

from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

TOOLS_DIR = REPO_ROOT / "tools"
CHECK_SCRIPT = TOOLS_DIR / "check_offline.py"


def _load_check():
    import importlib.util

    spec = importlib.util.spec_from_file_location("offline_check", CHECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, relpath, content):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def check_offline():
    return _load_check().check_offline


def test_repo_scan_finds_no_network_imports(check_offline):
    hits = check_offline(REPO_ROOT)
    assert hits == []


@pytest.mark.parametrize(
    "content,module",
    [
        ("import socket\n", "socket"),
        ("import urllib.request\n", "urllib.request"),
        ("from urllib.request import urlopen\n", "urllib.request"),
        ("from http.client import HTTPConnection\n", "http.client"),
        ("import requests\n", "requests"),
        ("import smtplib\n", "smtplib"),
        ("import ftplib\n", "ftplib"),
        ("import urllib3\n", "urllib3"),
    ],
)
def test_network_imports_are_flagged(tmp_path, check_offline, content, module):
    _write(tmp_path, "net.py", content)
    hits = check_offline(tmp_path)
    assert len(hits) == 1
    file, lineno, name = hits[0]
    assert Path(file).name == "net.py"
    assert lineno == 1
    assert name == module


def test_non_network_imports_pass(tmp_path, check_offline):
    _write(
        tmp_path,
        "safe.py",
        "import os\nimport ast\nfrom pathlib import Path\n"
        "from math import sqrt\nimport numpy as np\n",
    )
    assert check_offline(tmp_path) == []


def test_tests_directory_is_excluded(tmp_path, check_offline):
    _write(tmp_path, "ok.py", "import os\n")
    _write(tmp_path, "tests/evil.py", "import socket\n")
    assert check_offline(tmp_path) == []


@pytest.mark.parametrize(
    "relpath",
    ["venv/x.py", ".venv/x.py", "env/x.py", "build/x.py", "dist/x.py", "node_modules/x.py"],
)
def test_vendored_directories_are_excluded(tmp_path, check_offline, relpath):
    _write(tmp_path, relpath, "import socket\n")
    assert check_offline(tmp_path) == []


def test_cli_exits_zero_and_prints_passed():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "offline check passed" in result.stdout


def test_cli_exits_nonzero_and_reports_network_import(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "net.py").write_text("import socket\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "network imports found" in result.stdout
    assert "socket" in result.stdout
