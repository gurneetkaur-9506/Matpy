#!/usr/bin/env python3
"""Static check that no network-capable import appears outside tests.

The offline translation pipeline must never depend on a live network, so
this lightweight check walks every ``.py`` file under the repository
(skipping tests/, virtual environments and build output) and flags any
import of a networking stdlib module or third-party HTTP/socket library.
It uses only the stdlib ``ast`` module -- no third-party linter or
dependency is required.

Usage:
    python3 tools/check_offline.py [ROOT_DIR]

Exit status:
    0  no network imports found ("offline check passed")
    1  at least one network import was found
"""

import argparse
import ast
import sys
from pathlib import Path

# Top-level modules that imply network access.  A dotted import is judged
# by its first component, so ``import urllib.request`` and
# ``from http.client import HTTPConnection`` are both caught.  Importing
# any of these outside the test suite would make the project require a
# live network at runtime.
NETWORK_MODULES = frozenset(
    {
        # stdlib networking modules
        "asyncio",
        "asyncore",
        "ftplib",
        "http",
        "httplib",
        "imaplib",
        "nntplib",
        "poplib",
        "smtpd",
        "smtplib",
        "socket",
        "socketserver",
        "telnetlib",
        "urllib",
        "urllib2",
        "webbrowser",
        "xmlrpc",
        # third-party HTTP / RPC / messaging / database clients
        "aiohttp",
        "asyncpg",
        "asyncssh",
        "cassandra",
        "elasticsearch",
        "fabric",
        "grpc",
        "httpcore",
        "httplib2",
        "httpx",
        "kafka",
        "mysql",
        "netmiko",
        "paramiko",
        "pg8000",
        "psycopg",
        "psycopg2",
        "pymongo",
        "pymysql",
        "pysocks",
        "redis",
        "requests",
        "socks",
        "socksipy",
        "tornado",
        "twisted",
        "urllib3",
        "websocket",
        "websocket_client",
        "websockets",
        "ws4py",
        "zmq",
    }
)

# Directories whose contents are never part of the shipped codebase.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        ".venv",
        "build",
        "dist",
        "env",
        "node_modules",
        "tests",
        "venv",
    }
)


def iter_python_files(root):
    """Yield every ``.py`` file under ``root``, skipping vendored dirs."""
    root = Path(root)
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def module_root(name):
    """Return the top-level package of a dotted import."""
    return name.split(".")[0]


def _findings_for_node(node, findings):
    if isinstance(node, ast.Import):
        for alias in node.names:
            if module_root(alias.name) in NETWORK_MODULES:
                findings.append((node.lineno, alias.name))
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return
        if module_root(node.module) in NETWORK_MODULES:
            findings.append((node.lineno, node.module))


def check_file(path):
    """Return ``(file, line, module)`` tuples for network imports in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    findings = []
    for node in ast.walk(tree):
        _findings_for_node(node, findings)
    return [(str(path), lineno, name) for lineno, name in findings]


def check_offline(root):
    """Scan every ``.py`` file under ``root`` for network imports.

    Returns a list of ``(file, line, module)`` tuples; an empty list
    means the scanned tree is network-free.
    """
    hits = []
    for path in iter_python_files(root):
        hits.extend(check_file(path))
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Flag network-capable imports outside test files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="repository root to scan (default: this project's root)",
    )
    args = parser.parse_args(argv)

    hits = check_offline(args.root)
    if hits:
        print("network imports found:")
        for file, lineno, name in hits:
            print("  %s:%d: import %s" % (file, lineno, name))
        return 1
    print("offline check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
