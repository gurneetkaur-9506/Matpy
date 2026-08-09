#!/bin/sh
# Entrypoint for the MatPy container.
#
# Runs first-time setup checks before launching the GUI so that missing
# dependencies or a broken tree-sitter MATLAB grammar produce a clear,
# actionable error message instead of a silent crash at startup.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
APP_DIR="${APP_DIR:-$SCRIPT_DIR}"
APP_ENTRY="$APP_DIR/ui/translator_window.py"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "== MatPy entrypoint: running setup checks =="

# 1. Working directory must contain the project sources.
[ -f "$APP_ENTRY" ] || fail "project sources not found (expected '$APP_ENTRY'). Are you running the built image with the project at APP_DIR=$APP_DIR?"

# 2. Python and the pinned dependencies must be present.
python3 -c "import tree_sitter, numpy, scipy" 2>/dev/null \
    || fail "missing core dependency (tree_sitter / numpy / scipy). Check requirements.txt."
python3 -c "import PyQt5" 2>/dev/null \
    || fail "PyQt5 is not installed; the GUI cannot start."

# 3. The tree-sitter MATLAB grammar must build and produce a working parser.
python3 - <<'PY'
import sys
try:
    from tree_sitter import Language, Parser
    from tree_sitter_matlab import language
    parser = Parser(Language(language()))
    tree = parser.parse(b"x = 1;")
    if tree is None or tree.root_node is None:
        sys.exit(1)
    print("tree-sitter MATLAB grammar OK: parsed 'x = 1;'")
except ImportError as exc:
    print(f"missing import: {exc}", file=sys.stderr)
    sys.exit(2)
except Exception as exc:
    print(f"grammar failed to initialize: {exc}", file=sys.stderr)
    sys.exit(2)
PY
[ $? -eq 0 ] || fail "tree-sitter MATLAB grammar is not available/built. Reinstall tree-sitter-matlab."

echo "== MatPy entrypoint: all checks passed, launching GUI =="
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$APP_DIR"
exec python3 -m ui.translator_window "$@"
