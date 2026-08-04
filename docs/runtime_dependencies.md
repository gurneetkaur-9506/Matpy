# Runtime Dependencies Audit

**Conclusion: No block requires external (remote/internet) network access at
runtime. The pipeline is fully offline.** The only network-adjacent behavior in
the entire codebase is one optional loopback HTTP call to a *locally running*
Ollama instance from the Assistant block (details below). It never connects to
a remote host.

## Summary

| Block | Imports / runtime dependencies | Remote network | Loopback (localhost) |
|-------|--------------------------------|----------------|----------------------|
| Reader | stdlib (`dataclasses`, `pprint`), `tree-sitter`, `tree-sitter-matlab` | None | None |
| Rulebook | stdlib (`re`) only | None | None |
| Specialist Library | `numpy` only | None | None |
| Assistant | stdlib (`json`, `re`, `urllib.request`), local modules | None | Optional HTTP POST to `localhost:11434` (Ollama) |
| Checker | stdlib (`hashlib`, `re`, `importlib.util`, `inspect`, `os`, `sys`, `uuid`), `numpy` | None | None |
| UI | `PyQt5`, local modules | None | None |
| Top-level `translator.py` | `tree-sitter`, local modules, `tempfile` | None | None |

All third-party libraries (`tree-sitter`, `tree-sitter-matlab`, `numpy`,
`scipy`, `PyQt5`) are pure computation / parsing / GUI libraries. None of them
perform network I/O in the code paths this application uses.

## Block-by-block details

### Reader
- **Files:** `reader/load_matlab_file.py`, `reader/extract_structure.py`,
  `reader/structure.py`
- **Dependencies:** file I/O, `tree-sitter` + `tree-sitter-matlab` (parsing),
  `dataclasses`
- **Network:** none. Reads a MATLAB file from disk and builds a syntax tree in
  memory.

### Rulebook
- **Files:** `rulebook/builtin_rules.py`, `complex_rules.py`,
  `indexing_rules.py`, `operator_rules.py`, `translator.py`
- **Dependencies:** stdlib `re` (regular expressions) only
- **Network:** none. Pure string/pattern transformations.

### Specialist Library
- **Files:** `specialist_lib/array_factor.py`, `beamform.py`,
  `steering_vector.py`
- **Dependencies:** `numpy` only
- **Network:** none. Pure array math.

### Assistant
- **Files:** `assistant/orchestrator.py`, `assistant/draft_translation.py`
- **Dependencies:** stdlib `json`, `re`, `urllib.request`, Rulebook
- **Remote network:** none.
- **Loopback caveat:** `draft_translation.py` posts the prompt to
  `http://localhost:11434/api/generate` (`OLLAMA_ENDPOINT`), the default
  endpoint of a local Ollama server, to draft translations of *unresolved*
  MATLAB functions.
  - This is a **loopback-only** connection (never leaves the machine) and
    requires a local Ollama service to be running; it is *not* a remote/cloud
    dependency.
  - It is triggered **only** when a translated function contains at least one
    `UNRESOLVED` statement (`orchestrator.draft_unresolved_functions`). Scripts
    such as `fft_basic.m` contain no functions, so this path is never reached
    for them and no localhost call is made.
  - Tests mock `_call_ollama`, so the test suite never opens a socket.

### Checker
- **Files:** `checker/verify.py`, `run_matlab_mock.py`, `run_python.py`,
  `compare_outputs.py`
- **Dependencies:** stdlib `hashlib`, `re`, `importlib.util`, `inspect`, `os`,
  `sys`, `uuid`, and `numpy`
- **Network:** none. Loads the translated Python module in-process via
  `importlib`; the MATLAB side is simulated with seeded random arrays.

### UI
- **Files:** `ui/minimal_window.py`, `ui/translator_window.py`, `ui/summary.py`
- **Dependencies:** `PyQt5`, local modules
- **Network:** none. The PyQt5 widgets used (`QPlainTextEdit`, `QSplitter`,
  `QPushButton`, `QLabel`, `QMainWindow`) perform no network I/O. File save
  ("Save correction") is a local write to `reference_set/`.

### Top-level pipeline (`translator.py`)
- Orchestrates Reader -> Rulebook -> Assistant -> Checker. Uses `tempfile` to
  stage the generated Python file for verification.
- **Network:** none (inherits only the Assistant loopback caveat above).

## Why the packaged executable is offline

The PyInstaller bundle (`dist/matpy`, built from `matpy.spec`) contains only the
modules above plus the grammar data for `tree-sitter-matlab`. It performs no
remote requests at startup or during translation of script files; the sole
loopback Ollama call is gated behind the unresolved-function path and targets
`localhost` only.

## Verification method

- Grepped the repository for network primitives and indicators:
  `urllib`, `requests`, `urlopen`, `urlretrieve`, `http`, `socket`, `connect`,
  `subprocess`, `os.system`, `curl`, `wget`, `pip`, `apt`, `download`,
  hostnames, and endpoint strings.
- The only matches were in `assistant/draft_translation.py` (loopback Ollama)
  and tests that patch that call.
- Reviewed the import graph of every block and confirmed all runtime calls are
  local file I/O, in-memory computation, or in-process module loading.
