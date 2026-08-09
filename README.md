# MATPY Translator

An architecture for translating MATLAB code to Python (and back). The system
is built from five cooperating blocks and is fully offline at runtime.

## Features

- **Bidirectional translation** — MATLAB -> Python (default) and
  Python -> MATLAB (reverse).
- **Rulebook-driven** — declarative mappings for indexing, loops, operators,
  builtins, plotting commands, and multi-output assignments.
- **Specialist library** — focused numpy/scipy idioms for array factor,
  beamforming, steering vectors, AWGN, chirp, convolution, and scan-file I/O.
- **Numeric verification** — the Checker cross-checks translated output
  against the original source using seeded inputs; conclusive when a real
  MATLAB engine is available.
- **Atomic-block safety** — a partially translated block construct is emitted
  as one fully-commented `UNRESOLVED` unit, so raw MATLAB syntax never leaks
  into the Python output.
- **Desktop UI** — a PyQt5 window with stage-by-stage status, an accuracy
  score, and a translation report.

## Requirements

- Python 3.8+
- Runtime dependencies: `tree-sitter`, `tree-sitter-matlab`, `numpy`, `scipy`
- UI only: `PyQt5`
- Optional: a local Ollama server (`localhost:11434`) to draft translations of
  unresolved functions, and a MATLAB engine for conclusive numeric checks

Install the core dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command line

Translate a MATLAB file to Python (defaults to `sample_matlab/indexing_ops.m`):

```bash
python run_translation.py [path/to/file.m]
```

### Library

```python
from translator import translate_file, translate_source
from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB

# From a file on disk
result = translate_file("sample_matlab/fft_basic.m")

# From a source string (no file needed)
result = translate_source("x = 1:10;", name="demo.m")

# Reverse direction: Python -> MATLAB
result = translate_source("import numpy as np\nx = np.arange(10)", direction=PYTHON_TO_MATLAB)
```

Each result contains the generated code (`python` or `matlab`), the extracted
`functions`, and a `sections` dict reporting the status of each pipeline stage
(`reader`, `rulebook`, `assistant`, `checker`).

### Desktop UI

```bash
python ui/minimal_window.py [path/to/source.m]
```

Use the direction combo to switch between MATLAB -> Python and
Python -> MATLAB, then press **Translate**. Open a file, review the stage
status markers, and use **Save correction** to write a validated
MATLAB/Python pair into `reference_set/`.

### Packaging

A PyInstaller spec is included for building a standalone executable:

```bash
pyinstaller matpy.spec
```

The bundle (`dist/matpy`) is fully offline; the only optional network call is
the loopback Ollama draft path. See `docs/runtime_dependencies.md` for the
audit.

## Architecture

### Reader

The Reader ingests MATLAB source files and parses them into a structured
intermediate representation. It relies on tree-sitter with the MATLAB grammar
to tokenize functions, scripts, and statements, producing a syntax tree that
downstream blocks can traverse. The Reader also captures file-level metadata
such as function signatures, comments, and line mappings so that translated
output can be traced back to its origin.

### Rulebook

The Rulebook holds the knowledge base of translation rules: declarative
mappings between MATLAB idioms and their Python equivalents. Each rule
describes a pattern to match, conditions under which it applies, and the
transformation to emit, covering areas such as indexing, loops, function
definitions, and built-in names. Rules are organized by topic so they can be
consulted independently or combined into larger transformations.

### Specialist Library

The Specialist Library is a collection of small, focused translation modules,
each expert in one domain such as matrix operations, signal processing, or
plotting. Specialists implement the trickier one-to-many conversions that a
single rule cannot express, producing Python that relies on numpy and scipy
idioms. They register with the Rulebook so that the Assistant can invoke them
when the matched pattern falls into their specialty.

### Assistant

The Assistant is the orchestrator. It walks the intermediate representation
produced by the Reader, applies matching rules and invokes the appropriate
Specialists, and assembles the final Python output. It tracks context such as
variable types and scope to keep the translation consistent across function
boundaries and to report warnings or unresolved constructs. When a function
contains unresolved statements, it can request a draft from a local Ollama
server, attaching a confidence score and uncertainty notes to the result.

### Checker

The Checker validates the generated Python before it is accepted. It
re-parses the output, performs structural and semantic checks, and optionally
cross-checks numeric behavior against the original MATLAB for supported
constructs. Findings are fed back to the Assistant to refine the translation,
closing the loop until the output is clean or the remaining issues are
reported to the user.

## Accuracy Scoring

The accuracy score reflects real correctness, not merely how many rules matched:

1. **Every line must parse.** Each generated statement and the whole generated
   module are validated with `ast.parse`; a line that does not parse counts as
   unresolved (weight 0.0).
2. **Numeric comparison where possible.** When the Checker runs a numeric
   cross-check against real output, its verdict drives the score: a
   `verified` verdict earns every resolved line full weight (1.0), while a
   `failed` verdict zeroes the resolved lines (0.0), even if rules matched.
3. **Fallback to provenance weights.** Without a conclusive numeric verdict
   (no MATLAB engine, no inputs, or an inconclusive result), resolved lines
   are weighted by source: rulebook lines 1.0, Assistant drafts by their
   reported confidence, unresolved lines 0.0.

The score is a percentage in 0-100. Each result reports the weighted
contribution per source (`breakdown`) and the `method` that produced the
score, so callers can tell whether it came from a numeric comparison or from
rulebook matching.

## Companion Modules

The project also includes a UI for driving the translation process and a
`reference_set` of paired MATLAB and Python examples used to develop and
validate rules. `reference_store.save_reference_entry` writes new pairs
without overwriting existing ones.

## Run with Docker

The app is a PyQt5 desktop GUI. Running it in Docker displays the window on your machine, so no MATLAB or Python installation is needed.

### Windows (one-time X server setup)

The container has no screen of its own, so Windows needs an X server to draw the window:

1. Install and start **VcXsrv** (XLaunch): choose "Multiple windows", Display number `0`, and on the "Extra settings" page tick **`-ac`** (disable access control).

### Every platform

Copy-paste these commands in order:

```
docker compose up
```

That's it — `docker compose up` builds the image and launches the GUI on your display. If you prefer plain Docker instead of Compose, run:

```
docker build -t matpy .
docker run --rm -e DISPLAY=host.docker.internal:0 matpy
```

On Linux, replace the `DISPLAY` value with your own (e.g. `:0`) and add `-v /tmp/.X11-unix:/tmp/.X11-unix:rw` to the `docker run` command.

## Project Layout

```
reader/           Parse MATLAB/Python into a structured IR (tree-sitter)
rulebook/         Declarative translation rules and the statement translator
specialist_lib/   Domain specialists (array_factor, beamform, awgn, ...)
assistant/        Orchestration and unresolved-function drafting
checker/          Validation and numeric cross-checking
ui/               PyQt5 translator window
reference_set/    Paired MATLAB/Python validation examples
sample_matlab/    Example MATLAB inputs
sample_matlab_real/  Larger, real-world MATLAB sources
sample_python/    Example Python outputs
docs/             Design and dependency documentation
tests/            Pytest suite
translator.py     Top-level pipeline (Reader -> Rulebook -> Assistant -> Checker)
run_translation.py  CLI entry point
reference_store.py  Reference-set writer
repo_paths.py     Central path helpers
matpy.spec        PyInstaller spec
```

## Testing

Run the test suite with pytest:

```bash
python -m pytest tests/
```

The suite covers translation rules (forward and reverse), indexing, operators,
specialists, the checker, and pipeline resilience. Tests mock the Ollama call,
so no local server is required.
