# MATPY Translator

An architecture for translating MATLAB code to Python (and back). The system
is built from four cooperating blocks and is fully offline at runtime.

## Features

- **Bidirectional translation** — MATLAB -> Python (default) and
  Python -> MATLAB (reverse).
- **Rulebook-driven** — declarative mappings for indexing, loops, operators,
  builtins, plotting commands, and multi-output assignments.
- **Specialist library** — focused numpy/scipy idioms for array factor,
  beamforming, steering vectors, AWGN, chirp, convolution, and scan-file I/O.
- **Numeric verification** — the Checker compares translated output against
  a deterministic seeded reference derived from the original source and the
  supplied inputs, with no MATLAB runtime required.
- **Atomic-block safety** — a partially translated block construct is emitted
  as one fully-commented `UNRESOLVED` unit, so raw MATLAB syntax never leaks
  into the Python output.
- **Desktop UI** — a PyQt5 window with stage-by-stage status, an accuracy
  score, and a translation report.

## Requirements

- Python 3.8+
- Runtime dependencies: `tree-sitter`, `tree-sitter-matlab`, `numpy`, `scipy`
- UI only: `PyQt5`
- No MATLAB runtime is required; the Checker validates against a deterministic
  seeded mock reference.

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
(`reader`, `rulebook`, `checker`).

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

The bundle (`dist/matpy`) is fully offline. See `docs/runtime_dependencies.md`
for the audit.

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
each expert in one domain such as array processing, signal generation, or noise
modelling. Specialists implement the trickier one-to-many conversions that a
single rule cannot express, producing Python that relies on numpy and scipy
idioms.

Specialists register with the Rulebook: each domain function is an entry in the
Rulebook's builtin table (`rulebook/builtin_rules.py`), so when a MATLAB call
matches a specialist's domain it is translated to a `specialist_lib` call at
translation time rather than passed through or silently substituted with a
numpy/scipy equivalent. The wired mappings are:

| MATLAB call | specialist_lib function |
| --- | --- |
| `chirp(...)` | `specialist_lib.chirp` |
| `conv(...)` | `specialist_lib.conv` |
| `awgn(...)` / `comm.AWGNChannel(...)` | `specialist_lib.awgn` |
| `steervec(...)` | `specialist_lib.steering_vector` |
| `phased.ArrayResponse(...)` | `specialist_lib.array_factor` |
| `phased.Beamformer(...)` | `specialist_lib.beamform` |

The `fscanf` file-reading idiom is wired separately through the scan rules
(`rulebook/scan_rules.py`), which emit `specialist_lib.read_matlab_scan_file`.
Its companion `format_spec_to_columns` is an internal helper called by that
function and is never emitted directly by the Rulebook.

### Checker

The Checker validates the generated Python before it is accepted. It
re-parses the output, performs structural and semantic checks, and compares
the numeric behavior of the translated output against a reference derived
from the original source and the supplied inputs. Findings are reported to
the user when a function or line could not be resolved.

The reference comparison is two-tiered:

1. **Real MATLAB verification.** When the MATLAB Engine for Python
   (`matlab.engine`) is installed, the Checker runs the original `.m` source
   through a live MATLAB Engine session, converts the returned arrays back to
   numpy, runs the translated Python, and compares the two with
   `numpy.allclose`. MATLAB vectors are stored as 1xN / Nx1 matrices, so
   singleton dimensions are squeezed out before the comparison to match the
   1-D numpy result. The engine session is started lazily and always quit
   when verification finishes. A `failed` verdict in this mode is a genuine
   numeric mismatch.
2. **Seeded mock fallback.** Without MATLAB, the Checker falls back to a
   deterministic seeded mock reference derived from the original source and
   inputs (`run_matlab_mock`). The mock produces random-but-reproducible
   values, so it is **not proof of MATLAB equivalence**: a `failed`
   comparison against the mock is reported as the inconclusive
   `inconclusive_no_matlab` verdict rather than a real failure.

## Accuracy Scoring

The accuracy score reflects observable output consistency, not merely how many rules matched:

1. **Every line must parse.** Each generated statement and the whole generated
   module are validated with `ast.parse`; a line that does not parse counts as
   unresolved (weight 0.0).
2. **Reference comparison where possible.** When the Checker compares the
   translated output against the reference (a live MATLAB Engine when
   available, otherwise the seeded mock), its verdict drives the score: a
   `verified` verdict earns every resolved line full weight (1.0), while a
   `failed` verdict zeroes the resolved lines (0.0), even if rules matched.
   The mock-only `inconclusive_no_matlab` verdict is not treated as a real
   failure; it falls through to the provenance weights below.
3. **Fallback to provenance weights.** Without a numeric verdict (no inputs,
   or an inconclusive result), resolved lines are weighted by source:
   rulebook lines 1.0, unresolved lines 0.0.

The score is a percentage in 0-100. Each result reports the weighted
contribution per source (`breakdown`) and the `method` that produced the
score, so callers can tell whether it came from the reference comparison or
from rulebook matching.

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
checker/          Validation and reference comparison
ui/               PyQt5 translator window
reference_set/    Paired MATLAB/Python validation examples
sample_matlab/    Example MATLAB inputs
sample_matlab_real/  Larger, real-world MATLAB sources
sample_python/    Example Python outputs
docs/             Design and dependency documentation
tests/            Pytest suite
translator.py     Top-level pipeline (Reader -> Rulebook -> Checker)
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
specialists, the checker, and pipeline resilience. It runs fully offline with
no external server required.
