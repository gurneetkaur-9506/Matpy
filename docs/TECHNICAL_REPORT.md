# MATPY — Final Technical Report

Source of truth: the actual implementation in the repository, inspected once.
README content is used only where it matches the code.

---

## 1. Project Summary

MATPY is a **fully offline, bidirectional MATLAB <-> Python translator**.
It parses MATLAB (or Python) source into a structured intermediate
representation, applies a rulebook of declarative translation rules, and
emits Python (or MATLAB) plus an accuracy score, advisory warnings, and
advisory symbolic insights.

Purpose and features:

- MATLAB -> Python translation (default) and Python -> MATLAB (reverse).
- Rulebook-driven translation: indexing, loops, operators, builtins,
  plotting commands, format strings (`fprintf`), file I/O idioms
  (`fopen`/`fscanf`/`while ~feof`), and multi-output assignments.
- A specialist library (`specialist_lib/`) holding MATLAB-toolbox-faithful
  numpy/scipy equivalents (array factor, beamforming, steering vectors,
  AWGN, chirp, convolution, scan-file I/O, signal-processing wrappers).
- A numeric Checker that compares translated output against a
  **deterministic seeded mock reference** (internal-consistency check, not
  MATLAB-fidelity proof). A live MATLAB Engine backend exists but is only
  used when `matlab.engine` is importable, which is not bundled.
- Static offline advisory stages: `validation` (undefined variables,
  unsupported constructs, leftover operators, unresolved functions,
  shadowing) and `symbolic` (constant detection, simplification, math
  reasoning).
- Atomic-block safety: partially translated block constructs are collapsed
  into a single fully-commented `UNRESOLVED` unit so raw MATLAB syntax never
  leaks into the Python output.
- A PyQt5 desktop GUI showing per-stage status markers, an accuracy score,
  a plain-language report, and problem-line highlighting.

Inputs / outputs:

- **Input**: MATLAB `.m` source (or Python source for reverse), optionally
  with a dict of argument values.
- **Output**: translated Python (or MATLAB) code, extracted function
  metadata, per-statement provenance/translations, per-stage status
  (`sections`), a numeric accuracy score (0-100), and a list of
  issues/warnings (`build_translation_report`).

Current status: all tests pass (`1104 passed, 1 skipped, 97 subtests`),
all work is committed and pushed to `origin/main` (HEAD `e8647e5`).

---

## 2. Complete Folder / File Structure

```
repo root
├── translator.py              Top-level pipeline (Reader -> Rulebook -> Checker)
├── run_translation.py         CLI entry point
├── reference_store.py         Writes paired reference_set entries (never overwrites)
├── repo_paths.py               Central path helpers
├── requirements.txt            Pinned dependencies
├── matpy.spec                  PyInstaller spec (standalone bundle)
├── Dockerfile                  PyQt5 GUI container image
├── docker-compose.yml          Runs the GUI on the host display
├── entrypoint.sh               Container startup checks + GUI launch
├── README.md                   Usage / architecture documentation
├── reader/                     Parse MATLAB/Python into a structured IR
│   ├── __init__.py             Public exports
│   ├── extract_structure.py    tree-sitter/AST -> refs, loops, function bodies
│   ├── load_matlab_file.py     Read a .m file
│   ├── load_python_file.py     Read + ast.dump a .py file (dev helper)
│   ├── load_structure.py       Parse source -> Structure; direction constants
│   └── structure.py            IR dataclasses + build_structure
├── rulebook/                   Declarative rules + statement translator
│   ├── __init__.py             Public exports
│   ├── translator.py           Core expression/statement/loop translation (both directions)
│   ├── builtin_rules.py        BUILTIN_RULES table + apply (fwd/rev)
│   ├── keyword_rules.py        Reserved-word renaming (lambda -> lambda_)
│   ├── indexing_rules.py       Index shifting / slicing (fwd/rev)
│   ├── index_shift.py          Shared +-1 shift primitive
│   ├── operator_rules.py       Operator translation incl. matrix vs element-wise
│   ├── complex_rules.py        MATLAB imaginary literal 3i -> 3j
│   ├── format_rules.py         fprintf -> print conversion
│   ├── multi_output_rules.py   [a,b]=func(...) decomposition registry
│   ├── scan_rules.py           fopen/fscanf/while~feof idiom recognition
│   ├── sequence_rules.py       np.arange/linspace/zeros/ones -> MATLAB (reverse)
│   ├── attribute_rules.py      obj.shape/T/dtype -> MATLAB (reverse)
│   ├── shape_inference.py      Offline scalar/vector/matrix inference pass
│   └── symbolic.py             Offline advisory symbolic analysis stage
├── specialist_lib/             MATLAB-toolbox-faithful numpy/scipy equivalents
│   ├── __init__.py             Exports (all specialists + signal wrappers)
│   ├── array_factor.py         Uniform-linear-array array factor
│   ├── beamform.py             phased.Beamformer replacement
│   ├── steering_vector.py      steervec replacement
│   ├── awgn.py                 Communications Toolbox awgn
│   ├── chirp.py                Signal Processing chirp
│   ├── conv.py                 conv with full/same/valid
│   ├── read_scan_file.py       fscanf file-reading helper
│   └── signal_tools.py         scipy-backed wrappers (square/findpeaks/xcorr/
│                               detrend/medfilt1/filter_with_state/freqz)
├── checker/                    Validation, scoring, reference comparison
│   ├── __init__.py             Public exports
│   ├── verify.py               Compare translated Python vs reference
│   ├── run_matlab_mock.py      Seeded mock reference (the default)
│   ├── run_matlab_real.py      Optional live MATLAB Engine backend
│   ├── run_python.py           Load+run the translated module
│   ├── compare_outputs.py      numpy.allclose-based verdict
│   ├── accuracy.py             Accuracy scoring (0-100)
│   ├── validate.py             Advisory static validation warnings
│   └── report.py               Plain-language issue report
├── ui/                         PyQt5 desktop interface
│   ├── __init__.py
│   ├── minimal_window.py       MinimalTranslatorWindow
│   ├── translator_window.py    TranslatorWindow (entrypoint target)
│   ├── summary.py              Status/accuracy text helpers
│   └── highlight.py            Problem-line syntax highlighter
├── reference_set/              Paired MATLAB/Python validation examples
│   ├── __init__.py
│   ├── beamform_basic.py
│   └── indexing_ops.py
├── sample_matlab/              Example MATLAB inputs
│   ├── beamform_basic.m
│   ├── fft_basic.m
│   ├── indexing_ops.m
│   └── shape_inference.m
├── sample_matlab_real/         Larger real-world MATLAB source
│   └── atlasDisplay.m
├── sample_python/              Example Python outputs
│   ├── beamform_basic_py.py
│   ├── fft_basic_py.py
│   └── indexing_ops_py.py
├── docs/
│   ├── runtime_dependencies.md Offline runtime audit
│   └── TECHNICAL_REPORT.md     This report
├── tools/
│   └── check_offline.py        Static scan for network imports
└── tests/                      47 pytest modules (see section 9)
```

---

## 3. File & Function Analysis

### 3.1 `translator.py` (top-level pipeline)

- `translate_file(matlab_path, inputs=None, tolerance=1e-8, direction=...)`
  reads a file and delegates to `translate_source`. On `OSError` it returns
  an `error` result with a `reader` section describing the failure.
- `translate_source(source, inputs, tolerance, direction, name)` — the
  central orchestrator:
  1. Parse via `_parse` -> `load_structure_from_source`.
  2. Record `reader` section (function names, statement count).
  3. Reverse direction: `translate_with_rulebook_reverse`. Forward:
     `infer_shapes` (record `inference` section) then
     `translate_with_rulebook(structure, shapes=inference)`.
  4. Compute `unresolved` count over all statements + function bodies;
     set `rulebook` section and possibly `status="unresolved"`.
  5. Emit code via `code_for_result` / `code_for_result_reverse`; collect
     `problems` (line indices of UNRESOLVED output lines).
  6. Forward: run `validate_translation` -> `validation` section; run
     `analyze_translation` -> `symbolic` section. Reverse: both `skipped`.
  7. If no inputs: `checker` is `skipped` (engine present) or
     `inconclusive_no_matlab`. Otherwise stage MATLAB + Python in a
     tempdir and call `verify`; a `failed` verdict without a live engine is
     downgraded to `inconclusive_no_matlab`.
- `matlab_engine_available()` — lazy probe of `matlab.engine`.
- `_emit_block` / `_emit_function` — walk translated statement dicts into
  final Python lines (recursing into loop bodies, skipping empty commands,
  emitting UNRESOLVED blocks as fully-commented atomic units).
- `_code_references(statements, functions, token)` — decides whether the
  generated header needs `import scipy.signal` / `import specialist_lib`.
- `code_for_result` / `code_for_result_reverse` — assemble import header +
  script statements + functions (or reverse MATLAB output).

### 3.2 `reader/`

- `load_structure.py`: `MATLAB_TO_PYTHON` / `PYTHON_TO_MATLAB` direction
  constants; `join_line_continuations` merges `...`-continued lines;
  `load_structure_from_source` parses MATLAB via tree-sitter or Python via
  `ast`; `load_structure(path, direction)` file variant.
- `structure.py`: IR dataclasses `IndexExpr`, `PlainRef`, `Loop`, `Statement`,
  `Function`, `Structure`; `build_structure` converts a parse tree (tree-sitter
  tree or `ast.AST`) into the IR, extracting function name/params/outputs,
  statements (loops as `Loop` with nested statements), and refs.
- `extract_structure.py`: `split_top_level`, `split_range`, `is_range`;
  `_collect_refs` walks tree-sitter nodes to collect plain/index refs;
  `_collect_loops`; Python-side equivalents (`_py_collect_refs`,
  `_extract_python`); `extract_structure` top-level entry.
- `load_matlab_file.py`: reads a file (trivial). `load_python_file.py`:
  parses and `ast.dump`s (dev/debug helper, prints to stdout).

### 3.3 `rulebook/`

- `translator.py` (core, ~1890 lines). Key functions:
  - `_translate_expr(expr, scalars, declared)` — recursive expression
    translator: matrices, strings, ranges, calls (disp/fprintf/fclose/
    numpy calls/reductions/degree trig/fft/diff/var/std/builtins/plot/
    length/numel/find/interp1/imagesc/surf/view), dotted calls, operators,
    logical NOT, transpose, constants (`pi`, `eps`), complex literals.
  - `_translate_statement` — commands, while/feof, assignments (fopen/
    fscanf/multi-output/plain), function calls.
  - `_translate_loop` — `for`/`while` loop translation; feof-read loops
    collapse to a scan call; `range()`/`np.arange` reconstruction.
  - `_preallocations_for_loop` — emits `np.zeros_like`/`np.zeros(0)`
    preallocation for MATLAB implicit array growth inside loops.
  - `_resolve_axis_xy` — attaches `origin='lower'` to a preceding imshow.
  - `collapse_unresolved_blocks` — enforces the atomic-block invariant.
  - `assert_block_invariant` — verifies the invariant post-translation.
  - `_compute_renames` / `_translate_function` / `translate_with_rulebook`.
  - Reverse side: `_translate_expr_reverse`, `_translate_statement_reverse`,
    `translate_with_rulebook_reverse`, `_translate_plot_command_reverse`,
    `_percent_print_to_fprintf`, `_fstring_to_fprintf`.
  - `PLOT_COMMANDS` — table mapping MATLAB plotting/config commands to
    matplotlib calls.
- `builtin_rules.py`: `BUILTIN_RULES` dict (name -> `{"python", "arg_mode"}`);
  `apply_builtin_rule` (fwd) with `arg_mode` in {same, tuple_dims, size,
  randn}; `apply_builtin_rule_reverse` (only for `np.`-target rules).
- `keyword_rules.py`: `PYTHON_KEYWORDS`, `PYTHON_BUILTINS`,
  `MATLAB_KEYWORDS`, `MATLAB_BUILTIN_CALLS`, `should_rename`, `rename_for`,
  `rename_comment`, `identifier_tokens`, `rename_text`.
- `indexing_rules.py`: `apply_indexing_rule` (MATLAB -> Python, e.g.
  `x(5)` -> `x[4]`, `end-k` -> `-k-1`, ranges, `length`->`len`); reverse
  variant. `index_shift.py`: `shift_index` primitive (literal shift,
  identifier pass-through, indexed access recursion, arithmetic folding,
  ranges, floor division); `floor_divide_top_level`.
- `operator_rules.py`: `OPERATOR_RULES`, `REVERSE_OPERATOR_RULES`,
  `_find_last_operator` (precedence-aware, protects scientific literals),
  `_split_transpose`, `apply_transpose_rule`, `apply_operator_rule`,
  `apply_operator_rule_reverse`, `is_scalar_like` / `is_known_scalar`,
  `scientific_literals`.
- `complex_rules.py`: `apply_complex_rule` — `3i` -> `3j`.
- `format_rules.py`: `convert_fprintf` (MATLAB `fprintf` -> Python `print`
  with `%` formatting), `format_spec_count`, `matlab_string_literal_to_python`.
- `multi_output_rules.py`: `MULTI_OUTPUT_RULES` registry with kinds
  `pair` (max/min/sort), `shape` (size), `where` (find), `meshgrid`,
  `tuple` (butter/filter/findpeaks/freqz/xcorr). `_translate_pair`,
  `_translate_shape`, `_translate_where`, `_translate_meshgrid`,
  `_translate_tuple`, `_resolve_nested` (peels nested reductions),
  `_dim_to_axis`; entry `translate_multi_output_assignment`.
- `scan_rules.py`: `translate_fopen`, `translate_fscanf`,
  `translate_feof_loop`, `translate_feof_statement`; emit
  `read_matlab_scan_file`.
- `sequence_rules.py`: reverse `np.arange`/`linspace`/`zeros`/`ones`.
- `attribute_rules.py`: reverse `obj.shape`/`.T`/`.dtype`/`.size`.
- `shape_inference.py`: `shape_of_expr` (textual scalar/vector/matrix/
  unknown), `infer_shapes` (order-aware per-scope pass), `ScopeInfo` /
  `InferenceResult`; `_non_scalar_names` feeds matrix-power decisions.
- `symbolic.py`: `analyze_expression(expr_text, source_text)` ->
  list of insights; `analyze_translation(result)` -> `symbolic` section.
  Categories: `constant_detection` (safe folding via `_eval_constant`),
  `simplification` (`0*x`, `x*1`, `x+0`, `x-x`, `x/1`, `x**1`, `x**0`,
  `-(-x)`, known values at 0/1), `math_reasoning` (abs>=0, exp>0,
  sin/cos in [-1,1], sqrt>=0, x^2>=0). Confidence HIGH/MEDIUM/LOW.

### 3.4 `specialist_lib/`

- `signal_tools.py`: MATLAB-faithful wrappers over scipy:
  - `square(t, duty=50)` — percent duty -> fraction.
  - `findpeaks(x)` — `(pks, locs)` with 1-based locs.
  - `xcorr(x, y=None, maxlag=None, scaleopt='none')` — `(r, lags)` with
    biased/unbiased/coeff normalizations and maxlag clipping.
  - `detrend(x, type='constant', bp=0)` — MATLAB default/arg order.
  - `medfilt1(x, n=3)` — even width nudged to odd.
  - `filter_with_state(b, a, x)` — `(y, zf)` with zero initial conditions.
  - `freqz(b, a=1, n=512)` — returns `(h, w)` (MATLAB order; scipy returns
    `(w, h)`).
- `awgn.py`: `awgn(x, snr, sigpower='measured', mode='dB', seed=None)`.
- `chirp.py`: `chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True)`
  mirroring scipy's phase formulas.
- `conv.py`: `conv(u, v, shape='full')` over `np.convolve`.
- `array_factor.py` / `beamform.py` / `steering_vector.py`: ULA math.
- `read_scan_file.py`: `format_spec_to_columns`, `read_matlab_scan_file`.

### 3.5 `checker/`

- `verify.py`: `verify(matlab_file, python_file, inputs, tolerance,
  use_real_matlab=None)` — parses signature, matches inputs positionally
  between MATLAB args and Python params (`_match_inputs`), runs the mock or
  real engine + `run_python`, then `compare_outputs`.
- `run_matlab_mock.py`: `_parse_signature` regex for `function [out]=name(...)`;
  `run_matlab_mock` fabricates deterministic seeded arrays per output
  (SHA-1 of file+output+index), shaped like a provided reference input.
- `run_matlab_real.py`: lazy `matlab.engine` integration; `start_engine`,
  `matlab_engine_session`, `run_matlab_engine`, numpy<->MATLAB conversions.
- `run_python.py`: loads the translated module via importlib, finds the
  function by file stem, binds args, runs it, collects outputs.
- `compare_outputs.py`: returns `verified` / `failed` / `review needed`
  based on allclose + shape/finiteness/name alignment.
- `accuracy.py`: `accuracy(result)` and `score_mix(items)` — 0-100 score
  driven by checker verdict (`verified`=1.0, `failed`=0.0) else provenance
  weights (rulebook=1.0, unresolved=0.0), with per-line `ast.parse` checks.
- `validate.py`: `validate_translation(result)` — advisory warnings:
  `undefined_variable`, `unsupported_construct`, `suspicious_operator`,
  `unresolved_function`, `unsafe_translation`.
- `report.py`: `build_translation_report(result)` — flattens unresolved
  lines, syntax errors, validation warnings, and checker verdicts into
  human-readable entries.

### 3.6 `ui/`

- `translator_window.py` (`TranslatorWindow`) — full-featured window; stage
  markers for reader/inference/rulebook/validation/symbolic/checker;
  `_translate`, `_save_correction`, `_update_report`, `_update_summary`.
- `minimal_window.py` (`MinimalTranslatorWindow`) — similar, with an extra
  accuracy label; packaged by `matpy.spec`.
- `summary.py` — `summarize`, `status_counts`, `status_line`, `summary_line`,
  `accuracy_text`, `accuracy_style`, `report_text`, `summarize_translation`.
- `highlight.py` — `ProblemLineHighlighter` (QSyntaxHighlighter).

---

## 4. Actual Architecture

Pipeline: layered, offline, single-pass per direction.

```
         MATLAB .m  /  Python .py
                 |
                 v
  +-------------------------------------+
  |  READER  (reader/)                  |
  |  tree-sitter-matlab  OR  ast.parse  |
  |  -> Structure IR (Loop/Statement/   |
  |     Function/refs/indices)          |
  +-------------------------------------+
                 |
                 v
  +-------------------------------------+
  |  RULEBOOK  (rulebook/)              |
  |  shape_inference -> scalars set     |
  |  translate_with_rulebook[(_reverse)]|
  |   per-statement dict:               |
  |   {kind, source, python/matlab,     |
  |    comment, body?, renamed?}        |
  |  collapse_unresolved_blocks         |
  |  assert_block_invariant             |
  +-------------------------------------+
                 |
                 v
  +-------------------------------------+
  |  EMIT  (translator.py code_for_*)   |
  |  import header (np / scipy.signal / |
  |    specialist_lib, conditionally)   |
  |  script statements + functions      |
  |  problems: line indices of UNRESOLVED
  +-------------------------------------+
                 |
                 v
  +-------------------------------------+
  |  ADVISORY STAGES                    |
  |  validation (validate.py)           |
  |  symbolic   (symbolic.py)           |
  +-------------------------------------+
                 |
                 v
  +-------------------------------------+
  |  CHECKER  (checker/)                |
  |  run_matlab_mock OR run_matlab_real |
  |  run_python(translated)             |
  |  compare_outputs -> verdict         |
  +-------------------------------------+
                 |
                 v
   result dict: python/matlab, functions,
   statements, sections{reader,inference,
   rulebook,validation,symbolic,checker},
   problems, status
                 |
                 v
   UI (PyQt5)  /  accuracy score  /  report
```

The reverse direction is symmetric: `ast` reader, `translate_with_rulebook_reverse`,
`code_for_result_reverse`, `validation`/`symbolic` skipped, checker may compare
Python-generated MATLAB against a mock reference when inputs are supplied.

---

## 5. Actual Workflow

Entry points:

1. **CLI** — `run_translation.py`: `main()` calls
   `translate_file(path)` (default `sample_matlab/indexing_ops.m`) and prints
   `result["python"]`.
2. **Library** — `translator.translate_file` / `translator.translate_source`.
3. **GUI** — `ui/translator_window.py` `main()` (the Docker entrypoint uses
   `python3 -m ui.translator_window`); `ui/minimal_window.py` for the
   PyInstaller bundle. Both call `translate_source` on button press.
4. **Container** — `entrypoint.sh` runs dependency/grammar checks then
   `python3 -m ui.translator_window`.

Forward (MATLAB -> Python) trace for `sample_matlab/shape_inference.m`:

1. `translate_file` reads the file, calls `translate_source`.
2. `load_structure_from_source(source, MATLAB_TO_PYTHON)` joins `...`
   continuations, parses with tree-sitter, builds a `Structure` with one
   `Function` (`shape_inference`, params `fs, numPoints`) and body
   statements/loops.
3. `infer_shapes` records per-scope shapes; `translate_with_rulebook` runs.
   `f1 = 50` -> `f1 = 50`; `t = 0:1/fs:1-1/fs` -> `np.arange(0, 1 - 1/fs, 1/fs)`;
   `sin(2*pi*f1*t)` -> `np.sin(2 * np.pi * f1 * t)`; `P1 = P2(1:len(P2)/2+1)`
   -> `P2[0:len(P2)//2+1]`; `for n = 1:numPoints ... acc(n) = n*2; end` ->
   preallocation `acc = np.zeros((1, numPoints))` + `for n in range(numPoints)`
   body. Loops collapse; invariant asserted.
4. `code_for_result` emits `import numpy as np` + the function.
5. `validate_translation` produces warnings (none for this file);
   `analyze_translation` produces symbolic insights.
6. No inputs -> checker `inconclusive_no_matlab`.
7. Returns the result dict (status `ok`).

Reverse trace (Python -> MATLAB): `ast.parse` -> `build_structure` ->
`translate_with_rulebook_reverse` (`_translate_statement_reverse` ->
`_translate_expr_reverse`, sequence/attribute/plot/operator reverse rules) ->
`code_for_result_reverse` (`_emit_block_reverse` adds trailing `;`) ->
validation/symbolic `skipped` -> checker as above.

---

## 6. Data Flow

- **Source text** -> `load_structure_from_source` -> tree -> `Structure` IR
  (`Function` has `statements` (list of `Statement`/`Loop`), `parameters`,
  `outputs`, `refs`, `indices`).
- **Structure** -> `translate_with_rulebook` -> list of statement dicts.
  Each dict carries `kind` (`assignment`, `function_call`, `command`, `loop`),
  `source` (original text), `python` (translated, or `"UNRESOLVED"`),
  optional `comment`, optional `renamed` (rename notes), optional `body`
  (nested statements for loops/preallocations).
- **Statement dicts** -> `code_for_result` -> string lines. UNRESOLVED
  statements become `# UNRESOLVED: <source>` comment blocks and are recorded
  in `problems` (1-based line indices) for the GUI highlighter.
- **Result dict** is the single contract consumed by the UI, accuracy
  scoring, and report builder. `sections` holds per-stage status; `checker`
  holds the numeric verdict.
- **Checker data flow**: MATLAB source + inputs -> mock/real reference
  arrays; translated module + inputs -> Python output arrays;
  `compare_outputs` -> `verified`/`failed`/`review needed`.

---

## 7. Technologies & Libraries

| Library | Version | Where used | Why |
|---|---|---|---|
| Python | 3.8+ (3.11 image) | all | runtime language |
| `tree-sitter` | 0.26.0 | reader | incremental parsing, grammar abstraction |
| `tree-sitter-matlab` | 1.3.0 | reader | MATLAB grammar for `.m` parsing |
| `numpy` | 2.4.6 | rulebook, specialist_lib, checker, generated code | array math, FFT, linalg, random |
| `scipy` | 1.17.1 | specialist_lib, generated code | signal processing (lfilter, butter, freqz, windows, convolve2d, resample_poly) |
| `PyQt5` | 5.15.11 | ui/ | desktop GUI |
| `pytest` | 9.1.1 | tests/ | test framework |

stdlib used: `ast`, `re`, `hashlib`, `math`, `importlib.util`, `inspect`,
`os`, `sys`, `tempfile`, `uuid`, `contextlib`, `dataclasses`, `keyword`,
`builtins`, `pathlib`, `argparse`.

Everything is pinned in `requirements.txt`. `scipy` is a runtime dependency
only because the specialist wrappers and generated DSP code use it; the
pipeline core otherwise needs only tree-sitter + numpy.

---

## 8. Algorithms & Core Logic

- **Rulebook translation** is regex/string transformation over statement
  text, guided by the IR and a shape-inference `scalars` set. Operators are
  resolved lowest-precedence-first (`_find_last_operator`) so scalar-vs-matrix
  decisions apply to true operands. `*` -> `@` only when both operands are
  known arrays; `/` -> `np.linalg.solve` matrix right-divide only when both
  are arrays, else element-wise `/`; `\` -> `np.linalg.solve`; `^` ->
  `np.linalg.matrix_power` for proven arrays, else `**`.
- **Index shifting** (`index_shift.shift_index`): literal `-1` forward /
  `+1` reverse; identifiers pass through; indexed access recurses; arithmetic
  folds the offset into a trailing constant (`i+1` -> `i`); ranges shift only
  the start; floor division for integer slice bounds.
- **Atomic-block invariant**: after translation, any block construct with an
  UNRESOLVED descendant is collapsed to a single UNRESOLVED statement whose
  source spans the whole block, guaranteed by `collapse_unresolved_blocks`
  and checked by `assert_block_invariant`.
- **Shape inference**: textual single-pass over statements; per-scope dict of
  name -> scalar/vector/matrix/unknown, order-aware (reassignment overwrites);
  only names assigned exactly once become definite scalars.
- **Symbolic analysis**: `ast.parse(mode="eval")` + `ast.unparse`; safe
  constant folding restricted to numeric literals, arithmetic, unary sign,
  math/numpy constants, and a whitelisted 1-arg math-function set; identity
  detection over `ast.walk`; math-reasoning properties; never raises.
- **Mock reference** (`run_matlab_mock`): deterministic seeded random arrays
  (SHA-1 of file/output/index) shaped like the first array input.
- **Accuracy**: checker verdict overrides provenance weights; otherwise
  rulebook-resolved=1.0 / unresolved=0.0; per-line `ast.parse` validation
  (loop headers parsed with their bodies).
- **Rename pass**: collects identifiers (skipping strings/comments/fields),
  maps Python keywords/builtins that aren't MATLAB keywords or recognized
  builtin calls to `name_`, applied consistently throughout a scope.

---

## 9. Testing

- Framework: pytest. 47 test modules. Full suite: **1104 passed, 1 skipped,
  97 subtests passed** (verified twice during this session; the skip is the
  live-MATLAB test in `tests/test_run_matlab_real.py`).
- Coverage by module (approx. test counts from source):
  - translator/translation: `test_translator.py` (98), `test_translate_file.py`
    (19), `test_translate_reverse.py` (49), `test_block_atomicity.py` (12),
    `test_line_continuation.py` (7), `test_pipeline_resilience.py` (2),
    `test_scientific_notation.py` (17).
  - rules forward/reverse: `test_builtin_rules.py` (41), `test_builtin_rules_reverse.py`
    (22), `test_indexing_rules.py` (13), `test_indexing_rules_reverse.py` (11),
    `test_index_shift.py` (58), `test_indexing_operators_edges.py` (20),
    `test_operator_rules.py` (27), `test_operator_rules_reverse.py` (23),
    `test_transpose_rules.py` (39), `test_keyword_renames.py` (17),
    `test_complex_rules.py` (5), `test_format_rules.py` (16),
    `test_scan_rules.py` (23), `test_multi_output_rules.py` (44),
    `test_sequence_rules_reverse.py` (12), `test_attribute_rules_reverse.py`
    (7), `test_stat_builtin_rules.py` (16), `test_extract_python.py` (9),
    `test_shape_inference.py` (37).
  - specialists: `test_array_factor.py` (6), `test_awgn.py` (12),
    `test_beamform.py` (5), `test_chirp.py` (14), `test_conv.py` (12),
    `test_steering_vector.py` (4), `test_signal_tools.py` (19),
    `test_read_scan_file.py` (10).
  - checker: `test_accuracy.py` (23), `test_compare_outputs.py` (11),
    `test_run_matlab_mock.py` (8), `test_run_matlab_real.py` (12, one
    skipped), `test_run_python.py` (7), `test_verify.py` (9),
    `test_verify_reverse.py` (2), `test_validate.py` (36),
    `test_report.py` (35), `test_symbolic.py` (19), `test_offline_check.py`
    (7).
  - UI: `test_translator_window.py` (51).

Run with `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/ -q`.

---

## 10. Bugs & Issues

### Confirmed bugs

- None reproducible in the current test suite. The suite passes fully and
  both an end-to-end forward translation (`shape_inference.m`) and a reverse
  translation were manually confirmed during this session.

### Potential issues (by inspection)

- **`code_for_result` header for `read_matlab_scan_file`**: the scan rules
  emit `read_matlab_scan_file(...)` but `_code_references` only tests for
  `specialist_lib.`. If a file uses only the fscanf idiom and no other
  specialist call, the emitted Python references `read_matlab_scan_file`
  without `import specialist_lib` in the header. (Not yet reproduced; the
  fscanf path returns `read_matlab_scan_file(...)` and `_code_references`
  scans for `specialist_lib.`.) Worth a regression test.
- **`translate_file` result key mismatch**: the reverse direction stores
  output under `result["python"]` (not `matlab`); callers expecting
  `result["matlab"]` would find nothing. The README documents the same key.
- **Double-output `freqz` vs `filter`**: `freqz(b, a, 256)` single-output
  form now maps to `specialist_lib.freqz` (returns the `(h, w)` tuple) which
  is only correct for the `[h, w] =` form; single-output `h = freqz(...)`
  yields a tuple. MATLAB's single-output `freqz` returns only `h`.
- **Generated-code runtime deps**: `scipy` calls with length constraints
  (`filtfilt` needs >15 samples, `decimate` >27) will raise at runtime for
  short inputs — a translation-correctness limitation, not a translation bug.
- **Command `_OTHER_BUILTINS`**: unhandled `clc`/`clear`/`close`/`cos`/
  `exp`/`log`/`sin`/`sqrt`/`tan` calls are returned as UNRESOLVED even though
  `cos`/`exp`/etc. are in `BUILTIN_RULES` (checked first). Only `cos` etc.
  not present in BUILTIN_RULES fall through — this set is effectively a
  safety net for names without a rule.

---

## 11. Limitations

- **Not a full MATLAB compiler.** `if`/`switch`/`try` blocks, `while` loops
  (except the feof-read idiom), `global`/`persistent`, cell arrays, struct
  field access, anonymous functions, function handles, and double-quoted
  string arrays are intentionally left UNRESOLVED and reported as
  unsupported constructs.
- **Numeric verification is internal-consistency only.** The default checker
  compares against a fabricated seeded mock, so `verified` never proves
  MATLAB equivalence. Real MATLAB requires `matlab.engine`, which is not
  bundled; without it, `failed` is downgraded to `inconclusive_no_matlab`.
- **Only one function per file is supported** for the checker's reference
  comparison, and scripts without functions produce a `result` output.
- **Static textual translation** has no type system, no dataflow, and limited
  control-flow understanding; correctness depends heavily on the `scalars`
  set and shape inference heuristics.
- **1-based vs 0-based indexing** is handled for the common forms but
  edge cases (negative literals, complex compound indices) are limited;
  `shift_index` reports negative literals as UNRESOLVED.
- **Single output from multi-output-capable functions** (e.g.
  `h = freqz(...)`, `b = butter(...)`) returns the full tuple rather than the
  MATLAB single output.
- **Reverse direction is narrower** than forward: only `np.`-target builtins
  mirror back, Python-only constructs are flagged UNRESOLVED, and plot
  keyword/limit forms are only partially handled.
- **No concurrency / no plugin system**; translation is single-threaded and
  rule-driven.

---

## 12. Implemented vs Partial vs Missing

### Implemented (complete)

- Bidirectional translation pipeline (Reader -> Rulebook -> Emit -> Checker).
- MATLAB parsing via tree-sitter; Python parsing via `ast`.
- Builtin/operator/indexing/loop/plot/format/multi-output/scan rules.
- Specialist library: array_factor, beamform, steering_vector, awgn, chirp,
  conv, read_scan_file, and the scipy.signal wrappers (square, findpeaks,
  xcorr, detrend, medfilt1, filter_with_state, freqz).
- Conditional header imports (`scipy.signal` / `specialist_lib`).
- Atomic-block invariant + enforcement + tests.
- Shape-inference pass feeding scalar/matrix operator decisions.
- Advisory validation and symbolic stages integrated in pipeline and GUI.
- Accuracy scoring with checker-verdict and provenance methods.
- Mock + optional real MATLAB checker backends; positional input matching.
- PyQt5 GUIs, report builder, reference-store writer, Docker packaging,
  PyInstaller spec, offline-import static check.

### Partial

- Plot command coverage (table-driven; some commands are no-ops/comments).
- Reverse translation (subset of builtins/plot/sequence/attribute forms).
- Single-output forms of multi-output MATLAB functions (return tuples).
- `fscanf`/feof handling is specific to the recognized idiom.
- Live MATLAB Engine verification (available but not bundled).

### Missing

- Full control-flow translation (`if`/`switch`/`try`/`while`), cells,
  structs, handles, anonymous functions, object-oriented MATLAB
  (`classdef`).
- True MATLAB-fidelity numeric validation without an installed engine.
- Multi-function-file translation and richer cross-function analysis.
- A dedicated regression test for the `fscanf`-only import-header path
  (see section 10).

---

## 13. Code Quality / Security

Quality:

- Clear separation of concerns (reader/rulebook/specialist/checker/ui) with
  package `__init__` exports and consistent public APIs.
- Heavy docstrings explaining *why* (MATLAB vs scipy semantic differences).
- Table-driven design keeps adding builtins/plot commands to a single table.
- The atomic-block invariant is enforced and tested (`test_block_atomicity`,
  `assert_block_invariant`).
- Extensive, well-targeted test suite (1104 passing tests).
- `tools/check_offline.py` statically enforces the offline constraint.

Security:

- No network access at runtime; no external endpoints; the offline scan
  (`tools/check_offline.py`) guards against accidental network imports.
- `checker/run_python.py` executes translated code via importlib in-process
  — inherent code-execution risk if inputs are untrusted; acceptable for a
  local translation tool but worth noting.
- The mock reference uses only hashing + RNG; no user data exfiltrated.
- No hardcoded secrets/keys anywhere in the repository.
- File I/O is local (reading `.m`/`.py` sources, writing to `reference_set/`).

---

## 14. Final Summary

Strengths:

- Coherent, well-documented architecture with genuinely bidirectional
  translation and a strict atomic-block safety guarantee.
- Deeply tested (1104 tests, all passing) with targeted rule and specialist
  coverage, including the recently added scipy.signal mappings and the
  symbolic-analysis stage.
- Fully offline runtime with a static enforcement check; Docker and
  PyInstaller packaging paths exist.
- Advisory validation + symbolic stages give a human actionable review
  guidance without ever rejecting valid code.

Weaknesses:

- Not a complete MATLAB language implementation; control flow, data
  structures, and OO remain unsupported by design.
- Numeric "verification" is a mock consistency check unless a live MATLAB
  engine is present — easy to overstate.
- Static textual heuristics (scalar sets, shape inference) can silently
  mis-decide matrix vs element-wise operations on unknown shapes.

Current status: all committed and pushed to `origin/main` (`e8647e5`);
test suite green.

Recommended improvements:

1. Add a regression test for the `fscanf`-only generated module ensuring
   `import specialist_lib` appears in the header (`_code_references` only
   scans for the `specialist_lib.` token, which the fscanf path never emits).
2. Distinguish single-output vs multi-output forms of `freqz`/`butter`/
   `filter` so `h = freqz(...)` returns `h`, not `(h, w)`.
3. Use `result["matlab"]` consistently for reverse output (or document the
   `python`-key convention).
4. Grow control-flow translation (at least `if/else`) beyond UNRESOLVED.
5. Consider optionally documenting input-length requirements for
   `filtfilt`/`decimate` to avoid silent runtime errors on short signals.
