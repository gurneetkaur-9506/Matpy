# MATPY Translator

An architecture for translating MATLAB code to Python. The system is built from five cooperating blocks.

## Architecture

### Reader

The Reader ingests MATLAB source files and parses them into a structured intermediate representation. It relies on tree-sitter with the MATLAB grammar to tokenize functions, scripts, and statements, producing a syntax tree that downstream blocks can traverse. The Reader also captures file-level metadata such as function signatures, comments, and line mappings so that translated output can be traced back to its origin.

### Rulebook

The Rulebook holds the knowledge base of translation rules: declarative mappings between MATLAB idioms and their Python equivalents. Each rule describes a pattern to match, conditions under which it applies, and the transformation to emit, covering areas such as indexing, loops, function definitions, and built-in names. Rules are organized by topic so they can be consulted independently or combined into larger transformations.

### Specialist Library

The Specialist Library is a collection of small, focused translation modules, each expert in one domain such as matrix operations, signal processing, or plotting. Specialists implement the trickier one-to-many conversions that a single rule cannot express, producing Python that relies on numpy and scipy idioms. They register with the Rulebook so that the Assistant can invoke them when the matched pattern falls into their specialty.

### Assistant

The Assistant is the orchestrator. It walks the intermediate representation produced by the Reader, applies matching rules and invokes the appropriate Specialists, and assembles the final Python output. It tracks context such as variable types and scope to keep the translation consistent across function boundaries and to report warnings or unresolved constructs.

### Checker

The Checker validates the generated Python before it is accepted. It re-parses the output, performs structural and semantic checks, and optionally cross-checks numeric behavior against the original MATLAB for supported constructs. Findings are fed back to the Assistant to refine the translation, closing the loop until the output is clean or the remaining issues are reported to the user.

## Companion Modules

The project also includes a UI for driving the translation process and a reference_set of paired MATLAB and Python examples used to develop and validate rules.

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
