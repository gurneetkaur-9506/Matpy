# MAT2PY Translator

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
