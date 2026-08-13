"""Rename MATLAB identifiers that collide with Python reserved names.

A MATLAB variable may legally be named ``lambda``, ``class``, ``type``,
``in``, ``not``, etc. -- names Python reserves as keywords or uses as
builtins.  Emitting them verbatim produces invalid Python (keywords) or
code that silently shadows a builtin (builtins).  This module detects such
collisions and builds a per-file rename map (``lambda`` -> ``lambda_``) so
the rulebook can rename them consistently everywhere they occur.
"""

import builtins
import keyword

PYTHON_KEYWORDS = frozenset(keyword.kwlist)

PYTHON_BUILTINS = frozenset(
    name for name in dir(builtins) if name.isidentifier() and not name.startswith("_")
)

# MATLAB reserved words.  These are part of MATLAB's own syntax (for/while/
# if/else/end/...), never legal variable names, so they must not be touched
# even though some (for, if, while, else, return, ...) are also Python
# keywords.
MATLAB_KEYWORDS = frozenset(
    {
        "arguments",
        "break",
        "case",
        "catch",
        "classdef",
        "continue",
        "else",
        "elseif",
        "end",
        "enumeration",
        "events",
        "for",
        "function",
        "global",
        "if",
        "import",
        "methods",
        "otherwise",
        "package",
        "parfor",
        "persistent",
        "properties",
        "return",
        "spmd",
        "switch",
        "try",
        "while",
    }
)

# MATLAB builtin function/command names the rulebook recognizes.  These are
# call names, not variables, and must never be renamed (sum(x) must stay
# sum(x) even though 'sum' is a Python builtin).
MATLAB_BUILTIN_CALLS = frozenset(
    {
        "abs",
        "clc",
        "clear",
        "close",
        "cos",
        "disp",
        "exp",
        "fft",
        "figure",
        "filter",
        "find",
        "grid",
        "ifft",
        "interp1",
        "legend",
        "length",
        "linspace",
        "log",
        "max",
        "mean",
        "min",
        "numel",
        "ones",
        "plot",
        "reshape",
        "sin",
        "size",
        "sqrt",
        "sum",
        "tan",
        "title",
        "xlabel",
        "ylabel",
        "zeros",
    }
)

_RESERVED = PYTHON_KEYWORDS | PYTHON_BUILTINS
_SUFFIX = "_"

_RENAME_COMMENT = "# renamed: MATLAB '%s' -> Python '%s' (%s)"
_KEYWORD_REASON = "reserved keyword"
_BUILTIN_REASON = "shadowed builtin"


def should_rename(name):
    """True when a MATLAB variable named ``name`` must be renamed in Python."""
    if name in MATLAB_KEYWORDS or name in MATLAB_BUILTIN_CALLS:
        return False
    return name in _RESERVED


def rename_for(name):
    """Return the Python name a colliding MATLAB ``name`` is renamed to."""
    return name + _SUFFIX


def rename_comment(name):
    """Return the '#' comment noting the rename of ``name``."""
    reason = _KEYWORD_REASON if name in PYTHON_KEYWORDS else _BUILTIN_REASON
    return _RENAME_COMMENT % (name, rename_for(name), reason)


def identifier_tokens(text):
    """Yield ``(start, end, name)`` for identifier tokens used as variables.

    Tokens inside single-quoted strings and ``%`` comments and struct field
    names (preceded by a dot) are skipped because they are not variable
    usages.  Call names are not special-cased here: a name that collides
    with a Python reserved word is renamed consistently everywhere it is
    used (including indexing like ``class(1)``), while names that are never
    used as variables never enter the rename map in the first place.
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "%":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "'":
            i += 1
            while i < n:
                if text[i] == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[i:j]
            prev = text[i - 1] if i > 0 else ""
            if name and prev != ".":
                tokens.append((i, j, name))
            i = j
            continue
        i += 1
    return tokens


def rename_text(text, renames):
    """Replace variable identifiers in ``text`` according to ``renames``.

    Only variable-position tokens (see :func:`identifier_tokens`) are
    replaced, so strings, comments, field access and call names are left
    untouched.  Returns the original text unchanged when there is nothing to
    rename.
    """
    if not renames:
        return text
    parts = []
    last = 0
    for start, end, name in identifier_tokens(text):
        replacement = renames.get(name)
        if replacement is None:
            continue
        parts.append(text[last:start])
        parts.append(replacement)
        last = end
    if not parts:
        return text
    parts.append(text[last:])
    return "".join(parts)
