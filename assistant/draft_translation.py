import json
import re
import urllib.request

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from specialist_lib import collect_numpy_operations, reverse_lookup

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

UNRESOLVED = "UNRESOLVED"

_SYSTEM_INSTRUCTION = """\
You are an expert MATLAB-to-Python translator. Translate the given MATLAB
function into idiomatic Python using numpy.

Respond in EXACTLY this format and nothing else:

CODE
<python code, no markdown code fences>
END CODE
CONFIDENCE
<one number from 0.0 to 1.0>
END CONFIDENCE
UNSURE
<one line per item, each prefixed with "- ", listing EVERY point you are not
fully certain about: assumptions, guesses, ambiguous MATLAB constructs,
unknown shapes or semantics. If you are certain about everything, write "none".>
END UNSURE

Rules:
- Never present uncertain behavior as certain. Every assumption or guess
  MUST be listed in the UNSURE section.
- Confidence MUST be 0.0 if there are any unresolved or guessed constructs.
"""

_SYSTEM_INSTRUCTION_REVERSE = """\
You are an expert Python-to-MATLAB translator. Translate the given Python
function into idiomatic MATLAB using the Phased Array Toolbox where
appropriate.

The reverse-lookup candidates below map numpy operations to the MATLAB
Phased Array Toolbox functions they might correspond to. Several candidates
may apply; pick the best fit from context and flag every ambiguous choice in
the UNSURE section.

Respond in EXACTLY this format and nothing else:

CODE
<matlab code, no markdown code fences>
END CODE
CONFIDENCE
<one number from 0.0 to 1.0>
END CONFIDENCE
UNSURE
<one line per item, each prefixed with "- ", listing EVERY point you are not
fully certain about: assumptions, guesses, ambiguous MATLAB constructs,
unknown shapes or semantics. If you are certain about everything, write "none".>
END UNSURE

Rules:
- Never present uncertain behavior as certain. Every assumption or guess
  MUST be listed in the UNSURE section.
- Confidence MUST be 0.0 if there are any unresolved or guessed constructs.
"""

_UNCERTAINTY_HINTS = (
    "i think",
    "i believe",
    "maybe",
    "perhaps",
    "not sure",
    "not certain",
    "assum",
    "guess",
    "probably",
    "might",
    "could be",
)


def _source_text(function_struct):
    return "\n".join(
        s.get("source", "")
        for s in function_struct.get("statements", [])
    )


def _reverse_lookup_context(function_struct):
    source = _source_text(function_struct)
    context = {}
    for op in collect_numpy_operations(source):
        context[op] = reverse_lookup(op)
    return context


def _build_prompt(function_struct, context, direction):
    source = _source_text(function_struct)
    instruction = (
        _SYSTEM_INSTRUCTION_REVERSE
        if direction == PYTHON_TO_MATLAB
        else _SYSTEM_INSTRUCTION
    )
    source_label = (
        "PYTHON function" if direction == PYTHON_TO_MATLAB else "MATLAB function"
    )
    context_block = ""
    if context:
        context_label = (
            "reverse-lookup candidates"
            if direction == PYTHON_TO_MATLAB
            else "specialist library"
        )
        context_block = "\n\nAvailable %s:\n%s" % (
            context_label,
            json.dumps(context, indent=2),
        )
    return "%s\n\n%s:\n%s%s" % (
        instruction,
        source_label,
        source,
        context_block,
    )


def _call_ollama(prompt):
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("response", "")


_SECTION_MARKERS = (
    "CODE",
    "END CODE",
    "CONFIDENCE",
    "END CONFIDENCE",
    "UNSURE",
    "END UNSURE",
)


def _extract_section(text, start, end):
    """Extract a section body bounded by ``start``/``end`` markers.

    The strict format requires both markers.  Real models sometimes omit the
    ``END ...`` closing marker, so when it is missing the body is taken up to
    the next known section marker (or the end of the text).
    """
    pattern = re.compile(
        r"%s\s*\n(?P<body>.*?)\n\s*%s" % (re.escape(start), re.escape(end)),
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        head = re.search(r"%s\s*\n" % re.escape(start), text, re.DOTALL)
        if not head:
            return None
        body_end = len(text)
        for marker in _SECTION_MARKERS:
            if marker == start:
                continue
            pos = text.find(marker, head.end())
            if pos != -1 and pos < body_end:
                body_end = pos
        match = {"body": text[head.end():body_end]}
    body = match["body"].strip()
    body = re.sub(r"^```(?:python)?\s*\n", "", body)
    body = re.sub(r"\n```\s*$", "", body)
    return body


def _extract_confidence(text):
    match = re.search(
        r"CONFIDENCE\s*\n\s*([0-9]+(?:\.[0-9]+)?)", text
    )
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return None


def _extract_unsure(text):
    section = _extract_section(text, "UNSURE", "END UNSURE")
    if section is None or section.strip().lower() in ("", "none", "n/a", "na"):
        return []
    items = []
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        items.append(line)
    return items


def _has_uncertainty_hints(text):
    lowered = text.lower()
    return any(hint in lowered for hint in _UNCERTAINTY_HINTS)


def parse_response(text):
    code = _extract_section(text, "CODE", "END CODE")
    confidence = _extract_confidence(text)
    unsure = _extract_unsure(text)

    notes = []
    if code is None:
        code = ""
        notes.append("no CODE section found in model response")
    if confidence is None:
        confidence = 0.0
        notes.append("model did not state a CONFIDENCE value")
    if not unsure and _has_uncertainty_hints(text):
        notes.append(
            "model used uncertain language but did not flag items in UNSURE"
        )
    for item in unsure:
        notes.append("uncertainty flagged: %s" % item)

    return {"code": code, "confidence": confidence, "notes": notes}


def draft_translation(function_struct, context, direction=MATLAB_TO_PYTHON):
    if direction == PYTHON_TO_MATLAB:
        context = _reverse_lookup_context(function_struct)
    prompt = _build_prompt(function_struct, context, direction)
    response = _call_ollama(prompt)
    return parse_response(response)
