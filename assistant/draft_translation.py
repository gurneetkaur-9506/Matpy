import json
import re
import urllib.request

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


def _build_prompt(matlab_function_struct, specialist_lib_contents):
    source = "\n".join(
        s.get("source", "")
        for s in matlab_function_struct.get("statements", [])
    )
    spec = ""
    if specialist_lib_contents:
        spec = "\n\nAvailable specialist library:\n%s" % (
            json.dumps(specialist_lib_contents, indent=2)
        )
    return "%s\n\nMATLAB function:\n%s%s" % (
        _SYSTEM_INSTRUCTION,
        source,
        spec,
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


def _extract_section(text, start, end):
    pattern = re.compile(
        r"%s\s*\n(?P<body>.*?)\n\s*%s" % (re.escape(start), re.escape(end)),
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    body = match.group("body").strip()
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


def draft_translation(matlab_function_struct, specialist_lib_contents):
    prompt = _build_prompt(matlab_function_struct, specialist_lib_contents)
    response = _call_ollama(prompt)
    return parse_response(response)
