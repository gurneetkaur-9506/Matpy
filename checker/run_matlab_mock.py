import hashlib
import re

import numpy as np

_FUNC_LINE = re.compile(
    r"^\s*function\s*"
    r"(?:\[(?P<outputs>[^\]]*)\]\s*=\s*|(?P<single_out>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<args>[^)]*)\))?\s*$",
    re.MULTILINE,
)


def _split_names(text):
    return [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]


def _parse_signature(text):
    match = _FUNC_LINE.search(text)
    if not match:
        return None, [], []
    outputs = []
    if match.group("outputs"):
        outputs = _split_names(match.group("outputs"))
    elif match.group("single_out"):
        outputs = [match.group("single_out")]
    args = _split_names(match.group("args")) if match.group("args") else []
    return match.group("name"), outputs, args


def _reference_input(inputs):
    for value in inputs.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return arr
    return None


def _fake_value(file_path, output_name, inputs, index):
    seed = int(
        hashlib.sha1(
            ("%s::%s::%d" % (file_path, output_name, index)).encode("utf-8")
        ).hexdigest(),
        16,
    ) % (2 ** 32)
    rng = np.random.default_rng(seed)
    reference = _reference_input(inputs)
    if reference is None:
        return np.asarray(rng.random())
    if np.iscomplexobj(reference):
        return rng.random(reference.shape) + 1j * rng.random(reference.shape)
    return rng.random(reference.shape)


def run_matlab_mock(file_path, inputs=None):
    inputs = inputs or {}
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    name, outputs, args = _parse_signature(text)
    notes = []
    for arg in args:
        if arg not in inputs:
            notes.append("missing input %r" % arg)

    result = {}
    if outputs:
        for index, out in enumerate(outputs):
            result[out] = _fake_value(file_path, out, inputs, index)
    else:
        result["result"] = _fake_value(file_path, "result", inputs, 0)

    return {
        "success": True,
        "file": file_path,
        "function": name,
        "inputs": {k: np.asarray(v) for k, v in inputs.items()},
        "outputs": result,
        "notes": notes,
    }
