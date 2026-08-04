import importlib.util
import inspect
import os
import sys
import uuid

import numpy as np


def _load_module(file_path):
    module_name = "translated_module_%s" % uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    directory = os.path.dirname(os.path.abspath(file_path))
    sys.path.insert(0, directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    return module


def _find_function(module, file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    locals_functions = [
        obj
        for name, obj in vars(module).items()
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    ]
    for name, obj in vars(module).items():
        if name == stem and inspect.isfunction(obj):
            return obj
    return locals_functions[0] if locals_functions else None


def _bind_args(func, inputs):
    signature = inspect.signature(func)
    kwargs = {}
    missing = []
    for pname, param in signature.parameters.items():
        if pname in inputs:
            kwargs[pname] = inputs[pname]
        elif param.default is inspect.Parameter.empty:
            missing.append(pname)
    return kwargs, missing


def _result(success, file_path, func_name, inputs, outputs, notes):
    return {
        "success": success,
        "file": file_path,
        "function": func_name,
        "inputs": {k: np.asarray(v) for k, v in inputs.items()},
        "outputs": outputs,
        "notes": notes,
    }


def run_python(file_path, inputs=None, output_names=None):
    inputs = inputs or {}
    notes = []

    try:
        module = _load_module(file_path)
    except Exception as exc:
        return _result(
            False, file_path, None, inputs, {}, ["failed to load module: %s" % exc]
        )

    func = _find_function(module, file_path)
    if func is None:
        return _result(
            False, file_path, None, inputs, {}, ["no function found in module"]
        )

    kwargs, missing = _bind_args(func, inputs)
    if missing:
        for name in missing:
            notes.append("missing input %r" % name)
        return _result(False, file_path, func.__name__, inputs, {}, notes)

    try:
        value = func(**kwargs)
    except Exception as exc:
        notes.append("execution failed: %s" % exc)
        return _result(False, file_path, func.__name__, inputs, {}, notes)

    if value is None:
        notes.append("function returned None")
        return _result(True, file_path, func.__name__, inputs, {}, notes)

    values = value if isinstance(value, tuple) else (value,)
    if output_names:
        names = output_names
    else:
        names = [func.__name__] if len(values) == 1 else [
            "out%d" % (i + 1) for i in range(len(values))
        ]

    outputs = {}
    for index, name in enumerate(names):
        if index < len(values):
            outputs[name] = np.asarray(values[index])
    if len(values) > len(names):
        notes.append(
            "function returned %d values but only %d output names given"
            % (len(values), len(names))
        )

    return _result(True, file_path, func.__name__, inputs, outputs, notes)
