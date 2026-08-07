import os

from repo_paths import REFERENCE_SET_DIR


def save_reference_entry(matlab_source, python_source, base_name,
                         directory=None):
    """Write a new reference_set entry (MATLAB input + Python reference).

    Creates ``<base>_py.m`` (the MATLAB input) and ``<base>.py`` (the
    corrected Python reference) as a new pair. If a pair already exists
    for ``base_name``, a numeric suffix is appended so existing reference
    entries are never overwritten.

    Returns the two written paths as ``(matlab_path, python_path)``.
    """
    directory = directory or REFERENCE_SET_DIR
    base = _unique_base_name(base_name, directory)
    matlab_path = os.path.join(directory, base + "_py.m")
    python_path = os.path.join(directory, base + ".py")
    with open(matlab_path, "w", encoding="utf-8") as f:
        f.write(matlab_source)
    with open(python_path, "w", encoding="utf-8") as f:
        f.write(python_source)
    return matlab_path, python_path


def _unique_base_name(base_name, directory):
    candidate = base_name
    index = 1
    while (
        os.path.exists(os.path.join(directory, candidate + "_py.m"))
        or os.path.exists(os.path.join(directory, candidate + ".py"))
    ):
        index += 1
        candidate = "%s_%d" % (base_name, index)
    return candidate
