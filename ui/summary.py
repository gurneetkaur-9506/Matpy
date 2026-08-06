import os

import numpy as np

from repo_paths import sample_matlab
from translator import translate_file

SAMPLE_FILES = [
    sample_matlab("builtins_demo.m"),
    sample_matlab("indexing_ops.m"),
    sample_matlab("fft_basic.m"),
    sample_matlab("beamform_basic.m"),
]

BEAMFORM_INPUTS = {
    "N": 8,
    "d": 0.5,
    "lamb": 1.0,
    "theta": np.linspace(0, np.pi, 91),
    "theta0": 0.0,
}


def summarize(result):
    rulebook = result["sections"]["rulebook"]
    total = rulebook["total"]
    unresolved = rulebook["unresolved"]
    drafted = result["sections"]["assistant"]["drafted"]
    rulebook_pct = (total - unresolved) / total * 100 if total else 100.0
    assistant_pct = unresolved / total * 100 if total and drafted else 0.0
    return {
        "rulebook_pct": rulebook_pct,
        "assistant_pct": assistant_pct,
        "checker": result["sections"]["checker"]["status"],
    }


def status_counts(result):
    rulebook = result["sections"].get("rulebook", {})
    total = rulebook.get("total", 0)
    unresolved = rulebook.get("unresolved", 0)
    drafted = len(result["sections"].get("assistant", {}).get("drafted") or [])
    checker = result["sections"].get("checker", {}).get("status", "skipped")
    verified = total if checker == "verified" else 0
    return total - unresolved, unresolved + drafted, verified


def status_line(result):
    auto, review, verified = status_counts(result)
    return (
        "%d lines translated automatically, %d need your review, %d verified yet"
        % (auto, review, verified)
    )


def summarize_translation(paths=None, beamform_inputs=None):
    paths = paths or SAMPLE_FILES
    rows = []
    for path in paths:
        name = os.path.basename(path)
        inputs = None
        if beamform_inputs is not None and "beamform" in name:
            inputs = beamform_inputs
        result = translate_file(path, inputs=inputs)
        summary = summarize(result)
        rows.append(
            {
                "file": name,
                "rulebook_pct": summary["rulebook_pct"],
                "assistant_pct": summary["assistant_pct"],
                "checker": summary["checker"],
            }
        )
    return rows


def print_summary(rows):
    header = "%-20s %12s %12s  %s" % ("filename", "rulebook%", "assistant%", "verification")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-20s %11.1f%% %11.1f%%  %s"
            % (row["file"], row["rulebook_pct"], row["assistant_pct"], row["checker"])
        )


def main():
    rows = summarize_translation(beamform_inputs=BEAMFORM_INPUTS)
    print_summary(rows)


if __name__ == "__main__":
    main()
