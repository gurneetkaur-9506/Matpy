import os

import numpy as np

from repo_paths import sample_matlab
from translator import translate_file

SAMPLE_FILES = [
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
    rulebook_pct = (total - unresolved) / total * 100 if total else 100.0
    return {
        "rulebook_pct": rulebook_pct,
        "checker": result["sections"]["checker"]["status"],
    }


def status_counts(result):
    rulebook = result["sections"].get("rulebook", {})
    total = rulebook.get("total", 0)
    unresolved = rulebook.get("unresolved", 0)
    checker = result["sections"].get("checker", {}).get("status", "skipped")
    verified = total if checker == "verified" else 0
    return total - unresolved, unresolved, verified


def status_line(result):
    auto, review, verified = status_counts(result)
    return (
        "%d lines translated automatically, %d need your review, %d verified yet"
        % (auto, review, verified)
    )


def summary_line(result):
    """One-line summary of a translation result, e.g.
    "12 lines translated, 2 need review, accuracy 87%."

    The accuracy clause is omitted when no score can be computed.
    """
    auto, review, _ = status_counts(result)
    try:
        from checker import accuracy

        score = accuracy(result)["score"]
    except Exception:
        score = None
    if score is None:
        return "%d lines translated, %d need review." % (auto, review)
    return "%d lines translated, %d need review, accuracy %d%%." % (
        auto,
        review,
        round(score),
    )


ACCURACY_STYLE = {
    "high": "color: #1e8e3e; font-weight: bold;",
    "mid": "color: #b58900; font-weight: bold;",
    "low": "color: #c62828; font-weight: bold;",
    "unknown": "color: #95a5a6;",
}


def accuracy_text(score):
    """Return the label text for an accuracy score (0-100) or None."""
    if score is None:
        return "Accuracy: --"
    return "Accuracy: %d%%" % round(score)


def accuracy_style(score):
    """Pick a color style for an accuracy score.

    Green above 90%, yellow for 70-90%, red below 70%; a neutral grey when
    no score is available yet.
    """
    if score is None:
        return ACCURACY_STYLE["unknown"]
    if score > 90:
        return ACCURACY_STYLE["high"]
    if score >= 70:
        return ACCURACY_STYLE["mid"]
    return ACCURACY_STYLE["low"]


def report_text(entry):
    """Format one build_translation_report() entry as a plain-language line.

    Example:
        "Line 47: interp1 with 3 outputs - not yet supported, left as
        TODO comment."
    """
    where = (
        "Line %d" % entry["line"]
        if entry.get("line")
        else entry.get("stage", "checker").title()
    )
    source = entry.get("source") or ""
    reason = entry.get("reason") or ""
    if source and entry.get("stage") != "checker":
        return "%s: %s - %s" % (where, source, reason)
    return "%s: %s" % (where, reason)


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
                "checker": summary["checker"],
            }
        )
    return rows


def print_summary(rows):
    header = "%-20s %12s  %s" % ("filename", "rulebook%", "verification")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-20s %11.1f%%  %s"
            % (row["file"], row["rulebook_pct"], row["checker"])
        )


def main():
    rows = summarize_translation(beamform_inputs=BEAMFORM_INPUTS)
    print_summary(rows)


if __name__ == "__main__":
    main()
