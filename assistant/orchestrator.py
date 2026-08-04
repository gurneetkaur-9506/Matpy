from rulebook import UNRESOLVED

from .draft_translation import draft_translation


def draft_unresolved_functions(result, specialist_lib_contents):
    for func in result["functions"]:
        if any(s["python"] == UNRESOLVED for s in func["statements"]):
            func["draft"] = draft_translation(func, specialist_lib_contents)
    return result
