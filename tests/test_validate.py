"""Tests for the static-validation stage (checker.validate)."""

import unittest

from checker import validate_translation
from reader import PYTHON_TO_MATLAB
from rulebook import UNRESOLVED
from tests.paths import sample_matlab
from translator import translate_file, translate_source


def _stmt(kind, source, python="x = 1", body=None):
    stmt = {"kind": kind, "source": source, "python": python}
    if body is not None:
        stmt["body"] = body
    return stmt


def _func(name, statements, parameters=None):
    return {
        "name": name,
        "parameters": parameters or [],
        "outputs": [],
        "statements": statements,
    }


def _source_text(functions, statements):
    lines = []
    for func in functions:
        lines.append("function out = %s(%s)" % (func["name"], ", ".join(func["parameters"])))
        for stmt in func["statements"]:
            lines.extend(stmt["source"].splitlines())
        lines.append("end")
    for stmt in statements:
        lines.extend(stmt["source"].splitlines())
    return "\n".join(lines)


def _result(functions=None, statements=None, source=None, direction="matlab_to_python"):
    functions = functions or []
    statements = statements or []
    if source is None:
        source = _source_text(functions, statements)
    return {
        "file": "input.m",
        "source": source,
        "direction": direction,
        "status": "ok",
        "python": "",
        "functions": functions,
        "statements": statements,
        "sections": {},
    }


def _categories(warnings):
    return [w["category"] for w in warnings]


class TestDirectionAndEmpty(unittest.TestCase):
    def test_reverse_direction_returns_empty(self):
        result = _result(direction=PYTHON_TO_MATLAB)
        self.assertEqual(validate_translation(result), [])

    def test_empty_result_returns_empty(self):
        self.assertEqual(validate_translation({}), [])

    def test_clean_function_returns_empty(self):
        result = _result(
            functions=[
                _func("f", [_stmt("assignment", "y = x + 1", "y = x + 1")], ["x"])
            ]
        )
        self.assertEqual(validate_translation(result), [])

    def test_clean_script_returns_empty(self):
        result = _result(
            statements=[
                _stmt("assignment", "x = 1", "x = 1"),
                _stmt("assignment", "y = x + 1", "y = x + 1"),
            ]
        )
        self.assertEqual(validate_translation(result), [])


class TestUndefinedVariable(unittest.TestCase):
    def test_function_undefined_high_confidence(self):
        result = _result(
            functions=[_func("f", [_stmt("assignment", "y = missing + 1")])]
        )
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["undefined_variable"])
        warning = warnings[0]
        self.assertEqual(warning["confidence"], "HIGH")
        self.assertEqual(warning["stage"], "validation")
        self.assertIn("missing", warning["message"])

    def test_script_undefined_medium_confidence(self):
        result = _result(statements=[_stmt("assignment", "y = missing + 1")])
        warnings = validate_translation(result)
        self.assertEqual(warnings[0]["confidence"], "MEDIUM")

    def test_assignment_in_order_seeds_scope(self):
        result = _result(
            statements=[
                _stmt("assignment", "a = 1", "a = 1"),
                _stmt("assignment", "b = a + 1", "b = a + 1"),
            ]
        )
        self.assertEqual(validate_translation(result), [])

    def test_loop_variable_seeds_scope(self):
        body = [_stmt("assignment", "total = i * 2")]
        loop = _stmt("loop", "for i = 1:N", "for i in range(N):", body)
        result = _result(functions=[_func("f", [loop], ["N"])])
        categories = _categories(validate_translation(result))
        self.assertNotIn("undefined_variable", categories)

    def test_parameter_seeds_scope(self):
        result = _result(
            functions=[_func("f", [_stmt("assignment", "y = x * 2")], ["x"])]
        )
        self.assertEqual(validate_translation(result), [])

    def test_keyword_renamed_parameter_seeds_original_name(self):
        result = _result(
            functions=[
                _func(
                    "beamform",
                    [_stmt("assignment", "k = 2 * pi / lambda")],
                    ["lambda_"],
                )
            ]
        )
        self.assertEqual(validate_translation(result), [])

    def test_matlab_constant_not_undefined(self):
        result = _result(
            functions=[_func("f", [_stmt("assignment", "y = pi * x")], ["x"])]
        )
        self.assertEqual(validate_translation(result), [])

    def test_rulebook_call_name_not_undefined(self):
        result = _result(
            functions=[
                _func(
                    "f",
                    [_stmt("assignment", "y = smoothdata(x) + 1", "y = smoothdata(x) + 1")],
                    ["x"],
                )
            ]
        )
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unresolved_function"])
        self.assertNotIn("undefined_variable", warnings[0]["category"])

    def test_multi_output_assignment_seeds_all_names(self):
        result = _result(
            functions=[
                _func(
                    "f",
                    [
                        _stmt("assignment", "[v, i] = max(x)"),
                        _stmt("assignment", "y = v + i"),
                    ],
                    ["x"],
                )
            ]
        )
        self.assertEqual(validate_translation(result), [])


class TestUnsupportedConstruct(unittest.TestCase):
    def test_if_statement_high_confidence(self):
        stmt = {"kind": "if_statement", "source": "if x > 0\n    y = 1;\nend",
                "python": UNRESOLVED}
        warnings = validate_translation(_result(statements=[stmt]))
        self.assertEqual(warnings[0]["category"], "unsupported_construct")
        self.assertEqual(warnings[0]["confidence"], "HIGH")
        self.assertIn("if/elseif/else", warnings[0]["message"])

    def test_switch_and_try_and_global_and_persistent(self):
        cases = [
            ("switch_statement", "switch x"),
            ("try_statement", "try"),
            ("global_operator", "global x"),
            ("persistent_operator", "persistent acc"),
        ]
        for kind, source in cases:
            stmt = {"kind": kind, "source": source, "python": UNRESOLVED}
            warnings = validate_translation(_result(statements=[stmt]))
            self.assertEqual(
                warnings[0]["category"], "unsupported_construct", source
            )
            self.assertEqual(warnings[0]["confidence"], "HIGH", source)

    def test_while_loop_high_confidence(self):
        body = [_stmt("assignment", "k = k - 1")]
        loop = {"kind": "loop", "source": "while k > 0\n    k = k - 1;\nend",
                "python": UNRESOLVED, "body": body}
        warnings = validate_translation(_result(functions=[_func("f", [loop])]))
        self.assertEqual(_categories(warnings), ["unsupported_construct"])
        self.assertIn("while", warnings[0]["message"])
    def test_cell_array_medium_confidence(self):
        result = _result(statements=[_stmt("assignment", "c = {1, 2}")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsupported_construct"])
        self.assertEqual(warnings[0]["confidence"], "MEDIUM")
        self.assertIn("cell", warnings[0]["message"])

    def test_struct_field_access_medium_confidence(self):
        result = _result(statements=[_stmt("assignment", "s.field = 3")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsupported_construct"])
        self.assertIn("struct", warnings[0]["message"])

    def test_struct_field_access_inside_call_is_not_struct(self):
        result = _result(
            functions=[
                _func(
                    "f",
                    [_stmt("assignment", "y = obj.method(x)", "y = obj.method(x)")],
                    ["obj", "x"],
                )
            ]
        )
        self.assertEqual(validate_translation(result), [])

    def test_anonymous_function_high_confidence(self):
        result = _result(statements=[_stmt("assignment", "f = @(t) t.^2")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsupported_construct"])
        self.assertEqual(warnings[0]["confidence"], "HIGH")

    def test_function_handle_medium_confidence(self):
        result = _result(statements=[_stmt("assignment", "h = @sin")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsupported_construct"])
        self.assertEqual(warnings[0]["confidence"], "MEDIUM")

    def test_double_quoted_string_low_confidence(self):
        result = _result(statements=[_stmt("assignment", 'y = "text"')])
        warnings = validate_translation(result)
        self.assertEqual(warnings[0]["confidence"], "LOW")
        self.assertIn("double-quoted", warnings[0]["message"])


class TestSuspiciousOperator(unittest.TestCase):
    def test_leftover_matlab_operator_flagged(self):
        stmt = _stmt("assignment", "y = x", "y = x .* 2")
        result = _result(functions=[_func("f", [stmt], ["x"])])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["suspicious_operator"])
        self.assertEqual(warnings[0]["confidence"], "HIGH")

    def test_clean_python_not_flagged(self):
        stmt = _stmt("assignment", "y = x", "y = x * 2")
        result = _result(functions=[_func("f", [stmt], ["x"])])
        self.assertEqual(validate_translation(result), [])


class TestUnresolvedFunction(unittest.TestCase):
    def test_unknown_call_high_confidence(self):
        stmt = _stmt("assignment", "y = smoothdata(x)", "y = smoothdata(x)")
        result = _result(functions=[_func("f", [stmt], ["x"])])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unresolved_function"])
        self.assertIn("smoothdata", warnings[0]["message"])

    def test_numpy_qualified_and_builtin_calls_not_flagged(self):
        statements = [
            _stmt("assignment", "y = sin(x)", "y = np.sin(x)"),
            _stmt("assignment", "n = length(x)", "n = len(x)"),
        ]
        result = _result(functions=[_func("f", statements, ["x"])])
        self.assertEqual(validate_translation(result), [])

    def test_same_file_function_not_flagged(self):
        stmt = _stmt("assignment", "y = helper(x)", "y = helper(x)")
        result = _result(
            functions=[
                _func("main", [stmt], ["x"]),
                _func("helper", [_stmt("assignment", "z = x", "z = x")], ["x"]),
            ]
        )
        self.assertEqual(validate_translation(result), [])


class TestUnsafeTranslation(unittest.TestCase):
    def test_np_shadow_high_confidence(self):
        result = _result(statements=[_stmt("assignment", "np = 5", "np = 5")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsafe_translation"])
        self.assertEqual(warnings[0]["confidence"], "HIGH")
        self.assertIn("np", warnings[0]["message"])

    def test_generator_builtin_shadow_medium_confidence(self):
        result = _result(statements=[_stmt("assignment", "len = 5", "len = 5")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsafe_translation"])
        self.assertEqual(warnings[0]["confidence"], "MEDIUM")

    def test_other_builtin_shadow_low_confidence(self):
        result = _result(statements=[_stmt("assignment", "id = 5", "id = 5")])
        warnings = validate_translation(result)
        self.assertEqual(_categories(warnings), ["unsafe_translation"])
        self.assertEqual(warnings[0]["confidence"], "LOW")

    def test_harmless_numpy_qualified_names_not_flagged(self):
        result = _result(
            statements=[_stmt("assignment", "sum = 5", "sum = 5")]
        )
        self.assertEqual(validate_translation(result), [])


class TestPipelineIntegration(unittest.TestCase):
    def test_warnings_from_real_translation(self):
        result = translate_source(
            "function y = demo(x)\n    y = smoothdata(x);\n    if x > 0\n        y = y + 1;\n    end\nend"
        )
        section = result["sections"]["validation"]
        self.assertEqual(section["status"], "ok")
        self.assertIn("unresolved_function", section["counts"])
        self.assertIn("unsupported_construct", section["counts"])
        self.assertEqual(section["counts"]["unresolved_function"], 1)

    def test_clean_sample_has_empty_warnings(self):
        result = translate_file(sample_matlab("fft_basic.m"))
        section = result["sections"]["validation"]
        self.assertEqual(section["status"], "ok")
        self.assertEqual(section["warnings"], [])
        self.assertEqual(section["counts"], {})

    def test_reverse_direction_section_skipped(self):
        result = translate_source("x = 1", direction=PYTHON_TO_MATLAB)
        self.assertEqual(
            result["sections"]["validation"],
            {"status": "skipped", "warnings": []},
        )

    def test_warnings_are_advisory_never_change_status(self):
        result = translate_source(
            "function y = demo(x)\n    y = smoothdata(x);\nend"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sections"]["validation"]["status"], "ok")

    def test_line_numbers_are_1_based(self):
        result = translate_source(
            "function y = demo(x)\n    a = 1;\n    y = smoothdata(x);\nend"
        )
        warnings = result["sections"]["validation"]["warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["line"], 3)


if __name__ == "__main__":
    unittest.main()
