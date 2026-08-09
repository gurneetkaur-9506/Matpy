import ast
import keyword
import re
import unittest

from rulebook.keyword_rules import (
    MATLAB_BUILTIN_CALLS,
    MATLAB_KEYWORDS,
    PYTHON_KEYWORDS,
    rename_comment,
    rename_for,
    should_rename,
)
from translator import translate_source


def _bare_regex(name):
    return re.compile(r"\b%s\b" % re.escape(name))


def _bare_uses(code, name):
    """Lines of generated code that use ``name`` as a bare word.

    Comment lines and the ``import numpy as np`` header are excluded so the
    rename note and the import statement do not count as usages.
    """
    pattern = _bare_regex(name)
    return [
        line
        for line in code.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and line.strip() != "import numpy as np"
        and pattern.search(line)
    ]


class TestReservedKeywordRenames(unittest.TestCase):
    def _translate(self, source):
        result = translate_source(source)
        self.assertEqual(result["status"], "ok", result["python"])
        return result

    def _assert_renamed(self, source, name):
        """Translate and assert ``name`` is renamed everywhere to valid Python."""
        result = self._translate(source)
        code = result["python"]
        self.assertIn(rename_for(name), code)
        self.assertEqual(_bare_uses(code, name), [])
        ast.parse(code)
        return result

    def test_lambda_renamed_everywhere(self):
        result = self._assert_renamed(
            "lambda = 5;\ny = lambda * 2;\nz = lambda + y;\n", "lambda"
        )
        self.assertIn("lambda_ = 5", result["python"])
        self.assertIn("y = lambda_ * 2", result["python"])
        self.assertIn("z = lambda_ + y", result["python"])

    def test_class_renamed_everywhere(self):
        result = self._assert_renamed(
            "class = 3;\ntotal = class + 1;\nvalue = class * total;\n", "class"
        )
        self.assertIn("class_ = 3", result["python"])
        self.assertIn("total = class_ + 1", result["python"])

    def test_type_renamed_everywhere(self):
        result = self._assert_renamed("type = 7;\nresult = type * 2;\n", "type")
        self.assertIn("type_ = 7", result["python"])
        self.assertIn("result = type_ * 2", result["python"])

    def test_lambda_in_expression_arguments_renamed(self):
        result = self._translate(
            "lambda = 2;\nwavelength = pi / lambda;\ndisp(lambda);\n"
        )
        self.assertIn("wavelength = np.pi / lambda_", result["python"])
        self.assertIn("print(lambda_)", result["python"])
        self.assertEqual(_bare_uses(result["python"], "lambda"), [])
        ast.parse(result["python"])

    def test_function_parameter_renamed(self):
        result = self._translate(
            "function af = f(N, lambda, class)\n"
            "    af = N + lambda + class;\n"
            "end\n"
        )
        self.assertIn("def f(N, lambda_, class_):", result["python"])
        self.assertIn("af = N + lambda_ + class_", result["python"])
        self.assertEqual(_bare_uses(result["python"], "lambda"), [])
        ast.parse(result["python"])

    def test_loop_variable_renamed(self):
        result = self._translate(
            "function y = f()\n"
            "    total = 0;\n"
            "    for class = 1:3\n"
            "        total = total + class;\n"
            "    end\n"
            "    y = total;\n"
            "end\n"
        )
        self.assertIn("for class_ in range(3):", result["python"])
        self.assertIn("total = total + class_", result["python"])
        self.assertEqual(_bare_uses(result["python"], "class"), [])
        ast.parse(result["python"])

    def test_comment_emitted_once_at_first_occurrence(self):
        code = self._translate("lambda = 1;\ny = lambda;\nz = lambda + 1;\n")[
            "python"
        ]
        expected = "# renamed: MATLAB 'lambda' -> Python 'lambda_' (reserved keyword)"
        self.assertEqual(code.count(expected), 1)
        self.assertLess(code.index(expected), code.index("lambda_ = 1"))

    def test_struct_field_access_not_renamed(self):
        code = self._translate(
            "class = 5;\nname = class;\nobj.type = 3;\n" "obj.value = 4;\n"
        )["python"]
        self.assertIn("class_ = 5", code)
        self.assertIn("name = class_", code)
        self.assertIn("obj.type = 3", code)
        self.assertIn("obj.value = 4", code)
        ast.parse(code)

    def test_indexed_variable_renamed_too(self):
        code = self._translate("class = [1 2 3];\nfirst = class(1);\n")["python"]
        self.assertIn("class_ = np.array([[1, 2, 3]])", code)
        self.assertIn("first = class_[0]", code)
        ast.parse(code)

    def test_source_preserves_original_matlab_text(self):
        result = self._translate("lambda = 5;\ny = lambda * 2;\n")
        sources = {s["source"] for s in result["statements"]}
        self.assertIn("lambda = 5", sources)
        self.assertIn("y = lambda * 2", sources)

    def test_builtin_comment_notes_shadowing(self):
        code = self._translate("type = 7;\n")["python"]
        self.assertIn(
            "# renamed: MATLAB 'type' -> Python 'type_' (shadowed builtin)",
            code,
        )

    def test_keyword_comment_notes_reserved(self):
        self.assertEqual(
            rename_comment("lambda"),
            "# renamed: MATLAB 'lambda' -> Python 'lambda_' (reserved keyword)",
        )

    def test_translated_output_never_uses_bare_keyword(self):
        for name in ("lambda", "class", "type"):
            with self.subTest(name=name):
                self._assert_renamed("%s = 1;\ny = %s;\n" % (name, name), name)


class TestAllReservedKeywords(unittest.TestCase):
    """The collision check must cover ALL Python reserved words, not just
    the documented examples.  Every keyword that MATLAB itself does not
    reserve must be renamed when used as a variable."""

    RENAMABLE = sorted(
        name
        for name in PYTHON_KEYWORDS
        if name.isidentifier()
        and name not in MATLAB_KEYWORDS
        and name not in MATLAB_BUILTIN_CALLS
    )

    def test_every_renamable_keyword_is_renamed(self):
        self.assertTrue(self.RENAMABLE)
        for name in self.RENAMABLE:
            with self.subTest(name=name):
                self.assertTrue(should_rename(name))
                result = translate_source("%s = 1;\nvalue = %s + 1;\n" % (name, name))
                self.assertEqual(result["status"], "ok", name)
                code = result["python"]
                self.assertIn(rename_for(name), code, name)
                self.assertEqual(_bare_uses(code, name), [], (name, code))
                ast.parse(code)

    def test_matlab_keywords_and_builtins_never_renamed(self):
        for name in MATLAB_KEYWORDS:
            with self.subTest(name=name):
                self.assertFalse(should_rename(name))
        for name in MATLAB_BUILTIN_CALLS:
            with self.subTest(name=name):
                self.assertFalse(should_rename(name))

    def test_documented_examples_included(self):
        self.assertIn("lambda", self.RENAMABLE)
        self.assertIn("class", self.RENAMABLE)
        self.assertIn("def", self.RENAMABLE)

    def test_renamed_names_are_valid_python_identifiers(self):
        for name in self.RENAMABLE:
            self.assertTrue(rename_for(name).isidentifier())
        for name in keyword.kwlist:
            self.assertNotEqual(name, rename_for(name))


if __name__ == "__main__":
    unittest.main()
