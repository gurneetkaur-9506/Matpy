import unittest

from reader import (
    MATLAB_TO_PYTHON,
    join_line_continuations,
    load_structure_from_source,
)
from rulebook import translate_with_rulebook


class TestJoinLineContinuations(unittest.TestCase):
    def test_joins_two_lines_ending_in_ellipsis(self):
        source = "fprintf('The value is %d', ...\n    x);\n"
        self.assertEqual(
            join_line_continuations(source),
            "fprintf('The value is %d', x);\n",
        )

    def test_multiple_continuations_joined(self):
        source = (
            "disp('first', ...\n"
            "  'second', ...\n"
            "  'third');\n"
        )
        self.assertEqual(
            join_line_continuations(source),
            "disp('first', 'second', 'third');\n",
        )

    def test_non_continuation_lines_untouched(self):
        source = "a = 1;\nb = 2;\n"
        self.assertEqual(join_line_continuations(source), source)

    def test_empty_source(self):
        self.assertEqual(join_line_continuations(""), "")


class TestLineContinuationParsing(unittest.TestCase):
    def _translate(self, source):
        structure = load_structure_from_source(source, MATLAB_TO_PYTHON)
        result = translate_with_rulebook(structure)
        return [s["python"] for s in result["statements"]]

    def test_two_line_fprintf_split_by_ellipsis(self):
        source = "fprintf('The value is %d', ...\n    x);\n"
        lines = self._translate(source)
        self.assertEqual(lines, ['print("The value is %d" % (x))'])
        self.assertNotIn("...", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
