import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure, load_matlab_file
from rulebook import UNRESOLVED, translate_with_rulebook


class TestTranslateWithRulebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = Parser(Language(language()))
        tree = parser.parse(load_matlab_file("/sample_matlab/indexing_ops.m").encode("utf-8"))
        cls.structure = build_structure(tree)

    def _translations(self):
        result = translate_with_rulebook(self.structure)
        return [s["python"] for s in result["statements"]]

    def test_fully_resolved(self):
        translations = self._translations()
        self.assertNotIn(UNRESOLVED, translations)

    def test_expected_lines(self):
        translations = [t for t in self._translations() if t]
        expected = [
            "A = np.array([[1, 2, 3], [4, 5, 6]])",
            "B = np.array([[7, 8], [9, 10], [11, 12]])",
            "print('A(1,1) (first row, first col):')",
            "print(A[0, 0])",
            "print('A(2,3) (second row, third col):')",
            "print(A[1, 2])",
            "print('Matrix multiplication A * B:')",
            "print(A @ B)",
            "print('Element-wise multiplication A .* A:')",
            "print(A * A)",
            "print('First row of A via 1-based indexing:')",
            "print(A[0, :])",
        ]
        self.assertEqual(translations, expected)


if __name__ == "__main__":
    unittest.main()
