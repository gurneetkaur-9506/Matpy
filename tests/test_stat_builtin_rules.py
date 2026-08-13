import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader.structure import Statement, Structure
from rulebook import UNRESOLVED, translate_with_rulebook

parser = Parser(Language(language()))


def _translate(text):
    """Translate a single MATLAB assignment statement through the full
    rulebook pipeline and return the resulting Python expression."""
    structure = Structure(statements=[Statement("assignment", text)])
    result = translate_with_rulebook(structure)
    return result["statements"][0]["python"]


class TestDimBuiltins(unittest.TestCase):
    """prod/median/cumsum/cumsum/diff reduce the whole array with one
    argument and take an explicit numpy axis when given the MATLAB
    1-based dimension argument."""

    def test_prod(self):
        self.assertEqual(_translate("p = prod(x)"), "p = np.prod(x)")
        self.assertEqual(_translate("p = prod(x, 2)"), "p = np.prod(x, axis=1)")
        self.assertEqual(_translate("p = prod(A, dim)"), "p = np.prod(A, axis=(dim - 1))")

    def test_median(self):
        self.assertEqual(_translate("m = median(x)"), "m = np.median(x)")
        self.assertEqual(_translate("m = median(x, 2)"), "m = np.median(x, axis=1)")

    def test_cumsum(self):
        self.assertEqual(_translate("s = cumsum(x)"), "s = np.cumsum(x)")
        self.assertEqual(_translate("s = cumsum(x, 2)"), "s = np.cumsum(x, axis=1)")

    def test_cumprod(self):
        self.assertEqual(_translate("c = cumprod(x)"), "c = np.cumprod(x)")
        self.assertEqual(_translate("c = cumprod(x, 2)"), "c = np.cumprod(x, axis=1)")

    def test_dim_builtin_composition(self):
        self.assertEqual(
            _translate("s = cumsum(abs(x))"), "s = np.cumsum(np.abs(x))"
        )
        self.assertEqual(
            _translate("p = prod(find(x > 0))"), "p = np.prod(np.where(x > 0)[0])"
        )


class TestDiffRule(unittest.TestCase):
    def test_diff_one_arg(self):
        self.assertEqual(_translate("d = diff(x)"), "d = np.diff(x)")

    def test_diff_order(self):
        self.assertEqual(_translate("d = diff(x, 2)"), "d = np.diff(x, 2)")

    def test_diff_dim(self):
        self.assertEqual(_translate("d = diff(x, 2, 1)"), "d = np.diff(x, 2, axis=0)")
        self.assertEqual(_translate("d = diff(x, 1, 2)"), "d = np.diff(x, 1, axis=1)")
        self.assertEqual(
            _translate("d = diff(x, 2, dim)"), "d = np.diff(x, 2, axis=(dim - 1))"
        )


class TestVarStdRule(unittest.TestCase):
    """MATLAB var/std default to N-1 normalization, i.e. numpy ddof=1;
    w=1 means N normalization, i.e. numpy's default ddof=0."""

    def test_var_default(self):
        self.assertEqual(_translate("v = var(x)"), "v = np.var(x, ddof=1)")

    def test_var_w_forms(self):
        self.assertEqual(_translate("v = var(x, 0)"), "v = np.var(x, ddof=1)")
        self.assertEqual(_translate("v = var(x, 1)"), "v = np.var(x)")

    def test_var_dim(self):
        self.assertEqual(_translate("v = var(x, 0, 2)"), "v = np.var(x, ddof=1, axis=1)")
        self.assertEqual(_translate("v = var(x, 1, 2)"), "v = np.var(x, axis=1)")

    def test_var_weight_vector_stays_unresolved(self):
        self.assertEqual(_translate("v = var(x, w)"), UNRESOLVED)

    def test_std_default(self):
        self.assertEqual(_translate("s = std(x)"), "s = np.std(x, ddof=1)")

    def test_std_w_forms(self):
        self.assertEqual(_translate("s = std(x, 0)"), "s = np.std(x, ddof=1)")
        self.assertEqual(_translate("s = std(x, 1)"), "s = np.std(x)")

    def test_std_dim(self):
        self.assertEqual(_translate("s = std(x, 0, 1)"), "s = np.std(x, ddof=1, axis=0)")
        self.assertEqual(_translate("s = std(x, 1, 1)"), "s = np.std(x, axis=0)")

    def test_std_weight_vector_stays_unresolved(self):
        self.assertEqual(_translate("s = std(x, w)"), UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
