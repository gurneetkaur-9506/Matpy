import unittest

from rulebook import apply_indexing_rule


class TestApplyIndexingRule(unittest.TestCase):
    def test_integer_index(self):
        self.assertEqual(apply_indexing_rule("1"), "0")
        self.assertEqual(apply_indexing_rule("2"), "1")
        self.assertEqual(apply_indexing_rule("7"), "6")

    def test_colon(self):
        self.assertEqual(apply_indexing_rule(":"), ":")

    def test_variable_index(self):
        self.assertEqual(apply_indexing_rule("i"), "i")
        self.assertEqual(apply_indexing_rule("row"), "row")

    def test_end_keyword(self):
        self.assertEqual(apply_indexing_rule("end"), "-1")
        self.assertEqual(apply_indexing_rule("end-1"), "-2")
        self.assertEqual(apply_indexing_rule("end - 2"), "-3")

    def test_range(self):
        self.assertEqual(apply_indexing_rule("1:5"), "0:5")
        self.assertEqual(apply_indexing_rule("6:10"), "5:10")
        self.assertEqual(apply_indexing_rule("2:end-1"), "1:-1")
        self.assertEqual(apply_indexing_rule("1:end"), "0:")
        self.assertEqual(apply_indexing_rule(":end"), ":")

    def test_length_in_range(self):
        self.assertEqual(apply_indexing_rule("1:length(P2)/2+1"), "0:len(P2)/2+1")
    def test_full_index_expr(self):
        self.assertEqual(apply_indexing_rule("A(1,1)"), "A[0, 0]")
        self.assertEqual(apply_indexing_rule("A(2,3)"), "A[1, 2]")
        self.assertEqual(apply_indexing_rule("A(1,:)"), "A[0, :]")
        self.assertEqual(apply_indexing_rule("x(1:5)"), "x[0:5]")
        self.assertEqual(apply_indexing_rule("x(6:10)"), "x[5:10]")
        self.assertEqual(apply_indexing_rule("P1(2:end-1)"), "P1[1:-1]")

    def test_length_call(self):
        self.assertEqual(apply_indexing_rule("length(P2)"), "len(P2)")

    def test_unknown_passthrough(self):
        self.assertEqual(apply_indexing_rule("a+b"), "a+b")


if __name__ == "__main__":
    unittest.main()
