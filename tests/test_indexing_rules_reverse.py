import unittest

from rulebook import apply_indexing_rule_reverse


class TestApplyIndexingRuleReverse(unittest.TestCase):
    def test_integer_index(self):
        self.assertEqual(apply_indexing_rule_reverse("0"), "1")
        self.assertEqual(apply_indexing_rule_reverse("1"), "2")
        self.assertEqual(apply_indexing_rule_reverse("6"), "7")

    def test_colon(self):
        self.assertEqual(apply_indexing_rule_reverse(":"), ":")

    def test_variable_index(self):
        self.assertEqual(apply_indexing_rule_reverse("i"), "i")
        self.assertEqual(apply_indexing_rule_reverse("row"), "row")

    def test_negative_index(self):
        self.assertEqual(apply_indexing_rule_reverse("-1"), "end")
        self.assertEqual(apply_indexing_rule_reverse("-2"), "end-1")
        self.assertEqual(apply_indexing_rule_reverse("-3"), "end-2")

    def test_range(self):
        self.assertEqual(apply_indexing_rule_reverse("0:5"), "1:5")
        self.assertEqual(apply_indexing_rule_reverse("5:10"), "6:10")
        self.assertEqual(apply_indexing_rule_reverse("1:-1"), "2:end-1")
        self.assertEqual(apply_indexing_rule_reverse("0:"), "1:end")
        self.assertEqual(apply_indexing_rule_reverse(":5"), "1:5")
        self.assertEqual(apply_indexing_rule_reverse(":"), ":")

    def test_length_in_range(self):
        self.assertEqual(
            apply_indexing_rule_reverse("0:len(P2)/2+1"), "1:length(P2)/2+1"
        )

    def test_full_index_expr(self):
        self.assertEqual(apply_indexing_rule_reverse("A[0, 0]"), "A(1, 1)")
        self.assertEqual(apply_indexing_rule_reverse("A[1, 2]"), "A(2, 3)")
        self.assertEqual(apply_indexing_rule_reverse("A[0, :]"), "A(1, :)")
        self.assertEqual(apply_indexing_rule_reverse("x[0:5]"), "x(1:5)")
        self.assertEqual(apply_indexing_rule_reverse("x[5:10]"), "x(6:10)")
        self.assertEqual(apply_indexing_rule_reverse("P1[1:-1]"), "P1(2:end-1)")

    def test_len_call(self):
        self.assertEqual(apply_indexing_rule_reverse("len(P2)"), "length(P2)")

    def test_unknown_passthrough(self):
        self.assertEqual(apply_indexing_rule_reverse("a+b"), "a+b")

    def test_round_trip_with_forward(self):
        from rulebook import apply_indexing_rule

        cases = [
            "A(1,1)",
            "A(2,3)",
            "A(1,:)",
            "x(1:5)",
            "x(6:10)",
            "P1(2:end-1)",
            "length(P2)",
            "1:length(P2)/2+1",
        ]
        for matlab in cases:
            self.assertEqual(
                apply_indexing_rule_reverse(apply_indexing_rule(matlab)).replace(" ", ""),
                matlab.replace(" ", ""),
            )


if __name__ == "__main__":
    unittest.main()
