import unittest

from rulebook import SEQUENCE_RULES, apply_sequence_rule_reverse


class TestSequenceRuleReverse(unittest.TestCase):
    def test_table_has_all_family_members(self):
        self.assertEqual(
            SEQUENCE_RULES,
            {"arange": "range", "linspace": "linspace", "zeros": "zeros", "ones": "ones"},
        )

    def test_arange_three_args(self):
        self.assertEqual(apply_sequence_rule_reverse("np.arange(0, 10, 2)"), "0:2:10")
        self.assertEqual(apply_sequence_rule_reverse("np.arange(-pi, pi, 0.01)"), "-pi:0.01:pi")

    def test_arange_two_args(self):
        self.assertEqual(apply_sequence_rule_reverse("np.arange(0, 10)"), "0:10")
        self.assertEqual(apply_sequence_rule_reverse("np.arange(1, len(x))"), "1:len(x)")

    def test_arange_one_arg(self):
        self.assertEqual(apply_sequence_rule_reverse("np.arange(N)"), "0:N-1")
        self.assertEqual(apply_sequence_rule_reverse("np.arange(5)"), "0:5-1")

    def test_arange_unmatched_arg_count_passthrough(self):
        self.assertEqual(
            apply_sequence_rule_reverse("np.arange(1, 10, 2, 5)"), "np.arange(1, 10, 2, 5)"
        )

    def test_linspace_direct_match(self):
        self.assertEqual(
            apply_sequence_rule_reverse("np.linspace(0, 1, 100)"), "linspace(0, 1, 100)"
        )
        self.assertEqual(
            apply_sequence_rule_reverse("np.linspace(-pi, pi, 50)"), "linspace(-pi, pi, 50)"
        )

    def test_zeros_scalar_prepends_row(self):
        self.assertEqual(apply_sequence_rule_reverse("np.zeros(3)"), "zeros(1, 3)")
        self.assertEqual(apply_sequence_rule_reverse("np.zeros(N)"), "zeros(1, N)")

    def test_zeros_tuple_shape(self):
        self.assertEqual(apply_sequence_rule_reverse("np.zeros((2, 5))"), "zeros(2, 5)")
        self.assertEqual(
            apply_sequence_rule_reverse("np.zeros((2, 5, 3))"), "zeros(2, 5, 3)"
        )

    def test_ones_scalar_prepends_row(self):
        self.assertEqual(apply_sequence_rule_reverse("np.ones(5)"), "ones(1, 5)")
        self.assertEqual(apply_sequence_rule_reverse("np.ones(M)"), "ones(1, M)")

    def test_ones_tuple_shape(self):
        self.assertEqual(apply_sequence_rule_reverse("np.ones((4, 1))"), "ones(4, 1)")

    def test_numpy_prefix_also_supported(self):
        self.assertEqual(apply_sequence_rule_reverse("numpy.arange(0, 5, 1)"), "0:1:5")
        self.assertEqual(apply_sequence_rule_reverse("numpy.zeros(2)"), "zeros(1, 2)")

    def test_other_calls_pass_through(self):
        self.assertEqual(
            apply_sequence_rule_reverse("np.sin(x)"), "np.sin(x)"
        )
        self.assertEqual(
            apply_sequence_rule_reverse("myfunc(x)"), "myfunc(x)"
        )


if __name__ == "__main__":
    unittest.main()
