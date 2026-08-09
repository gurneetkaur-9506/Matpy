import unittest

from rulebook import apply_builtin_rule_reverse


class TestApplyBuiltinRuleReverse(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(apply_builtin_rule_reverse("np.zeros((2, 5))"), "zeros(2, 5)")
        self.assertEqual(apply_builtin_rule_reverse("np.zeros(3)"), "zeros(3)")
        self.assertEqual(apply_builtin_rule_reverse("np.zeros((2, 5, 3))"), "zeros(2, 5, 3)")

    def test_linspace(self):
        self.assertEqual(
            apply_builtin_rule_reverse("np.linspace(0, 2*pi, 10)"),
            "linspace(0, 2*pi, 10)",
        )
        self.assertEqual(
            apply_builtin_rule_reverse("np.linspace(0, 1, 100)"),
            "linspace(0, 1, 100)",
        )

    def test_reshape(self):
        self.assertEqual(apply_builtin_rule_reverse("np.reshape(m, (5, 2))"), "reshape(m, 5, 2)")
        self.assertEqual(apply_builtin_rule_reverse("np.reshape(x, 10)"), "reshape(x, 10)")
        self.assertEqual(apply_builtin_rule_reverse("np.reshape(m, (5, -1))"), "reshape(m, 5, -1)")

    def test_fft(self):
        self.assertEqual(apply_builtin_rule_reverse("np.fft.fft(x)"), "fft(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.fft.fft(x, 512)"), "fft(x, 512)")

    def test_abs(self):
        self.assertEqual(apply_builtin_rule_reverse("np.abs(Y)"), "abs(Y)")
        self.assertEqual(apply_builtin_rule_reverse("np.abs(-3)"), "abs(-3)")

    def test_randn(self):
        self.assertEqual(
            apply_builtin_rule_reverse("np.random.randn(*X.shape)"), "randn(size(X))"
        )
        self.assertEqual(apply_builtin_rule_reverse("np.random.randn(3, 5)"), "randn(3, 5)")
        self.assertEqual(apply_builtin_rule_reverse("np.random.randn(3)"), "randn(3)")

    def test_nested_abs_fft(self):
        self.assertEqual(
            apply_builtin_rule_reverse("np.abs(np.fft.fft(x))"), "abs(fft(x))"
        )

    def test_other_simple_builtins(self):
        self.assertEqual(apply_builtin_rule_reverse("np.sin(x)"), "sin(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.cos(theta)"), "cos(theta)")
        self.assertEqual(apply_builtin_rule_reverse("np.exp(-j*2*pi*f)"), "exp(-j*2*pi*f)")
        self.assertEqual(apply_builtin_rule_reverse("np.sqrt(x)"), "sqrt(x)")

    def test_unknown_call_passthrough(self):
        self.assertEqual(apply_builtin_rule_reverse("np.linalg.det(a)"), "np.linalg.det(a)")
        self.assertEqual(apply_builtin_rule_reverse("myfunc(x)"), "myfunc(x)")

    def test_operator_expression_not_misparsed_as_single_call(self):
        # A greedy 'name(.*)' regex reads 'np.sin(theta) - np.sin(theta0)'
        # as the call 'np.sin(theta) - np.sin(theta0)'. The balanced splitter
        # must refuse it so operators are handled by the operator rules.
        self.assertEqual(
            apply_builtin_rule_reverse("np.sin(theta) - np.sin(theta0)"),
            "np.sin(theta) - np.sin(theta0)",
        )
        self.assertEqual(
            apply_builtin_rule_reverse("np.abs(a) + np.cos(b)"),
            "np.abs(a) + np.cos(b)",
        )
        self.assertEqual(apply_builtin_rule_reverse("a + b"), "a + b")

    def test_round_trip_with_forward(self):
        from rulebook import apply_builtin_rule

        cases = [
            "zeros(2, 5)",
            "zeros(3)",
            "linspace(0, 2*pi, 10)",
            "linspace(0, 1, 100)",
            "reshape(m, 5, 2)",
            "reshape(x, 10)",
            "fft(x)",
            "fft(x, 512)",
            "abs(Y)",
            "abs(-3)",
            "abs(fft(x))",
        ]
        for matlab in cases:
            self.assertEqual(
                apply_builtin_rule_reverse(apply_builtin_rule(matlab)).replace(" ", ""),
                matlab.replace(" ", ""),
            )


class TestReverseIdenticalNamingBuiltins(unittest.TestCase):
    """Functions whose numpy and MATLAB names are identical must reverse
    translate with the name unchanged -- only the ``np.`` prefix is
    dropped, the call is never renamed or rewritten."""

    def test_round_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.round(x)"), "round(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.round(x, 2)"), "round(x, 2)")

    def test_abs_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.abs(x)"), "abs(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.abs(-3)"), "abs(-3)")

    def test_sqrt_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.sqrt(x)"), "sqrt(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.sqrt(a + b)"), "sqrt(a + b)")

    def test_exp_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.exp(x)"), "exp(x)")
        self.assertEqual(
            apply_builtin_rule_reverse("np.exp(-j*2*pi*f)"), "exp(-j*2*pi*f)"
        )

    def test_log_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.log(x)"), "log(x)")
        self.assertEqual(apply_builtin_rule_reverse("np.log(10)"), "log(10)")

    def test_sin_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.sin(x)"), "sin(x)")
        self.assertEqual(
            apply_builtin_rule_reverse("np.sin(2*pi*f*t)"), "sin(2*pi*f*t)"
        )

    def test_cos_passes_through(self):
        self.assertEqual(apply_builtin_rule_reverse("np.cos(theta)"), "cos(theta)")
        self.assertEqual(
            apply_builtin_rule_reverse("np.cos(2*pi*f*t)"), "cos(2*pi*f*t)"
        )


if __name__ == "__main__":
    unittest.main()
