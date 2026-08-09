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


if __name__ == "__main__":
    unittest.main()
