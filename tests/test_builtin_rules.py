import unittest

from rulebook import apply_builtin_rule


class TestApplyBuiltinRule(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(apply_builtin_rule("zeros(2, 5)"), "np.zeros((2, 5))")
        self.assertEqual(apply_builtin_rule("zeros(2,5)"), "np.zeros((2, 5))")
        self.assertEqual(apply_builtin_rule("zeros(3)"), "np.zeros(3)")

    def test_linspace(self):
        self.assertEqual(apply_builtin_rule("linspace(0, 2*pi, 10)"), "np.linspace(0, 2*pi, 10)")
        self.assertEqual(apply_builtin_rule("linspace(0, 1, 100)"), "np.linspace(0, 1, 100)")

    def test_reshape(self):
        self.assertEqual(apply_builtin_rule("reshape(m, 5, 2)"), "np.reshape(m, (5, 2))")
        self.assertEqual(apply_builtin_rule("reshape(x, 10)"), "np.reshape(x, 10)")

    def test_size(self):
        self.assertEqual(apply_builtin_rule("size(m)"), "m.shape")
        self.assertEqual(apply_builtin_rule("size(m, 1)"), "m.shape[0]")
        self.assertEqual(apply_builtin_rule("size(m, 2)"), "m.shape[1]")
        self.assertEqual(apply_builtin_rule("size(x, dim)"), "x.shape[(dim - 1)]")

    def test_nested_calls(self):
        self.assertEqual(apply_builtin_rule("zeros(size(theta))"), "np.zeros(theta.shape)")
        self.assertEqual(apply_builtin_rule("size(reshape(m, 5, 2))"), "np.reshape(m, (5, 2)).shape")

    def test_fft(self):
        self.assertEqual(apply_builtin_rule("fft(x)"), "np.fft.fft(x)")
        self.assertEqual(apply_builtin_rule("fft(x, 512)"), "np.fft.fft(x, 512)")

    def test_abs(self):
        self.assertEqual(apply_builtin_rule("abs(Y)"), "np.abs(Y)")
        self.assertEqual(apply_builtin_rule("abs(-3)"), "np.abs(-3)")

    def test_nested_abs_fft(self):
        self.assertEqual(apply_builtin_rule("abs(fft(x))"), "np.abs(np.fft.fft(x))")

    def test_sin(self):
        self.assertEqual(apply_builtin_rule("sin(x)"), "np.sin(x)")
        self.assertEqual(apply_builtin_rule("sin(2*pi*f*t)"), "np.sin(2*pi*f*t)")

    def test_cos(self):
        self.assertEqual(apply_builtin_rule("cos(x)"), "np.cos(x)")
        self.assertEqual(apply_builtin_rule("cos(theta)"), "np.cos(theta)")

    def test_tan(self):
        self.assertEqual(apply_builtin_rule("tan(x)"), "np.tan(x)")
        self.assertEqual(apply_builtin_rule("tan(phi, 1)"), "np.tan(phi, 1)")

    def test_sqrt(self):
        self.assertEqual(apply_builtin_rule("sqrt(x)"), "np.sqrt(x)")
        self.assertEqual(apply_builtin_rule("sqrt(a + b)"), "np.sqrt(a + b)")

    def test_exp(self):
        self.assertEqual(apply_builtin_rule("exp(x)"), "np.exp(x)")
        self.assertEqual(apply_builtin_rule("exp(-j*2*pi*f)"), "np.exp(-j*2*pi*f)")

    def test_log(self):
        self.assertEqual(apply_builtin_rule("log(x)"), "np.log(x)")
        self.assertEqual(apply_builtin_rule("log(abs(Y))"), "np.log(np.abs(Y))")

    def test_unknown_call_passthrough(self):
        self.assertEqual(apply_builtin_rule("myfunc(x)"), "myfunc(x)")

    def test_non_call_passthrough(self):
        self.assertEqual(apply_builtin_rule("a + b"), "a + b")
        self.assertEqual(apply_builtin_rule("x"), "x")


if __name__ == "__main__":
    unittest.main()
