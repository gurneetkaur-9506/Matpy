import unittest

import pytest

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

    def test_round(self):
        self.assertEqual(apply_builtin_rule("round(x)"), "np.round(x)")
        self.assertEqual(apply_builtin_rule("round(x, 2)"), "np.round(x, 2)")

    def test_floor(self):
        self.assertEqual(apply_builtin_rule("floor(x)"), "np.floor(x)")
        self.assertEqual(apply_builtin_rule("floor(2.7)"), "np.floor(2.7)")

    def test_ceil(self):
        self.assertEqual(apply_builtin_rule("ceil(x)"), "np.ceil(x)")
        self.assertEqual(apply_builtin_rule("ceil(2.1)"), "np.ceil(2.1)")

    def test_fix(self):
        self.assertEqual(apply_builtin_rule("fix(x)"), "np.trunc(x)")
        self.assertEqual(apply_builtin_rule("fix(-2.7)"), "np.trunc(-2.7)")

    def test_rounding_nested(self):
        self.assertEqual(apply_builtin_rule("floor(abs(x))"), "np.floor(np.abs(x))")
        self.assertEqual(apply_builtin_rule("round(ceil(x))"), "np.round(np.ceil(x))")
        self.assertEqual(apply_builtin_rule("fix(size(m))"), "np.trunc(m.shape)")

    def test_randn(self):
        self.assertEqual(
            apply_builtin_rule("randn(size(X))"), "np.random.randn(*X.shape)"
        )
        self.assertEqual(apply_builtin_rule("randn(3, 5)"), "np.random.randn(3, 5)")
        self.assertEqual(apply_builtin_rule("randn(3)"), "np.random.randn(3)")

    def test_randn_with_other_builtin_shape(self):
        self.assertEqual(
            apply_builtin_rule("randn(size(theta))"), "np.random.randn(*theta.shape)"
        )

    def test_hann(self):
        self.assertEqual(apply_builtin_rule("hann(N)"), "scipy.signal.windows.hann(N)")
        self.assertEqual(apply_builtin_rule("hann(256)"), "scipy.signal.windows.hann(256)")

    def test_unknown_call_passthrough(self):
        self.assertEqual(apply_builtin_rule("myfunc(x)"), "myfunc(x)")

    def test_non_call_passthrough(self):
        self.assertEqual(apply_builtin_rule("a + b"), "a + b")
        self.assertEqual(apply_builtin_rule("x"), "x")


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Inverse/reciprocal trigonometric functions
        ("acos(x)", "np.arccos(x)"),
        ("acosh(x)", "np.arccosh(x)"),
        ("asin(x)", "np.arcsin(x)"),
        ("asinh(x)", "np.arcsinh(x)"),
        ("atan(x)", "np.arctan(x)"),
        ("atanh(x)", "np.arctanh(x)"),
        ("atan2(y, x)", "np.arctan2(y, x)"),
        # Hyperbolic functions
        ("cosh(x)", "np.cosh(x)"),
        ("sinh(x)", "np.sinh(x)"),
        ("tanh(x)", "np.tanh(x)"),
        # Logarithms / exponentials
        ("log2(x)", "np.log2(x)"),
        ("log1p(x)", "np.log1p(x)"),
        ("expm1(x)", "np.expm1(x)"),
        ("pow2(x)", "np.exp2(x)"),
        # Sign, rounding-modulo family
        ("sign(x)", "np.sign(x)"),
        ("mod(a, b)", "np.mod(a, b)"),
        ("rem(a, b)", "np.fmod(a, b)"),
        ("hypot(a, b)", "np.hypot(a, b)"),
        # Complex parts and phase
        ("real(Z)", "np.real(Z)"),
        ("imag(Z)", "np.imag(Z)"),
        ("angle(Z)", "np.angle(Z)"),
        # Linear algebra
        ("det(A)", "np.linalg.det(A)"),
        ("inv(A)", "np.linalg.inv(A)"),
        ("pinv(A)", "np.linalg.pinv(A)"),
        ("diag(v)", "np.diag(v)"),
        ("trace(A)", "np.trace(A)"),
        ("triu(A)", "np.triu(A)"),
        ("tril(A)", "np.tril(A)"),
        ("cross(a, b)", "np.cross(a, b)"),
        ("kron(A, B)", "np.kron(A, B)"),
        # FFT family
        ("ifft(X)", "np.fft.ifft(X)"),
        ("fftshift(X)", "np.fft.fftshift(X)"),
        ("ifftshift(X)", "np.fft.ifftshift(X)"),
        # Array manipulation / predicates
        ("squeeze(A)", "np.squeeze(A)"),
        ("flipud(A)", "np.flipud(A)"),
        ("unique(x)", "np.unique(x)"),
        ("any(x)", "np.any(x)"),
        ("all(x)", "np.all(x)"),
        # Constructors
        ("ones(2, 5)", "np.ones((2, 5))"),
        ("ones(3)", "np.ones(3)"),
        ("ones(m, n)", "np.ones((m, n))"),
        ("logspace(0, 2, 50)", "np.logspace(0, 2, 50)"),
        ("rand(3, 5)", "np.random.rand(3, 5)"),
        ("rand(size(X))", "np.random.rand(*X.shape)"),
    ],
)
def test_extended_builtin_rules(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        ("atan2(sin(y), cos(x))", "np.arctan2(np.sin(y), np.cos(x))"),
        ("inv(det(A))", "np.linalg.inv(np.linalg.det(A))"),
        ("kron(eye(2), ones(2, 2))", "np.kron(np.eye(2), np.ones((2, 2)))"),
        ("angle(fft(x))", "np.angle(np.fft.fft(x))"),
        ("real(ifft(Y))", "np.real(np.fft.ifft(Y))"),
    ],
)
def test_extended_builtin_rules_nested(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


class TestSpecialistBuiltinRules(unittest.TestCase):
    """MATLAB toolbox calls wired to specialist_lib functions via the
    Rulebook's builtin table.  Each must translate to a specialist_lib
    call, never to a numpy/scipy substitute."""

    def test_chirp(self):
        self.assertEqual(
            apply_builtin_rule("chirp(t, 0, 1, 100)"),
            "specialist_lib.chirp(t, 0, 1, 100)",
        )
        self.assertEqual(
            apply_builtin_rule("chirp(t, f0, t1, f1, 'quadratic', 90)"),
            "specialist_lib.chirp(t, f0, t1, f1, 'quadratic', 90)",
        )

    def test_conv(self):
        self.assertEqual(apply_builtin_rule("conv(u, v)"), "specialist_lib.conv(u, v)")
        self.assertEqual(
            apply_builtin_rule("conv(u, v, 'same')"),
            "specialist_lib.conv(u, v, 'same')",
        )

    def test_awgn(self):
        self.assertEqual(
            apply_builtin_rule("awgn(x, 10, 'measured')"),
            "specialist_lib.awgn(x, 10, 'measured')",
        )

    def test_comm_awgn_channel(self):
        self.assertEqual(
            apply_builtin_rule("comm.AWGNChannel(x, 5)"),
            "specialist_lib.awgn(x, 5)",
        )

    def test_steervec(self):
        self.assertEqual(
            apply_builtin_rule("steervec(fc, angles)"),
            "specialist_lib.steering_vector(fc, angles)",
        )

    def test_phased_array_response(self):
        self.assertEqual(
            apply_builtin_rule("phased.ArrayResponse('SensorArray', array)"),
            "specialist_lib.array_factor('SensorArray', array)",
        )

    def test_phased_beamformer(self):
        self.assertEqual(
            apply_builtin_rule("phased.Beamformer(signal, weights)"),
            "specialist_lib.beamform(signal, weights)",
        )

    def test_specialist_output_is_not_numpy_or_scipy(self):
        cases = [
            ("chirp(t, 0, 1, 100)", "specialist_lib.chirp(t, 0, 1, 100)"),
            ("conv(u, v, 'same')", "specialist_lib.conv(u, v, 'same')"),
            ("awgn(x, 10, 'measured')", "specialist_lib.awgn(x, 10, 'measured')"),
            ("steervec(fc, angles)", "specialist_lib.steering_vector(fc, angles)"),
            (
                "phased.ArrayResponse('SensorArray', array)",
                "specialist_lib.array_factor('SensorArray', array)",
            ),
            (
                "phased.Beamformer(signal, weights)",
                "specialist_lib.beamform(signal, weights)",
            ),
            ("comm.AWGNChannel(x, 5)", "specialist_lib.awgn(x, 5)"),
        ]
        for matlab, expected in cases:
            output = apply_builtin_rule(matlab)
            self.assertEqual(output, expected)
            self.assertNotIn("np.", output)
            self.assertNotIn("scipy.", output)

    def test_unknown_call_passthrough_unchanged(self):
        self.assertEqual(apply_builtin_rule("myfunc(x)"), "myfunc(x)")


@pytest.mark.parametrize(
    "matlab,expected",
    [
        ("conv(u, v, 'full')", "specialist_lib.conv(u, v, 'full')"),
        ("conv(u, v, 'valid')", "specialist_lib.conv(u, v, 'valid')"),
        ("awgn(x, 10)", "specialist_lib.awgn(x, 10)"),
        (
            "comm.AWGNChannel(x, 5, 'SignalPower', 1)",
            "specialist_lib.awgn(x, 5, 'SignalPower', 1)",
        ),
        (
            "steervec(fc, angles, 'WeightsNormalization', 'steer')",
            "specialist_lib.steering_vector(fc, angles, 'WeightsNormalization', 'steer')",
        ),
        (
            "chirp(t, 0, 1, 100, 'linear', 0)",
            "specialist_lib.chirp(t, 0, 1, 100, 'linear', 0)",
        ),
        (
            "phased.ArrayResponse('SensorArray', array, 'Weights', w)",
            "specialist_lib.array_factor('SensorArray', array, 'Weights', w)",
        ),
        (
            "phased.Beamformer(signal, weights, 'Weights', w)",
            "specialist_lib.beamform(signal, weights, 'Weights', w)",
        ),
    ],
)
def test_specialist_option_arguments_passthrough(matlab, expected):
    """Option/name-value arguments after the positional ones stay put."""
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        (
            "conv(chirp(t, 0, 1, 100), h)",
            "specialist_lib.conv(specialist_lib.chirp(t, 0, 1, 100), h)",
        ),
        (
            "awgn(fft(x), snr, 'measured')",
            "specialist_lib.awgn(np.fft.fft(x), snr, 'measured')",
        ),
        (
            "chirp(fft(t), 0, 1, 100)",
            "specialist_lib.chirp(np.fft.fft(t), 0, 1, 100)",
        ),
    ],
)
def test_specialist_nested_arguments_translate(matlab, expected):
    """Arguments nested inside a specialist call translate recursively."""
    assert apply_builtin_rule(matlab) == expected


if __name__ == "__main__":
    unittest.main()
