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
