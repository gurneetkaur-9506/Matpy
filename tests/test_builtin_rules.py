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


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Normal (positional) arguments.
        ("butter(4, 0.2)", "scipy.signal.butter(4, 0.2)"),
        ("butter(6, [0.2, 0.5], 'bandpass')", "scipy.signal.butter(6, [0.2, 0.5], 'bandpass')"),
        ("filter(b, a, x)", "scipy.signal.lfilter(b, a, x)"),
        ("filtfilt(b, a, x)", "scipy.signal.filtfilt(b, a, x)"),
        ("freqz(b, a, 256)", "specialist_lib.freqz(b, a, 256)"),
        ("detrend(x)", "specialist_lib.detrend(x)"),
        ("sawtooth(t, 0.5)", "scipy.signal.sawtooth(t, 0.5)"),
        ("square(t, 25)", "specialist_lib.square(t, 25)"),
        ("conv2(A, B)", "scipy.signal.convolve2d(A, B)"),
        ("decimate(x, 4)", "scipy.signal.decimate(x, 4)"),
        ("resample(x, 3, 2)", "scipy.signal.resample_poly(x, 3, 2)"),
        ("medfilt1(x, 5)", "specialist_lib.medfilt1(x, 5)"),
        ("hamming(64)", "scipy.signal.windows.hamming(64)"),
        ("blackman(64)", "scipy.signal.windows.blackman(64)"),
        ("kaiser(64, 5)", "scipy.signal.windows.kaiser(64, 5)"),
        ("cheby1(4, 3, 0.2)", "scipy.signal.cheby1(4, 3, 0.2)"),
        ("cheby2(4, 20, 0.2)", "scipy.signal.cheby2(4, 20, 0.2)"),
        ("ellip(4, 1, 40, [0.2, 0.5])", "scipy.signal.ellip(4, 1, 40, [0.2, 0.5])"),
        ("sosfilt(sos, x)", "scipy.signal.sosfilt(sos, x)"),
        ("upfirdn(h, x, 3, 2)", "scipy.signal.upfirdn(h, x, 3, 2)"),
        ("hilbert(x)", "scipy.signal.hilbert(x)"),
        ("periodogram(x, fs)", "scipy.signal.periodogram(x, fs)"),
        ("welch(x)", "scipy.signal.welch(x)"),
        ("pwelch(x)", "scipy.signal.welch(x)"),
        ("bartlett(64)", "scipy.signal.windows.bartlett(64)"),
        ("triang(64)", "scipy.signal.windows.triang(64)"),
        ("barthannwin(32)", "scipy.signal.windows.barthann(32)"),
        ("blackmanharris(128)", "scipy.signal.windows.blackmanharris(128)"),
        ("nuttallwin(64)", "scipy.signal.windows.nuttall(64)"),
        ("flattopwin(64)", "scipy.signal.windows.flattop(64)"),
        ("chebwin(64, 60)", "scipy.signal.windows.chebwin(64, 60)"),
        ("tukeywin(64, 0.5)", "scipy.signal.windows.tukey(64, 0.5)"),
        ("rectwin(16)", "np.ones(16)"),
    ],
)
def test_scipy_signal_builtin_rules(matlab, expected):
    """MATLAB DSP/toolbox calls map onto their scipy.signal equivalents."""
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Option/name-value arguments after the positional ones stay put.
        ("conv2(A, B, 'same')", "scipy.signal.convolve2d(A, B, 'same')"),
        ("conv2(A, B, 'valid')", "scipy.signal.convolve2d(A, B, 'valid')"),
        ("detrend(x, 'linear')", "specialist_lib.detrend(x, 'linear')"),
        ("detrend(x, 'linear', 3)", "specialist_lib.detrend(x, 'linear', 3)"),
        ("square(t, 75)", "specialist_lib.square(t, 75)"),
        ("medfilt1(x)", "specialist_lib.medfilt1(x)"),
        ("sawtooth(t)", "scipy.signal.sawtooth(t)"),
        ("kaiser(N, beta)", "scipy.signal.windows.kaiser(N, beta)"),
    ],
)
def test_scipy_signal_option_arguments_passthrough(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Nested arguments inside DSP calls translate recursively.
        ("butter(4, 2 * pi * fc / fs)", "scipy.signal.butter(4, 2 * pi * fc / fs)"),
        ("filter(butter(2, 0.3), x)", "scipy.signal.lfilter(scipy.signal.butter(2, 0.3), x)"),
        ("conv2(conv2(A, B), C)", "scipy.signal.convolve2d(scipy.signal.convolve2d(A, B), C)"),
        ("hamming(2 * N)", "scipy.signal.windows.hamming(2 * N)"),
        ("square(chirp(t, 0, 1, 10), 50)", "specialist_lib.square(specialist_lib.chirp(t, 0, 1, 10), 50)"),
        ("filtfilt(b, a, awgn(x, 10))", "scipy.signal.filtfilt(b, a, specialist_lib.awgn(x, 10))"),
    ],
)
def test_scipy_signal_nested_arguments_translate(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        ("[b, a] = butter(4, 0.2)", "b, a = scipy.signal.butter(4, 0.2)"),
        ("[y, zf] = filter(b, a, x)", "y, zf = specialist_lib.filter_with_state(b, a, x)"),
        ("[pks, locs] = findpeaks(x)", "pks, locs = specialist_lib.findpeaks(x)"),
        ("[r, lags] = xcorr(x, y)", "r, lags = specialist_lib.xcorr(x, y)"),
        ("[h, w] = freqz(b, a, 256)", "h, w = specialist_lib.freqz(b, a, 256)"),
        ("[~, a] = butter(4, 0.2)", "a = scipy.signal.butter(4, 0.2)[1]"),
    ],
)
def test_scipy_signal_multi_output(matlab, expected):
    from rulebook.multi_output_rules import translate_multi_output_assignment

    lines = translate_multi_output_assignment(
        matlab.split("=")[0].strip(), matlab.split("=")[1].strip(), lambda a: a
    )
    assert lines is not None
    assert lines[0] == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Normal (positional) arguments for numerical / linear-algebra /
        # engineering mappings.
        ("norm(A)", "np.linalg.norm(A)"),
        ("norm(A, 'fro')", "np.linalg.norm(A, 'fro')"),
        ("cond(A)", "np.linalg.cond(A)"),
        ("rank(A)", "np.linalg.matrix_rank(A)"),
        ("chol(A)", "np.linalg.cholesky(A)"),
        ("qr(A)", "np.linalg.qr(A)"),
        ("eig(A)", "np.linalg.eigvals(A)"),
        ("expm(A)", "scipy.linalg.expm(A)"),
        ("logm(A)", "scipy.linalg.logm(A)"),
        ("sqrtm(A)", "scipy.linalg.sqrtm(A)"),
        ("toeplitz(c)", "scipy.linalg.toeplitz(c)"),
        ("hankel(c)", "scipy.linalg.hankel(c)"),
        ("polyfit(x, y, 2)", "np.polyfit(x, y, 2)"),
        ("polyval(p, x)", "np.polyval(p, x)"),
        ("roots(p)", "np.roots(p)"),
        ("polyint(p)", "np.polyint(p)"),
        ("polyder(p)", "np.polyder(p)"),
        ("gradient(y)", "np.gradient(y)"),
        ("trapz(y, x)", "np.trapezoid(y, x)"),
        ("unwrap(ph)", "np.unwrap(ph)"),
        ("sinc(t)", "np.sinc(t)"),
        ("conj(z)", "np.conj(z)"),
        ("fft2(X)", "np.fft.fft2(X)"),
        ("ifft2(Y)", "np.fft.ifft2(Y)"),
        ("fftn(X)", "np.fft.fftn(X)"),
        ("ifftn(Y)", "np.fft.ifftn(Y)"),
        # Single-output svd keeps the singular-value vector.
        ("svd(A)", "np.linalg.svd(A, compute_uv=False)"),
    ],
)
def test_numerical_engineering_builtin_rules(matlab, expected):
    """MATLAB numerical/engineering calls map onto numpy/scipy equivalents."""
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Option/name-value arguments after the positional ones stay put.
        ("pwelch(x, 256)", "scipy.signal.welch(x, 256)"),
        ("pwelch(x, 256, 128, 512, fs)", "scipy.signal.welch(x, 256, 128, 512, fs)"),
        ("cheby1(4, 3, 0.2, 's')", "scipy.signal.cheby1(4, 3, 0.2, 's')"),
        ("ellip(4, 1, 40, [0.2, 0.5], 'bandpass')", "scipy.signal.ellip(4, 1, 40, [0.2, 0.5], 'bandpass')"),
        ("periodogram(x, [], 'power')", "scipy.signal.periodogram(x, [], 'power')"),
        ("norm(A, 'inf')", "np.linalg.norm(A, 'inf')"),
        ("polyfit(x, y, 2, 'west')", "np.polyfit(x, y, 2, 'west')"),
        ("tukeywin(64, 0.5)", "scipy.signal.windows.tukey(64, 0.5)"),
        ("blackmanharris(128)", "scipy.signal.windows.blackmanharris(128)"),
        ("chebwin(64, 60)", "scipy.signal.windows.chebwin(64, 60)"),
    ],
)
def test_numerical_engineering_option_arguments_passthrough(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        # Nested arguments inside numerical/engineering calls translate.
        ("norm(abs(x))", "np.linalg.norm(np.abs(x))"),
        ("qr(conv(A, B))", "np.linalg.qr(specialist_lib.conv(A, B))"),
        ("polyval(p, fft(x))", "np.polyval(p, np.fft.fft(x))"),
        ("gradient(sin(t))", "np.gradient(np.sin(t))"),
        ("unwrap(angle(fft(x)))", "np.unwrap(np.angle(np.fft.fft(x)))"),
        ("sinc(2 * pi * t)", "np.sinc(2 * pi * t)"),
        ("hilbert(awgn(x, 10))", "scipy.signal.hilbert(specialist_lib.awgn(x, 10))"),
        ("ellip(4, 1, 40, 2 * pi * fc / fs)", "scipy.signal.ellip(4, 1, 40, 2 * pi * fc / fs)"),
        ("svd(A)", "np.linalg.svd(A, compute_uv=False)"),
    ],
)
def test_numerical_engineering_nested_arguments_translate(matlab, expected):
    assert apply_builtin_rule(matlab) == expected


@pytest.mark.parametrize(
    "matlab,expected",
    [
        ("[U, S, V] = svd(A)", "U, S, V = specialist_lib.svd(A)"),
        ("[V, D] = eig(A)", "V, D = specialist_lib.eig(A)"),
        ("[b, a] = cheby1(4, 3, 0.2)", "b, a = scipy.signal.cheby1(4, 3, 0.2)"),
        ("[b, a] = cheby2(4, 20, 0.2)", "b, a = scipy.signal.cheby2(4, 20, 0.2)"),
        ("[b, a] = ellip(4, 1, 40, [0.2, 0.5])", "b, a = scipy.signal.ellip(4, 1, 40, [0.2, 0.5])"),
        ("[~, D] = eig(A)", "D = specialist_lib.eig(A)[1]"),
        ("[S, ~, ~] = svd(A)", "S = specialist_lib.svd(A)[0]"),
    ],
)
def test_numerical_engineering_multi_output(matlab, expected):
    from rulebook.multi_output_rules import translate_multi_output_assignment

    lines = translate_multi_output_assignment(
        matlab.split("=")[0].strip(), matlab.split("=")[1].strip(), lambda a: a
    )
    assert lines is not None
    assert lines[0] == expected


if __name__ == "__main__":
    unittest.main()
