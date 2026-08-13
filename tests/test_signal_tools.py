import unittest

import numpy as np
from scipy import signal as sp

from specialist_lib import (
    detrend,
    filter_with_state,
    findpeaks,
    freqz,
    medfilt1,
    square,
    xcorr,
)


class TestSquare(unittest.TestCase):
    def test_default_duty_is_50_percent(self):
        t = np.linspace(0, 1, 101)
        np.testing.assert_allclose(square(t), sp.square(t, duty=0.5))

    def test_duty_percent_converts_to_fraction(self):
        t = np.linspace(0, 1, 101)
        for duty in (10, 25, 75, 90):
            with self.subTest(duty=duty):
                np.testing.assert_allclose(square(t, duty), sp.square(t, duty=duty / 100.0))

    def test_three_state_hand_computed(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # duty=50 -> +1 for the first half of each period.
        np.testing.assert_array_equal(
            square(2 * np.pi * t), np.array([1.0, 1.0, -1.0, -1.0, 1.0])
        )


class TestFindpeaks(unittest.TestCase):
    def test_returns_values_and_one_based_locations(self):
        x = np.array([1, 3, 1, 2, 4, 2, 1])
        pks, locs = findpeaks(x)
        np.testing.assert_array_equal(pks, np.array([3, 4]))
        np.testing.assert_array_equal(locs, np.array([2, 5]))

    def test_matches_scipy_indices(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50)
        pks, locs = findpeaks(x)
        inds, _ = sp.find_peaks(x)
        np.testing.assert_array_equal(locs, inds + 1)
        np.testing.assert_allclose(pks, x[inds])


class TestXcorr(unittest.TestCase):
    def test_lags_and_values_match_scipy(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        r, lags = xcorr(x, y)
        np.testing.assert_allclose(r, sp.correlate(x, y, mode="full"))
        np.testing.assert_array_equal(lags, np.arange(-2, 3))

    def test_autocorrelation_lags_symmetric(self):
        x = np.array([1.0, 0.5, -0.5, 1.0])
        r, lags = xcorr(x)
        self.assertEqual(len(r), 7)
        np.testing.assert_array_equal(lags, np.arange(-3, 4))

    def test_coeff_normalization_peak_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        r, lags = xcorr(x, scaleopt="coeff")
        np.testing.assert_allclose(np.max(r), 1.0, atol=1e-12)
        self.assertEqual(lags[np.argmax(r)], 0)

    def test_maxlag_clips(self):
        x = np.arange(10.0)
        r, lags = xcorr(x, maxlag=2)
        np.testing.assert_array_equal(lags, np.arange(-2, 3))


class TestDetrend(unittest.TestCase):
    def test_default_removes_constant(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(detrend(x), sp.detrend(x, type="constant"))

    def test_linear_option_matches_scipy(self):
        x = np.linspace(0, 10, 21) + 0.5 * np.sin(np.linspace(0, 4, 21))
        np.testing.assert_allclose(detrend(x, "linear"), sp.detrend(x, type="linear"))

    def test_breakpoint_option(self):
        x = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        np.testing.assert_allclose(
            detrend(x, "linear", 3), sp.detrend(x, type="linear", bp=3)
        )


class TestMedfilt1(unittest.TestCase):
    def test_default_width_three(self):
        x = np.array([1, 100, 3, 4, 5])
        np.testing.assert_array_equal(medfilt1(x), sp.medfilt(x, kernel_size=3))

    def test_odd_width_matches_scipy(self):
        x = np.array([1, 2, 9, 4, 5, 6, 1])
        np.testing.assert_array_equal(medfilt1(x, 3), sp.medfilt(x, kernel_size=3))
        np.testing.assert_array_equal(medfilt1(x, 5), sp.medfilt(x, kernel_size=5))

    def test_even_width_nudged_to_odd(self):
        # MATLAB allows even widths; scipy requires odd, so the wrapper
        # nudges n=4 up to a 5-point window rather than failing.
        x = np.array([1, 2, 9, 4, 5, 6, 1])
        np.testing.assert_array_equal(medfilt1(x, 4), medfilt1(x, 5))


class TestFilterWithState(unittest.TestCase):
    def test_matches_scipy_lfilter_with_zero_ic(self):
        b = np.array([1.0])
        a = np.array([1.0, -0.5])
        x = np.ones(8)
        y, zf = filter_with_state(b, a, x)
        y_ref, zf_ref = sp.lfilter(b, a, x, zi=np.zeros(1))
        np.testing.assert_allclose(y, y_ref)
        np.testing.assert_allclose(zf, zf_ref)

    def test_final_state_last_sample(self):
        b = np.array([0.5, 0.5])
        a = np.array([1.0, -0.2])
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        _, zf = filter_with_state(b, a, x)
        self.assertEqual(zf.shape, (1,))


class TestFreqz(unittest.TestCase):
    def test_returns_response_then_frequencies(self):
        # MATLAB [h, w] = freqz(...) puts the response first; scipy returns
        # (w, h).  The wrapper must restore MATLAB's order.
        b = np.array([1.0])
        a = np.array([1.0, -0.5])
        h, w = freqz(b, a, 8)
        w_ref, h_ref = sp.freqz(b, a, worN=8)
        np.testing.assert_allclose(h, h_ref)
        np.testing.assert_allclose(w, w_ref)

    def test_default_512_points(self):
        b = np.array([1.0])
        a = np.array([1.0, -0.5])
        h, w = freqz(b, a)
        self.assertEqual(w.shape, (512,))
        self.assertEqual(h.shape, (512,))


if __name__ == "__main__":
    unittest.main()
