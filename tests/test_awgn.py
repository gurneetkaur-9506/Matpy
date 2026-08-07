import unittest

import numpy as np

from specialist_lib import awgn


class TestAwgnMeasuredMode(unittest.TestCase):
    def test_real_measured_hand_computed(self):
        # x = [1,2,3,4], snr = 10 dB, 'measured'.
        # signal power = mean(x^2) = 7.5, snr_lin = 10,
        # noise_var = 7.5/10 = 0.75.
        x = np.array([1.0, 2.0, 3.0, 4.0])
        expected = x + np.sqrt(0.75) * np.random.default_rng(7).standard_normal(4)
        np.testing.assert_allclose(awgn(x, 10, "measured", "dB", seed=7), expected)

    def test_seed_reproducible(self):
        x = np.random.default_rng(1).standard_normal(50)
        y1 = awgn(x, 15, "measured", seed=42)
        y2 = awgn(x, 15, "measured", seed=42)
        np.testing.assert_array_equal(y1, y2)

    def test_noise_statistics_match_snr(self):
        # Constant unit-power signal, snr = 0 dB -> noise_var = 1.
        x = np.ones(20000)
        y = awgn(x, 0, "measured", seed=3)
        noise = y - x
        self.assertAlmostEqual(np.std(noise), 1.0, delta=0.02)

    def test_complex_measured_total_noise_power(self):
        # x = 1+2j has |x|^2 = 5, snr = 3 dB -> snr_lin ~ 1.9953,
        # noise_var = 5/snr_lin, split equally real/imag.
        x = np.ones(20000) + 2j * np.ones(20000)
        y = awgn(x, 3, "measured", seed=11)
        noise = y - x
        snr_lin = 10 ** (3 / 10)
        expected_var = 5 / snr_lin
        self.assertAlmostEqual(np.mean(np.abs(noise) ** 2), expected_var, delta=0.03)
        self.assertAlmostEqual(
            np.mean(noise.real ** 2), np.mean(noise.imag ** 2), delta=0.03
        )


class TestAwgnExplicitSignalPower(unittest.TestCase):
    def test_explicit_dB_sigpower_hand_computed(self):
        # sigpower = 10*log10(4) dBW -> signal power 4, snr = 20 dB ->
        # noise_var = 4/100 = 0.04.
        x = np.ones(4)
        sigpower_db = 10 * np.log10(4)
        expected = x + np.sqrt(0.04) * np.random.default_rng(5).standard_normal(4)
        np.testing.assert_allclose(
            awgn(x, 20, sigpower_db, "dB", seed=5), expected, rtol=1e-12
        )

    def test_linear_mode_hand_computed(self):
        # snr given as a linear power ratio 100 == 20 dB.
        x = np.ones(4)
        expected = awgn(x, 20, 10 * np.log10(4), "dB", seed=5)
        np.testing.assert_allclose(
            awgn(x, 100, 4.0, "linear", seed=5), expected, rtol=1e-12
        )

    def test_explicit_power_equals_measured_when_matching(self):
        # For x with power P, explicit sigpower = 10*log10(P) dBW must
        # give the same noise as 'measured'.
        x = np.array([0.5, -1.0, 2.0, 3.0])
        power = np.mean(x ** 2)
        np.testing.assert_allclose(
            awgn(x, 12, "measured", seed=9),
            awgn(x, 12, 10 * np.log10(power), "dB", seed=9),
            rtol=1e-12,
        )

    def test_explicit_zero_sigpower(self):
        # sigpower 0 dBW -> power 1; snr 10 dB -> noise_var 0.1.
        x = np.zeros(3)
        expected = np.sqrt(0.1) * np.random.default_rng(2).standard_normal(3)
        np.testing.assert_allclose(awgn(x, 10, 0, "dB", seed=2), expected, rtol=1e-12)


class TestAwgnValidation(unittest.TestCase):
    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            awgn(np.ones(4), 10, mode="watts")

    def test_invalid_sigpower_string_raises(self):
        with self.assertRaises(ValueError):
            awgn(np.ones(4), 10, sigpower="auto")

    def test_nonpositive_snr_raises(self):
        with self.assertRaises(ValueError):
            awgn(np.ones(4), 0, "measured", "linear")

    def test_shape_preserved(self):
        x = np.random.default_rng(0).standard_normal((8, 8))
        self.assertEqual(awgn(x, 10, "measured", seed=1).shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
