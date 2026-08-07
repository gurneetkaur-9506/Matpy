import unittest

import numpy as np
from scipy import signal

from specialist_lib import chirp


class TestChirpHandComputed(unittest.TestCase):
    def test_linear_hand_computed(self):
        # t=0.25, f0=0, t1=1, f1=10: f(t) = 10*t = 2.5.
        # phase = 2*pi*0.5*10*0.25**2 = 2*pi*0.3125 = 5*pi/8.
        # chirp = cos(5*pi/8) = -0.3826834324.
        self.assertAlmostEqual(chirp(0.25, 0, 1, 10), np.cos(5 * np.pi / 8), places=12)

    def test_quadratic_hand_computed(self):
        # t=0.5, f0=0, t1=1, f1=10, vertex_zero=True:
        # beta = 2*pi*(f1-f0)/t1^2, phase = beta*t^3/3 = 2*pi*10*0.125/3.
        # 2*pi*10*0.125/3 = 2*pi*1.25/3 = (5*pi)/6.
        self.assertAlmostEqual(
            chirp(0.5, 0, 1, 10, method="quadratic"), np.cos(5 * np.pi / 6), places=12
        )

    def test_linear_phi_offset_hand_computed(self):
        # phi=90 deg adds pi/2: cos(5*pi/8 + pi/2) = cos(9*pi/8).
        self.assertAlmostEqual(
            chirp(0.25, 0, 1, 10, phi=90), np.cos(5 * np.pi / 8 + np.pi / 2), places=12
        )


class TestChirpScipyCrossCheck(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 1, 101)

    def assert_close(self, got, ref):
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(np.allclose(got, ref, atol=1e-12))

    def test_linear_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 0, 1, 10, method="linear"),
            signal.chirp(self.t, 0, 1, 10, method="linear"),
        )

    def test_quadratic_vertex_zero_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 0, 1, 10, method="quadratic", vertex_zero=True),
            signal.chirp(self.t, 0, 1, 10, method="quadratic", vertex_zero=True),
        )

    def test_quadratic_vertex_t1_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 5, 1, -3, method="quadratic", vertex_zero=False),
            signal.chirp(self.t, 5, 1, -3, method="quadratic", vertex_zero=False),
        )

    def test_logarithmic_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 1, 1, 100, method="logarithmic"),
            signal.chirp(self.t, 1, 1, 100, method="logarithmic"),
        )

    def test_hyperbolic_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 1, 1, 100, method="hyperbolic"),
            signal.chirp(self.t, 1, 1, 100, method="hyperbolic"),
        )

    def test_phi_degrees_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 0, 1, 10, method="linear", phi=60),
            signal.chirp(self.t, 0, 1, 10, method="linear", phi=60),
        )

    def test_method_abbreviation_matches_scipy(self):
        self.assert_close(
            chirp(self.t, 0, 1, 10, method="lin"),
            signal.chirp(self.t, 0, 1, 10, method="lin"),
        )
        self.assert_close(
            chirp(self.t, 1, 1, 100, method="log"),
            signal.chirp(self.t, 1, 1, 100, method="log"),
        )

    def test_scalar_input_returns_scalar(self):
        result = chirp(0.25, 0, 1, 10)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, np.cos(5 * np.pi / 8), places=12)

    def test_array_shape_preserved(self):
        t = np.linspace(0, 1, 64)
        self.assertEqual(chirp(t, 0, 1, 10).shape, (64,))

    def test_logarithmic_rejects_opposite_signs(self):
        with self.assertRaises(ValueError):
            chirp(np.linspace(0, 1, 10), -1, 1, 10, method="logarithmic")

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            chirp(np.linspace(0, 1, 10), 0, 1, 10, method="exponential")


if __name__ == "__main__":
    unittest.main()
