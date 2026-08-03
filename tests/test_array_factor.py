import unittest

import numpy as np

from specialist_lib import array_factor


class TestArrayFactor(unittest.TestCase):
    def test_boresight_peak(self):
        # N=4, d=0.5, theta=0: every element contributes exp(j*0)=1,
        # so AF = 4.
        self.assertAlmostEqual(array_factor(0.0, n_elements=4, spacing=0.5), 4.0, places=12)

    def test_null_half_wavelength(self):
        # N=4, d=0.5, theta=pi/6: steering vector is [1, j, -1, -j],
        # whose sum is 1 + j - 1 - j = 0.
        af = array_factor(np.pi / 6, n_elements=4, spacing=0.5)
        self.assertAlmostEqual(abs(af), 0.0, places=12)

    def test_two_element_null_at_endfire(self):
        # N=2, d=0.5, theta=pi/2: sv = [1, exp(j*pi)] = [1, -1],
        # sum = 0.
        af = array_factor(np.pi / 2, n_elements=2, spacing=0.5)
        self.assertAlmostEqual(abs(af), 0.0, places=12)

    def test_steered_at_target(self):
        # Steered toward theta0=pi/6, evaluated at theta=pi/6:
        # each term has zero phase -> AF = N.
        af = array_factor(np.pi / 6, n_elements=4, spacing=0.5, theta0=np.pi / 6)
        self.assertAlmostEqual(af, 4.0, places=12)

    def test_steered_null_boresight(self):
        # N=4, d=0.5, theta0=pi/6, theta=0:
        # terms exp(-j*pi*n/2) sum to conj([1, j, -1, -j]) -> 0.
        af = array_factor(0.0, n_elements=4, spacing=0.5, theta0=np.pi / 6)
        self.assertAlmostEqual(abs(af), 0.0, places=12)

    def test_array_of_angles_shape(self):
        result = array_factor(np.array([0.0, np.pi / 6]), n_elements=4, spacing=0.5)
        self.assertEqual(result.shape, (2,))


if __name__ == "__main__":
    unittest.main()
