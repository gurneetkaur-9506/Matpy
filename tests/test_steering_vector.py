import unittest

import numpy as np

from specialist_lib import steering_vector


class TestSteeringVector(unittest.TestCase):
    def test_boresight_half_wavelength(self):
        # By hand: phase = 2*pi*d*sin(theta) = 2*pi*0.5*0 = 0,
        # so every element has exp(j*0) = 1.
        expected = np.ones(4, dtype=complex)
        result = steering_vector(0.0, n_elements=4, spacing=0.5)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_off_boresight_half_wavelength(self):
        # theta = pi/6 -> sin(theta) = 0.5, d = 0.5 -> phase = pi/2.
        # sv[n] = exp(j*n*pi/2) = [1, j, -1, -j].
        expected = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
        result = steering_vector(np.pi / 6, n_elements=4, spacing=0.5)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_array_of_angles_shape(self):
        result = steering_vector(np.array([0.0, np.pi / 6]), n_elements=4, spacing=0.5)
        self.assertEqual(result.shape, (4, 2))

    def test_scalar_input_flattens_last_axis(self):
        result = steering_vector(0.0, n_elements=4, spacing=0.5)
        self.assertEqual(result.shape, (4,))


if __name__ == "__main__":
    unittest.main()
