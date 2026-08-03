import unittest

import numpy as np

from specialist_lib import beamform


class TestBeamform(unittest.TestCase):
    def test_unit_weights_are_element_sums(self):
        # signal: 3 elements, 2 samples; weights all ones.
        # y[t] = sum_e 1 * signal[e, t] -> [1+3+5, 2+4+6] = [9, 12].
        signal = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.array([1, 1, 1])
        np.testing.assert_allclose(beamform(signal, weights), np.array([9, 12]))

    def test_complex_weights_hand_computed(self):
        # weights [1j, 2, -1] -> conj = [-1j, 2, -1].
        # y[0] = -1j*1 + 2*3 + (-1)*5 = 1 - 1j
        # y[1] = -1j*2 + 2*4 + (-1)*6 = 2 - 2j
        signal = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.array([1j, 2, -1])
        expected = np.array([1 - 1j, 2 - 2j])
        np.testing.assert_allclose(beamform(signal, weights), expected)

    def test_steering_vector_gives_element_gain(self):
        # 4-element ULA, half-wavelength, plane wave from pi/6.
        # Matched weights -> |y| = 4 * |source|.
        sv = np.array([1, 1j, -1, -1j])
        source = np.array([1.0, 2.0, 3.0])
        signal = np.outer(sv, source)
        out = beamform(signal, sv)
        np.testing.assert_allclose(out, 4 * source)

    def test_output_shape(self):
        signal = np.zeros((4, 10))
        weights = np.ones(4)
        self.assertEqual(beamform(signal, weights).shape, (10,))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            beamform(np.zeros((4, 10)), np.ones(3))


if __name__ == "__main__":
    unittest.main()
