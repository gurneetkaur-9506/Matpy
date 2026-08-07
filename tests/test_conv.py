import unittest

import numpy as np

from specialist_lib import conv


class TestConvHandComputed(unittest.TestCase):
    def test_full_hand_computed(self):
        # (1,2,3) * (0,1,0.5):
        # c0 = 1*0          = 0
        # c1 = 1*1 + 2*0    = 1
        # c2 = 1*.5+2*1+3*0 = 2.5
        # c3 = 2*.5+3*1     = 4
        # c4 = 3*.5         = 1.5
        np.testing.assert_allclose(conv([1, 2, 3], [0, 1, 0.5]), [0, 1, 2.5, 4, 1.5])

    def test_same_hand_computed(self):
        # Same pair: 'same' is the central length-3 part starting at
        # index (len(v)-1)//2 = 1 of the full convolution.
        np.testing.assert_allclose(conv([1, 2, 3], [0, 1, 0.5], "same"), [1, 2.5, 4])

    def test_valid_hand_computed(self):
        # (1,2,3,4) * (1,2): full = [1,4,7,10,8], valid keeps the fully
        # overlapping length 4-2+1 = 3 tail of the correlation.
        np.testing.assert_allclose(conv([1, 2, 3, 4], [1, 2], "valid"), [4, 7, 10])

    def test_default_shape_is_full(self):
        np.testing.assert_array_equal(conv([1, 2], [3, 4]), conv([1, 2], [3, 4], "full"))


class TestConvNumpyCrossCheck(unittest.TestCase):
    def test_random_full(self):
        u = np.random.default_rng(0).standard_normal(7)
        v = np.random.default_rng(1).standard_normal(5)
        np.testing.assert_allclose(conv(u, v, "full"), np.convolve(u, v, "full"))

    def test_random_same(self):
        u = np.random.default_rng(2).standard_normal(6)
        v = np.random.default_rng(3).standard_normal(4)
        np.testing.assert_allclose(conv(u, v, "same"), np.convolve(u, v, "same"))

    def test_random_valid(self):
        u = np.random.default_rng(4).standard_normal(9)
        v = np.random.default_rng(5).standard_normal(2)
        np.testing.assert_allclose(conv(u, v, "valid"), np.convolve(u, v, "valid"))

    def test_unequal_same_length(self):
        # MATLAB 'same' returns length max(M, N).
        u = np.ones(3)
        v = np.ones(5)
        self.assertEqual(conv(u, v, "same").shape, (5,))

    def test_complex_inputs(self):
        u = np.array([1 + 1j, 2 - 1j])
        v = np.array([1j, 1 + 0j])
        np.testing.assert_allclose(conv(u, v, "full"), np.convolve(u, v, "full"))

    def test_scalar_inputs(self):
        np.testing.assert_allclose(conv(2, 3), np.convolve([2], [3]))

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            conv([1, 2], [3, 4], "circular")

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            conv(np.ones((2, 2)), [1, 2])


if __name__ == "__main__":
    unittest.main()
