import unittest

from specialist_lib import collect_numpy_operations, reverse_lookup


class TestReverseLookup(unittest.TestCase):
    def test_bare_name_returns_candidates(self):
        self.assertIn("steervec", reverse_lookup("np.exp"))
        self.assertIn("phased.SteeringVector", reverse_lookup("np.outer"))
        self.assertIn("beamscan", reverse_lookup("np.fft.fft"))

    def test_full_call_uses_outermost_operation(self):
        candidates = reverse_lookup("np.exp(1j * np.outer(indices, phase))")
        self.assertEqual(candidates, ["steervec", "phased.SteeringVector", "phased.ULA"])

    def test_may_return_multiple_candidates(self):
        candidates = reverse_lookup("np.conj")
        self.assertGreaterEqual(len(candidates), 2)
        self.assertIn("phased.MVDRBeamformer", candidates)

    def test_linear_algebra_adaptive_beamformers(self):
        self.assertIn("phased.MVDRBeamformer", reverse_lookup("np.linalg.inv"))
        self.assertIn("phased.MUSICEstimator", reverse_lookup("np.linalg.svd"))

    def test_unknown_returns_empty_list(self):
        self.assertEqual(reverse_lookup("np.unknown_op"), [])

    def test_non_numpy_input_returns_empty_list(self):
        self.assertEqual(reverse_lookup("scipy.signal.welch(x)"), [])
        self.assertEqual(reverse_lookup("a + b"), [])

    def test_non_string_input_returns_empty_list(self):
        self.assertEqual(reverse_lookup(None), [])
        self.assertEqual(reverse_lookup(42), [])

    def test_numpy_alias_normalized(self):
        self.assertEqual(reverse_lookup("numpy.conj"), reverse_lookup("np.conj"))

    def test_returns_fresh_list(self):
        candidates = reverse_lookup("np.abs")
        candidates.append("phased.FakeCandidate")
        self.assertNotIn("phased.FakeCandidate", reverse_lookup("np.abs"))

    def test_linspace_scanning_estimators(self):
        candidates = reverse_lookup("np.linspace(0, np.pi, 91)")
        self.assertIn("beamscan", candidates)
        self.assertIn("phased.BeamscanEstimator", candidates)
        self.assertIn("phased.MUSICEstimator", candidates)

    def test_beamform_basic_array_factor_expression(self):
        expr = "np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)"
        candidates = reverse_lookup(expr)
        self.assertIn("steervec", candidates)
        self.assertIn("phased.SteeringVector", candidates)
        self.assertIn("phased.ULA", candidates)
        self.assertIn("phased.ArrayGain", candidates)
        self.assertIn("phased.Beamformer", candidates)

    def test_newaxis_broadcast_is_ambiguous(self):
        candidates = reverse_lookup("n[:, np.newaxis]")
        self.assertEqual(
            candidates, ["steervec", "phased.SteeringVector", "phased.ULA"]
        )

    def test_sum_method_normalized_to_np_sum(self):
        self.assertEqual(reverse_lookup("x.sum(axis=0)"), reverse_lookup("np.sum(x, axis=0)"))

    def test_shape_attribute_maps_to_size(self):
        candidates = reverse_lookup("af.shape")
        self.assertEqual(candidates, ["size", "numel"])

    def test_full_beamform_basic_operations_resolved(self):
        source = (
            "k = 2 * np.pi / lamb\n"
            "phase = k * d * (np.sin(theta) - np.sin(theta0))\n"
            "n = np.arange(N)\n"
            "af = np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)\n"
            "theta = np.linspace(0, np.pi, 91)\n"
            "print('array factor shape:', af.shape)\n"
        )
        candidates = reverse_lookup(source)
        self.assertIn("steervec", candidates)
        self.assertIn("beamscan", candidates)
        self.assertIn("size", candidates)


class TestCollectNumpyOperations(unittest.TestCase):
    def test_returns_distinct_operations_in_order(self):
        ops = collect_numpy_operations(
            "np.exp(x).sum(axis=0) + np.sin(y) + af.shape"
        )
        self.assertEqual(ops, ["np.exp", "np.sin", "np.sum", ".shape"])

    def test_dedupes_operations(self):
        ops = collect_numpy_operations("np.sin(a) + np.sin(b) + c.sum()")
        self.assertEqual(ops, ["np.sin", "np.sum"])

    def test_empty_input(self):
        self.assertEqual(collect_numpy_operations(""), [])
        self.assertEqual(collect_numpy_operations(None), [])


if __name__ == "__main__":
    unittest.main()
