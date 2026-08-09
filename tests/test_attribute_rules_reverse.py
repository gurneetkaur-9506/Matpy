import unittest

from rulebook import apply_attribute_rule_reverse


class TestApplyAttributeRuleReverse(unittest.TestCase):
    def test_shape_maps_to_size_on_different_names(self):
        self.assertEqual(apply_attribute_rule_reverse("x.shape"), "size(x)")
        self.assertEqual(apply_attribute_rule_reverse("t.shape"), "size(t)")
        self.assertEqual(apply_attribute_rule_reverse("af.shape"), "size(af)")
        self.assertEqual(apply_attribute_rule_reverse("theta.shape"), "size(theta)")

    def test_size_maps_to_numel_on_different_names(self):
        self.assertEqual(apply_attribute_rule_reverse("x.size"), "numel(x)")
        self.assertEqual(apply_attribute_rule_reverse("t.size"), "numel(t)")
        self.assertEqual(apply_attribute_rule_reverse("data.size"), "numel(data)")

    def test_transpose_maps_to_transpose_on_different_names(self):
        self.assertEqual(apply_attribute_rule_reverse("x.T"), "transpose(x)")
        self.assertEqual(apply_attribute_rule_reverse("t.T"), "transpose(t)")
        self.assertEqual(apply_attribute_rule_reverse("A.T"), "transpose(A)")

    def test_dtype_maps_to_class_on_different_names(self):
        self.assertEqual(apply_attribute_rule_reverse("x.dtype"), "class(x)")
        self.assertEqual(apply_attribute_rule_reverse("t.dtype"), "class(t)")
        self.assertEqual(apply_attribute_rule_reverse("arr.dtype"), "class(arr)")

    def test_shape_dimension_maps_to_size(self):
        self.assertEqual(apply_attribute_rule_reverse("x.shape[0]"), "size(x, 1)")
        self.assertEqual(apply_attribute_rule_reverse("t.shape[1]"), "size(t, 2)")

    def test_whitespace_is_tolerated(self):
        self.assertEqual(apply_attribute_rule_reverse("x.shape"), apply_attribute_rule_reverse("  x.shape  "))

    def test_non_attribute_expression_passthrough(self):
        self.assertEqual(apply_attribute_rule_reverse("x + y"), "x + y")
        self.assertEqual(apply_attribute_rule_reverse("x.foo"), "x.foo")
        self.assertEqual(apply_attribute_rule_reverse("np.abs(x)"), "np.abs(x)")
        self.assertEqual(apply_attribute_rule_reverse(""), "")


if __name__ == "__main__":
    unittest.main()
