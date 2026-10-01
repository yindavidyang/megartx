"""Independent rounding and rejection fixtures for the CPU boundary oracle."""
import unittest

import numpy as np

from layer0_reference import bit_difference, projection_diagnostic
from router_reference import bf16_rne_bits


class ArithmeticContractTests(unittest.TestCase):
    def bits(self, values):
        return bf16_rne_bits(np.asarray(values, dtype=np.float32))

    def test_exact_one_term_dots_and_wrong_scale_rejection(self):
        x, w = self.bits([[2.0]]), self.bits([[3.0], [-4.0]])
        result, _ = projection_diagnostic(x, w, self.bits([[6.0, -8.0]]))
        self.assertTrue(result["ideal_fp64_dot_then_f32_bf16"]["raw_bits_equal"])
        self.assertTrue(result["conditional_f32_interval_pass"])
        self.assertFalse(result["native_accumulation_qualified"])
        wrong, _ = projection_diagnostic(x, w, self.bits([[12.0, -8.0]]))
        self.assertFalse(wrong["conditional_f32_interval_pass"])
        self.assertEqual(wrong["outside_conditional_interval"], 1)

    def test_bf16_midpoint_neighbours_can_both_fit_an_unfitted_f32_interval(self):
        x, w = self.bits([[1.0, 2.0 ** -8]]), self.bits([[1.0, 1.0]])
        low, raw = projection_diagnostic(x, w, self.bits([[1.0]]))
        high, _ = projection_diagnostic(x, w, self.bits([[1.0 + 2.0 ** -7]]))
        self.assertEqual(float(raw["dot"][0, 0]), 1.0 + 2.0 ** -8)
        self.assertTrue(low["conditional_f32_interval_pass"])
        self.assertTrue(high["conditional_f32_interval_pass"])
        self.assertTrue(low["ideal_fp64_dot_then_f32_bf16"]["raw_bits_equal"])
        self.assertFalse(high["ideal_fp64_dot_then_f32_bf16"]["raw_bits_equal"])

    def test_subnormal_products_withhold_the_conditional_gate(self):
        tiny = self.bits([[1e-40]])
        result, _ = projection_diagnostic(tiny, self.bits([[1.0]]), tiny)
        self.assertFalse(result["conditional_f32_interval_usable"])
        self.assertIsNone(result["conditional_f32_interval_pass"])

    def test_signed_zero_is_reported_as_a_raw_bit_difference(self):
        result = bit_difference(np.array([0], dtype=np.uint16), np.array([0x8000], dtype=np.uint16))
        self.assertFalse(result["raw_bits_equal"])
        self.assertEqual(result["signed_zero_differences"], 1)
        self.assertEqual(result["max_absolute_difference"], 0)

    def test_nonfinite_or_wrong_geometry_is_not_accepted(self):
        for candidate in (np.array([[0x7f80]], dtype=np.uint16), self.bits([[1, 1]])):
            with self.assertRaises(ValueError):
                projection_diagnostic(self.bits([[1]]), self.bits([[1]]), candidate)


if __name__ == "__main__":
    unittest.main()
