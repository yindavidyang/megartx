"""Saved-input negative controls and paired-stage localization; CPU only."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import controlled_comparison as comparison
import controlled_reference as reference
import test_controlled_reference as golden_cases

f32bits = golden_cases.f32bits


class CommonInputComparisonTests(unittest.TestCase):
    def setUp(self):
        golden = golden_cases.IndependentReplayTests()
        golden.setUp()
        self.native, self.originals = golden.expert_case()
        self.paired = copy.deepcopy(self.native)
        self.negative, _ = golden.expert_case(negative=True)

    def negative_check(self):
        return comparison.compare_negative_common_input(self.native, self.paired, self.negative,
                                                         self.originals, a1_bits=f32bits(1))

    def test_single_scale_change_agrees_wrong_oracle_and_disagrees_correct_one(self):
        report = self.negative_check()
        self.assertTrue(report["negative_matches_wrong_alpha_oracle"])
        self.assertTrue(report["negative_differs_from_correct_alpha_oracle"])
        self.assertTrue(report["negative_control_effect_observed"])
        self.assertEqual(report["wrong_alpha_oracle_comparison"]["max_absolute_difference"], 0)
        self.assertEqual(report["correct_alpha_counterfactual_comparison"]["max_absolute_difference"], 4)
        self.assertFalse(report["native_mma_or_model_qualified"])
        self.assertEqual(self.originals["up"][2], 0.5)

    def test_unchanged_control_alpha_does_not_count_as_a_negative_control(self):
        self.negative["up_alpha_bits"] = f32bits(0.5)
        with self.assertRaisesRegex(ValueError, "did not change only"):
            self.negative_check()

    def test_wrong_positive_alpha_is_rejected_even_when_control_matches_it(self):
        self.native["up_alpha_bits"] = f32bits(0.25)
        with self.assertRaisesRegex(ValueError, "original up product"):
            self.negative_check()

    def test_different_negative_input_or_gate_cannot_be_attributed_to_the_scale(self):
        for name in ("input_bits", "gate_bits", "gate_alpha_bits"):
            negative = copy.deepcopy(self.negative)
            self.negative[name].reshape(-1)[0] += 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "common actual input/gate"):
                self.negative_check()
            self.negative = negative

    def test_changed_nonzero_logical_q1_operand_blocks_common_input_claim(self):
        self.negative["q1"][0, 0] = 0x11
        with self.assertRaisesRegex(ValueError, "common logical q1"):
            self.negative_check()

    def test_zero_up_projection_makes_the_negative_control_inconclusive(self):
        self.originals["up"] = (np.zeros((16, 8), dtype=np.uint8), self.originals["up"][1], 0.5)
        for item in (self.native, self.paired, self.negative):
            item["up_bits"][:] = 0
        report = self.negative_check()
        self.assertTrue(report["alphas_distinct"])
        self.assertTrue(report["negative_matches_wrong_alpha_oracle"])
        self.assertFalse(report["negative_differs_from_correct_alpha_oracle"])
        self.assertFalse(report["negative_control_effect_observed"])

    def test_padding_differences_are_reported_without_changing_logical_operands(self):
        # Logical row0/block0 is physical byte0. Last padded byte is unused.
        self.paired["sf1"].reshape(-1)[-1] = 0x7F
        report = comparison.compare_positive_stages(self.native, self.paired)
        self.assertTrue(report["strict_observed_stage_gate_pass"])
        self.assertFalse(report["all_raw_payloads_equal"])
        self.assertFalse(report["activation_operands"]["a1"]["physical_sf_bytes_equal"])
        self.assertTrue(report["activation_operands"]["a1"]["logical_sf_bytes_equal"])

    def test_zero_scale_payload_differences_preserve_only_effective_equivalence(self):
        self.native["sf1"][:] = 0
        self.paired["sf1"][:] = 0
        self.paired["q1"][:] = 0x77
        report = comparison.compare_positive_stages(self.native, self.paired)
        self.assertTrue(report["strict_observed_stage_gate_pass"])
        self.assertFalse(report["all_raw_payloads_equal"])
        self.assertFalse(report["activation_operands"]["a1"]["packed_bytes_equal"])
        self.assertTrue(report["activation_operands"]["a1"]["decoded_units_equal"])

    def test_nonzero_operand_change_is_localized_before_downstream_comparison(self):
        self.paired["q1"][0, 0] = 0x11
        report = comparison.compare_positive_stages(self.native, self.paired)
        self.assertFalse(report["strict_observed_stage_gate_pass"])
        self.assertEqual(report["first_differing_stage"], "a1")

    def test_same_gemm_with_different_activation_fails_shared_activation_contract(self):
        self.paired["activation_bits"].reshape(-1)[0] += 1
        report = comparison.compare_positive_stages(self.native, self.paired)
        self.assertFalse(report["strict_observed_stage_gate_pass"])
        self.assertEqual(report["first_differing_stage"], "activation_bits")


if __name__ == "__main__":
    unittest.main()
