"""Accounting witnesses, not device timing or draft feasibility measurements."""

import unittest
from megartx.speculative_cost import break_even, expected_yield, expert_reuse, staging_budget


class CostTests(unittest.TestCase):
    def test_exact_KV_page_and_logits_bounds_do_not_claim_peak_fit(self):
        aligned = staging_budget(candidates=1, absolute_start=2048, page_tokens=16)
        self.assertEqual(aligned["kv_bytes_per_row"], 225280)
        self.assertEqual(aligned["packed_kv_payload_bytes"], 450560)
        self.assertEqual(aligned["new_or_COW_storage_bytes"], 3604480)
        self.assertEqual(aligned["page_storage_plus_one_row_logits_bytes"], 4653056)
        edge = staging_budget(candidates=7, absolute_start=1023, page_tokens=16)
        self.assertEqual(edge["packed_kv_payload_bytes"], 1802240)
        self.assertEqual(edge["FP32_batched_logits_bytes"], 8 << 20)
        self.assertEqual(edge["new_or_COW_storage_bytes"], 7208960)
        self.assertEqual(edge["remaining_under_8MiB_before_other_scratch"], 131072)
        self.assertLess(staging_budget(candidates=1, absolute_start=2048, page_tokens=128)[
            "remaining_under_8MiB_before_other_scratch"], 0)
        for kwargs in ({"candidates": 8}, {"page_tokens": 0}, {"absolute_start": True}):
            values = {"candidates": 1, "absolute_start": 2048, "page_tokens": 16}
            values.update(kwargs)
            with self.assertRaises(ValueError):
                staging_budget(**values)

    def test_yield_uses_prefix_survival_and_budget_censoring(self):
        self.assertEqual(expected_yield(()), 1)
        self.assertAlmostEqual(expected_yield((.8, .6, .4)), 2.8)
        self.assertEqual(expected_yield((.8, .6), budget=1), 1)
        self.assertAlmostEqual(expected_yield((.8, .6), budget=2), 1.8)
        for survival in ((.2, .4), (1.1,), (float("nan"),), (-.1,)):
            with self.assertRaises(ValueError):
                expected_yield(survival)

    def test_complete_cost_and_draft_budget(self):
        # Arbitrary scalar witness, not a 5090 performance prediction.
        result = break_even(target_only_ms=10, verify_ms=16, draft_ms=3,
                            overhead_ms=1, committed_yield=2.5)
        self.assertEqual(result, {"cycle_ms": 20, "ms_per_committed_token": 8,
                                  "speedup": 1.25, "max_profitable_draft_ms": 8})
        result = break_even(target_only_ms=10, verify_ms=30, draft_ms=3,
                            overhead_ms=1, committed_yield=2)
        self.assertLess(result["max_profitable_draft_ms"], 0)
        with self.assertRaises(ValueError):
            break_even(target_only_ms=10, verify_ms=float("inf"), draft_ms=0,
                       overhead_ms=0, committed_yield=1)

    def test_union_counts_all_rows_and_does_not_imply_residency(self):
        self.assertEqual(expert_reuse(((0, 1), (1, 2), (0, 1)), experts=4, top_k=2),
                         {"rows": 3, "assignments": 6, "union": 3,
                          "repeated_assignments": 3, "potential_reuse_fraction": .5})
        self.assertEqual(expert_reuse((tuple(range(8)),) * 8)["union"], 8)
        self.assertEqual(expert_reuse(tuple(tuple(range(i * 8, (i + 1) * 8)) for i in range(8)))["union"], 64)
        for rows in ((), ((1, 1),), ((0, 4),), ((0, True),)):
            with self.assertRaises(ValueError):
                expert_reuse(rows, experts=4, top_k=2)


if __name__ == "__main__":
    unittest.main()
