"""Pair provenance guards, independently of checkpoint and arithmetic work."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import layer0_reference as ref


class PairProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.hashes = {field: str(i) * 64 for i, field in enumerate(("q", "k", "v"), 1)}
        binding = {"schema": 2, "path": "full", "sources": ref.PINNED_SOURCES,
                   "qkv_quant_method": "same", "qkv_gemm": "same", "norms": "same", "rope": "same",
                   **{"loaded_" + field + "_sha256": value for field, value in self.hashes.items()},
                   "allow_bf16_reduced_precision_reduction": True, "batch_invariant": False,
                   "torch_version": "2.13.0+cu130", "cuda_version": "13.0"}
        rows = {p: {name: np.zeros(width, np.uint16) for name, width in {**ref.WIDTHS, **ref.EXTRA_WIDTHS}.items()}
                for p in (31, 32)}
        for row in rows.values():
            row["rope_cache_bits"] = np.zeros(256, np.uint16)
        self.full = {"binding": binding, "constants": {}, "rows": rows, "binding_sha256": "f" * 64}
        self.cached = copy.deepcopy(self.full)
        self.cached["binding"]["path"] = "cached"

    def run_pair(self):
        def original(checkpoint, field):
            return np.zeros((1, 1), np.uint16), self.hashes[field]

        def diagnostic(inputs, weights, outputs):
            return {}, {"dot": np.zeros(outputs.shape), "bound": np.zeros(outputs.shape),
                        "ideal_bits": np.zeros(outputs.shape, np.uint16)}

        plan = {"token_sha256": "test_token", "schedule_sha256": "test_schedule"}
        with patch.object(ref.evidence, "load_plan", return_value=plan), \
             patch.object(ref.evidence, "digest_file", return_value=ref.evidence.reference.CONFIG_SHA256), \
             patch.object(ref, "load_case", side_effect=[self.full, self.cached]), \
             patch.object(ref, "original_bf16_projection", side_effect=original) as read, \
             patch.object(ref, "projection_diagnostic", side_effect=diagnostic) as arithmetic:
            self.read, self.arithmetic = read, arithmetic
            return ref.compare_pair("unused", "unused-full", "unused-cached", "unused-checkpoint")

    def test_changed_cached_q_hash_is_rejected_before_original_reads(self):
        self.cached["binding"]["loaded_q_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "positive path binding changed: loaded_q_sha256"):
            self.run_pair()
        self.read.assert_not_called()
        self.arithmetic.assert_not_called()

    def test_schema_two_requires_q_hash_in_both_paths(self):
        for paths in (("full",), ("cached",), ("full", "cached")):
            with self.subTest(paths=paths):
                original = [copy.deepcopy(c) for c in (self.full, self.cached)]
                for path in paths:
                    getattr(self, path)["binding"].pop("loaded_q_sha256")
                with self.assertRaisesRegex(ValueError, "schema-2 Q provenance is missing"):
                    self.run_pair()
                self.read.assert_not_called()
                self.arithmetic.assert_not_called()
                self.full, self.cached = original

    def test_shared_wrong_q_hash_is_rejected_against_original_payload(self):
        for case in (self.full, self.cached):
            case["binding"]["loaded_q_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "differ from original checkpoint: q / full"):
            self.run_pair()
        self.arithmetic.assert_not_called()

    def test_matching_schema_two_verifies_both_original_weight_bindings(self):
        report = self.run_pair()
        for field in ("q", "k", "v"):
            self.assertTrue(report["projections"][field]["loaded_original_weight_hash_verified"])
            self.assertEqual([p["path"] for p in report["projections"][field]["paths"]], ["full", "cached"])
        self.assertFalse(report["full_cached_handoff_accepted"])
        self.assertFalse(report["native_accumulation_qualified"])

    def test_historical_schema_one_q_remains_unverified(self):
        for case in (self.full, self.cached):
            case["binding"]["schema"] = 1
            del case["binding"]["loaded_q_sha256"]
        report = self.run_pair()
        self.assertFalse(report["projections"]["q"]["loaded_original_weight_hash_verified"])
        self.assertTrue(report["projections"]["k"]["loaded_original_weight_hash_verified"])
        self.assertTrue(report["projections"]["v"]["loaded_original_weight_hash_verified"])

    def test_partial_historical_q_provenance_cannot_report_pair_verified(self):
        for case in (self.full, self.cached):
            case["binding"]["schema"] = 1
        del self.full["binding"]["loaded_q_sha256"]
        self.assertFalse(self.run_pair()["projections"]["q"]["loaded_original_weight_hash_verified"])
        self.cached["binding"]["loaded_q_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "differ from original checkpoint: q / cached"):
            self.run_pair()


if __name__ == "__main__":
    unittest.main()
