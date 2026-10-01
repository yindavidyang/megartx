"""Golden router math and provenance checks; NumPy/stdlib only, no GPU."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import router_reference as ref
import check_router_capture as capture


def bits(values):
    return ref.bf16_rne_bits(np.asarray(values, dtype=np.float32))


class RouterMathTests(unittest.TestCase):
    def test_bf16_known_values_signed_zero_and_all_finite_roundtrips(self):
        codes = np.asarray([0, 0x8000, 0x3F80, 0xBF80, 1], dtype=np.uint16)
        decoded = ref.bf16_decode(codes)
        np.testing.assert_array_equal(decoded, [0, -0.0, 1, -1, 2 ** -133])
        self.assertTrue(np.signbit(decoded[1]))
        all_codes = np.arange(65536, dtype=np.uint16)
        finite = all_codes[(all_codes & 0x7F80) != 0x7F80]
        self.assertEqual(len(finite), 65280)
        np.testing.assert_array_equal(ref.bf16_rne_bits(ref.bf16_decode(finite)), finite)

    def test_bf16_midpoints_round_to_even(self):
        midpoint = np.asarray([1.00390625, 1.01171875, -1.00390625, -1.01171875], dtype=np.float32)
        np.testing.assert_array_equal(ref.bf16_rne_bits(midpoint), [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_separate_root_dimension_stores_are_not_folded(self):
        norm = np.asarray(0x3F01, dtype=np.uint16)  # 0.50390625
        root = np.asarray(0x3C9A, dtype=np.uint16)  # 0.018798828125
        dimension = np.asarray(0x3F70, dtype=np.uint16)  # 0.9375
        intermediate = ref.bf16_product_bits(norm, root)
        result = ref.bf16_product_bits(intermediate, dimension)
        folded = ref.bf16_rne_bits(ref.bf16_decode(norm).astype(np.float64) *
                                  ref.bf16_decode(root).astype(np.float64) *
                                  ref.bf16_decode(dimension).astype(np.float64))
        self.assertEqual(int(intermediate), 0x3C1B)
        self.assertEqual(int(result), 0x3C11)
        self.assertEqual(int(folded), 0x3C12)

    def test_nonfinite_bf16_and_scores_fail_closed(self):
        for code in (0x7F80, 0xFF80, 0x7FC1):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "nonfinite"):
                ref.bf16_decode(np.asarray([code], dtype=np.uint16))
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                ref.pinned_bit_order(np.asarray([[value]], dtype=np.float32))

    def test_projection_golden_dot_and_conditional_scope(self):
        report = ref.projection_diagnostic(bits([[1, 2, 3]]), bits([[4, 5, 6], [-1, 2, -3]]),
                                           np.asarray([[32, -6]], dtype=np.float32))
        np.testing.assert_array_equal(report["dot_f64"], [[32, -6]])
        self.assertTrue(report["conditional_envelope_pass"])
        self.assertFalse(report["native_accumulation_qualified"])
        self.assertEqual(report["observed_ieee32_value_equal_fraction"], 1)

    def test_cancellation_requires_absolute_not_relative_dot_bound(self):
        # A legitimate F32 order loses the middle one: (2**24+1)-2**24 == 0.
        report = ref.projection_diagnostic(bits([[2 ** 24, 1, -(2 ** 24)]]), bits([[1, 1, 1]]),
                                           np.asarray([[0]], dtype=np.float32))
        self.assertEqual(float(report["dot_f64"][0, 0]), 1)
        self.assertEqual(report["max_absolute_error"], 1)
        self.assertTrue(report["conditional_envelope_pass"])
        self.assertFalse(report["native_accumulation_qualified"])

    def test_wrong_projection_permutation_fails_unfitted_envelope(self):
        report = ref.projection_diagnostic(bits([[1, 2, 3]]), bits([[1, 0, 0], [0, 1, 0]]),
                                           np.asarray([[2, 1]], dtype=np.float32))
        self.assertFalse(report["conditional_envelope_pass"])

    def test_f32_subnormal_product_does_not_get_a_normal_product_bound(self):
        report = ref.projection_diagnostic(np.asarray([[1]], dtype=np.uint16),
                                           np.asarray([[1]], dtype=np.uint16),
                                           np.asarray([[0]], dtype=np.float32))
        self.assertFalse(report["conditional_envelope_usable"])
        self.assertIsNone(report["conditional_envelope_pass"])
        self.assertFalse(report["native_accumulation_qualified"])

    def test_gamma_has_a_fixed_mathematical_value_and_rejects_invalid_lengths(self):
        self.assertEqual(ref.gamma(2816, 2 ** -24), 2816 / (2 ** 24 - 2816))
        for n in (0, 2 ** 24):
            with self.subTest(n=n), self.assertRaises(ValueError):
                ref.gamma(n, 2 ** -24)

    def test_bit_order_preserves_signed_zero_and_expert_id_ties(self):
        row = np.asarray([[2, -1, 0.0, -0.0, 1, 1, 2 ** -149, -(2 ** -149)]], dtype=np.float32)
        np.testing.assert_array_equal(ref.pinned_bit_order(row), [[0, 4, 5, 6, 2, 3, 7, 1]])

    def test_exact_same_score_bits_tie_by_ascending_original_id(self):
        row = np.full((2, 128), np.float32(-0.25), dtype=np.float32)
        np.testing.assert_array_equal(ref.pinned_bit_order(row), np.tile(np.arange(128), (2, 1)))

    def test_score_order_requires_actual_f32_storage(self):
        with self.assertRaisesRegex(ValueError, "F32 matrix"):
            ref.pinned_bit_order(np.asarray([[1]], dtype=np.float64))

    def test_learned_factors_apply_after_selection_without_second_renormalization(self):
        logits = np.zeros((1, 8), dtype=np.float32)
        scales = np.ones(8, dtype=np.float32)
        scales[3], scales[7] = 2, 100
        ids = ref.pinned_bit_order(logits)[:, :4]
        np.testing.assert_array_equal(ids, [[0, 1, 2, 3]])
        profile = ref.weight_profiles(logits, ids, bits(scales))
        np.testing.assert_array_equal(profile["semantic_weights_f64"], [[0.25, 0.25, 0.25, 0.5]])
        np.testing.assert_array_equal(profile["named_f32_weights"], [[0.25, 0.25, 0.25, 0.5]])
        self.assertEqual(float(np.sum(profile["semantic_weights_f64"])), 1.25)
        self.assertFalse(profile["native_weight_kernel_qualified"])

    def test_duplicate_and_nonlocal_selected_ids_fail_closed(self):
        for ids in ([[0, 0]], [[0, -1]], [[0, 4]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                ref.weight_profiles(np.zeros((1, 4), dtype=np.float32),
                                    np.asarray(ids, dtype=np.int32), bits(np.ones(4)))

    def test_numeric_tie_interval_is_distinct_from_exact_bit_rank(self):
        scores = np.zeros((1, 4), dtype=np.float32)
        report = ref.score_diagnostic(scores, np.asarray([[0, 1]], dtype=np.int32),
                                     np.asarray([[0.5, 0.5]], dtype=np.float32), bits(np.ones(4)), [3], topk=2)
        target = report["row_metrics"][0]["targets"][0]
        self.assertEqual(target["pinned_bit_rank"], 4)
        self.assertEqual((target["numeric_tie_minimum_rank"], target["numeric_tie_maximum_rank"]), (1, 4))
        self.assertFalse(target["selected"])
        self.assertFalse(report["weight_comparison_has_acceptance_bound"])

    def test_native_id_discrepancy_is_reported_without_sorting_it_away(self):
        report = ref.score_diagnostic(np.asarray([[4, 3, 2, 1]], dtype=np.float32),
                                     np.asarray([[1, 0]], dtype=np.int32),
                                     np.asarray([[0.25, 0.75]], dtype=np.float32), bits(np.ones(4)), [0], topk=2)
        self.assertFalse(report["selected_ids_match_pinned_bit_order"])
        self.assertFalse(report["native_weight_kernel_qualified"])

    def test_interval_rank_distinguishes_safe_and_uncertain_membership(self):
        scores = np.asarray([4, 3, 2, 1], dtype=np.float64)
        narrow = np.full(4, 0.01)
        self.assertEqual(ref.interval_rank(scores, narrow, 0, 2)["conditional_topk_membership"], "inside")
        self.assertEqual(ref.interval_rank(scores, narrow, 3, 2)["conditional_topk_membership"], "outside")
        self.assertEqual(ref.interval_rank(scores, np.full(4, 0.6), 1, 2)["conditional_topk_membership"], "uncertain")

    def test_exact_interval_ties_are_conservatively_unresolved(self):
        report = ref.interval_rank(np.asarray([1, 1, 0]), np.zeros(3), 0, topk=1)
        self.assertEqual((report["conditional_minimum_rank"], report["conditional_maximum_rank"]), (1, 2))
        self.assertEqual(report["conditional_topk_membership"], "uncertain")


class ForwardStageTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "weight_bits": bits(np.eye(4)), "dimension_scale_bits": bits([1, 2, 3, 4]),
            "expert_scale_bits": bits(np.ones(4)), "root_cast_bf16_bits": bits([0.5]),
        }
        logits = np.asarray([[0.5, 1, 1.5, 2]], dtype=np.float32)
        ids = np.asarray([[3, 2]], dtype=np.int32)
        weight = ref.weight_profiles(logits, ids, self.reference["expert_scale_bits"])["named_f32_weights"]
        self.arrays = {
            "positions": np.asarray([1024], dtype=np.int64), "token_ids": np.asarray([42], dtype=np.int64),
            "checked_indices": np.asarray([0], dtype=np.int64), "residual_bits": bits([[1, 1, 1, 1]]),
            "norm_bits": bits([[1, 1, 1, 1]]), "projection_bits": bits([[0.5, 1, 1.5, 2]]),
            "logits_f32": logits, "selected_ids": ids, "selected_weights_f32": weight,
        }

    def analyze(self):
        return ref.analyze_forward(self.arrays, self.reference, [0], hidden=4, experts=4, topk=2)

    def test_independent_stage_analysis_retains_its_scope(self):
        report = self.analyze()
        self.assertTrue(report["projection_reconstructed_from_norm_root_dimension_bits_equal"])
        self.assertEqual(report["checked_projection_rows"], 1)
        self.assertFalse(report["whole_router_numerically_qualified"])
        self.assertFalse(report["rms_semantic_diagnostic"]["native_rms_qualified"])

    def test_wrong_projection_stage_is_an_observed_discrepancy(self):
        self.arrays["projection_bits"][0, 3] = bits(2.5)
        report = self.analyze()
        self.assertFalse(report["projection_reconstructed_from_norm_root_dimension_bits_equal"])
        self.assertFalse(report["projection_profile"]["conditional_envelope_pass"])

    def test_duplicate_checked_row_cannot_expand_oracle_coverage(self):
        self.arrays["checked_indices"] = np.asarray([0, 0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "duplicate or outside"):
            self.analyze()

    def test_ninth_checked_row_exceeds_bounded_oracle(self):
        self.arrays["checked_indices"] = np.arange(9, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "bounded integer vector"):
            self.analyze()

    def test_missing_stage_is_not_silently_substituted(self):
        del self.arrays["norm_bits"]
        with self.assertRaises(KeyError):
            self.analyze()


class CaptureProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="router-cpu-unit-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_npz(self, arrays):
        path = self.directory / "fixture.npz"
        np.savez(path, **arrays)
        return path.name, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_npz_loader_checks_hash_and_exact_member_scope(self):
        name, sha = self.write_npz({"bits": np.asarray([1], dtype=np.uint16)})
        loaded = capture.read_npz(self.directory, name, sha, {"bits"})
        np.testing.assert_array_equal(loaded["bits"], [1])
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            capture.read_npz(self.directory, name, "0" * 64, {"bits"})
        with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
            capture.read_npz(self.directory, name, sha, {"other"})

    def test_npz_loader_rejects_path_escape(self):
        with self.assertRaisesRegex(ValueError, "local basename"):
            capture.read_npz(self.directory, "../outside.npz", "0" * 64, {"bits"})

    def test_npz_loader_never_loads_object_pickles(self):
        name, sha = self.write_npz({"bits": np.asarray([{"not": "numeric"}], dtype=object)})
        with self.assertRaises(ValueError):
            capture.read_npz(self.directory, name, sha, {"bits"})

    def request(self):
        prompt = list(range(1025))
        return {"id": "one-request", "prompt_token_ids": prompt,
                "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                "completion_token_ids": [10, 11], "usage": {"prompt_tokens": 1025, "completion_tokens": 2}}

    def test_client_prefix_digest_and_usage_are_bound_to_actual_ids(self):
        request = self.request()
        self.assertIs(capture.validate_request([request]), request)
        request["prompt_token_ids"][0] = 20
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            capture.validate_request([request])

    def test_client_scope_cannot_expand_to_two_requests_or_nine_outputs(self):
        request = self.request()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            capture.validate_request([request, request])
        request["completion_token_ids"] = list(range(9))
        with self.assertRaisesRegex(ValueError, "eight-output scope"):
            capture.validate_request([request])

    def original(self):
        arrays = {"weight_bits": bits(np.zeros((128, 2816))), "dimension_scale_bits": bits(np.ones(2816)),
                  "expert_scale_bits": bits(np.ones(128)), "root_raw_bits": np.asarray([0x9A, 0x3C], dtype=np.uint8),
                  "root_cast_bf16_bits": np.asarray([0x3C9A], dtype=np.uint16)}
        original = []
        for label, suffix in (("weight_bits", "proj.weight"), ("dimension_scale_bits", "scale"),
                              ("expert_scale_bits", "per_expert_scale")):
            payload = arrays[label].astype("<u2").tobytes()
            original.append({"tensor": "model.language_model.layers.0.router." + suffix,
                             "dtype": "torch.bfloat16", "shape": list(arrays[label].shape),
                             "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                             "loaded_original_bytes_equal": True})
        metadata = {"layer": 0, "expert_ids": [42, 82], "loader_ordinal": 0,
                    "projection_lane": "torch.mm(BF16,BF16,out_dtype=float32)",
                    "projection_flags": {"allow_specialized_router_gemm": False, "allow_fp32_router_gemm": False,
                                         "allow_bf16x3_router_gemm": False, "allow_ll_bf16_gemm": False,
                                         "allow_cublas_router_gemm": True},
                    "norm_eps": 1e-6, "norm_has_weight": False, "norm_pass_weight": False,
                    "norm_pass_weight_add": False, "norm_variance_size_override": None,
                    "root_storage_dtype": "torch.bfloat16", "root_storage_shape": [],
                    "original_tensors": original}
        return metadata, arrays

    def test_original_payload_hashes_and_actual_root_cast_match(self):
        metadata, arrays = self.original()
        report = capture.validate_reference(metadata, arrays)
        self.assertTrue(report["original_tensor_payloads_match_manifest_sha256"])
        self.assertTrue(report["root_cast_matches_storage_bits"])
        self.assertIn("not reread", report["loaded_original_byte_equality"])

    def test_payload_tampering_cannot_preserve_original_weight_provenance(self):
        metadata, arrays = self.original()
        arrays["weight_bits"][0, 0] = 0x3F80
        with self.assertRaisesRegex(ValueError, "hash/loaded-byte proof differs"):
            capture.validate_reference(metadata, arrays)

    def test_missing_loaded_original_byte_proof_is_rejected(self):
        metadata, arrays = self.original()
        metadata["original_tensors"][0]["loaded_original_bytes_equal"] = False
        with self.assertRaisesRegex(ValueError, "hash/loaded-byte proof differs"):
            capture.validate_reference(metadata, arrays)

    def test_root_cast_cannot_be_replaced_without_matching_raw_storage(self):
        metadata, arrays = self.original()
        arrays["root_cast_bf16_bits"][0] += 1
        with self.assertRaisesRegex(ValueError, "Root castBF16 differs"):
            capture.validate_reference(metadata, arrays)

    def test_non_cublas_or_weighted_norm_cannot_use_this_profile(self):
        metadata, arrays = self.original()
        changed = copy.deepcopy(metadata)
        changed["projection_flags"]["allow_specialized_router_gemm"] = True
        with self.assertRaisesRegex(ValueError, "pinned BF16 cuBLAS"):
            capture.validate_reference(changed, arrays)
        metadata["norm_has_weight"] = True
        with self.assertRaisesRegex(ValueError, "RMS norm contract"):
            capture.validate_reference(metadata, arrays)


class ReportPreservationTests(unittest.TestCase):
    """Exercise real CLI writing while replacing only expensive math analysis."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="router-report-unit-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.capture_directory = self.directory / "router-scores"
        self.capture_directory.mkdir()
        self.requests = self.directory / "quality-requests.json"
        self.requests.write_bytes(b"original client request evidence")
        self.report = {"captured_files": 60, "natural_input_rows_per_layer": 1032,
                       "layers": [{"summary": {"checked_projection_rows": 8,
                                               "selected_ids_match_pinned_bit_order": True}}]}

    def invoke(self, output, requests=None):
        arguments = ["check_router_capture.py", str(self.capture_directory), "--output", str(output)]
        if requests is not None:
            arguments.extend(["--requests", str(requests)])
        with mock.patch("sys.argv", arguments), \
                mock.patch.object(capture, "check_capture", return_value=self.report) as analysis, \
                redirect_stdout(io.StringIO()):
            self.analysis = analysis
            capture.main()

    def test_capture_manifest_cannot_be_overwritten(self):
        evidence = self.capture_directory / "capture-manifest.json"
        original = b"immutable captured manifest"
        evidence.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "outside capture directory"):
            self.invoke(evidence)
        self.analysis.assert_not_called()
        self.assertEqual(evidence.read_bytes(), original)

    def test_new_report_cannot_be_added_inside_capture_directory(self):
        evidence = self.capture_directory / "new-report.json"
        with self.assertRaisesRegex(ValueError, "outside capture directory"):
            self.invoke(evidence)
        self.analysis.assert_not_called()
        self.assertFalse(evidence.exists())

    def test_default_client_request_cannot_be_overwritten(self):
        original = self.requests.read_bytes()
        with self.assertRaisesRegex(ValueError, "distinct from request evidence"):
            self.invoke(self.requests)
        self.analysis.assert_not_called()
        self.assertEqual(self.requests.read_bytes(), original)

    def test_explicit_client_request_cannot_be_overwritten(self):
        request = self.directory / "explicit-request.json"
        original = b"separate explicit request evidence"
        request.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "distinct from request evidence"):
            self.invoke(request, requests=request)
        self.analysis.assert_not_called()
        self.assertEqual(request.read_bytes(), original)

    def test_symlink_directory_cannot_bypass_capture_protection(self):
        alias = self.directory / "capture-alias"
        alias.symlink_to(self.capture_directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside capture directory"):
            self.invoke(alias / "new-report.json")
        self.analysis.assert_not_called()
        self.assertFalse((self.capture_directory / "new-report.json").exists())

    def test_symlink_request_cannot_bypass_request_protection(self):
        alias = self.directory / "request-alias.json"
        alias.symlink_to(self.requests)
        original = self.requests.read_bytes()
        with self.assertRaisesRegex(ValueError, "distinct from request evidence"):
            self.invoke(alias)
        self.analysis.assert_not_called()
        self.assertEqual(self.requests.read_bytes(), original)

    def test_existing_external_report_is_preserved_by_exclusive_create(self):
        output = self.directory / "existing-report.json"
        original = b"earlier immutable CPU report"
        output.write_bytes(original)
        with self.assertRaises(FileExistsError):
            self.invoke(output)
        self.analysis.assert_called_once()
        self.assertEqual(output.read_bytes(), original)

    def test_new_external_report_is_written_without_modifying_evidence(self):
        output = self.directory / "router-scores-report.json"
        original = self.requests.read_bytes()
        self.invoke(output)
        self.analysis.assert_called_once_with(self.capture_directory, self.requests)
        self.assertEqual(json.loads(output.read_text()), self.report)
        self.assertEqual(self.requests.read_bytes(), original)
        self.assertEqual(list(self.capture_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
