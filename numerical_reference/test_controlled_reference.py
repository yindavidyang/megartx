"""Golden arithmetic and adversarial controlled-fixture CPU guards; no GPU."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import nvfp4_reference as formats
import controlled_reference as ref


def f32bits(value):
    return np.asarray([value], dtype=np.float32).view(np.uint32)


def schedule():
    return ref.make_schedule(np.arange(33, dtype=np.int64) % 4)


def dispatch(table):
    hits = [{**ref.CONTROLLED_ORIGIN, "layer": layer, "expert": expert, "position": ref.TARGET_POSITIONS[layer, expert],
             "slot": 7, "weight_bits": int(f32bits(0.125)[0]), "mode": "native",
             "completed": True, "finite_output": True, "nonzero_output_elements": 2816,
             "registered_binding_verified": True, "execution_proof_sha256": "d" * 64,
             "cuda_trace_sha256": "e" * 64, "stage_capture_sha256": "f" * 64} for layer, expert in ref.TARGETS]
    return {**ref.CONTROLLED_ORIGIN, "token_sha256": table["token_sha256"], "schedule_sha256": table["schedule_sha256"],
            "mode": "native", "layers": np.arange(30, dtype=np.int32),
            "positions": np.arange(33, dtype=np.int64), "tokens": table["tokens"].copy(),
            "ids": table["ids"].copy(), "weight_bits": table["weight_bits"].copy(),
            "executed_interventions": hits, "execution_proof_sha256": "d" * 64, "cuda_trace_sha256": "e" * 64}


class RouteBindingTests(unittest.TestCase):
    def setUp(self):
        self.table = schedule()
        self.actual = dispatch(self.table)

    def test_exact_six_targets_and_ordinary_rows_are_predeclared(self):
        self.assertEqual(self.table["ids"][0, 31].tolist(), [0, 1, 2, 3, 4, 5, 6, 42])
        self.assertEqual(self.table["ids"][0, 32].tolist(), [0, 1, 2, 3, 4, 5, 6, 82])
        self.assertEqual(self.table["ids"][29, 32].tolist(), [0, 1, 2, 3, 4, 5, 6, 8])
        report = ref.validate_dispatch(self.table, self.actual)
        self.assertEqual(report["reported_completed_intervention_records"], 6)
        self.assertFalse(report["independent_execution_proof_trace_stage_verification"])
        self.assertFalse(report["model_or_natural_quality_qualified"])

    def test_reordered_complete_rows_align_by_absolute_position(self):
        order = np.arange(33)[::-1]
        for field in ("positions", "tokens"):
            self.actual[field] = self.actual[field][order]
        for field in ("ids", "weight_bits"):
            self.actual[field] = self.actual[field][:, order]
        self.assertTrue(ref.validate_dispatch(self.table, self.actual)["all_reported_dispatched_rows_exact"])

    def test_changed_ordinary_route_cannot_hide_behind_positive_hit_counts(self):
        self.actual["ids"][29, 10, 7] = 9
        with self.assertRaisesRegex(ValueError, "Actual dispatched"):
            ref.validate_dispatch(self.table, self.actual)

    def test_changed_weight_raw_bits_cannot_pass_numerical_relabelling(self):
        self.actual["weight_bits"][7, 10, 2] += 1
        with self.assertRaisesRegex(ValueError, "Actual dispatched"):
            ref.validate_dispatch(self.table, self.actual)

    def test_modified_schedule_cannot_be_legitimized_by_rehashing(self):
        self.table["ids"][0, 31, 7] = 8
        self.table["schedule_sha256"] = ref.schedule_sha256(self.table)
        with self.assertRaisesRegex(ValueError, "predeclared deterministic"):
            ref.validate_schedule(self.table)

    def test_nonpositive_intervention_weight_is_rejected_even_after_rehash(self):
        self.table["weight_bits"][0, 31, 7] = 0
        self.table["schedule_sha256"] = ref.schedule_sha256(self.table)
        with self.assertRaisesRegex(ValueError, "predeclared F32"):
            ref.validate_schedule(self.table)

    def test_duplicate_position_cannot_replace_missing_real_decode_input(self):
        self.actual["positions"][-1] = 31
        with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
            ref.validate_dispatch(self.table, self.actual)

    def test_client_token_mismatch_is_rejected_before_arithmetic(self):
        self.actual["tokens"][-1] = 3
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            ref.validate_dispatch(self.table, self.actual)

    def test_missing_and_duplicate_executed_hit_do_not_count_as_coverage(self):
        hit = self.actual["executed_interventions"].pop()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            ref.validate_dispatch(self.table, self.actual)
        self.actual["executed_interventions"].append(copy.deepcopy(self.actual["executed_interventions"][0]))
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            ref.validate_dispatch(self.table, self.actual)
        self.actual["executed_interventions"][-1] = hit
        self.assertEqual(ref.validate_dispatch(self.table, self.actual)["reported_completed_intervention_records"], 6)

    def test_old_fused_control_is_not_the_single_alpha_negative_mode(self):
        self.actual["mode"] = "control"
        with self.assertRaisesRegex(ValueError, "Unknown controlled"):
            ref.validate_dispatch(self.table, self.actual)

    def test_wrong_layer_or_weight_in_hit_is_rejected(self):
        self.actual["executed_interventions"][0]["weight_bits"] = int(f32bits(0.25)[0])
        with self.assertRaisesRegex(ValueError, "correction weight"):
            ref.validate_dispatch(self.table, self.actual)

    def test_short_or_large_prefix_and_changed_digest_are_rejected(self):
        for length in (32, 34, 1034):
            with self.subTest(length=length), self.assertRaises(ValueError):
                ref.make_schedule(np.zeros(length, dtype=np.int64))
        self.actual["token_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prefix/route-table"):
            ref.validate_dispatch(self.table, self.actual)

    def test_plan_records_cannot_claim_completed_finite_registered_output(self):
        for field in ("completed", "finite_output", "registered_binding_verified"):
            changed = copy.deepcopy(self.actual)
            changed["executed_interventions"][0][field] = False
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "execution assertions"):
                ref.validate_dispatch(self.table, changed)

    def test_zero_or_unbounded_output_cannot_count_as_positive_execution(self):
        for count in (0, -1, 2817, True):
            changed = copy.deepcopy(self.actual)
            changed["executed_interventions"][0]["nonzero_output_elements"] = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "bounded nonzero"):
                ref.validate_dispatch(self.table, changed)

    def test_mixed_execution_mode_or_proof_trace_stage_hashes_are_rejected(self):
        for field, value in (("mode", "paired_reference"), ("execution_proof_sha256", "1" * 64),
                             ("cuda_trace_sha256", "2" * 64), ("stage_capture_sha256", "missing")):
            changed = copy.deepcopy(self.actual)
            changed["executed_interventions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ref.validate_dispatch(self.table, changed)

    def test_controlled_artifacts_cannot_be_relabelled_natural_at_any_scope(self):
        for location in ("schedule", "dispatch", "hit"):
            table, actual = copy.deepcopy(self.table), copy.deepcopy(self.actual)
            record = table if location == "schedule" else actual if location == "dispatch" else actual["executed_interventions"][0]
            record["route_origin"] = "natural"
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "not natural coverage"):
                ref.validate_dispatch(table, actual)


class IndependentReplayTests(unittest.TestCase):
    def setUp(self):
        # Exactly known units: A and W are one at sixteen K coordinates.
        self.q = np.full((1, 8), 0x22, dtype=np.uint8)
        self.sf = formats.swizzle_128x4(np.asarray([[0x38]], dtype=np.uint8))
        self.original = (np.full((1, 8), 0x22, dtype=np.uint8),
                         np.asarray([[0x38]], dtype=np.uint8), 0.5)

    def replay(self, candidate=8, alpha=0.5):
        return ref.replay_projection(self.q, self.sf, self.original, input_global_bits=f32bits(1),
                                     alpha_bits=f32bits(alpha), candidate_bits=ref.bf16_bits([[candidate]]))

    def test_known_original_global_dot_matches_without_global_folding(self):
        report = self.replay()
        self.assertTrue(report["comparison"]["raw_bf16_bits_equal"])
        self.assertFalse(report["comparison"]["native_mma_qualified"])
        self.assertEqual(int(report["expected_bits"][0, 0]), 0x4100)

    def test_wrong_gate_only_scale_is_detected_without_fitting_a_margin(self):
        report = self.replay(candidate=4)
        self.assertFalse(report["comparison"]["raw_bf16_bits_equal"])
        self.assertFalse(report["comparison"]["conditional_f32_envelope_pass"])
        self.assertEqual(report["comparison"]["max_absolute_difference"], 4)

    def test_rounded_wrong_alpha_fails_provenance_before_dot_comparison(self):
        with self.assertRaisesRegex(ValueError, "original projection/global product"):
            self.replay(candidate=4, alpha=0.25)

    def test_e4m3_bytes_are_decoded_as_format_not_uint8_numeric_values(self):
        self.sf = formats.swizzle_128x4(np.asarray([[0x40]], dtype=np.uint8))  # 2, not 64.
        report = self.replay(candidate=16)
        self.assertTrue(report["comparison"]["raw_bf16_bits_equal"])

    def test_low_high_nibble_order_has_a_coordinate_negative_control(self):
        self.q[0, 0] = 0x12  # first A is 1, second A is 0.5.
        weight = np.zeros((1, 8), dtype=np.uint8)
        weight[0, 0] = 0x02  # select first coordinate only.
        self.original = (weight, np.asarray([[0x38]], dtype=np.uint8), 0.5)
        self.assertTrue(self.replay(candidate=0.5)["comparison"]["raw_bf16_bits_equal"])
        self.assertFalse(self.replay(candidate=0.25)["comparison"]["conditional_f32_envelope_pass"])

    def test_four_rows_and_large_k_n_cannot_bypass_memory_bounds(self):
        for rows, k, n in ((4, 16, 1), (1, 2832, 1), (1, 16, 2817)):
            q = np.zeros((rows, k // 2), dtype=np.uint8)
            original = (np.zeros((n, k // 2), dtype=np.uint8), np.zeros((n, k // 16), dtype=np.uint8), 0.5)
            with self.subTest(rows=rows, k=k, n=n), self.assertRaisesRegex(ValueError, "bounded"):
                ref.replay_projection(q, self.sf, original, input_global_bits=f32bits(1),
                                      alpha_bits=f32bits(0.5), candidate_bits=np.zeros((rows, n), dtype=np.uint16))

    def test_signed_zero_does_not_become_bit_exact_by_value_comparison(self):
        report = ref._compare_bits(np.asarray([0x8000], dtype=np.uint16), np.asarray([0], dtype=np.uint16))
        self.assertTrue(report["numerical_values_equal"])
        self.assertFalse(report["raw_bf16_bits_equal"])

    def test_named_quantizer_payload_differences_report_effective_zero_equality(self):
        q = np.full((1, 8), 0x77, dtype=np.uint8)
        zero_sf = formats.swizzle_128x4(np.zeros((1, 1), dtype=np.uint8))
        report = ref.quantizer_diagnostics(np.zeros((1, 16), dtype=np.uint16),
                                          input_global_bits=f32bits(1), actual_quant_global_bits=f32bits(1),
                                          packed=q, physical_sf=zero_sf)
        self.assertFalse(report["flashinfer_strict_rn"]["packed_bytes_equal"])
        self.assertTrue(report["flashinfer_strict_rn"]["decoded_units_equal"])
        self.assertFalse(report["cuda_quantizer_qualified"])

    def test_wrong_quantizer_reciprocal_is_not_reconstructed_away(self):
        with self.assertRaisesRegex(ValueError, "explicit F32 reciprocal"):
            ref.quantizer_diagnostics(np.zeros((1, 16), dtype=np.uint16), input_global_bits=f32bits(2),
                                     actual_quant_global_bits=f32bits(1), packed=self.q, physical_sf=self.sf)

    def test_original_reader_reads_bounded_payload_and_preserves_raw_hashes(self):
        with tempfile.TemporaryDirectory(prefix="controlled-original-unit-") as temporary:
            payload, header = bytearray(), {}
            for label, global_scale in (("gate", 0.25), ("up", 0.5), ("down", 0.125)):
                prefix = f"model.language_model.layers.0.experts.42.{label}_proj"
                for suffix, dtype, value in (("weight", "U8", self.original[0]),
                                              ("weight_scale", "F8_E4M3", self.original[1]),
                                              ("weight_scale_2", "F32", np.asarray(global_scale, dtype="<f4"))):
                    raw = value.tobytes()
                    begin = len(payload)
                    payload.extend(raw)
                    header[prefix + "." + suffix] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [begin, len(payload)]}
            encoded = json.dumps(header).encode()
            path = Path(temporary) / "original.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            reader = formats.CheckpointReader(Path(temporary))
            originals, hashes = ref.original_expert(reader, 0, 42)
            self.assertEqual(originals["up"][2], 0.5)
            self.assertEqual(hashes["up"]["weight_sha256"], hashlib.sha256(self.original[0].tobytes()).hexdigest())
            self.assertEqual(path.read_bytes(), struct.pack("<Q", len(encoded)) + encoded + payload)
            with self.assertRaisesRegex(ValueError, "six affected"):
                ref.original_expert(reader, 0, 0)

    def expert_case(self, negative=False):
        # Golden captured-operand arithmetic. Quantizer profiles are deliberately
        # separate diagnostics: this synthetic q is not asserted to come from CUDA.
        original = {"gate": (np.full((16, 8), 0x11, dtype=np.uint8), np.full((16, 1), 0x38, dtype=np.uint8), 0.25),
                    "up": (np.full((16, 8), 0x22, dtype=np.uint8), np.full((16, 1), 0x38, dtype=np.uint8), 0.5),
                    "down": (np.full((16, 8), 0x22, dtype=np.uint8), np.full((16, 1), 0x38, dtype=np.uint8), 0.125)}
        stages = {"input_bits": ref.bf16_bits(np.ones((1, 16))), "q1": self.q.copy(), "sf1": self.sf.copy(),
                  "q2": self.q.copy(), "sf2": self.sf.copy(), "quant1_global_bits": f32bits(1),
                  "quant2_global_bits": f32bits(1), "gate_alpha_bits": f32bits(0.25),
                  "up_alpha_bits": f32bits(0.25 if negative else 0.5), "down_alpha_bits": f32bits(0.125),
                  "gate_bits": ref.bf16_bits(np.full((1, 16), 2)),
                  "up_bits": ref.bf16_bits(np.full((1, 16), 4 if negative else 8)),
                  "activation_bits": ref.bf16_bits(np.full((1, 16), 7.8125 if negative else 15.625)),
                  "down_bits": ref.bf16_bits(np.full((1, 16), 2))}
        return stages, original

    def test_three_projection_replay_does_not_promote_shared_quantization(self):
        stages, originals = self.expert_case()
        report = ref.replay_expert(stages, originals, a1_bits=f32bits(1), a2_bits=f32bits(1), mode="native")
        self.assertTrue(all(report[label]["raw_bf16_bits_equal"] for label in ("gate", "up", "down")))
        self.assertFalse(report["a1_quantizer_profiles"]["flashinfer_strict_rn"]["decoded_units_equal"])
        self.assertFalse(report["native_activation_qualified"])
        self.assertFalse(report["whole_expert_or_model_qualified"])

    def test_wrong_up_alpha_is_allowed_only_in_explicit_negative_mode(self):
        stages, originals = self.expert_case(negative=True)
        with self.assertRaisesRegex(ValueError, "original projection/global product"):
            ref.replay_expert(stages, originals, a1_bits=f32bits(1), a2_bits=f32bits(1), mode="native")
        report = ref.replay_expert(stages, originals, a1_bits=f32bits(1), a2_bits=f32bits(1), mode="gate_only_negative_control")
        self.assertTrue(all(report[label]["raw_bf16_bits_equal"] for label in ("gate", "up", "down")))
        self.assertEqual(originals["up"][2], 0.5)
        np.testing.assert_array_equal(originals["up"][0], np.full((16, 8), 0x22, dtype=np.uint8))


def logit_snapshot(table, positions=(31, 32)):
    positions = np.asarray(positions, dtype=np.int64)
    return {**ref.CONTROLLED_ORIGIN, "token_sha256": table["token_sha256"], "schedule_sha256": table["schedule_sha256"],
            "row_identity_verified": True, "input_positions": positions,
            "input_token_ids": table["tokens"][positions],
            "logits": np.asarray([[0, 1, 2, 3] for _ in positions], dtype=np.float32)}


class LogitIdentityTests(unittest.TestCase):
    def setUp(self):
        self.table = schedule()
        self.a = logit_snapshot(self.table)
        self.b = logit_snapshot(self.table)

    def compare(self):
        return ref.compare_logits(self.a, self.b, self.table, expected_vocab=4)

    def test_full_distribution_identity_scores_once_at_real_next_token(self):
        report = self.compare()
        self.assertTrue(report["full_f32_bits_equal"])
        self.assertEqual(report["prediction_positions"], [32, 33])
        self.assertEqual(report["unique_scored_prediction_positions"], 1)
        expected = np.log(np.exp(np.arange(4) - 3).sum()) + 3  # token at 32 is zero.
        self.assertAlmostEqual(report["scored_nll"][0]["candidate"], expected)
        self.assertFalse(report["quality_gate_passed"])

    def test_call_or_capture_order_does_not_determine_logit_alignment(self):
        self.b = logit_snapshot(self.table, positions=(32, 31))
        self.assertTrue(self.compare()["full_f32_bits_equal"])

    def test_one_changed_logit_is_reported_without_auto_accepting_small_error(self):
        self.a["logits"][1, 0] = np.nextafter(np.float32(0), np.float32(1))
        report = self.compare()
        self.assertFalse(report["full_f32_bits_equal"])
        self.assertEqual(report["first_changed_input_position"], 32)
        self.assertIsNone(report["numerical_acceptance_tolerance"])

    def test_signed_zero_is_a_raw_logit_difference_even_with_zero_value_error(self):
        self.a["logits"][0, 0] = -0.0
        report = self.compare()
        self.assertFalse(report["full_f32_bits_equal"])
        self.assertEqual(report["max_absolute_logit_difference"], 0)

    def test_duplicate_position_cannot_be_scored_or_compared_twice(self):
        self.a = logit_snapshot(self.table, positions=(31, 31))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compare()

    def test_wrong_token_or_hidden_row_identity_is_rejected(self):
        self.a["input_token_ids"][1] = 1
        with self.assertRaisesRegex(ValueError, "wrong actual input token"):
            self.compare()
        self.a = logit_snapshot(self.table)
        self.a["row_identity_verified"] = False
        with self.assertRaisesRegex(ValueError, "correspondence"):
            self.compare()

    def test_mismatched_position_set_or_prefix_cannot_compare(self):
        self.b = logit_snapshot(self.table, positions=(30, 32))
        with self.assertRaisesRegex(ValueError, "required handoff logit positions"):
            self.compare()
        self.b = logit_snapshot(self.table)
        self.b["token_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.compare()

    def test_top_five_or_nonfinite_rows_cannot_impersonate_full_vocabulary(self):
        self.a["logits"] = self.a["logits"][:, :3]
        with self.assertRaisesRegex(ValueError, "full-vocabulary"):
            self.compare()
        self.a = logit_snapshot(self.table)
        self.a["logits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.compare()

    def test_large_or_degenerate_vocabulary_is_rejected_before_analysis(self):
        for vocabulary in (1, 262145):
            with self.subTest(vocabulary=vocabulary), self.assertRaisesRegex(ValueError, "fixed full-vocabulary bound"):
                ref.compare_logits(self.a, self.b, self.table, expected_vocab=vocabulary)

    def test_one_handoff_row_cannot_establish_the_two_row_contract(self):
        for positions in ((31,), (32,), (0,)):
            self.a = logit_snapshot(self.table, positions=positions)
            self.b = copy.deepcopy(self.a)
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "required handoff"):
                self.compare()

    def test_logit_origin_cannot_be_natural_even_with_matching_tokens(self):
        self.a["routing_intervention"] = False
        with self.assertRaisesRegex(ValueError, "not natural coverage"):
            self.compare()


def cache_contract():
    # Synthetic geometry deliberately differs from model config: no layer%6 assumption.
    contract = {"provenance": {"scope": "synthetic_cpu_fixture", "config_sha256": "a" * 64,
                               "source_sha256": "b" * 64, "kv_owner_layout_confirmed": False},
                "layers": [{"layer": i, "attention_type": "full_attention", "kv_heads": 1,
                            "head_dim": 4, "window_size": None, "logical_layout": "position,kv_head,head_dim",
                            "dtype": "bf16"} for i in range(30)]}
    contract["sha256"] = ref.cache_contract_sha256(contract)
    return contract


def cache_snapshot(table, contract, layer=0):
    return {**ref.CONTROLLED_ORIGIN, "token_sha256": table["token_sha256"], "schedule_sha256": table["schedule_sha256"],
            "cache_contract_sha256": contract["sha256"], "layer": layer, "committed_length": 33,
            "attention_type": "full_attention", "kv_heads": 1, "head_dim": 4, "mask_window_start": 0,
            "logical_mapping_verified": True, "logical_positions": np.asarray([31, 32], dtype=np.int64),
            "key_bits": ref.bf16_bits(np.ones((2, 1, 4))), "value_bits": ref.bf16_bits(np.full((2, 1, 4), 2))}


class CacheDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.table = schedule()
        self.contract = cache_contract()
        self.a = cache_snapshot(self.table, self.contract)
        self.b = cache_snapshot(self.table, self.contract)

    def compare(self):
        return ref.compare_cache(self.a, self.b, self.table, cache_contract=self.contract)

    def test_logical_comparison_uses_explicit_geometry_and_keeps_k_v_distinct(self):
        report = self.compare()
        self.assertTrue(report["key"]["raw_bf16_bits_equal"])
        self.assertTrue(report["value"]["raw_bf16_bits_equal"])
        self.assertEqual(report["cache_descriptor_scope"], "synthetic_cpu_fixture")
        self.assertFalse(report["installed_owner_layout_independently_verified"])
        self.assertFalse(report["independent_cache_correctness_qualified"])

    def test_key_value_swap_is_detected_without_assuming_projection_sharing(self):
        self.b["key_bits"], self.b["value_bits"] = self.b["value_bits"], self.b["key_bits"]
        report = self.compare()
        self.assertFalse(report["key"]["raw_bf16_bits_equal"])
        self.assertFalse(report["value"]["raw_bf16_bits_equal"])

    def test_physical_block_identity_is_ignored_after_logical_mapping(self):
        self.a["physical_block_id"] = 100
        self.b["physical_block_id"] = 17
        order = [1, 0]
        for field in ("logical_positions", "key_bits", "value_bits"):
            self.b[field] = self.b[field][order]
        self.assertTrue(self.compare()["key"]["raw_bf16_bits_equal"])

    def test_stale_uncommitted_duplicate_or_wrong_length_state_is_rejected(self):
        for positions in ([31, 33], [31, 31], [-1, 32]):
            self.a["logical_positions"] = np.asarray(positions, dtype=np.int64)
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "stale, evicted or uncommitted"):
                self.compare()
        self.a = cache_snapshot(self.table, self.contract)
        self.a["committed_length"] = 32
        with self.assertRaisesRegex(ValueError, "committed sequence"):
            self.compare()

    def test_unverified_mapping_or_wrong_descriptor_identity_fails_closed(self):
        self.a["logical_mapping_verified"] = False
        with self.assertRaisesRegex(ValueError, "mapping was not verified"):
            self.compare()
        self.a["logical_mapping_verified"] = True
        self.a["cache_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen descriptor"):
            self.compare()

    def test_mutated_descriptor_cannot_preserve_its_frozen_hash(self):
        self.contract["layers"][0]["head_dim"] = 5
        with self.assertRaisesRegex(ValueError, "digest differs"):
            self.compare()

    def test_config_only_provenance_cannot_assert_runtime_owner_confirmation(self):
        self.contract["provenance"]["scope"] = "checkpoint_config_only"
        self.contract["provenance"]["kv_owner_layout_confirmed"] = True
        self.contract["sha256"] = ref.cache_contract_sha256(self.contract)
        with self.assertRaisesRegex(ValueError, "exceeds its descriptor provenance"):
            self.compare()

    def test_wrong_head_geometry_or_cache_layer_cannot_compare(self):
        self.a["head_dim"] = 3
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            self.compare()
        self.a = cache_snapshot(self.table, self.contract, layer=1)
        with self.assertRaisesRegex(ValueError, "logical layer/position"):
            self.compare()

    def test_descriptor_memory_bounds_and_bf16_finiteness_are_enforced(self):
        self.contract["layers"][0]["head_dim"] = 513
        self.contract["sha256"] = ref.cache_contract_sha256(self.contract)
        with self.assertRaisesRegex(ValueError, "bounded head geometry"):
            self.compare()
        self.contract = cache_contract()
        self.a["key_bits"][0, 0, 0] = 0x7F80
        with self.assertRaisesRegex(ValueError, "Nonfinite captured BF16"):
            self.compare()

    def test_selected_nominal_config_hash_and_tampering_guard(self):
        config_path = Path(__file__).resolve().parents[1] / "deliverables/review-copy/results/sources/config.json"
        if config_path.exists():
            contract = ref.nominal_cache_contract(config_path)
            self.assertEqual(contract["provenance"]["config_sha256"], ref.CONFIG_SHA256)
            self.assertEqual((contract["layers"][0]["kv_heads"], contract["layers"][0]["head_dim"]), (8, 256))
            self.assertEqual((contract["layers"][5]["kv_heads"], contract["layers"][5]["head_dim"]), (2, 512))
            self.assertFalse(contract["provenance"]["kv_owner_layout_confirmed"])
        with tempfile.TemporaryDirectory(prefix="controlled-config-unit-") as temporary:
            path = Path(temporary) / "changed-config.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable checkpoint hash"):
                ref.nominal_cache_contract(path)

    def test_missing_handoff_cache_row_cannot_establish_consistency(self):
        for field in ("logical_positions", "key_bits", "value_bits"):
            self.a[field] = self.a[field][:1]
            self.b[field] = self.b[field][:1]
        with self.assertRaisesRegex(ValueError, "required logical handoff"):
            self.compare()

    def test_cache_collection_requires_every_nominal_layer_once(self):
        left = [cache_snapshot(self.table, self.contract, layer=i) for i in range(30)]
        right = copy.deepcopy(left)
        report = ref.compare_cache_collection(left, right, self.table, cache_contract=self.contract)
        self.assertTrue(report["all_selected_logical_k_v_bits_equal"])
        self.assertFalse(report["installed_owner_layout_independently_verified"])
        with self.assertRaisesRegex(ValueError, "missing nominal layer coverage"):
            ref.compare_cache_collection(left[:-1], right, self.table, cache_contract=self.contract)
        left[-1] = copy.deepcopy(left[0])
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            ref.compare_cache_collection(left, right, self.table, cache_contract=self.contract)

    def test_cache_origin_cannot_be_relabelled_as_natural(self):
        self.b["scope"] = "natural_quality"
        with self.assertRaisesRegex(ValueError, "not natural coverage"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
