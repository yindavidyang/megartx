"""Explicit fresh-run maps and fail-closed handoff comparisons; CPU only."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import compare_controlled_paths as check
import controlled_reference as ref
import test_controlled_reference as examples


class RunMapGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = examples.schedule()
        self.filename = self.root / "run-map.json"
        self.payload = {**ref.CONTROLLED_ORIGIN, "token_sha256": self.plan["token_sha256"],
            "schedule_sha256": self.plan["schedule_sha256"], "runs": {}}
        for path, modes in (("full", check.MODES), ("cached", check.MODES[:2])):
            self.payload["runs"][path] = {}
            for mode in modes:
                name = path + "-" + mode
                directory = self.root / name / "controlled" / path
                directory.mkdir(parents=True)
                for field in check.FROZEN_METADATA:
                    (directory / field).write_text("{}")
                (directory / "kv-binding.json").write_text(json.dumps({"layers": [
                    {"layer": i, "resolved_layout": "LBNHC"} for i in range(30)]}))
                self.payload["runs"][path][mode] = name
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.filename.write_text(json.dumps(self.payload))

    def load(self):
        return check.load_run_map(self.filename, self.plan)

    def result(self, gate):
        return {"all_strict_observed_operator_gates_pass": gate}

    def test_explicit_fresh_roots_resolve_from_map_parent_with_positive_only_cached(self):
        actual = self.load()
        self.assertEqual(set(actual["full"]), set(check.MODES))
        self.assertEqual(set(actual["cached"]), {"native", "paired_reference"})
        self.assertNotEqual(actual["full"]["native"], actual["cached"]["native"])
        self.assertEqual(actual["cached"]["native"], (self.root / "cached-native").resolve())

    def test_native_only_paths_are_valid_before_paired_evidence_exists(self):
        for path in self.payload["runs"]:
            self.payload["runs"][path] = {"native": self.payload["runs"][path]["native"]}
        self.save()
        self.assertTrue(all(set(values) == {"native"} for values in self.load().values()))

    def test_natural_origin_and_changed_prefix_table_are_rejected(self):
        for field, value in (("route_origin", "natural"), ("routing_intervention", False),
                             ("token_sha256", "0" * 64), ("schedule_sha256", "1" * 64)):
            old = self.payload[field]
            self.payload[field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()
            self.payload[field] = old

    def test_negative_without_paired_lane_and_unknown_mode_are_rejected(self):
        for entries in ({"native": "full-native", "gate_only_negative_control": "full-gate_only_negative_control"},
                        {"native": "full-native", "control": "full-native"}):
            self.payload["runs"]["full"] = entries
            self.save()
            with self.assertRaisesRegex(ValueError, "optional paired"):
                self.load()

    def test_unknown_path_or_extra_schema_cannot_silently_broaden_work(self):
        self.payload["runs"]["timed_8k"] = {"native": "full-native"}
        self.save()
        with self.assertRaisesRegex(ValueError, "execution paths"):
            self.load()
        del self.payload["runs"]["timed_8k"]
        self.payload["quality_gate_passed"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "schema"):
            self.load()

    def test_duplicate_json_key_cannot_hide_changed_run_origin(self):
        text = json.dumps(self.payload)
        self.filename.write_text(text[:-1] + ',"route_origin":"natural"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.load()

    def test_missing_directory_and_file_in_place_of_run_are_rejected(self):
        for name in ("not-present", self.filename.name):
            self.payload["runs"]["cached"]["native"] = name
            self.save()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "directory is missing"):
                self.load()

    def test_absent_duplicate_or_empty_requested_paths_reject_before_original_reads(self):
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check.base.formats, "CheckpointReader", side_effect=AssertionError("No original read")):
            for paths in ([], ["full", "full"], ["chunked"]):
                with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "duplicate or absent"):
                    check.compare_mapped_runs("unused", self.filename, "unused", paths)

    def test_two_lane_cached_path_uses_positive_only_gate_and_cannot_claim_complete_matrix(self):
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check.base, "compare_runs", return_value=self.result(False)) as original, patch.object(check, "compare_positive_only_runs", return_value=self.result(False)) as positive:
            report = check.compare_mapped_runs("unused", self.filename, "unused")
        self.assertEqual(original.call_count, 1)
        self.assertEqual(positive.call_count, 1)
        self.assertEqual(positive.call_args.args[-1], "cached")
        self.assertFalse(report["complete_paired_matrix_for_requested_paths"])
        self.assertFalse(report["controlled_handoff_equivalence_observed"])
        self.assertFalse(report["quality_gate_passed"])

    def test_failed_cached_operator_gate_withholds_cross_path_readers(self):
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check.base, "compare_runs", return_value=self.result(True)), patch.object(check, "compare_positive_only_runs", return_value=self.result(False)), patch.object(check.base.reference, "nominal_cache_contract", return_value={}), patch.object(check.base, "load_case", side_effect=AssertionError("No downstream reader")):
            report = check.compare_mapped_runs("unused", self.filename, "unused")
        self.assertIn("blocker", report["within_mode_path_comparisons"]["cached"])
        self.assertFalse(report["controlled_handoff_equivalence_observed"])

    def test_metadata_mutation_during_operator_replay_rejects_promotion(self):
        def mutation(*args):
            path = self.root / "full-native" / "controlled" / "full" / "controlled-manifest.json"
            path.write_text('{"changed":true}')
            return self.result(True)
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check.base, "compare_runs", side_effect=mutation), self.assertRaisesRegex(ValueError, "changed during"):
            check.compare_mapped_runs("unused", self.filename, "unused", ["full"])

    def test_unknown_physical_enum_rejects_before_original_projection_reads(self):
        filename = self.root / "full-native" / "controlled" / "full" / "kv-binding.json"
        binding = json.loads(filename.read_text())
        binding["layers"][0]["resolved_layout"] = "unrecognized_BHNC"
        filename.write_text(json.dumps(binding))
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check.base, "compare_runs", side_effect=AssertionError("No original read")), self.assertRaisesRegex(ValueError, "unknown installed physical layout"):
            check.compare_mapped_runs("unused", self.filename, "unused", ["full"])

    def test_known_physical_permutations_are_valid_logical_bhnc_descriptors(self):
        filename = self.root / "full-native" / "controlled" / "full" / "kv-binding.json"
        binding = json.loads(filename.read_text())
        for name in check.KV_LAYOUT_NAMES:
            binding["layers"][0]["resolved_layout"] = name
            filename.write_text(json.dumps(binding))
            self.assertEqual(len(check.metadata_hashes(self.root / "full-native", "full")), 5)

    def test_output_cannot_replace_run_map_or_evidence(self):
        for output in (self.filename, self.root / "cached-native" / "new-report.json"):
            args = ["--plan", str(self.root / "full-native"), "--run-map", str(self.filename),
                "--checkpoint", str(self.root / "cached-native"), "--output", str(output)]
            before = self.filename.read_bytes()
            with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check, "compare_mapped_runs", side_effect=AssertionError("Reject before replay")), self.assertRaisesRegex(ValueError, "outside"):
                check.main(args)
            self.assertEqual(self.filename.read_bytes(), before)

    def test_existing_outside_report_is_preserved(self):
        output = self.root / "report.json"
        output.write_text("prior report")
        args = ["--plan", str(self.root / "full-native"), "--run-map", str(self.filename),
            "--checkpoint", str(self.root / "cached-native"), "--output", str(output)]
        with patch.object(check.base, "load_plan", return_value=self.plan), patch.object(check, "compare_mapped_runs", return_value={
            "all_strict_observed_operator_gates_pass": False, "complete_paired_matrix_for_requested_paths": False}), self.assertRaises(FileExistsError):
            check.main(args)
        self.assertEqual(output.read_text(), "prior report")


class HandoffScopeGuards(unittest.TestCase):
    def setUp(self):
        self.plan = examples.schedule()
        values = np.zeros((2, 262144), dtype=np.float32)
        primary = {**ref.CONTROLLED_ORIGIN, "token_sha256": self.plan["token_sha256"], "schedule_sha256": self.plan["schedule_sha256"],
            "row_identity_verified": True, "input_positions": np.asarray([31, 32], np.int64),
            "input_token_ids": self.plan["tokens"][[31, 32]], "logits": values}
        self.full = {"primary": primary, "primary_source_rows": {31: 33, 32: 33},
            "rows": {(31, 33): values[0], (32, 33): values[1], (32, 1): values[1].copy()}}
        self.cached = {"primary": copy.deepcopy(primary), "primary_source_rows": {31: 32, 32: 1},
            "rows": {(31, 32): values[0].copy(), (32, 1): values[1].copy()}}
        self.cache = {"contract": {}, "snapshots": []}

    def pair(self, cache_equal=False):
        with patch.object(check.base.reference, "compare_cache_collection", return_value={"all_selected_logical_k_v_bits_equal": cache_equal}):
            return check._compare_paths(self.cached, self.full, self.cache, self.cache, self.plan)

    def test_different_prompt_head_batches_remain_explicit_with_same_m1_sampler(self):
        result = self.pair()
        self.assertTrue(result["head_batch_shape_confound_present"])
        self.assertTrue(result["same_single_row_sampler_position32"]["raw_f32_bits_equal"])
        self.assertFalse(result["quality_gate_passed"])
        self.assertFalse(result["independent_cache_correctness_qualified"])

    def test_small_observed_sampler_difference_is_reported_without_fitted_tolerance(self):
        self.cached["rows"][32, 1][50] = np.float32(2**-20)
        report = self.pair()["same_single_row_sampler_position32"]
        self.assertFalse(report["raw_f32_bits_equal"])
        self.assertEqual(report["differing_raw_values"], 1)
        self.assertIsNone(report["numerical_acceptance_tolerance"])

    def test_missing_same_m1_head_cannot_be_replaced_by_different_batch(self):
        del self.full["rows"][32, 1]
        with self.assertRaisesRegex(ValueError, "single-row sampler"):
            self.pair()

    def test_missing_prefill_handoff_position_is_rejected(self):
        self.cached["primary"]["input_positions"] = np.asarray([30, 32], np.int64)
        self.cached["primary"]["input_token_ids"] = self.plan["tokens"][[30, 32]]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.pair()


if __name__ == "__main__":
    unittest.main()
