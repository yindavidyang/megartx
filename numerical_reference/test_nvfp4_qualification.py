"""CPU artifact-gate regressions, using only small temporary JSON/gzip files.

This tests the real stdlib qualification module, not Torch or a copied gate.
After copying into the repo, the source resolves relative to the repo root.
Set MEGARTX_QUALIFICATION_SOURCE for a task-owned review copy.
Successful fixture qualification is limited to artifact guards and coverage;
these synthetic files establish neither CUDA execution nor numerical quality.
"""

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(os.environ.get(
    "MEGARTX_QUALIFICATION_SOURCE",
    str(Path(__file__).resolve().parents[1] / "src/megartx/nvfp4_qualification.py"),
))
spec = importlib.util.spec_from_file_location("_megartx_qualification_subject", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load qualification helper: {SOURCE}")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


PAIRS = ((0, 42), (0, 82), (1, 126), (2, 89), (3, 7), (5, 12))
LABELS = (
    "megartx::corrected_expert_native", "megartx::corrected_expert_reference"
)
DENSE_KERNEL = "test::MainloopSm120TmaWarpSpecializedBlockScaled<fixture>"
GELU_KERNEL = "_gelu_product"


def layer_name(index):
    return f"model.language_model.layers.{index}.moe"


class QualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="megartx-qualification-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.events = [
            {"name": label, "ph": "X", "cat": "user_annotation",
             "ts": index, "dur": 1, "pid": 1, "tid": 1}
            for label in LABELS for index in range(6)
        ]
        self.events += [
            {"name": name, "ph": "X", "cat": "kernel",
             "ts": index, "dur": 1, "pid": 2, "tid": 1}
            for name, count in ((DENSE_KERNEL, 18), (GELU_KERNEL, 6))
            for index in range(count)
        ]
        self.proof = {
            "forced_all_six_executed": True,
            "natural_model_forward_verified": True,
            "registered_layers": [
                {"layer_name": layer_name(index)} for index in range(30)
            ],
            "original_tensor_captures": [
                {"tensor": (
                    f"model.language_model.layers.{layer}.experts.{expert}."
                    f"{projection}_proj.{suffix}"
                ), "bytes": 1, "sha256": "a" * 64,
                 "original_bytes_equal": True}
                for layer, expert in PAIRS
                for projection in ("gate", "up", "down")
                for suffix in ("weight", "weight_scale", "weight_scale_2")
            ],
            "forced_fixtures": [
                {"layer_name": layer_name(layer), "expert": expert,
                 "forced_correction_rows": 2, "observed_bf16_value_equal": True,
                 "max_absolute_difference": 0, "nonzero_output_elements": 1,
                 "original_router_factor": 1.0}
                for layer, expert in PAIRS
            ],
        }
        self.write_trace()
        self.requests = []
        for case_id, tokens in (("case-a", [1, 2, 3]), ("case-b", [4, 5, 6])):
            self.requests.append({
                "id": case_id, "prompt_token_ids": tokens, "teacher_forced": True,
                "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            })
        self.manifests = [
            {"loader_ordinal": layer, "layer": layer_name(layer), "mode": "native",
             "correction_active": True,
             "affected_experts": [{"expert": expert} for l, expert in PAIRS if l == layer]}
            for layer in sorted({layer for layer, _ in PAIRS})
        ]
        self.hits = [
            {"loader_ordinal": layer, "expert": expert, "layer_name": layer_name(layer),
             "mode": "native", "case_id": self.requests[0]["id"],
             "prompt_sha256": self.requests[0]["prompt_sha256"],
             "routed_rows": index + 2, "nonzero_route_weights": index + 1}
            for index, (layer, expert) in enumerate(PAIRS)
        ]
        self.hits.append({
            **self.hits[0], "case_id": self.requests[1]["id"],
            "prompt_sha256": self.requests[1]["prompt_sha256"],
            "routed_rows": 4, "nonzero_route_weights": 3,
        })
        self.write_natural_inputs()

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def write_proof(self):
        self.write_json("activation-proof.json", self.proof)

    def write_trace(self):
        payload = gzip.compress(json.dumps({"traceEvents": self.events}).encode(), mtime=0)
        (self.directory / "activation-forced.json.gz").write_bytes(payload)
        self.proof["forced_trace_sha256"] = hashlib.sha256(payload).hexdigest()
        self.write_proof()

    def write_natural_inputs(self):
        self.write_json("quality-requests.json", self.requests)
        for name, rows in (("adapter-manifest.jsonl", self.manifests),
                           ("route-hits.jsonl", self.hits)):
            (self.directory / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

    def assert_activation_rejected(self, message):
        self.write_proof()
        with self.assertRaisesRegex(RuntimeError, message):
            helper.activation(self.directory)

    def assert_natural_rejected(self, message):
        self.write_natural_inputs()
        with self.assertRaisesRegex(RuntimeError, message):
            helper.natural_coverage(self.directory, "native")

    def test_complete_forced_artifacts_have_only_dispatch_qualification(self):
        report = helper.activation(self.directory)
        self.assertTrue(report["forced_all_six_executed"])
        self.assertTrue(report["guarded_model_dispatch"])
        self.assertEqual(report["forced_adapter_spans"], dict.fromkeys(LABELS, 6))
        self.assertEqual(report["native_dense_sm120_launches"], 18)
        self.assertEqual(report["adapter_gelu_launches"], 6)
        self.assertEqual(report["forced_trace_sha256"], self.proof["forced_trace_sha256"])
        self.assertNotIn("quality_gate_passed", report)
        self.assertNotIn("all_six_naturally_exercised", report)

    def test_invalidation_marker_overrides_an_otherwise_valid_proof(self):
        self.write_json("QUALIFICATION-INVALIDATED.json", {
            "reason": "Old hook never ran in the natural model path"
        })
        self.assert_activation_rejected("explicitly invalidated")

    def test_missing_forced_execution_flag_is_rejected(self):
        self.proof.pop("forced_all_six_executed")
        self.assert_activation_rejected("Missing forced correction")

    def test_missing_model_dispatch_flag_is_rejected(self):
        self.proof.pop("natural_model_forward_verified")
        self.assert_activation_rejected("guarded model dispatch proof")

    def test_missing_each_part_of_frozen_scope_is_rejected(self):
        for key in ("registered_layers", "original_tensor_captures", "forced_fixtures"):
            with self.subTest(scope=key):
                original = self.proof[key]
                self.proof[key] = original[:-1]
                self.assert_activation_rejected("Incomplete frozen integration scope")
                self.proof[key] = original

    def test_duplicate_registered_layer_is_rejected(self):
        self.proof["registered_layers"][-1] = self.proof["registered_layers"][0]
        self.assert_activation_rejected("Duplicate integration identities")

    def test_duplicate_original_tensor_capture_is_rejected(self):
        self.proof["original_tensor_captures"][-1] = self.proof["original_tensor_captures"][0]
        self.assert_activation_rejected("Duplicate integration identities")

    def test_duplicate_forced_expert_cannot_count_as_six_experts(self):
        self.proof["forced_fixtures"][-1] = dict(self.proof["forced_fixtures"][0])
        self.assert_activation_rejected("Duplicate or unregistered forced expert")

    def test_forced_expert_on_an_unregistered_layer_is_rejected(self):
        self.proof["forced_fixtures"][-1]["layer_name"] = layer_name(30)
        self.assert_activation_rejected("Duplicate or unregistered forced expert")

    def test_original_tensor_byte_mismatch_is_rejected(self):
        self.proof["original_tensor_captures"][0]["original_bytes_equal"] = False
        self.assert_activation_rejected("Original tensor capture mismatch")

    def test_forced_correction_missing_a_mode_execution_is_rejected(self):
        self.proof["forced_fixtures"][0]["forced_correction_rows"] = 1
        self.assert_activation_rejected("fixture was bypassed or differs")

    def test_forced_unequal_bf16_output_is_rejected(self):
        self.proof["forced_fixtures"][0]["observed_bf16_value_equal"] = False
        self.assert_activation_rejected("fixture was bypassed or differs")

    def test_nonzero_forced_error_is_not_accepted_as_zero(self):
        self.proof["forced_fixtures"][0]["max_absolute_difference"] = 2 ** -16
        self.assert_activation_rejected("fixture was bypassed or differs")

    def test_all_zero_forced_output_cannot_prove_correction(self):
        self.proof["forced_fixtures"][0]["nonzero_output_elements"] = 0
        self.assert_activation_rejected("fixture was bypassed or differs")

    def test_missing_trace_hash_is_rejected(self):
        self.proof.pop("forced_trace_sha256")
        self.write_proof()
        with self.assertRaises(KeyError):
            helper.activation(self.directory)

    def test_changed_trace_bytes_are_rejected_before_span_counting(self):
        self.events.append({"name": "unrelated", "ph": "X", "cat": "user_annotation"})
        changed = gzip.compress(json.dumps({"traceEvents": self.events}).encode(), mtime=0)
        (self.directory / "activation-forced.json.gz").write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, "trace hash differs"):
            helper.activation(self.directory)

    def test_missing_trace_file_is_rejected(self):
        (self.directory / "activation-forced.json.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            helper.activation(self.directory)

    def test_missing_adapter_span_is_rejected_despite_matching_hash(self):
        self.events.pop(0)
        self.write_trace()
        with self.assertRaisesRegex(RuntimeError, "adapter-specific forced correction"):
            helper.activation(self.directory)

    def test_trace_view_annotations_cannot_fill_missing_actual_spans(self):
        self.events.pop(0)
        self.events += [
            {"name": LABELS[0], "ph": "X", "cat": "Trace view"} for _ in range(20)
        ]
        self.write_trace()
        with self.assertRaisesRegex(RuntimeError, "adapter-specific forced correction"):
            helper.activation(self.directory)

    def test_noncomplete_trace_events_cannot_count_as_correction_spans(self):
        self.events[0]["ph"] = "B"
        self.write_trace()
        with self.assertRaisesRegex(RuntimeError, "adapter-specific forced correction"):
            helper.activation(self.directory)

    def test_missing_dense_gpu_kernels_are_rejected(self):
        self.events = [event for event in self.events if event["name"] != DENSE_KERNEL]
        self.write_trace()
        self.assert_activation_rejected("Missing native SM120 dense correction")

    def test_seventeen_dense_launches_cannot_qualify_all_six_experts(self):
        self.events.remove(next(e for e in self.events if e["name"] == DENSE_KERNEL))
        self.write_trace()
        self.assert_activation_rejected("Missing native SM120 dense correction")

    def test_missing_adapter_gelu_gpu_kernels_are_rejected(self):
        self.events = [event for event in self.events if event["name"] != GELU_KERNEL]
        self.write_trace()
        self.assert_activation_rejected("adapter GELU kernels")

    def test_five_gelu_launches_cannot_qualify_six_corrections(self):
        self.events.remove(next(e for e in self.events if e["name"] == GELU_KERNEL))
        self.write_trace()
        self.assert_activation_rejected("adapter GELU kernels")

    def test_gelu_name_substring_cannot_replace_exact_adapter_kernel(self):
        for event in self.events:
            if event["name"] == GELU_KERNEL:
                event["name"] = "unrelated::_gelu_product"
        self.write_trace()
        self.assert_activation_rejected("adapter GELU kernels")

    def test_cpu_annotations_cannot_fill_missing_dense_gpu_launches(self):
        event = next(e for e in self.events if e["name"] == DENSE_KERNEL)
        event["cat"] = "user_annotation"
        self.write_trace()
        self.assert_activation_rejected("Missing native SM120 dense correction")

    def test_noncomplete_gpu_events_cannot_fill_missing_gelu_launches(self):
        event = next(e for e in self.events if e["name"] == GELU_KERNEL)
        event["ph"] = "B"
        self.write_trace()
        self.assert_activation_rejected("adapter GELU kernels")

    def test_any_case_marlin_gpu_kernel_invalidates_native_dispatch(self):
        for name in ("marlin_fp4", "MARLIN_FP4", "unrelated_MaRlIn_kernel"):
            with self.subTest(kernel=name):
                self.events.append({"name": name, "ph": "X", "cat": "kernel"})
                self.write_trace()
                self.assert_activation_rejected("unexpected Marlin dispatch")
                self.events.pop()

    def test_marlin_text_in_a_cpu_annotation_is_not_gpu_dispatch(self):
        self.events.append({"name": "MARLIN disabled", "ph": "X", "cat": "user_annotation"})
        self.write_trace()
        report = helper.activation(self.directory)
        self.assertEqual(report["native_dense_sm120_launches"], 18)
        self.assertEqual(report["adapter_gelu_launches"], 6)

    def test_valid_six_natural_hits_sum_rows_and_preserve_case_binding(self):
        report = helper.natural_coverage(self.directory, "native")
        self.assertTrue(report["all_six_naturally_exercised"])
        self.assertEqual(report["client_requests"], 2)
        self.assertEqual(report["mode"], "native")
        self.assertEqual([(r["loader_ordinal"], r["expert"]) for r in report["experts"]], list(PAIRS))
        first = report["experts"][0]
        self.assertEqual(first["routed_rows"], 6)
        self.assertEqual(first["nonzero_route_weights"], 4)
        self.assertEqual(first["cases"], ["case-a", "case-b"])
        self.assertEqual(
            json.loads((self.directory / "natural-coverage.json").read_text()), report
        )
        self.assertNotIn("quality_gate_passed", report)

    def test_valid_reference_mode_requires_consistent_manifest_and_audit(self):
        for row in self.manifests + self.hits:
            row["mode"] = "reference"
        self.write_natural_inputs()
        report = helper.natural_coverage(self.directory, "reference")
        self.assertTrue(report["all_six_naturally_exercised"])
        self.assertEqual(report["mode"], "reference")

    def test_natural_coverage_directly_rejects_invalidated_artifacts(self):
        self.write_json("QUALIFICATION-INVALIDATED.json", {"reason": "inactive hook"})
        self.assert_natural_rejected("explicitly invalidated")

    def test_control_and_unknown_modes_cannot_qualify_a_correction(self):
        for mode in ("control", "unknown"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "active correction mode"):
                    helper.natural_coverage(self.directory, mode)

    def test_mixed_manifest_mode_is_rejected(self):
        self.manifests[1]["mode"] = "reference"
        self.assert_natural_rejected("inactive or unmatched correction mode")

    def test_inactive_manifest_is_rejected_despite_nonzero_natural_hits(self):
        self.manifests[0]["correction_active"] = False
        self.assert_natural_rejected("inactive or unmatched correction mode")

    def test_empty_manifest_is_rejected(self):
        self.manifests = []
        self.assert_natural_rejected("inactive or unmatched correction mode")

    def test_forced_success_cannot_substitute_for_zero_natural_hits(self):
        helper.activation(self.directory)
        self.hits = []
        self.assert_natural_rejected("did not exercise corrected experts")
        failed = json.loads((self.directory / "natural-coverage.json").read_text())
        self.assertFalse(failed["all_six_naturally_exercised"])
        self.assertEqual(len(failed["experts"]), 6)
        self.assertTrue(all(r["routed_rows"] == 0 for r in failed["experts"]))

    def test_routed_rows_with_zero_router_weights_are_not_coverage(self):
        for hit in self.hits:
            hit["nonzero_route_weights"] = 0
        self.assert_natural_rejected("did not exercise corrected experts")

    def test_one_unexercised_expert_fails_the_whole_six_expert_claim(self):
        self.hits = [hit for hit in self.hits if hit["expert"] != 12]
        self.assert_natural_rejected(r"did not exercise corrected experts: \[\(5, 12\)\]")

    def test_mixed_execution_mode_is_rejected(self):
        self.hits[0]["mode"] = "reference"
        self.assert_natural_rejected("actual loaded experts/client prefix")

    def test_foreign_prompt_prefix_is_rejected(self):
        self.hits[0]["prompt_sha256"] = self.requests[1]["prompt_sha256"]
        self.assert_natural_rejected("actual loaded experts/client prefix")

    def test_foreign_case_id_is_rejected(self):
        self.hits[0]["case_id"] = "startup-forced-fixture"
        self.assert_natural_rejected("actual loaded experts/client prefix")

    def test_wrong_routed_layer_is_rejected(self):
        self.hits[0]["layer_name"] = layer_name(29)
        self.assert_natural_rejected("actual loaded experts/client prefix")

    def test_unloaded_expert_is_rejected(self):
        self.hits[0]["expert"] = 43
        self.assert_natural_rejected("actual loaded experts/client prefix")

    def test_invalid_nonzero_route_counters_are_rejected(self):
        for nonzero in (-1, 3):
            with self.subTest(nonzero=nonzero):
                self.hits[0]["nonzero_route_weights"] = nonzero
                self.assert_natural_rejected("Invalid natural route counters")

    def test_missing_route_counter_is_rejected(self):
        self.hits[0].pop("nonzero_route_weights")
        self.write_natural_inputs()
        with self.assertRaises(KeyError):
            helper.natural_coverage(self.directory, "native")

    def test_missing_natural_audit_file_is_rejected(self):
        (self.directory / "route-hits.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            helper.natural_coverage(self.directory, "native")

    def test_missing_sixth_manifest_expert_is_rejected(self):
        self.manifests.pop()
        self.assert_natural_rejected("six frozen experts and a client corpus")

    def test_empty_client_corpus_is_rejected(self):
        self.requests = []
        self.assert_natural_rejected("six frozen experts and a client corpus")


if __name__ == "__main__":
    unittest.main()
