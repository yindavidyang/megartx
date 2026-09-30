"""Synthetic CPU tests. None execute or validate a model, quantizer or GPU kernel."""

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from megartx.adapter import BackendUnavailable, FlashInferAdapter, check_dispatch_packet
from megartx.cli import main
from megartx.contracts import ContractError, load_json, readiness, validate
from megartx.inventory import collect_inventory
from megartx.memory import estimate_memory, kv_bytes, storage_bytes, tensor_bytes
from megartx.schema import SCHEMAS
from megartx.stats import paired_median_reduction, quantile, summarize

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return load_json(ROOT / "configs" / name)


def tensor(name="weight", storage_id="weight", shape=None, dtype="packed_fp4"):
    shape = [3, 3] if shape is None else shape
    return {"name": name, "logical_shape": shape, "storage_shape": shape,
            "storage_dtype": dtype, "storage_id": storage_id,
            "storage_bytes": storage_bytes(shape, dtype), "quantization_role": "payload",
            "ownership": "text"}


def layer(policy="full_context", window=None):
    return {"layer_id": "synthetic_layer", "kv_heads": 2, "key_head_dim": 4,
            "value_head_dim": 8, "key_dtype": "bfloat16", "value_dtype": "bfloat16",
            "storage_policy": policy, "window_tokens": window}


def synthetic_results(count=30):
    record = config("result_template.json")
    record["provenance"] = "synthetic"
    record["run_id"] = "cpu-unit-test-only"
    record["pairs"] = [{"pair_id": str(i), "order": "baseline_first" if i % 2 else "candidate_first",
                        "prompt_sha256": "a" * 64, "seed": i,
                        "baseline_itl_ms": [10, 10], "candidate_itl_ms": [8, 8],
                        "baseline_ttft_ms": 20, "candidate_ttft_ms": 20,
                        "baseline_total_response_ms": 40, "candidate_total_response_ms": 36}
                       for i in range(count)]
    return record


class ContractTests(unittest.TestCase):
    def test_all_proposed_configs_validate(self):
        for path in sorted((ROOT / "configs").glob("*.json")):
            with self.subTest(path=path.name):
                validate(load_json(path))

    def test_exported_schemas_match_canonical(self):
        for kind, schema in SCHEMAS.items():
            self.assertEqual(load_json(ROOT / "schemas" / f"{kind}.schema.json"), schema)

    def test_proposed_contract_is_not_ready(self):
        report = readiness(config("experiment.json"))
        self.assertFalse(report["freeze_ready"])
        self.assertGreater(len(report["blockers"]), 4)
        self.assertEqual(report["G0"], "pending_gpu_evidence")

    def test_frozen_requires_all_pins(self):
        doc = config("experiment.json")
        doc["status"] = "frozen"
        with self.assertRaises(ContractError):
            validate(doc)
        doc["checkpoint"].update(revision="a" * 40, tokenizer_sha256="b" * 64, chat_template_sha256="c" * 64)
        doc["acceptance"]["status"] = "frozen"
        doc["owner_approval_reference"] = "synthetic-test-approval"
        with self.assertRaises(ContractError):
            validate(doc)
        doc["workload"].update(prompt_set_sha256="d" * 64, sampling_policy="greedy", eos_policy="fixed_length_ignore_eos")
        self.assertTrue(readiness(doc)["freeze_ready"])
        self.assertEqual(readiness(doc)["G0"], "pending_gpu_evidence")

    def test_mutable_checkpoint_revision_rejected(self):
        doc = config("experiment.json")
        doc["checkpoint"]["revision"] = "main"
        with self.assertRaises(ContractError):
            validate(doc)

    def test_unknown_keys_bool_as_integer_and_nonfinite_rejected(self):
        for key, value in (("context_probes_tokens", [True]), ("output_capacity_tokens", -1), ("concurrency", 2)):
            doc = config("experiment.json")
            doc["workload"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate(doc)
        doc = config("experiment.json")
        doc["private_hostname"] = "not-allowed"
        with self.assertRaises(ContractError):
            validate(doc)
        doc = config("experiment.json")
        doc["acceptance"]["median_itl_reduction_fraction"] = math.nan
        with self.assertRaises(ContractError):
            validate(doc)

    def test_supported_path_needs_evidence(self):
        doc = config("compatibility.json")
        doc["paths"][0]["status"] = "supported"
        with self.assertRaises(ContractError):
            validate(doc)

    def test_pinned_environment_requires_metadata(self):
        doc = config("environment.lock.json")
        doc["status"] = "pinned"
        with self.assertRaises(ContractError):
            validate(doc)

    def test_captured_fixture_is_not_a_status_flip(self):
        doc = config("fixture_contract.json")
        doc["cases"][0]["status"] = "captured"
        with self.assertRaises(ContractError):
            validate(doc)

    def test_nonobject_duplicate_keys_and_nonfinite_json_rejected(self):
        for value in ([], None, "text", 7):
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate(value)
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.json"
                path.write_text(text)
                with self.assertRaises(ContractError):
                    load_json(path)


class MemoryTests(unittest.TestCase):
    def test_storage_bytes_padding_scalar_and_dtypes(self):
        self.assertEqual(storage_bytes([3, 3], "packed_fp4"), 5)
        self.assertEqual(storage_bytes([3, 4], "bfloat16"), 24)
        self.assertEqual(storage_bytes([], "float32"), 4)
        self.assertEqual(storage_bytes([16], "float8_e4m3fn"), 16)
        self.assertEqual(storage_bytes([7], "float8_e5m2"), 7)
        for shape, dtype in (([0], "uint8"), ([-1], "uint8"), ([True], "uint8"), ([2], "nf4")):
            with self.assertRaises(ContractError):
                storage_bytes(shape, dtype)

    def test_aliases_deduplicate_and_scalar_scale_validates(self):
        manifest = config("tensor_manifest.json")
        manifest["tensors"] = [tensor(), tensor("alias", "weight"), tensor("scale", "scale", [], "float32")]
        result = tensor_bytes(manifest)
        self.assertEqual(result["unique_storage_bytes"], 9)
        self.assertEqual(result["alias_bytes_excluded"], 5)

    def test_conflicting_alias_and_stored_bytes_rejected(self):
        manifest = config("tensor_manifest.json")
        manifest["tensors"] = [tensor(), tensor("alias", "weight", [4, 4])]
        with self.assertRaises(ContractError):
            tensor_bytes(manifest)
        with self.assertRaises(ContractError):
            validate(manifest)
        manifest["tensors"] = [tensor()]
        manifest["tensors"][0]["storage_bytes"] += 1
        with self.assertRaises(ContractError):
            tensor_bytes(manifest)
        with self.assertRaises(ContractError):
            validate(manifest)

    def test_validate_rejects_contradictory_kv_contracts(self):
        for layers in ([layer(), layer()], [layer("bounded_window")]):
            budget = config("memory_budget.json")
            budget["kv_layers"] = layers
            with self.assertRaises(ContractError):
                validate(budget)

    def test_distinct_kv_and_output_reserve(self):
        self.assertEqual(kv_bytes([layer()], 10, 2), 12 * 2 * (4 + 8) * 2)
        self.assertEqual(kv_bytes([layer()], 0, 0), 0)

    def test_local_mask_is_not_bounded_allocation(self):
        self.assertEqual(kv_bytes([layer("full_context", 4)], 10, 2), 576)
        self.assertEqual(kv_bytes([layer("bounded_window", 4)], 10, 2), 192)
        with self.assertRaises(ContractError):
            kv_bytes([layer("bounded_window")], 10, 2)
        with self.assertRaises(ContractError):
            kv_bytes([layer(), layer()], 10, 2)

    def test_unknown_coverage_cannot_establish_fit(self):
        report = estimate_memory(config("tensor_manifest.json"), config("memory_budget.json"))
        self.assertIsNone(report["estimated_within_capacity"])
        self.assertEqual(report["measured_fit"], "pending")
        self.assertGreater(len(report["unknown_components"]), 5)

    def test_complete_synthetic_budget_includes_reserve(self):
        manifest = config("tensor_manifest.json")
        manifest.update(complete=True, checkpoint_revision="a" * 40, tensors=[tensor()])
        budget = config("memory_budget.json")
        budget.update(complete=True, device_total_bytes=682, reserve_bytes=100,
                      cached_context_tokens=10, output_capacity_tokens=2, kv_layers=[layer()])
        budget["other_bytes"] = {key: 0 for key in budget["other_bytes"]}
        self.assertTrue(estimate_memory(manifest, budget)["estimated_within_capacity"])
        budget["device_total_bytes"] = 680
        self.assertFalse(estimate_memory(manifest, budget)["estimated_within_capacity"])
        self.assertEqual(estimate_memory(manifest, budget)["measured_fit"], "pending")


class StatisticsTests(unittest.TestCase):
    def test_microbenchmark_and_gpu_boundaries_are_not_end_to_end(self):
        for boundary in ("gpu_resident", "microbenchmark"):
            record = synthetic_results()
            record["boundary"] = boundary
            with self.assertRaises(ContractError):
                summarize(record, samples=100)

    def test_summary_keeps_omissions_and_pending_prerequisites(self):
        record = synthetic_results()
        record["omitted_critical_path_costs"] = ["host streaming"]
        report = summarize(record, samples=100)
        self.assertEqual(report["boundary"], "delivered_token_end_to_end")
        self.assertEqual(report["omitted_critical_path_costs"], ["host streaming"])
        self.assertEqual(report["prerequisite_statuses"]["correctness"], "pending")
        self.assertTrue(report["both_orders_present"])

    def test_quantile_definition(self):
        self.assertEqual(quantile([0, 10], 0.95), 9.5)
        with self.assertRaises(ContractError):
            quantile([], 0.5)

    def test_paired_bootstrap_deterministic_and_correlated(self):
        baseline = [10, 100, 1000, 10000]
        candidate = [v * 0.8 for v in baseline]
        a = paired_median_reduction(baseline, candidate, samples=100, seed=7)
        self.assertEqual(a, paired_median_reduction(baseline, candidate, samples=100, seed=7))
        self.assertAlmostEqual(a["reduction_fraction"], 0.2)
        for bound in a["confidence_interval"]:
            self.assertAlmostEqual(bound, 0.2)
        self.assertEqual(a["bootstrap_unit"], "matched_request_pair")

    def test_invalid_latency_inputs_rejected(self):
        for b, c in (([1], [1]), ([1, 2], [1]), ([1, 0], [1, 1]), ([1, math.inf], [1, 1])):
            with self.assertRaises(ContractError):
                paired_median_reduction(b, c)

    def test_summary_never_passes_a_gate(self):
        summary = summarize(synthetic_results(), samples=100)
        self.assertTrue(summary["minimum_pair_count_met"])
        self.assertAlmostEqual(summary["reduction_fraction"], 0.2)
        self.assertEqual(summary["gate_decision"], "not_evaluated")
        self.assertEqual(summary["provenance"], "synthetic")
        self.assertEqual(summary["baseline"]["token_interval_count"], 60)
        self.assertEqual(summary["request_pair_count"], 30)

    def test_small_sample_not_enough_and_template_rejected(self):
        self.assertFalse(summarize(synthetic_results(2), samples=100)["minimum_pair_count_met"])
        with self.assertRaises(ContractError):
            summarize(config("result_template.json"))

    def test_duplicate_pairs_lengths_timing_and_unpinned_measurements_rejected(self):
        for mutation in ("duplicate", "length", "boundary", "measured"):
            record = synthetic_results(2)
            if mutation == "duplicate":
                record["pairs"][1]["pair_id"] = record["pairs"][0]["pair_id"]
            elif mutation == "length":
                record["pairs"][0]["candidate_itl_ms"] = [8]
            elif mutation == "boundary":
                record["pairs"][0]["candidate_total_response_ms"] = 99
            else:
                record["provenance"] = "measured"
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                summarize(record, samples=100)


class InventoryAndCliTests(unittest.TestCase):
    def test_default_inventory_never_runs_subprocess(self):
        with patch("megartx.inventory.subprocess.run") as run:
            report = collect_inventory()
            run.assert_not_called()
        self.assertEqual(report["gpu_probe"], "not_requested")
        self.assertNotIn("hostname", report)
        self.assertNotIn("uuid", report)

    def test_missing_cuda_and_failed_probe_are_clean(self):
        with patch("megartx.inventory.shutil.which", return_value=None):
            self.assertEqual(collect_inventory(True)["gpu_probe"], "unavailable")
        with patch("megartx.inventory.shutil.which", return_value="/synthetic/tool"), \
             patch("megartx.inventory.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "PRIVATE_SECRET")):
            self.assertNotIn("PRIVATE_SECRET", json.dumps(collect_inventory(True)))

    def test_allowlisted_gpu_metadata_only(self):
        output = "NVIDIA GeForce RTX 5090, 999.1, 32768, 30000, 12.0\n"
        with patch("megartx.inventory.shutil.which", return_value="/synthetic/tool"), \
             patch("megartx.inventory.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")) as run:
            report = collect_inventory(True)
        self.assertEqual(report["gpus"][0]["total_bytes"], 32768 * 1024**2)
        self.assertNotIn("uuid", run.call_args.args[0][1])
        self.assertNotIn("serial", run.call_args.args[0][1])

    def test_cli_errors_nonzero_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for raw in (b"[]", b"{broken", b"\xff", b'{"kind": []}'):
                path.write_bytes(raw)
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    self.assertEqual(main(["validate", str(path)]), 2)
                self.assertNotIn("Traceback", error.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["validate", str(path) + ".missing"]), 2)


class AdapterTests(unittest.TestCase):
    def test_unverified_adapter_fails_closed(self):
        with self.assertRaises(BackendUnavailable):
            FlashInferAdapter().run()
        with self.assertRaises(BackendUnavailable):
            FlashInferAdapter().run(evidence={"flashinfer_installed": True})
        with self.assertRaises(ContractError):
            check_dispatch_packet(config("dispatch.json"), ROOT, "a" * 40, "b" * 64,
                                  "nvfp4_w4a4", {"attention": "flashinfer", "moe": "flashinfer"})

    def test_local_packet_hash_is_not_an_implemented_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "synthetic_trace.txt"
            trace.write_bytes(b"Synthetic test of hash plumbing, NOT GPU dispatch evidence")
            packet = config("dispatch.json")
            packet.update(classification="verified_flashinfer", checkpoint_revision="a" * 40,
                          environment_sha256="b" * 64, host_runtime="synthetic", host_revision="c" * 40,
                          flashinfer_revision="d" * 40, artifact_relative_path=trace.name,
                          artifact_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                          reviewer_reference="synthetic-test-only", non_flashinfer_components=["moe", "logits", "sampling"])
            packet["observations"] = [
                {"component": "attention", "provider": "flashinfer", "kernel_names": ["synthetic_attention"], "native_nvfp4": False},
                {"component": "moe", "provider": "cutlass", "kernel_names": ["synthetic_moe"], "native_nvfp4": True}]
            providers = {"attention": "flashinfer", "moe": "cutlass"}
            receipt = check_dispatch_packet(packet, directory, "a" * 40, "b" * 64, "nvfp4_w4a4", providers)
            with self.assertRaises(BackendUnavailable):
                FlashInferAdapter().run(evidence=receipt)
            packet["artifact_sha256"] = "0" * 64
            with self.assertRaises(ContractError):
                check_dispatch_packet(packet, directory, "a" * 40, "b" * 64, "nvfp4_w4a4", providers)
            packet["artifact_relative_path"] = "../escape"
            with self.assertRaises(ContractError):
                check_dispatch_packet(packet, directory, "a" * 40, "b" * 64, "nvfp4_w4a4", providers)


if __name__ == "__main__":
    unittest.main()
