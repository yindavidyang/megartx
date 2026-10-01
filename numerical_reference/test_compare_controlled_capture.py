"""Adversarial artifact/provenance guards; synthetic CPU files, never a GPU."""

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_reference"))
import compare_controlled_capture as check
import controlled_reference as ref
import test_controlled_reference as examples


def save_json(path, data):
    path.write_text(json.dumps(data))


def save_npz(directory, name, arrays):
    path = directory / name
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_trace(mode="native"):
    """Kernels execute later than their owning CPU launch spans on purpose."""
    events = []
    for i in range(6):
        base = 200 * i
        for name, start, duration in (("corrected", base, 100), ("controlled", base + 10, 80)):
            events.append({"ph": "X", "cat": "user_annotation", "name": "megartx::" + name + "_expert_" + mode,
                           "pid": 10, "tid": 20, "ts": start, "dur": duration})
        names = ([] if mode == "paired_reference" else ["MainloopSm120TmaWarpSpecializedBlockScaled_dense"] * 3) + ["_gelu_product"]
        for j, name in enumerate(names):
            correlation = i * 10 + j
            events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 20,
                           "ts": base + 20 + j * 10, "dur": 1, "args": {"correlation": correlation}})
            events.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7,
                           "ts": 10000 + base + j * 10, "dur": 2, "args": {"correlation": correlation}})
    return events


class ArtifactReadAndOutputGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.capture = self.root / "capture"
        self.capture.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_numeric_archive_is_a_read_only_copy(self):
        sha = save_npz(self.capture, "rows.npz", {"x": np.asarray([1, 2], dtype=np.uint16)})
        before = (self.capture / "rows.npz").read_bytes()
        loaded = check.read_npz(self.capture, "rows.npz", sha, {"x"})
        loaded["x"][0] = 3
        self.assertEqual((self.capture / "rows.npz").read_bytes(), before)
        self.assertEqual(check.read_npz(self.capture, "rows.npz", sha, {"x"})["x"].tolist(), [1, 2])

    def test_wrong_hash_rejects_before_numpy_loading(self):
        save_npz(self.capture, "rows.npz", {"x": np.zeros(1, np.uint8)})
        with patch.object(np, "load", side_effect=AssertionError("Must not load")), self.assertRaisesRegex(ValueError, "SHA256"):
            check.read_npz(self.capture, "rows.npz", "0" * 64, {"x"})

    def test_forged_huge_shape_rejects_before_allocation(self):
        source = io.BytesIO()
        np.lib.format.write_array_header_1_0(source, {"descr": "|u1", "fortran_order": False, "shape": (10**12,)})
        with zipfile.ZipFile(self.capture / "bad.npz", "w") as archive:
            archive.writestr("x.npy", source.getvalue())
        with patch.object(np, "load", side_effect=AssertionError("Must not allocate")), self.assertRaisesRegex(ValueError, "extent"):
            check.read_npz(self.capture, "bad.npz", None, {"x"})

    def test_pickle_object_arrays_are_rejected_before_numpy_loading(self):
        save_npz(self.capture, "bad.npz", {"x": np.asarray([{"not": "numeric"}], dtype=object)})
        with patch.object(np, "load", side_effect=AssertionError("Must not unpickle")), self.assertRaisesRegex(ValueError, "pickle"):
            check.read_npz(self.capture, "bad.npz", None, {"x"})

    def test_duplicate_npy_entries_cannot_override_a_verified_member(self):
        source = io.BytesIO()
        np.save(source, np.zeros(1, np.uint8))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.capture / "bad.npz", "w") as archive:
                archive.writestr("x.npy", source.getvalue())
                archive.writestr("x.npy", source.getvalue())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check.read_npz(self.capture, "bad.npz", None, {"x"})

    def test_extra_member_and_truncated_payload_cannot_hide_in_npz(self):
        save_npz(self.capture, "extra.npz", {"x": np.zeros(1, np.uint8), "other": np.zeros(1, np.uint8)})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            check.read_npz(self.capture, "extra.npz", None, {"x"})
        source = io.BytesIO()
        np.save(source, np.zeros(4, np.uint16))
        with zipfile.ZipFile(self.capture / "short.npz", "w") as archive:
            archive.writestr("x.npy", source.getvalue()[:-1])
        with self.assertRaisesRegex(ValueError, "extent"):
            check.read_npz(self.capture, "short.npz", None, {"x"})

    def test_traversal_and_symlink_cannot_read_outside_capture(self):
        (self.root / "outside").write_bytes(b"private")
        (self.capture / "linked").symlink_to(self.root / "outside")
        for name in ("../outside", "sub/rows.npz", "linked"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check.local_file(self.capture, name)

    def test_report_inside_capture_or_checkpoint_is_rejected(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        for destination in (self.capture / "controlled-manifest.json", checkpoint / "config.json", self.capture / "new-report.json"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "outside"):
                check.report_output_path(destination, (self.capture, checkpoint))

    def test_report_symlink_parent_into_capture_is_rejected(self):
        (self.root / "linked-dir").symlink_to(self.capture, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            check.report_output_path(self.root / "linked-dir" / "new-report.json", (self.capture,))

    def test_existing_outside_report_is_never_overwritten(self):
        destination = self.root / "report.json"
        destination.write_text("immutable previous report")
        args = ["--plan", str(self.capture), "--native", str(self.capture), "--checkpoint", str(self.capture), "--output", str(destination)]
        with patch.object(check, "compare_runs", return_value={"all_strict_observed_operator_gates_pass": True}), self.assertRaises(FileExistsError):
            check.main(args)
        self.assertEqual(destination.read_text(), "immutable previous report")

    def test_new_outside_native_only_report_is_exclusively_created(self):
        destination = self.root / "report.json"
        args = ["--plan", str(self.capture), "--native", str(self.capture), "--checkpoint", str(self.capture), "--output", str(destination)]
        with patch.object(check, "compare_runs", return_value={"all_strict_observed_operator_gates_pass": True}) as run, patch("builtins.print"):
            self.assertEqual(check.main(args), 0)
        self.assertIsNone(run.call_args.args[2])
        self.assertIsNone(run.call_args.args[3])
        self.assertTrue(json.loads(destination.read_text())["all_strict_observed_operator_gates_pass"])


class DedicatedTraceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def trace(self, events, mode="native"):
        path = self.directory / "controlled-trace.json.gz"
        with gzip.open(path, "wb") as stream:
            stream.write(json.dumps({"traceEvents": events}).encode())
        return check.trace_summary(self.directory, {"mode": mode, "cuda_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_async_native_kernels_bind_by_launch_correlation_not_gpu_time(self):
        report = self.trace(frame_trace())
        self.assertEqual(report["per_span_dense_launches"], [3] * 6)
        self.assertEqual(report["per_span_gelu_launches"], [1] * 6)
        self.assertTrue(report["cpu_launch_to_controlled_span_correlation_verified"])
        self.assertFalse(report["kernel_to_gate_up_down_stage_identity_qualified"])

    def test_paired_reference_has_no_native_dense_kernel_in_controlled_spans(self):
        report = self.trace(frame_trace("paired_reference"), "paired_reference")
        self.assertEqual(report["dense_sm120_launches"], 0)
        self.assertEqual(report["adapter_gelu_launches"], 6)

    def test_global_moe_kernels_cannot_replace_missing_controlled_dense_launch(self):
        events = frame_trace()
        # Kernel remains in the trace, but its CPU launch is outside every span.
        next(e for e in events if e["cat"] == "cuda_runtime")["ts"] = -100
        with self.assertRaisesRegex(ValueError, "per-expert"):
            self.trace(events)

    def test_paired_reference_native_dense_fallback_is_rejected(self):
        events = frame_trace("paired_reference")
        events.append({"ph": "X", "cat": "kernel", "name": "MainloopSm120TmaWarpSpecializedBlockScaled_dense", "args": {"correlation": 0}})
        with self.assertRaisesRegex(ValueError, "per-expert"):
            self.trace(events, "paired_reference")

    def test_missing_gelu_correlation_rejects_even_when_gelu_name_exists(self):
        events = frame_trace()
        next(e for e in events if e["cat"] == "kernel" and e["name"] == "_gelu_product")["args"] = {}
        with self.assertRaisesRegex(ValueError, "per-expert"):
            self.trace(events)

    def test_marlin_anywhere_and_duplicate_controlled_spans_are_rejected(self):
        events = frame_trace()
        with self.assertRaisesRegex(ValueError, "Marlin"):
            self.trace(events + [{"ph": "X", "cat": "kernel", "name": "Marlin_fallback"}])
        span = next(e for e in events if e["name"] == "megartx::controlled_expert_native")
        with self.assertRaisesRegex(ValueError, "six distinct"):
            self.trace(events + [copy.deepcopy(span)])


class CompletionCounterGuards(unittest.TestCase):
    def setUp(self):
        table = examples.schedule()
        self.manifest = examples.dispatch(table)
        for record in self.manifest["executed_interventions"]:
            record.update(controlled_count_before=2, controlled_count_after=3)
        self.manifest["natural_hits"] = {str(i): {} for i in range(30)}
        self.manifest["controlled_hits"] = {str(i): {} for i in range(30)}
        for layer, expert in ref.TARGETS:
            self.manifest["controlled_hits"][str(layer)][str(expert)] = 3

    def test_cumulative_counters_need_actual_single_row_delta(self):
        self.assertEqual(check.validate_counters(self.manifest)["natural_positive_counts"], 0)

    def test_planned_selection_without_completed_counter_increment_is_rejected(self):
        self.manifest["executed_interventions"][0]["controlled_count_after"] = 2
        with self.assertRaisesRegex(ValueError, "increment"):
            check.validate_counters(self.manifest)

    def test_natural_positive_counts_cannot_be_relabelled_controlled(self):
        self.manifest["natural_hits"]["0"]["42"] = 1
        with self.assertRaisesRegex(ValueError, "natural"):
            check.validate_counters(self.manifest)

    def test_extra_or_disagreeing_final_controlled_count_is_rejected(self):
        for layer, expert, count in (("0", "42", 4), ("29", "8", 1)):
            manifest = copy.deepcopy(self.manifest)
            manifest["controlled_hits"][layer][expert] = count
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "six completed"):
                check.validate_counters(manifest)

    def test_missing_layer_and_boolean_counter_are_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["natural_hits"]["29"]
        with self.assertRaisesRegex(ValueError, "thirty"):
            check.validate_counters(manifest)
        self.manifest["controlled_hits"]["0"]["42"] = True
        with self.assertRaisesRegex(ValueError, "count fields"):
            check.validate_counters(self.manifest)


class HeadBatchRoleGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.plan = examples.schedule()
        self.case = {"directory": self.directory, "manifest": {"mode": "native", "path": "full"}}
        self.records = []

    def tearDown(self):
        self.temp.cleanup()

    def add(self, positions, source_rows, offset=0):
        file = f"logits-{len(self.records):02d}.npz"
        logits = np.zeros((len(positions), 262144), dtype=np.float32)
        logits[:, 10] = offset
        sha = save_npz(self.directory, file, {"input_positions": np.asarray(positions, np.int64),
            "input_token_ids": self.plan["tokens"][positions], "logits": logits})
        record = {**ref.CONTROLLED_ORIGIN, "file": file, "sha256": sha, "mode": "native", "path": "full",
            "row_identity_verified": True, "row_correspondence": "verified_storage_view", "source_rows": source_rows,
            "token_sha256": self.plan["token_sha256"], "schedule_sha256": self.plan["schedule_sha256"]}
        self.records.append(record)
        (self.directory / "logits-records.jsonl").write_text("\n".join(json.dumps(item) for item in self.records) + "\n")

    def test_different_head_batchings_are_preserved_without_tolerance_or_collapse(self):
        self.add([32], 1, offset=0.125)
        self.add([31, 32], 33)
        loaded = check.load_logits(self.case, self.plan)
        self.assertEqual(set(loaded["rows"]), {(31, 33), (32, 33), (32, 1)})
        self.assertEqual(loaded["primary"]["logits"][1, 10], 0)
        alternate = loaded["within_position_different_head_batches"][0]
        self.assertEqual(alternate["comparison"]["max_absolute_difference"], 0.125)
        self.assertIsNone(alternate["comparison"]["numerical_acceptance_tolerance"])

    def test_differing_duplicate_of_same_head_shape_is_rejected(self):
        self.add([31, 32], 33)
        self.add([32], 33, offset=0.125)
        with self.assertRaisesRegex(ValueError, "same position and head batching"):
            check.load_logits(self.case, self.plan)

    def test_missing_required_prompt_role_cannot_be_replaced_with_sampler_only(self):
        self.add([31], 1)
        self.add([32], 1)
        with self.assertRaisesRegex(ValueError, "head batch sizes"):
            check.load_logits(self.case, self.plan)

    def test_modes_must_preserve_all_actual_head_roles(self):
        self.add([31, 32], 33)
        self.add([32], 1)
        loaded = check.load_logits(self.case, self.plan)
        missing = copy.deepcopy(loaded)
        del missing["rows"][32, 1]
        with self.assertRaisesRegex(ValueError, "call roles"):
            check.compare_logit_variants(loaded, missing, self.plan)

    def test_matching_variants_compare_every_row_with_quality_gate_false(self):
        self.add([31, 32], 33)
        self.add([32], 1)
        loaded = check.load_logits(self.case, self.plan)
        result = check.compare_logit_variants(loaded, loaded, self.plan)
        self.assertEqual(len(result["all_same_head_shape_variants"]), 3)
        self.assertTrue(all(row["raw_f32_bits_equal"] for row in result["all_same_head_shape_variants"]))
        self.assertFalse(result["quality_gate_passed"])


class CaseScopeGuards(unittest.TestCase):
    def test_case_invalidation_rejects_before_any_manifest_or_payload_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "controlled" / "full"
            directory.mkdir(parents=True)
            (directory / "INVALIDATED.json").write_text("{}")
            with patch.object(check, "read_json", side_effect=AssertionError("Must reject first")), self.assertRaisesRegex(ValueError, "invalidated"):
                check.load_case(root, "full", "native", examples.schedule())

    def test_whole_run_invalidation_rejects_before_any_manifest_or_payload_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "QUALIFICATION-INVALIDATED.json").write_text("{}")
            with patch.object(check, "read_json", side_effect=AssertionError("Must reject first")), self.assertRaisesRegex(ValueError, "invalidated"):
                check.load_case(root, "full", "native", examples.schedule())

    def test_partial_paired_matrix_is_rejected_before_original_weight_read(self):
        with patch.object(check.formats, "CheckpointReader", side_effect=AssertionError("Must reject first")), self.assertRaisesRegex(ValueError, "supplied together"):
            check.compare_runs("plan", "native", "paired", None, "checkpoint", ["full"])


class LogicalCacheFileGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.plan = examples.schedule()
        self.case = {"directory": self.directory, "manifest": {"mode": "native", "forward_calls": 1}}
        layers = [{"layer": i, "attention_type": "full_attention" if i % 6 == 5 else "sliding_attention",
            "kv_heads": 2 if i % 6 == 5 else 8, "head_dim": 512 if i % 6 == 5 else 256,
            "window_size": None if i % 6 == 5 else 1024, "logical_layout": "position,kv_head,head_dim", "dtype": "bf16"} for i in range(30)]
        source_sha = hashlib.sha256(json.dumps(check.KV_SOURCE_HASHES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.contract = {"checkpoint_revision": check.REVISION, "layers": layers,
            "provenance": {"scope": "installed_runtime_confirmed", "config_sha256": ref.CONFIG_SHA256,
                "source_sha256": source_sha, "kv_owner_layout_confirmed": True}}
        self.contract["sha256"] = ref.cache_contract_sha256(self.contract)
        self.nominal = copy.deepcopy(self.contract)
        self.binding = {**ref.CONTROLLED_ORIGIN, "source_sha256": check.KV_SOURCE_HASHES.copy(),
            "forward_calls": 1, "independent_cache_correctness_qualified": False,
            "scheduler_block_table_independently_reconstructed": False, "layers": []}
        self.records = []
        for layer, descriptor in enumerate(layers):
            h, d = descriptor["kv_heads"], descriptor["head_dim"]
            block = 32 if layer % 6 == 5 else 16
            self.binding["layers"].append({"layer": layer, "layer_name": f"language_model.model.layers.{layer}.self_attn.attn",
                "registry_owner_verified": True, "metadata_writer_slots_equal": True, "dtype": "bf16",
                "writer": "FlashInferImpl.do_kv_cache_update", "cache_view_shape": [4, h, block, 2*d],
                "cache_view_strides": [block*h*2*d, 2*d, h*2*d, 1], "resolved_layout": "LBNHC"})
            filename = f"kv-{layer:02d}.npz"
            sha = save_npz(self.directory, filename, {"logical_positions": np.asarray([31, 32], np.int64),
                "key_bits": np.full((2, h, d), 0x3F80, np.uint16),
                "value_bits": np.full((2, h, d), 0x4000, np.uint16)})
            self.records.append({**ref.CONTROLLED_ORIGIN, "layer": layer, "file": filename, "sha256": sha,
                "mode": "native", "token_sha256": self.plan["token_sha256"], "schedule_sha256": self.plan["schedule_sha256"],
                "cache_contract_sha256": self.contract["sha256"], "committed_length": 33,
                "attention_type": descriptor["attention_type"], "kv_heads": h, "head_dim": d,
                "mask_window_start": 0, "logical_mapping_verified": True, "writer_slots": [31, 32],
                "same_cache_owner_across_forwards": True,
                "logical_mapping_scope": "actual per-layer writer slot maps within one fresh short request"})
        self.save_metadata()

    def tearDown(self):
        self.temp.cleanup()

    def save_metadata(self):
        save_json(self.directory / "cache-contract.json", self.contract)
        save_json(self.directory / "kv-binding.json", self.binding)
        save_json(self.directory / "kv-records.json", self.records)

    def load(self):
        return check.load_cache(self.case, self.plan, self.nominal)

    def test_logical_bhnc_rows_keep_physical_lbnhc_and_separate_k_v(self):
        loaded = self.load()
        self.assertEqual(len(loaded["snapshots"]), 30)
        self.assertEqual(loaded["summary"]["recorded_physical_layouts"], ["LBNHC"])
        self.assertFalse(np.array_equal(loaded["snapshots"][0]["key_bits"], loaded["snapshots"][0]["value_bits"]))
        self.assertFalse(loaded["summary"]["independent_cache_correctness_qualified"])

    def test_changed_writer_source_is_rejected_without_inventing_runtime_proof(self):
        self.binding["source_sha256"]["vllm.forward_context"] = "0" * 64
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "changed source"):
            self.load()

    def test_changed_record_prefix_or_mode_is_rejected(self):
        self.records[0]["mode"] = "paired_reference"
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "mode/prefix"):
            self.load()

    def test_reused_or_out_of_allocation_writer_slots_are_rejected(self):
        for slots in ([31, 31], [31, 64]):
            self.records[0]["writer_slots"] = slots
            self.save_metadata()
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "writer slots"):
                self.load()

    def test_duplicate_logical_layer_cannot_replace_all_thirty_owners(self):
        self.records[29]["layer"] = 28
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "thirty distinct"):
            self.load()

    def test_rehashing_changed_nominal_head_geometry_does_not_make_it_valid(self):
        self.contract["layers"][0]["kv_heads"] = 1
        self.contract["sha256"] = ref.cache_contract_sha256(self.contract)
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "frozen config"):
            self.load()

    def test_payload_mutation_is_rejected_by_its_captured_hash(self):
        path = self.directory / self.records[0]["file"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_unverified_registry_or_missing_logical_handoff_row_is_rejected(self):
        self.binding["layers"][0]["registry_owner_verified"] = False
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "owner/writer"):
            self.load()
        self.binding["layers"][0]["registry_owner_verified"] = True
        item = self.records[0]
        item["sha256"] = save_npz(self.directory, item["file"], {"logical_positions": np.asarray([30, 31], np.int64),
            "key_bits": np.zeros((2, 8, 256), np.uint16), "value_bits": np.zeros((2, 8, 256), np.uint16)})
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.load()


if __name__ == "__main__":
    unittest.main()
