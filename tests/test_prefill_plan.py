import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from megartx import prefill_plan as prefill


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/prefill/profile-plan.json"
MANIFEST = ROOT / "docs/prefill/source-binding.json"


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = prefill.read_json(PLAN)

    def test_checked_in_plan_is_bound_and_reports_real_blockers(self):
        self.assertEqual(prefill.verify_source_binding(self.plan, MANIFEST, ROOT), 12)
        report = prefill.intake(self.plan)
        self.assertFalse(report["gpu_execution_available"])
        self.assertFalse(report["freeze_references_complete"])
        self.assertFalse(report["gpu_qualified"])
        self.assertIn("g1_acceptance_reference", report["blockers"])
        self.assertIn("sole_owner_handoff_reference", report["blockers"])
        self.assertEqual({c["prompt_tokens"] for c in report["planned_cells"]}, {2048, 8192})
        self.assertEqual(next(c for c in report["planned_cells"] if c["id"] == "p2048" and c["chunk_tokens"] == 255)["final_chunk_m"], 8)
        for workload in self.plan["workloads"]:
            self.assertIn(workload["id"] + ": resident peak fit including output reserve pending",
                          report["blockers"])
        for cell in report["planned_cells"]:
            with self.subTest(prompt=cell["prompt_tokens"], chunk=cell["chunk_tokens"]):
                self.assertEqual(cell["capacity_tokens"], cell["prompt_tokens"] + 256)

    def test_unsupported_scope_or_lane_fails_closed(self):
        changes = [("gpu_enabled", True), ("scope", "gpu_profile"), ("extra", 1)]
        for key, value in changes:
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prefill.validate_plan(plan)
        for key, value in [("kv_dtype", "fp8"), ("head_policy", "all_rows"), ("concurrency", True),
                           ("use_fused_finalize", 0), ("speculation", True), ("tp", 2),
                           ("enforce_eager", False), ("allow_tf32", True),
                           ("allow_bf16_reduced_precision_reduction", False),
                           ("lane", "checkpoint_per_expert_w4a4")]:
            plan = copy.deepcopy(self.plan)
            plan["controls"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prefill.validate_plan(plan)

    def test_complete_reference_strings_still_never_enable_a_runner(self):
        for key in self.plan["freeze"]:
            if key.endswith(("_sha256", "_commit")):
                self.plan["freeze"][key] = "a" * (64 if key.endswith("_sha256") else 40)
            else:
                self.plan["freeze"][key] = "review-reference"
        report = prefill.intake(self.plan)
        self.assertTrue(report["freeze_references_complete"])
        self.assertFalse(report["gpu_execution_available"])
        self.assertFalse(report["gpu_qualified"])
        self.assertIn("No GPU runner", report["runner_blocker"])

    def test_selected_checkpoint_metadata_binds_geometry_and_quantizer_distinction(self):
        manifest = prefill.read_json(MANIFEST)
        text = manifest["checkpoint_metadata"]["text"]
        self.assertEqual((text["hidden_size"], text["moe_intermediate_size"], text["intermediate_size"]), (2816, 704, 2112))
        self.assertEqual((text["num_experts"], text["top_k_experts"]), (128, 8))
        self.assertEqual((text["num_hidden_layers"], text["sliding_window"]), (30, 1024))
        self.assertEqual([i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"], [5, 11, 17, 23, 29])
        self.assertEqual(manifest["checkpoint_metadata"]["quantization"]["kv_cache_quant_algo"], "FP8")
        self.assertEqual(self.plan["controls"]["kv_dtype"], "bfloat16")

    def test_source_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.py").write_bytes(b"source")
            (root / "link.py").symlink_to(root / "target.py")
            for source_name in ("../target.py", "/tmp/target.py", "link.py"):
                manifest = prefill.read_json(MANIFEST)
                manifest["reviewed_source_overlays"] = []
                manifest["repo_files"] = {source_name: hashlib.sha256(b"source").hexdigest()}
                file = root / "manifest.json"
                file.write_text(json.dumps(manifest))
                self.plan["binding"]["source_manifest_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                with self.subTest(source_name=source_name), self.assertRaises(ValueError):
                    prefill.verify_source_binding(self.plan, file, root)
    def test_bad_sweep_capacity_or_32k_admission_rejected(self):
        for change in (lambda p: p["workloads"][0].update(chunk_tokens=[256, 256, 2048]),
                       lambda p: p["workloads"][0].update(chunk_tokens=[8192]),
                       lambda p: p["workloads"][0].update(chunk_tokens=[True, 2048]),
                       lambda p: p["workloads"][2].update(status="planned"),
                       lambda p: p["resource_bounds"].update(max_gpu_jobs=2),
                       lambda p: p["resource_bounds"].update(max_trace_bytes=0),
                       lambda p: p["freeze"].update(decode_control_commit="main")):
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.assertRaises(ValueError):
                prefill.validate_plan(plan)

    def test_source_drift_and_manifest_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prefill.read_json(MANIFEST)
            manifest["reviewed_source_overlays"] = []
            manifest["repo_files"] = {"source.py": hashlib.sha256(b"original").hexdigest()}
            file = root / "manifest.json"
            file.write_text(json.dumps(manifest))
            self.plan["binding"]["source_manifest_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            (root / "source.py").write_bytes(b"original")
            self.assertEqual(prefill.verify_source_binding(self.plan, file, root), 1)
            (root / "source.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source drift"):
                prefill.verify_source_binding(self.plan, file, root)
            file.write_text(file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "manifest digest"):
                prefill.verify_source_binding(self.plan, file, root)

    def test_duplicate_nonfinite_and_symlink_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"schema":1,"schema":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    prefill.read_json(path)
            link = path.with_name("link.json")
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                prefill.read_json(link)

    def test_cli_never_imports_device_packages_or_registers_runtime(self):
        code = """
import runpy, sys
sys.argv = ['prefill_plan', 'docs/prefill/profile-plan.json', '--source-manifest', 'docs/prefill/source-binding.json', '--root', '.']
try:
    runpy.run_module('megartx.prefill_plan', run_name='__main__')
except SystemExit as result:
    assert result.code == 0
assert not any(x.split('.')[0] in {'torch', 'triton', 'vllm', 'flashinfer', 'numpy'} for x in sys.modules)
assert 'megartx.vllm_scale_plugin' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class AccountingTests(unittest.TestCase):
    def test_whole_prompt_coverage_and_partial_tails(self):
        for length in (257, 1023, 1024, 1025, 2048, 8192):
            for chunk in (1, 255, 256, length):
                spans = prefill.chunk_spans(length, chunk, absolute_start=33)
                actual = [p for start, end in spans for p in range(start, end)]
                self.assertEqual(actual, list(range(33, 33 + length)))
                self.assertLessEqual(spans[-1][1] - spans[-1][0], chunk)

    def test_variable_expert_m_and_padding_are_not_full_prompt_per_expert(self):
        ids = [list(range(8)), list(range(4, 12)), list(range(120, 128))]
        weights = [[1.0] * 8, [1.0] * 8, [0.0] + [1.0] * 7]
        ledger = prefill.expert_row_ledger(ids, weights, padding_multiple=4)
        self.assertEqual(ledger["selected_slots"], 24)
        self.assertEqual(ledger["selected_m"][4:8], [2] * 4)
        self.assertEqual(ledger["selected_m"][120], 1)
        self.assertEqual(ledger["positive_weight_m"][120], 0)
        self.assertEqual(ledger["empty_experts"], 108)
        self.assertEqual(sum(ledger["projected_padded_m"]), 80)
        self.assertFalse(ledger["actual_executed_work_known"])
        for bad_ids, bad_weights in [(ids[:1], [[float("nan")] * 8]),
                                     ([[0] * 8], [[1] * 8]), ([[128] + list(range(7))], [[1] * 8]),
                                     (ids, weights[:1])]:
            with self.assertRaises(ValueError):
                prefill.expert_row_ledger(bad_ids, bad_weights)

    def test_one_user_many_rows_can_have_extreme_skew(self):
        ledger = prefill.expert_row_ledger([list(range(8)) for _ in range(256)], [[1] * 8 for _ in range(256)], 128)
        self.assertEqual(ledger["selected_m"][:8], [256] * 8)
        self.assertEqual(ledger["selected_m"][8:], [0] * 120)
        self.assertEqual(ledger["projected_padding_rows"], 0)
        self.assertEqual(ledger["input_m"], 256)

    def timing(self):
        return {"request_accept_ns": 0, "input_ready_ns": 100, "prompt_begin_ns": 200,
                "prompt_complete_ns": 1200, "kv_ready_ns": 1500, "first_token_ns": 2000,
                "prompt_tokens": 2048, "chunk_rows": [1024, 1024], "state": "warm",
                "observer_enabled": False, "preallocated": True, "bucket_warmed": True}

    def test_whole_prompt_latency_handoff_and_ttft_have_different_boundaries(self):
        result = prefill.summarize_timing(self.timing())
        self.assertEqual(result["prompt_latency_ns"], 1000)
        self.assertEqual(result["handoff_ns"], 300)
        self.assertEqual(result["ttft_ns"], 2000)
        self.assertEqual(result["prompt_tokens_per_second"], 2048e6)
        self.assertFalse(result["gpu_time_measured"])

    def test_observer_incomplete_prompt_and_unwarmed_timing_rejected(self):
        for key, value in [("observer_enabled", True), ("chunk_rows", [1024]),
                           ("chunk_rows", [2048, 2048]), ("kv_ready_ns", 1100),
                           ("bucket_warmed", False), ("preallocated", False),
                           ("prompt_complete_ns", 200), ("first_token_ns", True)]:
            record = self.timing()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prefill.summarize_timing(record)
        record = self.timing()
        record.update(state="cold", bucket_warmed=False, preallocated=False)
        self.assertEqual(prefill.summarize_timing(record)["state"], "cold")


if __name__ == "__main__":
    unittest.main()
