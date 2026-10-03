"""CPU source/host-contract checks. These are not native execution evidence."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from megartx.speculative_native_probe import (
    BASE, GPU_CAP, HEAD_PEAK, P, PagePlan, ProbeError, OwnedNativeProbe,
    allocation_lower_bound, check_admission, consumed_table, host_frame,
    inspect_sources, plan_pages, run_engine_core_probe, select_greedy, source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class PageContracts(unittest.TestCase):
    def test_actual_reserved_pages_at_p2048(self):
        for b in (16, 32, 64):
            with self.subTest(b=b):
                table = tuple(range(1, P // b + 1))
                reserved = table + (300, 301)
                p = plan_pages(table, reserved, cached=P, rows=2, block_size=b)
                self.assertEqual(p.table, table + (300,))
                self.assertEqual(p.slots, (300 * b, 300 * b + 1))
                self.assertEqual(p.copies, ())
                self.assertEqual(table, tuple(range(1, P // b + 1)))

    def test_partial_consumed_page_is_cow(self):
        table = tuple(range(1, 130))
        p = plan_pages(table, table + (300,), cached=P + 1, rows=1, block_size=16)
        self.assertEqual(p.copies, ((129, 300, 1),))
        self.assertEqual(p.table[-1], 300)
        self.assertNotIn(129 * 16 + 1, p.slots)

    def test_two_touched_pages_need_two_real_reservations(self):
        table = tuple(range(1, 65))
        with self.assertRaises(ProbeError):
            plan_pages(table, table + (300,), cached=1023, rows=2, block_size=16)
        p = plan_pages(table, table + (300, 301), cached=1023, rows=2, block_size=16)
        self.assertEqual(p.slots, (300 * 16 + 15, 301 * 16))
        self.assertEqual(consumed_table(p, cached=1023, consumed=1, block_size=16)[-1], 300)
        self.assertNotIn(301, consumed_table(p, cached=1023, consumed=1, block_size=16))

    def test_no_guessing_free_ids_or_aliases(self):
        for table, reserved in (((1,), (2, 3)), ((1,), (1, 1, 3)), ((1,), (0, 1, 3)),
                                ((1, 1), (1, 3)), ((1,), (1, True, 3))):
            with self.subTest(table=table, reserved=reserved), self.assertRaises(ProbeError):
                plan_pages(table, reserved, cached=16, rows=1, block_size=16)

    def test_unreviewed_shapes_rejected(self):
        for rows, b in ((3, 16), (1, 128), (0, 16), (True, 16)):
            with self.subTest(rows=rows, b=b), self.assertRaises(ProbeError):
                plan_pages((), (1, 2), cached=0, rows=rows, block_size=b)

    def test_consumed_count_never_includes_pending_output(self):
        p = PagePlan((10,), (160, 161), (), (10,))
        self.assertEqual(consumed_table(p, cached=0, consumed=1, block_size=16), (10,))
        with self.assertRaises(ProbeError):
            consumed_table(p, cached=0, consumed=3, block_size=16)

    def test_exact_host_metadata_and_prediction_positions(self):
        p = PagePlan((1, 20), (320, 321), (), (20,))
        f = host_frame(p, cached=16, rows=2)
        self.assertEqual(f, {"query_start_loc": (0, 2), "seq_lens": (18,), "num_reqs": 1,
            "num_actual_tokens": 2, "max_query_len": 2, "max_seq_len": 18,
            "block_table": (1, 20), "slot_mapping": (320, 321), "positions": (16, 17), "causal": True})
        self.assertEqual(tuple(x + 1 for x in f["positions"]), (17, 18))

    def test_misaligned_host_rows_fail(self):
        for slots in ((160,), (160, 160)):
            with self.assertRaises(ProbeError):
                host_frame(PagePlan((10,), slots, (), (10,)), cached=0, rows=2)


class SelectionAndAllocation(unittest.TestCase):
    def test_k0_anchor_is_already_emitted(self):
        self.assertEqual(select_greedy(99, (), (8,), 4), ((8,), 0, 1, False))

    def test_forced_rejection_consumes_only_anchor(self):
        self.assertEqual(select_greedy(99, (9,), (8, 7), 4), ((8,), 0, 1, False))

    def test_full_accept_consumes_anchor_and_candidate(self):
        self.assertEqual(select_greedy(99, (8,), (8, 7), 4), ((8, 7), 1, 2, False))

    def test_no_duplicate_bonus_at_budget(self):
        self.assertEqual(select_greedy(99, (8,), (8, 7), 1), ((8,), 1, 1, True))

    def test_eos_stays_pending(self):
        self.assertEqual(select_greedy(99, (1,), (1, 7), 4), ((1,), 1, 1, True))
        self.assertEqual(select_greedy(99, (8,), (8, 106), 4), ((8, 106), 1, 2, True))

    def test_rejected_eos_does_not_stop(self):
        self.assertEqual(select_greedy(99, (1,), (8, 7), 4), ((8,), 0, 1, False))

    def test_invalid_native_selection_fails(self):
        for candidates, rows, budget in (((2, 3), (2, 3, 4), 4), ((2,), (2,), 4),
                                         ((), (2,), 0), ((), (True,), 4)):
            with self.assertRaises(ProbeError):
                select_greedy(99, candidates, rows, budget)

    def test_softcap_coexistence_is_two_fp32_rows(self):
        self.assertEqual(HEAD_PEAK, 2097152)
        a = allocation_lower_bound((25 * 16 * 8192, 5 * 16 * 4096), 1024)
        self.assertEqual(a["private_page_bytes"], 3604480)
        self.assertEqual(a["known_bytes"], 3604480 + 2097152 + 11264 + 1024)
        self.assertLess(a["known_bytes"], GPU_CAP)
        self.assertNotIn("fit_admitted", a)

    def test_b32_lower_bound_already_infeasible_without_padding(self):
        with self.assertRaisesRegex(ProbeError, "exceeds 8 MiB"):
            allocation_lower_bound((25 * 32 * 8192, 5 * 32 * 4096), 0)

    def test_actual_allocator_padding_is_charged(self):
        logical = (25 * 16 * 8192, 5 * 16 * 4096)
        padded = (25 * 16 * 8192, 5 * 16 * 8192)
        a = allocation_lower_bound(logical, 0)
        b = allocation_lower_bound(padded, 0)
        self.assertEqual(b["known_bytes"] - a["known_bytes"], 327680)


class SourceAndDefaultOff(unittest.TestCase):
    def test_native_defaults_fail_before_access_or_import(self):
        with self.assertRaisesRegex(ProbeError, "default-off"):
            OwnedNativeProbe.from_runner(object(), {}, {})
        with self.assertRaisesRegex(ProbeError, "default-off"):
            run_engine_core_probe(object(), (), {})
        with self.assertRaises(ProbeError):
            check_admission({})

    def test_module_import_is_cpu_only(self):
        code = "import sys; import megartx.speculative_native_probe; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True,
                       env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True)

    def test_source_manifest_matches_upstream_and_previous_pins(self):
        m = source_manifest()
        self.assertEqual(len(m["files"]), 22)
        for name, f in m["files"].items():
            self.assertTrue(f["matches"])
            self.assertEqual(f["sha256"], f["upstream_sha256"])
            self.assertLess(f["bytes"], 1_000_000)
            self.assertIn("ced6857afa0ea7b2e3f0846a62e1394e90f15607/" + name, f["url"])
        old = json.loads((ROOT / "docs/evidence/supplied-candidate-source-review.json").read_text())
        for f in old["files"]:
            name = f["url"].split("ced6857afa0ea7b2e3f0846a62e1394e90f15607/")[-1]
            if name in m["files"]:
                self.assertEqual(f["sha256"], m["files"][name]["sha256"])

    def test_changed_or_missing_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ProbeError):
                inspect_sources(directory)

    def test_cli_freeze_has_no_native_launch_option(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_speculative_native_probe.py")],
                                check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertFalse(report["gpu_enabled"])
        self.assertFalse(report["native_executed"])
        rejected = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_speculative_native_probe.py"), "--execute"],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)

    def test_frozen_scope_preserves_base_and_cap(self):
        p = json.loads((ROOT / "docs/design/speculative-native-protocol.json").read_text())
        self.assertEqual(p["base_commit"], BASE)
        self.assertFalse(p["gpu_authorized"])
        self.assertEqual(p["candidate_counts"], [0, 1])
        self.assertEqual(p["bounds"]["max_total_incremental_gpu_bytes"], GPU_CAP)
        self.assertIsNone(p["bounds"]["actual_page_tokens"])
        self.assertEqual(p["required_launcher_exposure"]["shared_files_touched"], [])

    def test_actual_runtime_calls_are_concrete_and_no_provider_injection(self):
        # Supporting source check only, never a native correctness test.
        source = (ROOT / "src/megartx/speculative_native_probe.py").read_text()
        tree = ast.parse(source)
        from_runner = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == "from_runner")
        self.assertEqual([a.arg for a in from_runner.args.args], ["cls", "runner", "ticket", "admission"])
        for call in ("runner.get_model()", "pool.get_new_blocks(sum(counts))", "pool.free_blocks(blocks)",
                     "builder.build(0, commons[gid])", "self.model.compute_logits(hidden[row:row + 1])"):
            self.assertIn(call, source)
        self.assertNotIn("provider", [a.arg for a in from_runner.args.args])


if __name__ == "__main__":
    unittest.main()
