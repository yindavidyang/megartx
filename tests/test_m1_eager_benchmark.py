"""CPU contracts for new admission, safe request switches and SSE measurement."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_m1_eager_benchmark import make_plan
from m1_eager_benchmark_client import completion, summarize_run, validate_dispatch
from megartx.m1_eager_benchmark import DRIVER_SOURCES, EagerBenchmark, drain_marker, marker, validate_plan
from megartx.m1_execution import CONTROLLER_SOURCES, execution_mode
from megartx.m1_normal_plan import digest


def plan(trials=1, warmups=1):
    return make_plan({n: [3] * n for n in (2048, 8192)}, "a" * 40,
                     {n: "b" * 64 for n in CONTROLLER_SOURCES},
                     {n: "c" * 64 for n in DRIVER_SOURCES}, trials, warmups)


def finish(obj, row, generated=7):
    n = int(row["case"])
    for i in range(n // 256 + 255):
        prefill = i < n // 256
        positions = list(range(i * 256, (i + 1) * 256)) if prefill else [n + i - n // 256]
        obj.begin(marker(obj.plan, row), [3] * 256 if prefill else [generated], positions)
        for _ in range(30):
            obj.fallback() if prefill else obj.backend(int(row["lane"] == "fused"))
        obj.end()


class PlanTests(unittest.TestCase):
    def test_bounded_randomized_adjacent_balanced_pairs_and_warmups(self):
        p = plan(6, 2)
        self.assertEqual(len(p["schedule"]), 32)
        self.assertEqual(p, plan(6, 2))
        for n in ("2048", "8192"):
            cells = [r for r in p["schedule"] if r["phase"] == "measurement" and r["case"] == n]
            self.assertEqual(sum(r["lane"] == "stock" for r in cells[::2]), 3)

    def test_changed_sources_budget_prompt_schedule_and_digest_reject(self):
        p = plan()
        changes = [("outputs", 255), ("trials", 7), ("warmups", 0), ("seed", True),
                   ("checkpoint_revision", "d" * 40), ("controller_source_hashes", {}),
                   ("driver_source_hashes", {}), ("schedule", p["schedule"][:-1])]
        for key, value in changes:
            bad = copy.deepcopy(p);bad[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): validate_plan(bad)
        with self.assertRaisesRegex(RuntimeError, "source"):
            validate_plan(p, {n: "f" * 64 for n in CONTROLLER_SOURCES})
        bad = copy.deepcopy(p);bad["cases"][0]["prompt_token_ids"][0] = True
        with self.assertRaisesRegex(RuntimeError, "prompt"): validate_plan(bad)

    def test_only_explicit_isolated_observer_off_native_lane_admits(self):
        env = {"MEGARTX_M1_EXECUTION": "capture-free", "MEGARTX_SCALE_MODE": "native",
               "MEGARTX_M1_PREPARATION": "stock", "MEGARTX_M1_EAGER_BENCHMARK_PLAN": "plan",
               "MEGARTX_M1_EAGER_BENCHMARK_DIR": "bench"}
        with patch.dict(os.environ, env, clear=True): self.assertEqual(execution_mode(), "capture-free")
        for key, value in (("MEGARTX_M1_EXECUTION", "captured"), ("MEGARTX_SCALE_MODE", "control"),
                           ("MEGARTX_M1_EXTERNAL_OBSERVER_DIR", "observer"), ("MEGARTX_LOGITS_DIR", "logits"),
                           ("MEGARTX_CONTROLLED_DIR", "controlled"), ("MEGARTX_M1_ROUTE_CONTROLS", "1")):
            with patch.dict(os.environ, {**env, key: value}, clear=True), self.assertRaises(RuntimeError): execution_mode()


class LedgerTests(unittest.TestCase):
    def test_complete_switching_transcripts_backend_counts_and_untimed_drain(self):
        p = plan()
        with tempfile.TemporaryDirectory() as directory:
            obj = EagerBenchmark(p, Path(directory) / "ledger", "stock")
            records = []
            for row in p["schedule"]:
                with patch.object(Path, "open", side_effect=AssertionError("timed serialization")):
                    finish(obj, row)
                self.assertEqual(obj.lane, row["lane"])
                records.append({**row, "token_ids": [7] * 256, "usage": {"prompt_tokens": int(row["case"]), "completion_tokens": 256}})
            self.assertFalse(obj.directory.exists())
            self.assertFalse(obj.begin(drain_marker(p), [7], [0]))
            report = json.loads((obj.directory / "dispatch.json").read_text())
            validate_dispatch(p, report, records)
            shortened = dict(report, records=report["records"][:-2])
            with self.assertRaisesRegex(RuntimeError, "count"):
                validate_dispatch(p, shortened, records[:-2])
            self.assertEqual(obj.call_limit, 61200)
            with self.assertRaisesRegex(RuntimeError, "drained"): obj.begin(drain_marker(p), [7], [0])
            records[0]["token_ids"][0] = 8
            with self.assertRaisesRegex(RuntimeError, "transcript"): validate_dispatch(p, report, records)

    def test_incomplete_request_cannot_reset_budget_or_switch_lane(self):
        p = plan();obj = EagerBenchmark(p, "unused", "stock");row = p["schedule"][0]
        obj.begin(marker(p, row), [3] * 256, list(range(256)))
        with self.assertRaisesRegex(RuntimeError, "nested"): obj.begin(marker(p, p["schedule"][1]), [3] * 256, list(range(256)))
        with self.assertRaisesRegex(RuntimeError, "bypassed"): obj.end()
        self.assertEqual(obj.index, 0)

    def test_request_transitions_retain_references_and_never_hash_before_drain(self):
        p = plan()
        with tempfile.TemporaryDirectory() as directory:
            obj = EagerBenchmark(p, Path(directory) / "ledger", "stock")
            retained = []
            with patch("megartx.m1_eager_benchmark.digest", side_effect=AssertionError("timed transcript hashing")):
                for row in p["schedule"]:
                    finish(obj, row)
                    retained.append(obj.tokens)
            self.assertEqual(len(obj.completed_transcripts), len(p["schedule"]) - 1)
            for actual, expected in zip(obj.completed_transcripts, retained):
                self.assertIs(actual, expected)
            self.assertTrue(all("input_transcript_sha256" not in r for r in obj.records))
            with patch("megartx.m1_eager_benchmark.digest", wraps=digest) as hashes:
                obj.begin(drain_marker(p), [7], [0])
            self.assertEqual(hashes.call_count, len(p["schedule"]))
            self.assertEqual(obj.completed_transcripts, [])
            report = json.loads((obj.directory / "dispatch.json").read_text())
            self.assertEqual([r["input_transcript_sha256"] for r in report["records"]], [digest(t) for t in retained])

    def test_invalid_actual_token_position_and_premature_drain_fail(self):
        p = plan()
        for tokens, positions in (([4] * 256, list(range(256))), ([3] * 256, list(range(1, 257)))):
            obj = EagerBenchmark(p, "unused", "stock")
            with self.assertRaisesRegex(RuntimeError, "tokens/positions"):
                obj.begin(marker(p, p["schedule"][0]), tokens, positions)
        with self.assertRaisesRegex(RuntimeError, "premature"):
            EagerBenchmark(p, "unused", "stock").begin(drain_marker(p), [7], [0])

    def test_supported_native_fallback_does_not_mislabel_fused_measurement(self):
        p = plan();obj = EagerBenchmark(p, "unused", "stock")
        row = p["schedule"][0]
        finish(obj, row)
        obj.counts["fused" if row["lane"] == "fused" else "stock"] -= 1
        obj.counts["stock" if row["lane"] == "fused" else "fused"] += 1
        with self.assertRaisesRegex(RuntimeError, "ledger incomplete"): obj.finish_request()


class SSETests(unittest.TestCase):
    def response(self, multi=False, done=True, completion_tokens=2):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_lines(self, **kw):
                for ids in ([[7, 8]] if multi else [[7], [8]]):
                    yield b"data: " + json.dumps({"choices": [{"token_ids": ids}]}).encode()
                yield b"data: " + json.dumps({"choices": [{"finish_reason": "length"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": completion_tokens}}).encode()
                if done: yield b"data: [DONE]"
        return Response()

    def test_last_token_done_response_boundaries_and_multi_chunk_policy(self):
        for multi in (False, True):
            values = iter(range(0, 20_000_000, 1_000_000))
            session = type("Session", (), {"post": lambda *a, **kw: self.response(multi)})()
            result = completion(session, "http://127.0.0.1:18000", {"max_tokens": 2, "prompt": [3]}, lambda: next(values))
            self.assertLess(result["last_token_ms"], result["done_ms"])
            self.assertLess(result["done_ms"], result["response_ms"])
            self.assertEqual(result["itl_p50_ms"] is None, multi)
            self.assertEqual(result["multi_token_chunks"], int(multi))

    def test_missing_done_and_bad_usage_invalidate_result(self):
        for done, count in ((False, 2), (True, 1)):
            session = type("Session", (), {"post": lambda *a, **kw: self.response(done=done, completion_tokens=count)})()
            with self.assertRaises(RuntimeError): completion(session, "unused", {"max_tokens": 2, "prompt": [3]})

    def test_complete_private_packet_summary_checks_cleanup_and_discloses_zero_hits(self):
        p = plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);obj = EagerBenchmark(p, root / "eager-benchmark", "stock")
            records = []
            for i, row in enumerate(p["schedule"]):
                finish(obj, row)
                records.append({**row, "token_ids": [7] * 256, "token_elapsed_ns": list(range(1_000_000, 257_000_000, 1_000_000)),
                    "usage": {"prompt_tokens": int(row["case"]), "completion_tokens": 256},
                    "request_start_monotonic_ns": i * 1_000_000_000, "request_end_monotonic_ns": i * 1_000_000_000 + 258_000_000,
                    "ttft_ms": 1., "amortized_itl_ms": 1., "last_token_ms": 256., "done_ms": 257.,
                    "response_ms": 258., "multi_token_chunks": 0})
            obj.begin(drain_marker(p), [7], [0])
            (root / "eager-benchmark-plan.json").write_text(json.dumps(p))
            (root / "eager-requests.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            (root / "eager-benchmark-cleanup.json").write_text(json.dumps({"cleanup_complete": True}))
            for name in ("benchmark.exit", "run.exit"): (root / name).write_text("0\n")
            (root / "gpu-telemetry.jsonl").write_text("\n".join(json.dumps({"monotonic_ns": r["request_start_monotonic_ns"] + 1_000_000,
                "fields": ["memory.used", "memory.free"], "values": ["23000", "9000"], "exit": 0}) for r in records))
            result = summarize_run(root)
            self.assertFalse(result["performance_gate_passed"])
            self.assertFalse(result["minimum_30_pairs_met"])
            self.assertTrue(result["contexts"]["8192"]["stock"]["zero_natural_correction_hits"])
            self.assertEqual(result["memory"]["sampled_lifecycle_device_peak_mib"], 23000.)
            (root / "eager-benchmark-cleanup.json").write_text(json.dumps({"cleanup_complete": False}))
            with self.assertRaisesRegex(RuntimeError, "lifecycle"): summarize_run(root)


if __name__ == "__main__": unittest.main()
