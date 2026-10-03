"""CPU pipeline and closed-launch admission; fixtures are never live evidence."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megartx import prefill_launch as l
from megartx import prefill_runner as r
from test_prefill_runner import inputs, synthetic_records, DIGEST
from test_prefill_collect import observation, bootstrap, FakeEvents

ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    def __init__(self, protocol, prompts):
        self.protocol = protocol
        self.records = {x["run_id"]: x for x in synthetic_records(protocol, prompts)["runs"]}
        self.calls, self.tick, self.event_clock = [], 0, FakeEvents()
        self.fail_start, self.fail_observe, self.fail_cleanup = False, False, False
    def clock(self):
        self.tick += 1
        return self.tick
    def describe_contract(self):
        return {"contract": r.LAUNCHER_CONTRACT, **self.protocol["launcher"]}
    def admit_resources(self, bounds):
        self.calls.append("admit")
        return {"host_free_bytes": bounds["host_free_floor_bytes"], "gpu_free_bytes": bounds["gpu_free_floor_bytes"], "exclusive_slot": True}
    def startup(self, job):
        self.calls.append(("startup", job["run_id"]))
        self.tick = self.records[job["run_id"]]["run_begin_ns"]
        if self.fail_start: raise RuntimeError("partial owned startup failed")
        return {"sha256": DIGEST, "owned_only": True}
    def initialize(self, job):
        self.calls.append("initialize")
        return copy.deepcopy(self.records[job["run_id"]])
    def timing_request(self, job, payload):
        self.calls.append(("timing", job["state"]))
        return copy.deepcopy(self.records[job["run_id"]])
    def observe_request(self, job, payload, collector):
        self.calls.append(("observe", job["mode"], job["region"]))
        if self.fail_observe: raise RuntimeError("native observed forward failed")
        tokens = payload["prompt"]
        for start, end in job["spans"]:
            collector.begin_forward(tokens[start:end], list(range(start, end)))
            for layer in range(30):
                if job["mode"] == "correctness" or job["mode"] == "profile" and job["region"] == "expert":
                    m = end - start
                    collector.record_routes(layer, [list(range(8)) for _ in range(m)], [[1.0] * 8 for _ in range(m)],
                        [m] * 8 + [0] * 120, 0, 0)
                if job["mode"] == "profile" and job["region"] != "head":
                    component = {"attention": "attention", "dense": "qkv", "expert": "dispatch"}[job["region"]]
                    with collector.stage(component, layer, "synthetic-callable", DIGEST, (end-start, 2816, 2816), self.event_clock): pass
                collector.layer_complete(layer, observation(start, end, layer))
            collector.end_forward()
        record = copy.deepcopy(self.records[job["run_id"]])
        collector.handoff(record["handoff"])
        collector.bootstrap(bootstrap(job, tokens))
        if job["mode"] == "profile" and job["region"] == "head":
            with collector.stage("head", 29, "synthetic-last-row", DIGEST, (1, 262144, 2816), self.event_clock): pass
        for i in range(255):
            collector.begin_forward([17], [job["prompt_tokens"] + i], "decode")
            for layer in range(30): collector.layer_complete(layer)
            collector.end_forward()
        if job["mode"] == "profile" and job["region"] == "expert":
            record["profile"]["expert_rows"] = collector.routes
        return record
    def drain(self): self.calls.append("drain")
    def cleanup(self):
        self.calls.append("cleanup")
        if self.fail_cleanup: raise RuntimeError("owned cleanup failed")
        return {"sha256": DIGEST, "owned_only": True, "cleanup_complete": True}
    def poison(self): self.calls.append("poison")
    def host_clock_id(self): return "synthetic_cpu_monotonic_clock"
    def records_provenance(self):
        return {"kind": "synthetic_cpu", "origin_reference": "test_prefill_launch generated fixture",
                "artifact_sha256": DIGEST, "sanitized": True}


def synthetic_protocol(chunk=2048):
    manifest, plan, prompts = inputs(True)
    plan["resource_bounds"] = {k: l.CAPS[k] for k in r.plan_contract.RESOURCE_KEYS}
    protocol = r.compile_protocol(manifest, plan, "p2048", chunk)
    return protocol, prompts


class AdmissionTests(unittest.TestCase):
    def test_checked_in_packet_and_all_eleven_cells_remain_disabled(self):
        packet, manifest, plan = l.load_packet(ROOT)
        for prompt, chunks in l.CHUNKS.items():
            for chunk in chunks:
                result = l.compile_packet(packet, manifest, plan, "p" + str(prompt), chunk)
                self.assertFalse(result["gpu_enabled"])
                self.assertFalse(result["gpu_qualified"])
                self.assertFalse(result["protocol"]["gpu_execution_available"])
                self.assertEqual(result["protocol"]["request_count"], 8)
                self.assertEqual(sum(b-a for a,b in result["protocol"]["jobs"][1]["spans"]), prompt)
                self.assertIn("GPU handoff and frozen live review pending", result["blockers"])
                self.assertTrue(result["code_support_is_not_dispatch_qualification"])
        self.assertEqual(manifest["launcher"]["supported_chunk_tokens"], [])

    def test_execute_always_rejects_before_provider_factory_even_filled_receipts(self):
        packet, _, _ = l.load_packet(ROOT)
        packet["gpu_enabled"] = True
        packet["receipts"] = {k: DIGEST for k in l.REQUIRED}
        called = []
        launcher = l.SerializedLauncher(packet, lambda: called.append("factory"))
        with self.assertRaisesRegex(ValueError, "GPU execution disabled"):
            launcher.execute_serialized({"prerequisites_complete": True}, {})
        self.assertEqual(called, [])
        self.assertFalse(launcher.describe_contract()["provider_connected"])

    def test_source_drift_symlink_and_nested_bool_caps_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            l.load_packet(root)
            file = root / "src/megartx/prefill_observe.py"
            original = file.read_bytes()
            file.write_bytes(original + b"# drift\n")
            with self.assertRaisesRegex(ValueError, "Live source drift"): l.load_packet(root)
            file.write_bytes(original)
            target = root / "copy.py"
            target.write_bytes(original)
            file.unlink()
            file.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"): l.load_packet(root)
            file.unlink()
            file.write_bytes(original)
            plan_path = root / "docs/prefill/live-plan.json"
            content = json.loads(plan_path.read_text())
            for key, value in (("gpu_enabled", True), ("unknown", True)):
                changed = copy.deepcopy(content)
                changed[key] = value
                plan_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): l.load_packet(root)
            changed = copy.deepcopy(content)
            changed["caps"]["max_gpu_jobs"] = True
            plan_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "fixed live resource cap"): l.load_packet(root)

    def test_cli_does_not_import_native_stack_and_gpu_flag_rejects(self):
        script = '''import builtins,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','vllm','flashinfer','ctypes'}: raise RuntimeError('Native import attempted: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=blocked
sys.path.insert(0,'src')
from megartx.prefill_launch import main
main(['--root','.',*sys.argv[1:]])
'''
        for args, code in (([], 0), (["--execute-gpu"], 2)):
            result = subprocess.run([sys.executable, "-c", script, *args], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn("Native import attempted", result.stderr)
            if args: self.assertIn("GPU execution disabled", result.stderr)

    def test_cli_exclusive_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.json"
            with patch("sys.stdout"):
                self.assertEqual(l.main(["--root", str(ROOT), "--output", str(path)]), 0)
            content = path.read_bytes()
            with patch("sys.stderr"), self.assertRaises(SystemExit): l.main(["--root", str(ROOT), "--output", str(path)])
            self.assertEqual(path.read_bytes(), content)


class PipelineTests(unittest.TestCase):
    def test_complete_fake_pipeline_validates_pr19_and_cleans_every_run(self):
        packet, _, _ = l.load_packet(ROOT)
        protocol, prompts = synthetic_protocol()
        provider = FakeProvider(protocol, prompts)
        result = l.SerializedLauncher(packet)._collect_admitted(protocol, prompts, provider, provider.clock)
        self.assertTrue(result["records_consistent"])
        self.assertEqual(result["provenance_kind"], "synthetic_cpu")
        self.assertFalse(result["gpu_qualified"])
        self.assertFalse(result["performance_qualified"])
        self.assertEqual(provider.calls.count("cleanup"), 9)
        self.assertEqual(provider.calls.count("drain"), 9)
        self.assertEqual(sum(x[0] == "timing" for x in provider.calls if isinstance(x, tuple)), 2)
        self.assertEqual(len(result["collector_evidence"]), 6)
        self.assertTrue(all(len(x["forward_times"]) == 256 for x in result["collector_evidence"]))
        self.assertTrue(all(x["bootstrap"]["cached_length"] == 2048 for x in result["collector_evidence"]))

    def test_partial_startup_observation_and_cleanup_failure_stop_next_job(self):
        packet, _, _ = l.load_packet(ROOT)
        for failure in ("fail_start", "fail_observe", "fail_cleanup"):
            protocol, prompts = synthetic_protocol()
            provider = FakeProvider(protocol, prompts)
            setattr(provider, failure, True)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                l.SerializedLauncher(packet)._collect_admitted(protocol, prompts, provider, provider.clock)
            self.assertEqual(provider.calls.count("cleanup"), 2 if failure == "fail_observe" else 1)
            if failure != "fail_cleanup": self.assertIn("poison", provider.calls)

    def test_resource_and_provider_drift_fail_before_startup(self):
        packet, _, _ = l.load_packet(ROOT)
        for change in ("source", "host", "gpu", "exclusive", "bounds"):
            protocol, prompts = synthetic_protocol()
            provider = FakeProvider(protocol, prompts)
            if change == "source":
                provider.describe_contract = lambda: {"contract": "unknown"}
            elif change == "bounds":
                protocol["resource_bounds"]["max_gpu_jobs"] = True
            else:
                resource = {"host_free_bytes": l.CAPS["host_free_floor_bytes"], "gpu_free_bytes": l.CAPS["gpu_free_floor_bytes"], "exclusive_slot": True}
                if change == "exclusive": resource["exclusive_slot"] = False
                else: resource[change + "_free_bytes"] -= 1
                provider.admit_resources = lambda bounds: resource
            with self.subTest(change=change), self.assertRaises(ValueError):
                l.SerializedLauncher(packet)._collect_admitted(protocol, prompts, provider, provider.clock)
            self.assertFalse(any(isinstance(x, tuple) and x[0] == "startup" for x in provider.calls))

    def test_timing_observer_leak_rejects_and_owned_cleanup_runs(self):
        packet, _, _ = l.load_packet(ROOT)
        protocol, prompts = synthetic_protocol()
        protocol["jobs"] = [j for j in protocol["jobs"] if j["mode"] == "timing"]
        provider = FakeProvider(protocol, prompts)
        saved = provider.timing_request
        def leaked(job, payload):
            value = saved(job, payload)
            value["trace_bytes"] = 1
            return value
        provider.timing_request = leaked
        with self.assertRaisesRegex(ValueError, "Timing job activated"):
            l.SerializedLauncher(packet)._collect_admitted(protocol, prompts, provider, provider.clock)
        self.assertEqual(provider.calls.count("cleanup"), 1)
        self.assertIn("poison", provider.calls)


    def test_bad_completed_record_rejects_before_next_startup(self):
        packet, _, _ = l.load_packet(ROOT)
        protocol, prompts = synthetic_protocol()
        provider = FakeProvider(protocol, prompts)
        first = protocol["jobs"][0]["run_id"]
        provider.records[first]["initialization"]["build_rss_peak_bytes"] = l.CAPS["max_build_rss_bytes"] + 1
        with self.assertRaisesRegex(ValueError, "Build resource bound"):
            l.SerializedLauncher(packet)._collect_admitted(protocol, prompts, provider, provider.clock)
        self.assertEqual(provider.calls.count("cleanup"), 1)
        self.assertEqual(sum(x[0] == "startup" for x in provider.calls if isinstance(x, tuple)), 1)


if __name__ == "__main__": unittest.main()
