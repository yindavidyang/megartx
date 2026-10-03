import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ProcessProbeTests(unittest.TestCase):
    def test_fresh_spawn_process_records_actual_context_and_parent_child_identity(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "scripts/m1_process_probe/sitecustomize.py"
        expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            evidence.mkdir(mode=0o700)
            env = {
                "MEGARTX_M1_PROCESS_EVIDENCE_DIR": str(evidence),
                "MEGARTX_M1_PROCESS_EXPECTED_METHOD": "spawn",
                "MEGARTX_M1_PROCESS_PROBE_SHA256": expected_sha,
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "PYTHONPATH": str(source.parent),
            }
            code = ("import multiprocessing; c=multiprocessing.get_context('spawn'); "
                    "p=c.Process(target=abs,args=(-1,),name='EngineCore'); p.start(); "
                    "p.join(8); assert p.exitcode == 0, p.exitcode")
            subprocess.run([os.sys.executable, "-c", code], env=env, check=True, timeout=12)

            rows = [json.loads(line) for line in
                    (evidence / "process-events.jsonl").read_text().splitlines()]
            context_row = next(row for row in rows
                               if row["event"] == "multiprocessing_context_created")
            self.assertEqual(context_row["actual_start_method"], "spawn")
            self.assertEqual(context_row["probe_sha256"], expected_sha)
            child_row = next(row for row in rows
                             if row["event"] == "engine_core_process_started")
            self.assertEqual(child_row["actual_start_method"], "spawn")
            self.assertEqual(child_row["parent_pid"], context_row["pid"])
            self.assertEqual(child_row["parent_ppid"], context_row["ppid"])
            self.assertGreater(child_row["child_pid"], 0)
            self.assertEqual(child_row["child_name"], "EngineCore")
            ready = [row for row in rows if row["event"] == "process_probe_ready"]
            self.assertEqual(sum(row["pid"] == context_row["pid"] for row in ready), 1)
            self.assertEqual(sum(row["pid"] == child_row["child_pid"]
                                 and row["ppid"] == context_row["pid"] for row in ready), 1)

    def test_spawned_engine_core_waits_for_delayed_parent_start_record(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "scripts/m1_process_probe/sitecustomize.py"
        expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        script = """import json
import multiprocessing
import os
from pathlib import Path
import sitecustomize
import sys
import time

def validate_in_child(evidence, api_pid_path, probe_sha, waiting, observed_missing, result_path):
    from megartx import m1_process_lifecycle as lifecycle
    original_read = lifecycle._read_records
    def observe_missing_spawn(path, **kwargs):
        records = original_read(path, **kwargs)
        if not any(row.get("event") == "engine_core_process_started" for row in records):
            Path(observed_missing).write_text("child observed missing parent row")
        return records
    lifecycle._read_records = observe_missing_spawn
    Path(waiting).write_text("child entered validator")
    started = time.monotonic()
    identity = lifecycle.process_identity()
    result = lifecycle.validate_engine_core_spawn_owner(
        evidence, identity, api_pid_path, probe_sha)
    Path(result_path).write_text(json.dumps({"wait_seconds": time.monotonic() - started,
                                           "owner": result}))

if __name__ == "__main__":
    evidence, api_pid_path, probe_sha, waiting, observed_missing, result_path = sys.argv[1:]
    Path(api_pid_path).write_text(str(os.getpid()) + "\\n")
    original_append = sitecustomize._append_event
    def delay_engine_core_row(event, **fields):
        if event == "engine_core_process_started":
            deadline = time.monotonic() + 8
            while not Path(observed_missing).exists() and time.monotonic() < deadline:
                time.sleep(.01)
            if not Path(observed_missing).exists():
                raise RuntimeError("spawned child never observed the missing parent row")
            time.sleep(.35)
        original_append(event, **fields)
    sitecustomize._append_event = delay_engine_core_row
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=validate_in_child,
        args=(evidence, api_pid_path, probe_sha, waiting, observed_missing, result_path),
        name="EngineCore")
    child.start()
    child.join(12)
    if child.is_alive():
        child.terminate()
        child.join(2)
    assert child.exitcode == 0, child.exitcode
    assert Path(result_path).is_file(), "spawned EngineCore did not finish lifecycle validation"
"""
        with tempfile.TemporaryDirectory() as directory:
            root_dir = Path(directory)
            evidence = root_dir / "evidence"
            evidence.mkdir(mode=0o700)
            api_pid_path = root_dir / "api.pid"
            waiting = root_dir / "child-entered.txt"
            observed_missing = root_dir / "child-observed-missing.txt"
            result_path = root_dir / "validation.json"
            script_path = root_dir / "delayed_spawn.py"
            script_path.write_text(script)
            env = {
                "MEGARTX_M1_PROCESS_EVIDENCE_DIR": str(evidence),
                "MEGARTX_M1_PROCESS_EXPECTED_METHOD": "spawn",
                "MEGARTX_M1_PROCESS_PROBE_SHA256": expected_sha,
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "PYTHONPATH": os.pathsep.join((str(source.parent), str(root / "src"))),
            }
            subprocess.run([os.sys.executable, str(script_path), str(evidence),
                            str(api_pid_path), expected_sha, str(waiting),
                            str(observed_missing), str(result_path)],
                           env=env, check=True, timeout=18)
            result = json.loads(result_path.read_text())
            self.assertGreaterEqual(result["wait_seconds"], .20)
            self.assertEqual(result["owner"]["actual_start_method"], "spawn")
            self.assertEqual(result["owner"]["engine_core_process_name"], "EngineCore")
            self.assertTrue(waiting.is_file())
            self.assertTrue(observed_missing.is_file())


if __name__ == "__main__":
    unittest.main()
