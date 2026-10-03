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


if __name__ == "__main__":
    unittest.main()
