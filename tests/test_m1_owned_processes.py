"""CPU ownership/resource regressions, including real Linux orphan adoption."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m1_owned_processes import OwnedProcesses, Process, preserve_primary, signal_identity


def proc(pid, parent, *, start=None, rss=0, name="python", age=0, state="S"):
    return Process(pid, start or pid, parent, rss, name, state, age)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = proc(10, 1)
        self.owner = OwnedProcesses(self.supervisor)
        self.root = proc(20, 10)
        self.owner.register(self.root)

    def observe(self, *processes, now=100):
        return self.owner.observe({p.pid:p for p in (self.supervisor, *processes)}, now)

    def test_concurrent_groups_sum_rss_and_breach_blocks_dispatch(self):
        first = proc(30, 20, name="ninja", rss=100)
        # Group/session membership is deliberately absent from ownership.
        second = proc(40, 20, name="nvcc", rss=(2 << 30) - 100)
        self.observe(self.root, first, second, proc(41, 40, rss=1))
        self.assertEqual(self.owner.peak_compiler_rss, (2 << 30) + 1)
        self.assertIn("aggregate RSS", self.owner.failure)
        self.assertEqual(len(self.owner.failure_sample["compiler_identities"]), 3)

    def test_shared_budget_never_resets_for_new_jobs_or_idle_gap(self):
        self.observe(self.root, proc(30, 20, name="ninja", age=10), now=100)
        self.observe(self.root, now=200)
        self.observe(self.root, proc(40, 20, name="nvcc"), now=391)
        self.assertEqual(self.owner.compiler_elapsed, 301)
        self.assertIn("300 seconds", self.owner.failure)

    def test_reparent_retained_identity_and_unknown_subreaper_adoptee(self):
        self.observe(self.root, proc(30, 20), proc(40, 30, name="cicc"))
        owned = self.observe(proc(40, 1, name="cicc"), proc(50, 10))
        self.assertEqual({p.pid for p in owned}, {40, 50})
        self.assertIn((30, 30), self.owner.remembered)

    def test_reused_parent_does_not_own_unrelated_descendants(self):
        self.observe(self.root, proc(30, 20))
        owned = self.observe(proc(30, 1, start=300), proc(40, 30, start=400, name="nvcc"))
        self.assertEqual(owned, [])
        self.assertIsNone(self.owner.failure)

    def test_pidfd_recheck_protects_reused_pid(self):
        old = proc(30, 20)
        with patch("m1_owned_processes.os.pidfd_open", return_value=90, create=True), \
             patch("m1_owned_processes.read_process", return_value=proc(30, 1, start=300)), \
             patch("m1_owned_processes.signal.pidfd_send_signal", create=True) as send, \
             patch("m1_owned_processes.os.close") as close:
            self.assertFalse(signal_identity(old, signal.SIGKILL))
        send.assert_not_called(); close.assert_called_once_with(90)

    def test_first_fault_and_primary_error_survive_cleanup_fault(self):
        self.owner.fail("compiler bound", {"rss":123})
        self.owner.fail("later telemetry/cleanup")
        self.assertEqual(self.owner.failure, "compiler bound")
        original = ValueError("original request failure")
        preserve_primary(original, "cleanup failed")
        self.assertEqual(str(original), "original request failure")
        self.assertEqual(original.__notes__, ["cleanup failed"])
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            preserve_primary(None, "cleanup failed")

    def test_cleanup_checks_gpu_identity_and_protects_unrelated_jobs(self):
        unrelated = proc(99, 1, name="nvcc")
        snapshots = [{20:self.root,99:unrelated}, {99:unrelated}, {99:unrelated}]
        with patch("m1_owned_processes.snapshot", side_effect=snapshots), \
             patch("m1_owned_processes.signal_identity") as send:
            report = self.owner.cleanup(lambda:[99], lambda:None, term_seconds=0, kill_seconds=0)
        self.assertTrue(report["cleanup_complete"])
        send.assert_called_once_with(self.root, signal.SIGTERM)

    def test_cleanup_query_error_fails_closed(self):
        with patch("m1_owned_processes.snapshot", return_value={}):
            report = self.owner.cleanup(lambda:(_ for _ in ()).throw(RuntimeError("GPU query")), lambda:None)
        self.assertFalse(report["cleanup_complete"])
        self.assertIn("GPU query", report["cleanup_errors"])

    @unittest.skipUnless(sys.platform == "linux", "real /proc/subreaper regression requires Linux")
    def test_live_setsid_orphan_adoption_cleanup_and_unrelated_protection(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        code = '''
import json, os, subprocess, sys, time
from m1_owned_processes import OwnedProcesses, enable_subreaper, read_process, snapshot
owner=OwnedProcesses(enable_subreaper())
fixture="import os,time; p=os.fork(); os.setsid() if p==0 else None; time.sleep(30) if p==0 else time.sleep(.2)"
root=subprocess.Popen([sys.executable,"-c",fixture],start_new_session=True)
owner.register(read_process(root.pid))
owner.observe(snapshot(),time.monotonic()); root.wait(timeout=5)
time.sleep(.1)
alive=owner.observe(snapshot(),time.monotonic())
assert any(p.ppid==os.getpid() and p.pid!=root.pid for p in alive), alive
report=owner.cleanup(lambda:[],root.poll,term_seconds=2,kill_seconds=2)
assert report['cleanup_complete'], report
os.kill(int(sys.argv[1]),0)
print(json.dumps(report))
'''
        try:
            env = os.environ.copy(); env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "scripts")
            result = subprocess.run([sys.executable, "-c", code, str(unrelated.pid)], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["cleanup_complete"])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait(timeout=5)
