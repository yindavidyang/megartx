"""CPU ownership/resource regressions, including real Linux orphan adoption."""
import json
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m1_owned_processes import OwnedProcesses, Process, preserve_primary, read_process, signal_identity


def proc(pid, parent, *, start=None, rss=0, name="python", age=0, state="S", argv=None):
    return Process(pid, start or pid, parent, rss, name, state, age, argv)


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

    def test_pidfd_open_einval_requires_fresh_absent_or_reused_identity(self):
        old = proc(30, 20)
        for current in (FileNotFoundError(errno.ENOENT, "gone"), proc(30, 1, start=300)):
            with self.subTest(current=current), \
                 patch("m1_owned_processes.os.pidfd_open", side_effect=OSError(errno.EINVAL, "no TGID"), create=True), \
                 patch("m1_owned_processes.read_process", side_effect=[current]) as read, \
                 patch("m1_owned_processes.signal.pidfd_send_signal", create=True) as send:
                self.assertFalse(signal_identity(old, signal.SIGTERM))
                read.assert_called_once_with(30)
                send.assert_not_called()

    def test_pidfd_open_einval_for_same_identity_still_raises(self):
        old = proc(30, 20); error = OSError(errno.EINVAL, "invalid live pidfd")
        with patch("m1_owned_processes.os.pidfd_open", side_effect=error, create=True), \
             patch("m1_owned_processes.read_process", return_value=old), \
             self.assertRaises(OSError) as raised:
            signal_identity(old, signal.SIGKILL)
        self.assertIs(raised.exception, error)
        self.assertEqual(error.megartx_operation, "pidfd_open")

    def test_other_pidfd_errors_and_failed_identity_read_still_raise(self):
        old = proc(30, 20)
        for open_errno, read_result in ((errno.EPERM, None),
                                        (errno.EINVAL, PermissionError(errno.EACCES, "unreadable"))):
            with self.subTest(open_errno=open_errno), \
                 patch("m1_owned_processes.os.pidfd_open", side_effect=OSError(open_errno, "fault"), create=True), \
                 patch("m1_owned_processes.read_process", side_effect=read_result) as read, \
                 self.assertRaises(OSError):
                signal_identity(old, signal.SIGTERM)
            self.assertEqual(read.call_count, int(open_errno == errno.EINVAL))

    def test_pidfd_send_einval_is_never_treated_as_a_vanished_open(self):
        old = proc(30, 20)
        with patch("m1_owned_processes.os.pidfd_open", return_value=90, create=True), \
             patch("m1_owned_processes.read_process", return_value=old), \
             patch("m1_owned_processes.signal.pidfd_send_signal", side_effect=OSError(errno.EINVAL, "signal fault"), create=True), \
             patch("m1_owned_processes.os.close") as close, self.assertRaises(OSError) as raised:
            signal_identity(old, signal.SIGTERM)
        self.assertEqual(raised.exception.megartx_operation, "pidfd_send_signal")
        close.assert_called_once_with(90)

    def test_tileiras_and_descendants_are_retained_even_after_exit(self):
        assembler = proc(30, 20, name="tileiras", rss=10, age=.01, argv=("/usr/local/cuda/bin/tileiras","--help"))
        self.observe(self.root, assembler, proc(40, 30, rss=20))
        self.observe(self.root, proc(30,20,name="tileiras",state="Z"), now=101)
        report = self.owner.report()
        self.assertEqual(report["sampled_peak_compiler_rss_bytes"], 30)
        self.assertEqual({p["pid"] for p in report.get("sampled_compiler_identities", [])}, {30,40})
        self.assertIsNone(self.owner.failure)
        invocation = report["sampled_compiler_invocations"][0]
        self.assertEqual(invocation["compiler_argv"], assembler.compiler_argv)
        self.assertEqual(invocation["classification"], "tileiras_help_probe")
        with self.assertRaisesRegex(RuntimeError, "compiler activity"):
            self.owner.require_compiler_quiescence()

    def test_compiler_argv_capture_does_not_read_model_or_client_arguments(self):
        fields = ["S","20"] + ["0"] * 21
        fields[19] = "30"; fields[21] = "1"
        for name in ("tileiras","python"):
            with self.subTest(name=name), patch.object(Path,"read_text",return_value="30 (tool) " + " ".join(fields)), \
                 patch("m1_owned_processes.os.readlink", return_value="/bin/" + name), \
                 patch.object(Path,"stat",side_effect=FileNotFoundError), \
                 patch.object(Path,"read_bytes",return_value=b"/bin/tileiras\0--help\0") as read:
                process = read_process(30, uptime=100)
            self.assertEqual(read.call_count, int(name=="tileiras"))
            self.assertEqual(process.compiler_argv, ("/bin/tileiras","--help") if name=="tileiras" else None)

    def test_existing_compiler_pid_rechecks_argv_without_reading_other_process_arguments(self):
        # Both /proc states are synthetic: a real host PID 30 must not change
        # which source branch this privacy/identity regression exercises.
        fields = ["S", "20"] + ["0"] * 21
        fields[19] = "30"; fields[21] = "1"
        version = os.stat_result((0, 1, 1, 1, 1, 1, 4096, 1, 1, 1))
        for name in ("tileiras", "python"):
            with self.subTest(name=name), patch.object(Path, "read_text", return_value="30 (tool) " + " ".join(fields)), \
                 patch("m1_owned_processes.os.readlink", return_value="/bin/" + name), \
                 patch.object(Path, "stat", return_value=version) as stat, \
                 patch.object(Path, "read_bytes", return_value=b"/bin/tileiras\0--help\0") as read:
                process = read_process(30, uptime=100)
            self.assertEqual(read.call_count, 2 * int(name == "tileiras"))
            self.assertEqual(stat.call_count, 2 * int(name == "tileiras"))
            self.assertEqual(process.compiler_identity_verified, name == "tileiras")
            self.assertEqual(process.compiler_argv, ("/bin/tileiras", "--help") if name == "tileiras" else None)

    def test_tileiras_unknown_or_extra_arguments_are_conservative(self):
        for argv in (None, (), ("tileiras","input.tileir"), ("tileiras","--help","input.tileir")):
            self.assertEqual(proc(30,20,name="tileiras",argv=argv).compiler_invocation, "compiler_work_or_unknown")
        self.assertEqual(proc(30,20,name="tileiras",argv=("tileiras","--version")).compiler_invocation,
                         "tileiras_version_probe")

    def test_zero_rss_zombie_compiler_still_invalidates_timing(self):
        self.observe(self.root, proc(30,20,name="tileiras",state="Z"))
        self.assertEqual(self.owner.peak_compiler_rss, 0)
        self.assertEqual(self.owner.compiler_elapsed, 0)
        with self.assertRaisesRegex(RuntimeError, "compiler activity"):
            self.owner.require_compiler_quiescence()

    def test_unknown_cleanup_einval_stays_failed_with_exact_operation(self):
        error = OSError(errno.EINVAL, "invalid live pidfd")
        snapshots = [{20:self.root}, {}, {}, {}, {}]
        with patch("m1_owned_processes.snapshot", side_effect=snapshots), \
             patch("m1_owned_processes.os.pidfd_open", side_effect=error, create=True), \
             patch("m1_owned_processes.read_process", return_value=self.root):
            report = self.owner.cleanup(lambda:[], lambda:None, term_seconds=0, kill_seconds=0)
        self.assertFalse(report["cleanup_complete"])
        self.assertEqual(report["owned_identities_remaining"], [])
        detail = report["cleanup_error_details"][0]
        self.assertEqual((detail["operation"],detail["identity"],detail["errno"]), ("pidfd_open",(20,20),errno.EINVAL))

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
