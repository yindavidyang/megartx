"""CPU safety/reference cases for the single reviewed metadata timing policy."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m1_owned_processes import MetadataHelpTiming, OwnedProcesses, Process, decode_cmdline, file_version, read_process


def metadata_fixture(test):
    temp = tempfile.TemporaryDirectory(); test.addCleanup(temp.cleanup)
    root = Path(temp.name).resolve()
    binary = root / "tileiras"; binary.write_bytes(b"pinned fake CPU tool fixture")
    caller = root / "cutile/cutile_common.py"; caller.parent.mkdir(); caller.write_bytes(b"pinned caller fixture")
    for name, value in (("EXECUTABLE",str(binary)),("RESOLVED",str(binary)),
                        ("BINARY_SHA",hashlib.sha256(binary.read_bytes()).hexdigest()),
                        ("CALLER_SHA",hashlib.sha256(caller.read_bytes()).hexdigest())):
        patcher = patch.object(MetadataHelpTiming,name,value); patcher.start(); test.addCleanup(patcher.stop)
    policy = MetadataHelpTiming(root,"a" * 40)
    sample = Process(30,30,20,10,"tileiras","S",.01,policy.argv,policy.RESOLVED,
                     policy.binary_version,True)
    return policy, sample, binary, caller


class MetadataTimingTests(unittest.TestCase):
    def setUp(self):
        self.policy,self.sample,self.binary,self.caller = metadata_fixture(self)
        self.owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy)
        self.root = Process(20,20,10); self.owner.register(self.root)

    def observe(self, *samples, now=100):
        self.owner.observe({p.pid:p for p in (self.root,*samples)},now)

    def admit(self):
        self.owner.finalize_metadata()
        self.owner.require_compiler_quiescence()

    def test_only_verified_help_admits_and_keeps_resource_counters(self):
        self.observe(self.sample)
        with patch("m1_owned_processes.hash_file_version",side_effect=AssertionError("timed hashing")):
            self.observe(self.sample,now=101)
        self.admit()
        report = self.owner.report()
        self.assertEqual(report["sampled_peak_compiler_rss_bytes"],10)
        self.assertGreater(report["shared_compiler_elapsed_seconds"],0)
        self.assertEqual(report["timing_metadata_only_identities"],[(30,30)])
        self.assertEqual(report["timing_unknown_or_work_identities"],[])

    def test_default_without_policy_rejects_the_same_help(self):
        owner = OwnedProcesses(Process(10,10,1)); owner.register(self.root)
        owner.observe({20:self.root,30:self.sample},100)
        with self.assertRaisesRegex(RuntimeError,"compiler activity"):
            owner.require_compiler_quiescence()

    def test_missing_or_changed_evidence_and_extra_arguments_reject(self):
        for change in ({"compiler_argv":None}, {"compiler_executable":None},
                       {"compiler_file_version":None}, {"compiler_identity_verified":False},
                       {"compiler_file_version":tuple([*self.policy.binary_version[:-1],0])},
                       {"compiler_argv":(self.policy.EXECUTABLE,"--version")},
                       {"compiler_argv":(self.policy.EXECUTABLE,"--help","input.tileir")},
                       {"compiler_argv":(self.policy.EXECUTABLE,"--help-hidden")},
                       {"compiler_argv":("/alias/tileiras","--help")},
                       {"compiler_executable":"/alias/tileiras"}):
            with self.subTest(change=change):
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy); owner.register(self.root)
                owner.observe({20:self.root,30:replace(self.sample,**change)},100)
                owner.finalize_metadata()
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):owner.require_compiler_quiescence()

    def test_help_to_work_and_unknown_to_help_same_identity_are_permanent(self):
        unknown = replace(self.sample,compiler_argv=None)
        work = replace(self.sample,compiler_argv=(self.policy.EXECUTABLE,"input.tileir"))
        other_executable = replace(unknown,executable="unknown-worker",compiler_identity_verified=False)
        for samples in ((self.sample,work,self.sample),(unknown,self.sample),
                        (self.sample,other_executable,self.sample)):
            with self.subTest(samples=samples):
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy); owner.register(self.root)
                for i,p in enumerate(samples):owner.observe({20:self.root,30:p},100+i)
                owner.finalize_metadata()
                self.assertEqual(owner.report()["timing_metadata_only_identities"],[])
                self.assertIsNotNone(owner.report()["timing_classification_history"][0]["first_unknown_or_work_sample"])
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):owner.require_compiler_quiescence()

    def test_initial_zombie_rejects_but_matching_terminal_preserves_prior_evidence(self):
        zombie = replace(self.sample,state="Z",rss_bytes=0,compiler_argv=None,
                         compiler_executable=None,compiler_file_version=None)
        self.observe(zombie)
        with self.assertRaisesRegex(RuntimeError,"compiler activity"):self.admit()
        owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy); owner.register(self.root)
        owner.observe({20:self.root,30:self.sample},100)
        owner.observe({20:self.root,30:zombie},101); owner.finalize_metadata(); owner.require_compiler_quiescence()
        self.assertEqual(owner.report()["timing_metadata_only_identities"],[(30,30)])

    def test_terminal_work_or_conflicting_evidence_cannot_erase_prior_help(self):
        for change in ({"compiler_argv":(self.policy.EXECUTABLE,"input.tileir")},
                       {"compiler_argv":(self.policy.EXECUTABLE,"--help","")},
                       {"executable":"ptxas"}, {"compiler_executable":"/other/tileiras"},
                       {"compiler_file_version":tuple([*self.policy.binary_version[:-1],0])},
                       {"compiler_identity_verified":False}):
            with self.subTest(change=change):
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy);owner.register(self.root)
                owner.observe({20:self.root,30:self.sample},100)
                owner.observe({20:self.root,30:replace(self.sample,state="Z",rss_bytes=0,**change)},101)
                owner.observe({20:self.root,30:self.sample},102)
                owner.finalize_metadata()
                self.assertEqual(owner.report()["timing_metadata_only_identities"],[])
                self.assertIsNotNone(owner.report()["timing_classification_history"][0]["first_unknown_or_work_sample"])
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):owner.require_compiler_quiescence()

    def test_proc_cmdline_preserves_empty_arguments_and_requires_termination(self):
        self.assertEqual(decode_cmdline(b"tool\0--help\0"),("tool","--help"))
        self.assertEqual(decode_cmdline(b"tool\0--help\0\0"),("tool","--help",""))
        self.assertEqual(decode_cmdline(b"tool\0--help\0\0\0"),("tool","--help","",""))
        self.assertEqual(decode_cmdline(b"tool\0\0--help\0"),("tool","","--help"))
        self.assertIsNone(decode_cmdline(b""))
        self.assertIsNone(decode_cmdline(b"tool\0--help"))

    def test_reused_pid_terminal_or_missing_identity_cannot_inherit_help_evidence(self):
        self.observe(self.sample)
        for sample in (replace(self.sample,start_ticks=31,state="Z",compiler_argv=None),
                       replace(self.sample,state="Z",compiler_identity_verified=False)):
            with self.subTest(sample=sample):
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy); owner.register(self.root)
                owner.observe({20:self.root,30:self.sample},100)
                owner.observe({20:self.root,30:sample},101); owner.finalize_metadata()
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):owner.require_compiler_quiescence()

    def test_descendant_never_exempt_even_when_its_own_help_is_verified(self):
        child = replace(self.sample,pid=40,start_ticks=40,ppid=30)
        self.observe(self.sample,child)
        with self.assertRaisesRegex(RuntimeError,"compiler activity"):self.admit()
        self.assertIn((40,40),self.owner.non_metadata_identities)

    def test_same_inode_content_drift_with_restored_mtime_rejects(self):
        for path in (self.binary,self.caller):
            with self.subTest(path=path):
                policy = MetadataHelpTiming(self.caller.parents[1],"a"*40)
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=policy); owner.register(self.root)
                sample = replace(self.sample,compiler_file_version=policy.binary_version)
                original = path.read_bytes(); before = path.stat()
                # Linux cached filesystem time can coalesce rapid writes. The
                # version-drift case needs a distinct timestamp; independent
                # final-hash coverage below covers unchanged stat evidence.
                time.sleep(.05)
                path.write_bytes(b"X" + original[1:])
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
                self.assertEqual(path.stat().st_ino,before.st_ino)
                self.assertEqual(path.stat().st_size,before.st_size)
                self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)
                self.assertNotEqual(path.stat().st_ctime_ns,before.st_ctime_ns)
                owner.observe({20:self.root,30:sample},100)
                self.assertIn("file-version drift",owner.failure)
                final = owner.finalize_metadata()["final"]
                self.assertFalse(final["passed"])
                target = "binary" if path == self.binary else "caller"
                self.assertTrue(any(target + " hash/version differs" in e for e in final["errors"]))
                with self.assertRaises(RuntimeError):owner.require_compiler_quiescence()
                path.write_bytes(original)

    def test_final_hash_catches_content_mismatch_even_with_stale_stat_evidence(self):
        self.observe(self.sample); self.binary.write_bytes(b"X" + self.binary.read_bytes()[1:])
        # Even if version telemetry were substituted/stale, the untimed final
        # bytes must still reject; no real file timestamp manipulation claimed.
        with patch.object(self.policy,"version_error",return_value=None), \
             patch("m1_owned_processes.file_version",return_value=self.policy.binary_version):
            result = self.owner.finalize_metadata()
        self.assertFalse(result["final"]["passed"])
        self.assertIn("binary hash/version",self.owner.failure)

    def test_missing_final_verification_rejects(self):
        self.observe(self.sample)
        with self.assertRaisesRegex(RuntimeError,"final file verification"):
            self.owner.require_compiler_quiescence()

    def test_metadata_resource_breach_still_latches_rss_and_shared_time(self):
        for change in ({"rss_bytes":(2 << 30)+1},{"age_seconds":301}):
            with self.subTest(change=change):
                owner = OwnedProcesses(Process(10,10,1),metadata_timing=self.policy); owner.register(self.root)
                owner.observe({20:self.root,30:replace(self.sample,**change)},100)
                self.assertIn("Owned",owner.failure)
                self.assertEqual(owner.report()["timing_metadata_only_identities"],[(30,30)])

    def test_preflight_hash_and_alias_drift_fail_before_launch(self):
        for field in ("BINARY_SHA","CALLER_SHA","RESOLVED"):
            with self.subTest(field=field), patch.object(MetadataHelpTiming,field,"wrong"),self.assertRaises(RuntimeError):
                MetadataHelpTiming(self.caller.parents[1],"a"*40)

    def test_proc_evidence_rechecks_pid_start_executable_version_and_argv(self):
        fields = ["S","20"]+["0"]*21;fields[19]="30";fields[21]="1"
        original = "30 (tileiras) " + " ".join(fields)
        replaced = original.replace("30 (", "31 (")
        changed_start = fields.copy();changed_start[19]="300"
        variants = ((original,self.policy.RESOLVED,self.binary.stat(),b"/bin/tileiras\0--help\0",True),
                    (replaced,self.policy.RESOLVED,self.binary.stat(),b"/bin/tileiras\0--help\0",False),
                    ("30 (tileiras) " + " ".join(changed_start),self.policy.RESOLVED,self.binary.stat(),b"/bin/tileiras\0--help\0",False),
                    (original,"/different/tileiras",self.binary.stat(),b"/bin/tileiras\0--help\0",False),
                    (original,self.policy.RESOLVED,self.caller.stat(),b"/bin/tileiras\0--help\0",False),
                    (original,self.policy.RESOLVED,self.binary.stat(),b"/bin/tileiras\0input.tileir\0",False))
        for later,path,version,args,expected in variants:
            with self.subTest(expected=expected,path=path), \
                 patch.object(Path,"read_text",side_effect=[original,later]), \
                 patch("m1_owned_processes.os.readlink",side_effect=[self.policy.RESOLVED,path]), \
                 patch.object(Path,"stat",side_effect=[self.binary.stat(),version]), \
                 patch.object(Path,"read_bytes",side_effect=[b"/bin/tileiras\0--help\0",args]):
                observed = read_process(30,uptime=100)
                self.assertEqual(observed.compiler_identity_verified,expected)

    def test_unavailable_proc_evidence_stays_unknown(self):
        fields = ["S","20"]+["0"]*21;fields[19]="30";fields[21]="1"
        original = "30 (tileiras) " + " ".join(fields)
        with patch.object(Path,"read_text",side_effect=[original,original]), \
             patch("m1_owned_processes.os.readlink",side_effect=FileNotFoundError()), \
             patch.object(Path,"stat",side_effect=FileNotFoundError()), \
             patch.object(Path,"read_bytes",side_effect=PermissionError()):
            observed = read_process(30,uptime=100)
        self.assertFalse(observed.compiler_identity_verified)
        self.assertIsNone(observed.compiler_argv)
        self.assertFalse(self.policy.qualifies(observed))
