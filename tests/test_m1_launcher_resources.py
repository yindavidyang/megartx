"""Execute real launcher admission/cleanup blocks with a stalled watchdog.

Only external processes, GPU queries and /proc snapshots are substituted.
The launcher control flow and retained ownership implementation are unchanged.
"""
import ast
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m1_owned_processes import OwnedProcesses, Process, read_process
from test_m1_timing_metadata import metadata_fixture

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_scale_validation.py"
TREE = ast.parse(LAUNCHER.read_text())
RUN = next(n for n in TREE.body if isinstance(n, ast.Try) and n.finalbody)


def code(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(LAUNCHER), "exec")


def index(fragment):
    return next(i for i, node in enumerate(RUN.body) if fragment in ast.unparse(node))


class LauncherResourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.owner = OwnedProcesses(Process(10, 10, 1))
        self.root = Process(20, 20, 10)
        self.owner.register(self.root)
        self.summary = Mock()
        self.client_run = Mock(return_value=SimpleNamespace(returncode=0))
        self.events = []
        self.env = {"ownership":self.owner,"guard_failure":None,"time":time,"os":__import__("os"),
                    "sys":sys,"signal":signal,"json":json,"output":self.output,
                    "eager_benchmark":True,"args":SimpleNamespace(client="m1-eager-benchmark",
                    m1_external_observer=False,m1_eager_benchmark_plan=Path("private-plan"),
                    profile=False,m1_decode_profile=False,activation_only=False,routing_diagnostic=False,router_score_only=False),
                    "base":Path("private-runtime"),"project":LAUNCHER.parents[1],"env":{},
                    "benchmark_plan":{"plan_sha256":"p"},"result":SimpleNamespace(returncode=0),
                    "server":SimpleNamespace(pid=20,returncode=0,poll=lambda:0),
                    "gpu_jobs":lambda:[],"phase":lambda *a, **k:self.events.append((a,k)),
                    "phases":SimpleNamespace(close=Mock()),"stop_guard":SimpleNamespace(set=Mock()),
                    "stop_sample":SimpleNamespace(set=Mock()),"guard_thread":None,
                    "sample_thread":SimpleNamespace(ident=None),
                    "subprocess":SimpleNamespace(run=self.client_run,STDOUT=subprocess.STDOUT)}
        definitions = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "require_resources"]
        exec(code(definitions), self.env)

    def test_dispatch_rejects_authoritative_rss_or_shared_time_breach_without_watchdog(self):
        # Include the actual subprocess client dispatch, not only a helper.
        nodes = RUN.body[index("other = []"):index("benchmark.exit")]
        for rss, age in (((2 << 30) + 1, 0), (1, 301)):
            with self.subTest(rss=rss,age=age):
                owner = OwnedProcesses(Process(10,10,1)); owner.register(self.root)
                self.env["ownership"] = owner
                compiler = Process(30,30,20,rss,"nvcc",age_seconds=age)
                with patch("m1_owned_processes.snapshot", return_value={20:self.root,30:compiler}), \
                     self.assertRaisesRegex(RuntimeError, "Owned .*compiler"):
                    exec(code(nodes), self.env)
                self.assertIsNone(self.env["guard_failure"])
                self.assertIsNotNone(owner.failure)
                self.client_run.assert_not_called()
        self.assertFalse(any(a[0] == "client_launch" for a,k in self.events))

    def test_post_client_rejects_retained_breach_before_completion(self):
        self.owner.fail("Owned compiler breach with stalled watchdog")
        nodes = RUN.body[index("benchmark.exit"):]
        with self.assertRaisesRegex(RuntimeError, "compiler breach"):
            exec(code([ast.Try(body=nodes,handlers=RUN.handlers,orelse=[],finalbody=[])]), self.env)
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.assertFalse(any(a[0] == "bounded_eager_benchmark_complete" for a,k in self.events))

    def final(self, snapshots, primary=None, signal_error=None):
        (self.output / "run.exit").write_text("0\n" if primary is None else "1\n")
        with patch("m1_owned_processes.snapshot", side_effect=snapshots), \
             patch("m1_owned_processes.signal_identity", side_effect=signal_error), patch.object(signal,"signal"), \
             patch("m1_eager_benchmark_client.summarize_run",self.summary):
            if primary is None:
                exec(code(RUN.finalbody), self.env)
            else:
                self.env["primary"] = primary
                body = [ast.Raise(exc=ast.Name(id="primary",ctx=ast.Load()),cause=None)]
                wrapped = ast.fix_missing_locations(ast.Try(body=body,handlers=[],orelse=[],finalbody=RUN.finalbody))
                exec(code([wrapped]), self.env)

    def test_cleanup_first_breach_preserves_clean_cleanup_but_rejects_final_admission(self):
        compiler = Process(30,30,20,(2 << 30) + 1,"nvcc")
        with self.assertRaisesRegex(RuntimeError, "aggregate RSS"):
            self.final([{20:self.root,30:compiler},{},{}])
        cleanup = json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertTrue(cleanup["cleanup_complete"])
        self.assertEqual(cleanup["owned_identities_remaining"], [])
        self.assertIn("aggregate RSS", cleanup["failure"])
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.assertIsNone(self.env["guard_failure"])
        self.summary.assert_not_called()

    def test_actual_final_profile_path_requires_both_windows_and_never_summarizes(self):
        self.env["args"].m1_decode_profile = True
        self.env["benchmark_plan"]["source_head"] = "a" * 40
        profile = self.output / "decode-profile"
        profile.mkdir()
        for lane in ("stock", "fused"):
            (profile / (lane + "-scalars.json")).write_text(json.dumps({
                "lane": lane, "decode_steps": 4, "source_head": "a" * 40,
                "plan_sha256": "p", "timing_qualified": False}))
            (profile / (lane + ".json")).write_text('{}')
        self.final([{}, {}, {}])
        self.summary.assert_not_called()
        (profile / "fused.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "window/source"):
            self.final([{}, {}, {}])
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.summary.assert_not_called()

    def test_clean_final_admission_still_summarizes(self):
        self.final([{}, {}, {}])
        self.summary.assert_called_once_with(self.output)
        self.assertEqual((self.output / "run.exit").read_text(), "0\n")

    def test_below_bound_compiler_activity_preserves_cleanup_but_rejects_timing(self):
        for name in ("tileiras", "nvcc"):
            with self.subTest(name=name):
                self.owner = OwnedProcesses(Process(10,10,1)); self.owner.register(self.root)
                self.env["ownership"] = self.owner
                compiler = Process(30,30,20,1024,name,age_seconds=.01)
                with self.assertRaisesRegex(RuntimeError, "compiler activity"):
                    self.final([{20:self.root,30:compiler},{},{}])
                cleanup = json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
                self.assertTrue(cleanup["cleanup_complete"])
                self.assertIsNone(cleanup["failure"])
                self.assertEqual((self.output / "run.exit").read_text(), "1\n")
                self.summary.assert_not_called()

    def test_unknown_cleanup_error_rejects_summary_even_when_all_identities_disappear(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup did not complete"):
            self.final([{20:self.root},{},{},{}], signal_error=OSError(22,"unknown cleanup fault"))
        cleanup = json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertFalse(cleanup["cleanup_complete"])
        self.assertEqual(cleanup["owned_identities_remaining"], [])
        self.assertEqual(cleanup["cleanup_error_details"][0]["errno"], 22)
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.summary.assert_not_called()

    def metadata_owner(self):
        policy,sample,binary,caller = metadata_fixture(self)
        self.owner = OwnedProcesses(Process(10,10,1),metadata_timing=policy)
        self.owner.register(self.root); self.env["ownership"] = self.owner
        return policy,sample,binary

    def test_verified_metadata_actual_final_path_hashes_and_admits(self):
        policy,sample,binary = self.metadata_owner()
        self.owner.observe({20:self.root,30:sample},100)
        self.final([{},{},{}])
        self.summary.assert_called_once_with(self.output)
        final = json.loads((self.output / "timing-metadata-final.json").read_text())
        self.assertTrue(final["final"]["passed"])
        cleanup = json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertTrue(cleanup["cleanup_complete"])
        self.assertEqual(cleanup["sampled_peak_compiler_rss_bytes"],10)

    def test_same_identity_unknown_or_work_history_rejects_actual_final_path(self):
        for work_first in (False,True):
            with self.subTest(work_first=work_first):
                policy,sample,binary = self.metadata_owner()
                unknown = replace(sample,compiler_argv=None)
                work = replace(sample,compiler_argv=(policy.EXECUTABLE,"input.tileir"))
                for p in ((unknown,sample) if work_first else (sample,work,sample)):
                    self.owner.observe({20:self.root,30:p},100)
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):
                    self.final([{},{},{}])
                self.summary.assert_not_called()
                self.assertEqual((self.output / "run.exit").read_text(),"1\n")

    def test_actual_reader_terminal_changes_and_empty_argument_reject_final_admission(self):
        # The real /proc reader feeds the real launcher final block. External
        # process reads/cleanup are substituted; no tool or GPU is executed.
        for kind in ("work_at_exit","executable_at_exit","trailing_empty_argument",
                     "argv_recheck_at_exit","executable_recheck_at_exit",
                     "unreadable_recheck_at_exit","malformed_argv_at_exit",
                     "empty_argument_recheck_at_exit"):
            with self.subTest(kind=kind):
                policy,sample,binary = self.metadata_owner()
                if kind != "trailing_empty_argument":self.owner.observe({20:self.root,30:sample},100)
                other=binary.parent/"ptxas";other.write_bytes(b"different CPU compiler fixture")
                first_path=str(other if kind=="executable_at_exit" else binary)
                last_path=str(other if kind=="executable_recheck_at_exit" else first_path)
                first_args=(first_path,"input.tileir") if kind in {"work_at_exit","executable_at_exit"} else (first_path,"--help")
                if kind=="trailing_empty_argument":first_args+= ("",)
                last_args=(first_path,"input.tileir") if kind=="argv_recheck_at_exit" else first_args
                if kind=="empty_argument_recheck_at_exit":last_args+= ("",)
                encode=lambda args:b"\0".join(a.encode() for a in args)+b"\0"
                first_bytes=encode(first_args);last_bytes=encode(last_args)
                if kind=="malformed_argv_at_exit":first_bytes=first_bytes[:-1]
                if kind=="unreadable_recheck_at_exit":last_bytes=PermissionError("CPU fixture unreadable cmdline")
                fields=["S","20"]+["0"]*21;fields[19]="30";fields[21]="1"
                before="30 (tileiras) "+" ".join(fields)
                fields[0]="S" if kind=="trailing_empty_argument" else "Z"
                after="30 (tileiras) "+" ".join(fields)
                with patch.object(Path,"read_text",side_effect=[before,after]), \
                     patch("m1_owned_processes.os.readlink",side_effect=[first_path,last_path]), \
                     patch.object(Path,"stat",side_effect=[Path(first_path).stat(),Path(last_path).stat()]), \
                     patch.object(Path,"read_bytes",side_effect=[first_bytes,last_bytes]):
                    observed=read_process(30,uptime=100)
                if kind=="trailing_empty_argument":self.assertEqual(observed.compiler_argv,first_args)
                self.owner.observe({20:self.root,30:observed},101)
                with self.assertRaisesRegex(RuntimeError,"compiler activity"):self.final([{},{},{}])
                self.summary.assert_not_called()
                self.assertEqual((self.output/"run.exit").read_text(),"1\n")
                cleanup=json.loads((self.output/"eager-benchmark-cleanup.json").read_text())
                self.assertTrue(cleanup["cleanup_complete"])
                self.assertEqual(cleanup["timing_unknown_or_work_identities"],[[30,30]])
                self.assertIsNotNone(cleanup["timing_classification_history"][0]["first_unknown_or_work_sample"])

    def test_actual_reader_verified_help_then_information_loss_still_admits(self):
        policy,sample,binary=self.metadata_owner()
        self.owner.observe({20:self.root,30:sample},100)
        fields=["S","20"]+["0"]*21;fields[19]="30";fields[21]="1"
        before="30 (tileiras) "+" ".join(fields);fields[0]="Z"
        after="30 (tileiras) "+" ".join(fields)
        with patch.object(Path,"read_text",side_effect=[before,after]), \
             patch("m1_owned_processes.os.readlink",side_effect=FileNotFoundError()), \
             patch.object(Path,"stat",side_effect=FileNotFoundError()), \
             patch.object(Path,"read_bytes",return_value=b""):
            terminal=read_process(30,uptime=100)
        self.owner.observe({20:self.root,30:terminal},101)
        self.final([{},{},{}]);self.summary.assert_called_once_with(self.output)
        self.assertEqual((self.output/"run.exit").read_text(),"0\n")

    def test_metadata_resource_bounds_still_block_actual_client_dispatch(self):
        nodes = RUN.body[index("other = []"):index("benchmark.exit")]
        for change in ({"rss_bytes":(2 << 30)+1},{"age_seconds":301}):
            with self.subTest(change=change):
                policy,sample,binary = self.metadata_owner()
                sample = replace(sample,**change)
                with patch("m1_owned_processes.snapshot",return_value={20:self.root,30:sample}), \
                     self.assertRaisesRegex(RuntimeError,"Owned .*compiler"):
                    exec(code(nodes),self.env)
                self.client_run.assert_not_called()

    def test_final_file_drift_invalidates_timing_with_clean_cleanup_and_preserves_primary(self):
        policy,sample,binary = self.metadata_owner()
        self.owner.observe({20:self.root,30:sample},100)
        binary.write_bytes(b"X" + binary.read_bytes()[1:])
        with self.assertRaisesRegex(RuntimeError,"file-version drift|final binary hash/version differs"):
            self.final([{},{},{}])
        self.summary.assert_not_called()
        self.assertEqual((self.output / "run.exit").read_text(),"1\n")
        self.assertTrue(json.loads((self.output / "eager-benchmark-cleanup.json").read_text())["cleanup_complete"])
        self.assertFalse(json.loads((self.output / "timing-metadata-final.json").read_text())["final"]["passed"])
        primary = ValueError("primary model request failed")
        with self.assertRaises(ValueError) as raised:self.final([{},{},{}],primary)
        self.assertIs(raised.exception,primary)

    def test_cleanup_first_breach_preserves_existing_primary_exception(self):
        primary = ValueError("primary client failure")
        compiler = Process(30,30,20,(2 << 30) + 1,"nvcc")
        with self.assertRaises(ValueError) as raised:
            self.final([{20:self.root,30:compiler},{},{}],primary)
        self.assertIs(raised.exception,primary)
        self.assertTrue(json.loads((self.output / "eager-benchmark-cleanup.json").read_text())["cleanup_complete"])
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.summary.assert_not_called()
