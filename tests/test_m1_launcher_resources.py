"""Execute real launcher admission/cleanup blocks with a stalled watchdog.

Only external processes, GPU queries and /proc snapshots are substituted.
The launcher control flow and retained ownership implementation are unchanged.
"""
import ast
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
from m1_owned_processes import OwnedProcesses, Process

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
                    profile=False,activation_only=False,routing_diagnostic=False,router_score_only=False),
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

    def final(self, snapshots, primary=None):
        (self.output / "run.exit").write_text("0\n" if primary is None else "1\n")
        with patch("m1_owned_processes.snapshot", side_effect=snapshots), \
             patch("m1_owned_processes.signal_identity"), patch.object(signal,"signal"), \
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

    def test_clean_final_admission_still_summarizes(self):
        self.final([{}, {}, {}])
        self.summary.assert_called_once_with(self.output)
        self.assertEqual((self.output / "run.exit").read_text(), "0\n")

    def test_cleanup_first_breach_preserves_existing_primary_exception(self):
        primary = ValueError("primary client failure")
        compiler = Process(30,30,20,(2 << 30) + 1,"nvcc")
        with self.assertRaises(ValueError) as raised:
            self.final([{20:self.root,30:compiler},{},{}],primary)
        self.assertIs(raised.exception,primary)
        self.assertTrue(json.loads((self.output / "eager-benchmark-cleanup.json").read_text())["cleanup_complete"])
        self.assertEqual((self.output / "run.exit").read_text(), "1\n")
        self.summary.assert_not_called()
