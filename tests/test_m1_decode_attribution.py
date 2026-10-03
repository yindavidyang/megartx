"""Bounded profiler policy and failure cleanup, without CUDA or model loads."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from megartx.m1_live import DecodeAttribution, LivePreparation
from megartx.m1_eager_benchmark import EagerBenchmark, marker
from megartx.m1_execution import profile_scope


class Function:
    def __init__(self):
        self.calls = []
    def __call__(self, enabled, counters, count):
        self.calls.append(enabled)
        for i in range(count):
            counters[i] = 0 if enabled else i + 1
        return int(not enabled and getattr(self, "fail_release", False))


class AttributionTests(unittest.TestCase):
    def fixture(self, directory):
        b = SimpleNamespace(index=0, frame=0, drained=False, plan={
            "warmups": 1, "trials": 1, "source_head": "a" * 40, "plan_sha256": "b" * 64,
            "schedule": [{"id": "warm", "phase": "warmup", "case": "2048", "lane": "stock"},
                         {"id": "stock", "phase": "measurement", "case": "2048", "lane": "stock"},
                         {"id": "fused", "phase": "measurement", "case": "2048", "lane": "fused"},
                         {"id": "8k", "phase": "measurement", "case": "8192", "lane": "stock"}]})
        native = SimpleNamespace(megartx_m1_attribution_v1=Function())
        profiler = Mock()
        profiler.export_chrome_trace.side_effect = lambda p: Path(p).write_text('{}')
        torch = SimpleNamespace(profiler=SimpleNamespace(profile=Mock(return_value=profiler),
                                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")))
        return DecodeAttribution(b, native, Path(directory)/"profile"), b, native, profiler, torch

    def test_four_complete_steps_each_lane_and_no_warmup_prefill_or_8k(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, native, p, torch = self.fixture(d)
            token = SimpleNamespace(numel=lambda: 1)
            with patch.dict(sys.modules, {"torch": torch}):
                a.before_frame(token)
                self.assertFalse(a.directory.exists())
                for index in (1, 2):
                    b.index, b.frame = index, 7
                    a.before_frame(token)
                    self.assertFalse(a.active)
                    for frame in range(8, 12):
                        b.frame = frame
                        a.before_frame(token)
                        self.assertTrue(a.active)
                        self.assertEqual(p.stop.call_count, index-1)
                    b.frame = 12
                    a.before_frame(token)
                    self.assertFalse(a.active)
                    data = json.loads((a.directory/(b.plan["schedule"][index]["lane"]+"-scalars.json")).read_text())
                    self.assertEqual(data["decode_steps"], 4)
                    self.assertFalse(data["timing_qualified"])
                b.index, b.frame = 3, 32
                a.before_frame(token)
                a.require_complete()
                self.assertEqual(native.megartx_m1_attribution_v1.calls, [1, 0, 1, 0])
                self.assertEqual(torch.profiler.profile.call_count, 2)
                self.assertFalse(torch.profiler.profile.call_args.kwargs["record_shapes"])
                self.assertEqual(p.start.call_count, 2)
                self.assertEqual(p.stop.call_count, 2)

    def test_profiler_start_failure_releases_native_and_incomplete_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, native, p, torch = self.fixture(d)
            b.index, b.frame = 1, 8
            p.start.side_effect = RuntimeError("start fault")
            with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(RuntimeError, "start fault"):
                a.before_frame(SimpleNamespace(numel=lambda: 1))
            self.assertEqual(native.megartx_m1_attribution_v1.calls, [1, 0])
            self.assertFalse(a.active)
            with self.assertRaisesRegex(RuntimeError, "complete"):
                a.require_complete()

    def test_abort_releases_native_even_when_profiler_stop_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, native, p, torch = self.fixture(d)
            b.index, b.frame = 1, 8
            with patch.dict(sys.modules, {"torch": torch}):
                a.before_frame(SimpleNamespace(numel=lambda: 1))
            p.stop.side_effect = RuntimeError("stop fault")
            with self.assertRaisesRegex(RuntimeError, "stop fault"):
                a.abort()
            self.assertFalse(a.active)
            self.assertEqual(native.megartx_m1_attribution_v1.calls, [1, 0])
            self.assertEqual(a.completed, set())
            self.assertFalse((a.directory/"stock-scalars.json").exists())

    def test_profile_artifacts_or_manifest_block_summary_before_reading_timing(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from m1_eager_benchmark_client import summarize_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"launch-manifest.json").write_text(json.dumps({"m1_decode_profile_requested": True}))
            with self.assertRaisesRegex(RuntimeError, "instrumented"):
                summarize_run(root)
            (root/"launch-manifest.json").unlink()
            (root/"decode-profile").mkdir()
            with self.assertRaisesRegex(RuntimeError, "instrumented"):
                summarize_run(root)

    def test_actual_model_ledger_and_logits_errors_abort_without_retry_or_error_replacement(self):
        from test_m1_eager_benchmark import plan
        path = Path(__file__).resolve().parents[1]/"src/megartx/vllm_scale_plugin.py"
        for case in ("ledger", "head"):
            for cleanup_fails in ("none", "stop", "reset", "both"):
                with self.subTest(case=case, cleanup_fails=cleanup_fails), tempfile.TemporaryDirectory() as d:
                    a, unused, native, p, torch = self.fixture(d)
                    torch.profiler.record_function = lambda *args: nullcontext()
                    b = EagerBenchmark(plan(), Path(d)/"ledger", "stock")
                    b.index = next(i for i, r in enumerate(b.plan["schedule"])
                                   if r["phase"] == "measurement" and r["case"] == "2048")
                    b.frame = 8
                    a.benchmark = b
                    with patch.dict(sys.modules, {"torch": torch}):
                        a.before_frame(SimpleNamespace(numel=lambda: 1))
                    c = LivePreparation.__new__(LivePreparation)
                    c.native, c.attribution, c.benchmark, c.failed = native, a, b, False
                    c.forward = c.layer = None
                    native.megartx_m1_active = lambda: 0
                    c.begin_forward = lambda *args: None
                    env = {"m1": c, "profile_scope": profile_scope, "capture": None,
                           "counter": 0, "context": None}
                    if case == "ledger":
                        b.begin(marker(b.plan, b.plan["schedule"][b.index]), [7], [2048])
                        layer = SimpleNamespace(_megartx={"dispatch_calls": 0})
                        def forward(*args):
                            layer._megartx["dispatch_calls"] += 1
                            return object()
                        old = Mock(side_effect=forward)
                        env.update(integration_layers=[layer], integration_report={"natural_model_forward_verified": True},
                                   router_observer=None, controlled_observer=None, normal_observer=None, old_forward=old)
                        name, args, message = "model_forward", (None, None, None), "bypassed native dispatch"
                    else:
                        primary = RuntimeError("head fault")
                        old = Mock(side_effect=primary)
                        env["old_logits"] = old
                        name, args, message = "logits_forward", (None, None), "head fault"
                    function = next(n for n in ast.walk(ast.parse(path.read_text()))
                                    if isinstance(n, ast.FunctionDef) and n.name == name)
                    # Equivalent globals supply the production closure's nonlocals.
                    function.body = [ast.Global(names=n.names) if isinstance(n, ast.Nonlocal) else n for n in function.body]
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), env)
                    if cleanup_fails in {"stop", "both"}:
                        p.stop.side_effect = RuntimeError("cleanup stop fault")
                    if cleanup_fails in {"reset", "both"}:
                        native.megartx_m1_attribution_v1.fail_release = True
                    with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(RuntimeError, message) as caught:
                        env[name](*args)
                    if case == "head":
                        self.assertIs(caught.exception, primary)
                    if cleanup_fails != "none" and hasattr(caught.exception, "__notes__"):
                        self.assertIn("cleanup", str(caught.exception.__notes__).lower())
                    self.assertTrue(c.failed)
                    self.assertFalse(a.active)
                    self.assertEqual(p.stop.call_count, 1)
                    self.assertEqual(native.megartx_m1_attribution_v1.calls, [1, 0])
                    self.assertEqual(old.call_count, 1)
                    self.assertEqual(a.completed, set())
                    self.assertFalse((a.directory/(a.lane+"-scalars.json")).exists())
