"""CPU mode/admission checks; no GPU package, process or request is started."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from megartx.m1_execution import (execution_mode, profile_scope,
                                  synchronous_scheduler_args)
from megartx.m1_live import load_controller


FREE_ENV = {"MEGARTX_M1_EXECUTION": "capture-free", "MEGARTX_M1_PREPARATION": "fused",
            "MEGARTX_SCALE_MODE": "native", "MEGARTX_CONTROLLED_DIR": "controlled",
            "MEGARTX_CONTROLLED_PLAN": "plan", "MEGARTX_LOGITS_DIR": "logits"}


class ExecutionPolicyTests(unittest.TestCase):
    def test_every_m1_cli_lane_forces_synchronous_scheduling(self):
        lanes = (
            ("normal", "stock", "captured", False),
            ("normal", "fused", "captured", False),
            ("controlled", "stock", "captured", False),
            ("controlled", "fused", "captured", False),
            ("controlled", "stock", "capture-free", False),
            ("controlled", "fused", "capture-free", False),
            ("controlled", "stock", "capture-free", True),
            ("controlled", "fused", "capture-free", True),
        )
        for client, lane, execution, observer in lanes:
            with self.subTest(client=client, lane=lane, execution=execution,
                              observer=observer):
                self.assertEqual(synchronous_scheduler_args(SimpleNamespace(
                    client=client, m1_preparation=lane, m1_execution=execution,
                    m1_external_observer=observer)),
                                 ["--no-async-scheduling"])
        self.assertEqual(synchronous_scheduler_args(SimpleNamespace(
            client="controlled", m1_preparation=None, m1_execution="captured",
            m1_external_observer=False)), [])

    def test_absent_toggle_keeps_captured_lane(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(execution_mode(), "captured")

    def test_explicit_stock_and_fused_are_independent_of_diagnostics(self):
        for lane in ("stock", "fused"):
            with patch.dict(os.environ, {**FREE_ENV, "MEGARTX_M1_PREPARATION": lane}, clear=True), \
                 patch("megartx.m1_live.LivePreparation") as controller:
                load_controller("native")
                controller.assert_called_once_with(lane)

    def test_free_cannot_bypass_optin_native_scope_or_controlled_binding(self):
        for key in FREE_ENV:
            if key == "MEGARTX_M1_EXECUTION": continue
            env = dict(FREE_ENV);del env[key]
            with self.subTest(key=key), patch.dict(os.environ, env, clear=True), \
                 patch("megartx.m1_live.LivePreparation") as controller, self.assertRaises(RuntimeError):
                load_controller("native")
            controller.assert_not_called()

    def test_unknown_and_mixed_diagnostics_fail_closed(self):
        changes = ({"MEGARTX_M1_EXECUTION": "0"}, {"MEGARTX_M1_ROUTE_CONTROLS": "1"},
                   {"MEGARTX_LAYER0_BOUNDARIES": "1"}, {"MEGARTX_M1_CAPTURE_DIR": "prep"},
                   {"MEGARTX_ROUTE_AUDIT_PATH": "audit"}, {"MEGARTX_ROUTING_COVERAGE_PATH": "coverage"},
                   {"MEGARTX_ROUTER_SCORE_DIR": "router"}, {"MEGARTX_SCALE_MODE": "reference"})
        for change in changes:
                with self.subTest(change=change), patch.dict(os.environ, {**FREE_ENV, **change}, clear=True), \
                     self.assertRaises(RuntimeError): execution_mode()

    def test_external_observer_is_an_explicit_capture_free_only_opt_in(self):
        with patch.dict(os.environ, {**FREE_ENV, "MEGARTX_M1_EXTERNAL_OBSERVER_DIR": "/tmp/observer"}, clear=True):
            self.assertEqual(execution_mode(), "capture-free")
        with patch.dict(os.environ, {"MEGARTX_M1_EXTERNAL_OBSERVER_DIR": "/tmp/observer"}, clear=True), \
             self.assertRaisesRegex(RuntimeError, "capture-free"):
            execution_mode()
        with patch.dict(os.environ, {**FREE_ENV, "MEGARTX_M1_EXTERNAL_OBSERVER_DIR": "/tmp/observer",
                                     "MEGARTX_M1_ROUTE_CONTROLS": "1"}, clear=True), \
             self.assertRaisesRegex(RuntimeError, "route controls"):
            execution_mode()

    def test_disabled_scope_does_not_import_torch(self):
        code = ("import sys; from megartx.m1_execution import profile_scope; "
                "scope=profile_scope('unused',False); scope.__enter__(); scope.__exit__(None,None,None); "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    def test_cli_rejects_inconsistent_modes_before_any_lifecycle_work(self):
        runner = str(Path(__file__).resolve().parents[1] / "scripts/run_scale_validation.py")
        # The scaffold CI intentionally installs no optional HTTP dependency.
        # These rejected invocations cannot reach any HTTP or GPU lifecycle API.
        code = ("import runpy,sys,types; sys.modules['requests']=types.SimpleNamespace(); "
                "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
        common = [sys.executable, "-c", code, runner, "--label", "cpu-only-rejection", "--mode", "native"]
        for args, message in ((["--m1-execution", "capture-free"], "capture-free execution is limited"),
                (["--m1-preparation", "fused", "--m1-execution", "capture-free"], "bounded native"),
                (["--client", "controlled", "--controlled-plan", "unused", "--controlled-path", "cached",
                  "--m1-preparation", "stock", "--m1-bridge", "unused", "--m1-build-receipt", "unused",
                  "--m1-execution", "capture-free", "--m1-route-controls"], "route controls"),
                (["--client", "controlled", "--controlled-plan", "unused", "--controlled-path", "cached",
                  "--m1-preparation", "stock", "--m1-bridge", "unused", "--m1-build-receipt", "unused",
                  "--m1-external-observer"], "external observer requires one native capture-free"),
                (["--client", "benchmark"], "Timing is fail-closed")):
            with self.subTest(args=args):
                result = subprocess.run(common + args, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


if __name__ == "__main__": unittest.main()
