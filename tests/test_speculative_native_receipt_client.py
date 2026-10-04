"""CPU fault fixtures only; no fixture is an admitted native owner."""
import asyncio
from concurrent.futures import Future
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from types import ModuleType

from megartx.speculative_native_client import ReceiptSession, RECEIPT, RELEASE, EXPECTED_RELEASE, State
from megartx.speculative_native_compare import validate_scalar_receipt, verify_private_receipt, stream_digest
from megartx.speculative_native_evidence import PrivateEvidence, bounded_json_chunks, install_native_receipt_evidence
from megartx.speculative_native_plan import (PURPOSE, REVIEWED_LIFECYCLE, LIMITS, CLIENT_SOURCES,
    read_json, validate_authorization, validate_plan, engine_kwargs, engine_argv, environment)
from megartx.speculative_native_probe import ProbeError, REVISION, ADAPTER_FILES, source_manifest, allocation_lower_bound
from megartx.speculative_native_receipt import fit_decision
from megartx.speculative_native_receipt import FFI_SOURCES

ROOT = Path(__file__).resolve().parents[1]


def scalar(slack=0):
    allocation = allocation_lower_bound([100], 32, reject=False)
    return {"schema": "megartx-native-zero-forward-receipt-v1", "receipt_sha256": "a" * 64,
            "diagnostic_target_forwards": 0, "existing_startup_target_forwards": 2,
            "existing_forced_expert_fixture_pairs": 6, "existing_forced_correction_rows": 12,
            "original_native_head_dtype": "torch.bfloat16", "allocation_lower_bound": allocation,
            "gpu_free_bytes": 3 << 30, "host_free_bytes": 10 << 30,
            "decision": fit_decision(allocation, allocated=1000, reserved=1000 + slack,
                                      gpu_free=3 << 30, host_free=10 << 30)}


class SyncFixture:
    def __init__(self, fail=None, pause=None, result=None, release=None, shutdown_error=False):
        self.fail, self.pause = fail, pause
        self.result = scalar() if result is None else result
        self.release = EXPECTED_RELEASE.copy() if release is None else release
        self.calls, self.shutdowns, self.shutdown_error = [], [], shutdown_error

    def call_utility(self, name, *args):
        self.calls.append((name, args))
        if self.fail == name:
            raise TimeoutError("fixture uncertainty")
        if name == "pause_scheduler":
            return self.pause
        if name == RECEIPT:
            return self.result
        if name == RELEASE:
            return self.release
        raise AssertionError("Forbidden utility")

    def shutdown(self, timeout=None):
        self.shutdowns.append(timeout)
        if self.shutdown_error:
            raise OSError("fixture shutdown")


class AsyncFixture(SyncFixture):
    async def call_utility_async(self, name, *args):
        await asyncio.sleep(0)
        return self.call_utility(name, *args)


def session(client, resources=lambda: None):
    deadline = time.monotonic() + 120
    s = ReceiptSession(client, deadline=deadline, resources=resources)
    admission = {"phase": "zero_forward_receipt", "client_purpose": PURPOSE,
                 "client_plan_sha256": "b" * 64, "source_head": REVIEWED_LIFECYCLE,
                 "deadline_monotonic": deadline}
    return s, admission


class SyncLifecycleControls(unittest.TestCase):
    def test_success_exact_sequence_and_no_request_resume_probe(self):
        client = SyncFixture()
        s, admission = session(client)
        self.assertEqual(s.run_sync(admission), scalar())
        self.assertEqual(client.calls, [("pause_scheduler", ("keep", False)), (RECEIPT, (admission,)), (RELEASE, ())])
        self.assertEqual(len(client.shutdowns), 1)
        self.assertEqual(s.state, State.CLOSED)

    def test_completed_pause_future_before_receipt(self):
        future = Future()
        future.set_result(None)
        client = SyncFixture(pause=future)
        s, admission = session(client)
        s.run_sync(admission)
        self.assertEqual([name for name, _ in client.calls], ["pause_scheduler", RECEIPT, RELEASE])

    def test_failed_pause_future_never_receipt(self):
        future = Future()
        future.set_exception(RuntimeError("fixture pause"))
        client = SyncFixture(pause=future)
        s, admission = session(client)
        with self.assertRaises(RuntimeError):
            s.run_sync(admission)
        self.assertEqual([name for name, _ in client.calls], ["pause_scheduler"])
        self.assertEqual(len(client.shutdowns), 1)

    def test_failed_pause_does_not_reserve(self):
        client = SyncFixture(fail="pause_scheduler")
        s, admission = session(client)
        with self.assertRaises(TimeoutError):
            s.run_sync(admission)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(s.release_attempted)

    def test_partial_rpc_uncertainty_no_second_rpc_or_retry(self):
        client = SyncFixture(fail=RECEIPT)
        s, admission = session(client)
        with self.assertRaises(TimeoutError):
            s.run_sync(admission)
        self.assertEqual([name for name, _ in client.calls], ["pause_scheduler", RECEIPT])
        with self.assertRaisesRegex(ProbeError, "single-use"):
            s.run_sync(admission)
        self.assertEqual(len(client.shutdowns), 1)

    def test_uncertain_drain_is_one_attempt_then_shutdown(self):
        for release in ({"drained": False}, {"drained": True, "released": False, "scheduler_stays_paused": True},
                        {"drained": 1, "released": True, "scheduler_stays_paused": True}):
            with self.subTest(release=release):
                client = SyncFixture(release=release)
                s, admission = session(client)
                with self.assertRaises(ProbeError):
                    s.run_sync(admission)
                self.assertEqual(sum(name == RELEASE for name, _ in client.calls), 1)
                self.assertEqual(len(client.shutdowns), 1)

    def test_release_timeout_never_retried(self):
        client = SyncFixture(fail=RELEASE)
        s, admission = session(client)
        with self.assertRaises(TimeoutError):
            s.run_sync(admission)
        self.assertTrue(s.release_attempted)
        self.assertEqual(len(client.calls), 3)

    def test_wrong_purpose_malformed_source_close_without_dispatch(self):
        for field, value in (("client_purpose", "later_target_probe"), ("source_head", "bad"),
                             ("client_plan_sha256", None), ("phase", "probe")):
            with self.subTest(field=field):
                client = SyncFixture()
                s, admission = session(client)
                admission[field] = value
                with self.assertRaises(ProbeError):
                    s.run_sync(admission)
                self.assertEqual(client.calls, [])
                self.assertEqual(len(client.shutdowns), 1)

    def test_resource_failure_before_pause_still_shutdown(self):
        def fail():
            raise RuntimeError("fixture resource")
        client = SyncFixture()
        s, admission = session(client, fail)
        with self.assertRaises(RuntimeError):
            s.run_sync(admission)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(client.shutdowns), 1)

    def test_primary_error_survives_shutdown_failure(self):
        client = SyncFixture(fail=RECEIPT, shutdown_error=True)
        s, admission = session(client)
        with self.assertRaises(TimeoutError) as found:
            s.run_sync(admission)
        if hasattr(found.exception, "__notes__"):
            self.assertIn("OSError", found.exception.__notes__[0])
        self.assertEqual(s.state, State.UNCERTAIN)

    def test_duplicate_successful_receipt_rejected(self):
        client = SyncFixture()
        s, admission = session(client)
        s.run_sync(admission)
        with self.assertRaises(ProbeError):
            s.run_sync(admission)
        self.assertEqual(len(client.calls), 3)

    def test_deadline_reserves_cleanup_before_pause(self):
        client = SyncFixture()
        s, admission = session(client)
        s.deadline = time.monotonic() + 10
        admission["deadline_monotonic"] = s.deadline
        with self.assertRaises(TimeoutError):
            s.run_sync(admission)
        self.assertEqual(client.calls, [])


class AsyncLifecycleControls(unittest.IsolatedAsyncioTestCase):
    async def test_async_complete_pause_receipt_release(self):
        client = AsyncFixture()
        s, admission = session(client)
        await s.run_async(admission)
        self.assertEqual([name for name, _ in client.calls], ["pause_scheduler", RECEIPT, RELEASE])
        self.assertEqual(len(client.shutdowns), 1)

    async def test_async_partial_rpc_no_retry(self):
        client = AsyncFixture(fail=RECEIPT)
        s, admission = session(client)
        with self.assertRaises(TimeoutError):
            await s.run_async(admission)
        self.assertEqual(len(client.calls), 2)
        with self.assertRaises(ProbeError):
            await s.run_async(admission)

    async def test_cancellation_during_receipt_shutdown_without_release(self):
        entered = asyncio.Event()
        class CancelFixture(AsyncFixture):
            async def call_utility_async(self, name, *args):
                if name == RECEIPT:
                    self.calls.append((name, args))
                    entered.set()
                    await asyncio.Event().wait()
                return await super().call_utility_async(name, *args)
        client = CancelFixture()
        s, admission = session(client)
        task = asyncio.create_task(s.run_async(admission))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([name for name, _ in client.calls], ["pause_scheduler", RECEIPT])
        self.assertEqual(len(client.shutdowns), 1)

    async def test_async_wrong_purpose_before_dispatch(self):
        client = AsyncFixture()
        s, admission = session(client)
        admission["client_purpose"] = "probe"
        with self.assertRaises(ProbeError):
            await s.run_async(admission)
        self.assertEqual(client.calls, [])


class ScalarControls(unittest.TestCase):
    def test_fit_remains_measurement_and_unknown_bounds(self):
        self.assertFalse(validate_scalar_receipt(scalar())["decision"]["admitted"])
        self.assertIsNone(scalar()["decision"]["bounded_incremental_bytes"])

    def test_slack_excess_is_scalar_rejection_not_budget_change(self):
        value = scalar(9 << 20)
        validate_scalar_receipt(value)
        self.assertGreater(value["decision"]["required_bytes_above_cap"], 0)

    def test_nonallowlisted_private_payload_rejected(self):
        value = scalar()
        value["storage_ptr"] = 123
        with self.assertRaises(ProbeError):
            validate_scalar_receipt(value)

    def test_boolean_zero_or_extra_forward_rejected(self):
        for value in (False, True, 1):
            row = scalar()
            row["diagnostic_target_forwards"] = value
            with self.assertRaises(ProbeError):
                validate_scalar_receipt(row)

    def test_startup_budget_is_separate(self):
        value = scalar()
        value["existing_startup_target_forwards"] = 12
        with self.assertRaises(ProbeError):
            validate_scalar_receipt(value)

    def test_fabricated_fit_and_removed_unknown_bound_rejected(self):
        for field, value in (("admitted", True), ("bounded_incremental_bytes", 0), ("blockers", [])):
            row = scalar()
            row["decision"][field] = value
            with self.assertRaises(ProbeError):
                validate_scalar_receipt(row)

    def test_malformed_digest_and_arithmetic_rejected(self):
        for field, value in (("receipt_sha256", "bad"), ("existing_forced_expert_fixture_pairs", 5)):
            row = scalar()
            row[field] = value
            with self.assertRaises(ProbeError):
                validate_scalar_receipt(row)
        row = scalar()
        row["decision"]["known_plus_slack_bytes"] += 1
        with self.assertRaises(ProbeError):
            validate_scalar_receipt(row)


class PrivateEvidenceControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.directory.chmod(0o700)

    def test_canonical_digest_and_immutable_no_replace(self):
        row = {"a": [1, 2], "b": None}
        expected = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with PrivateEvidence(self.directory) as store:
            result = store.write("one.json", row)
            with self.assertRaises(FileExistsError):
                store.write("one.json", {"a": 3})
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual((self.directory / "one.json").read_bytes(), expected)
        self.assertEqual((self.directory / "one.json").stat().st_mode & 0o777, 0o400)

    def test_oversized_before_file_acquisition(self):
        with PrivateEvidence(self.directory) as store:
            with self.assertRaises(ProbeError):
                store.write("large.json", {"x": "y" * 4096}, cap=100)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_nonfinite_nonjson_depth_and_long_string_before_encoding(self):
        deep = None
        for _ in range(34):
            deep = [deep]
        for value in (float("nan"), {"x"}, "x" * 4097, deep):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ProbeError):
                    list(bounded_json_chunks(value))

    def test_invalid_directory_mode_rejected(self):
        self.directory.chmod(0o755)
        with self.assertRaises(ProbeError):
            PrivateEvidence(self.directory)

    def test_symlink_directory_rejected(self):
        link = self.directory / "link"
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(OSError):
            PrivateEvidence(link)

    def test_json_duplicate_nonfinite_oversized_rejected(self):
        path = self.directory / "input.json"
        for data, cap in ((b'{"x":1,"x":2}', 100), (b'{"x":NaN}', 100), (b" " * 101, 100)):
            path.write_bytes(data)
            with self.assertRaises(ProbeError):
                read_json(path, cap)

    def test_path_traversal_rejected(self):
        with PrivateEvidence(self.directory) as store:
            with self.assertRaises(ProbeError):
                store.write("../bad.json", {})

    def test_inventory_budget_rejected_before_next_acquisition(self):
        existing = self.directory / "existing.bin"
        with existing.open("wb") as stream:
            stream.truncate(33 << 20)
        with PrivateEvidence(self.directory) as store:
            with self.assertRaisesRegex(ProbeError, "budget exceeded before acquisition"):
                store.write("next.json", {}, cap=100)
        self.assertFalse((self.directory / "next.json").exists())

    def test_inventory_topology_bound(self):
        for i in range(65):
            (self.directory / (str(i) + ".json")).write_text("{}")
        with PrivateEvidence(self.directory) as store:
            with self.assertRaisesRegex(ProbeError, "inventory"):
                store.write("next.json", {})

    def test_default_off_evidence_no_runtime_import(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(install_native_receipt_evidence())


class EvidenceHookControls(unittest.TestCase):
    def fixture(self):
        from megartx import speculative_native_lifecycle as lifecycle
        from megartx import speculative_native_evidence as evidence
        calls = []
        def original(core, admission):
            calls.append("receipt")
            return scalar()
        class CoreFixture:
            megartx_owned_native_receipt = original
        modules = {name: ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core")}
        modules["vllm.v1.engine.core"].EngineCoreProc = CoreFixture
        return lifecycle, evidence, original, CoreFixture, modules, calls

    def test_wraps_only_existing_utility_and_duplicate_rejected(self):
        lifecycle, evidence, original, core, modules, calls = self.fixture()
        with patch.dict(sys.modules, modules), patch.object(lifecycle, "owned_receipt", original), \
                patch.dict(os.environ, {"MEGARTX_NATIVE_DIAGNOSTIC": "1", "MEGARTX_NATIVE_RECEIPT_EVIDENCE": "1"}, clear=True):
            self.assertTrue(evidence.install_native_receipt_evidence())
            self.assertIs(core.megartx_owned_native_receipt.__wrapped__, original)
            self.assertFalse(hasattr(core, "megartx_owned_native_probe"))
            with self.assertRaisesRegex(ProbeError, "duplicate"):
                evidence.install_native_receipt_evidence()

    def test_known_writer_failure_uses_existing_release_preserves_primary(self):
        lifecycle, evidence, original, core, modules, calls = self.fixture()
        primary = ProbeError("fixture writer failure")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(lifecycle, "owned_receipt", original), \
                patch.object(lifecycle, "_release", side_effect=TimeoutError("fixture uncertain drain")) as release, \
                patch.object(evidence, "persist_owned_receipt", side_effect=primary), \
                patch.dict(os.environ, {"MEGARTX_NATIVE_DIAGNOSTIC": "1", "MEGARTX_NATIVE_RECEIPT_EVIDENCE": "1"}, clear=True):
            Path(tmp).chmod(0o700)
            evidence.install_native_receipt_evidence()
            admission = {"client_purpose": PURPOSE, "client_evidence_source_sha256": hashlib.sha256(Path(evidence.__file__).read_bytes()).hexdigest(),
                         "private_receipt_directory": tmp}
            with self.assertRaises(ProbeError) as result:
                core().megartx_owned_native_receipt(admission)
            self.assertIs(result.exception, primary)
            release.assert_called_once()
            self.assertEqual(calls, ["receipt"])

    def test_wrong_writer_source_rejected_before_original_reservation(self):
        lifecycle, evidence, original, core, modules, calls = self.fixture()
        with patch.dict(sys.modules, modules), patch.object(lifecycle, "owned_receipt", original), \
                patch.dict(os.environ, {"MEGARTX_NATIVE_DIAGNOSTIC": "1", "MEGARTX_NATIVE_RECEIPT_EVIDENCE": "1"}, clear=True):
            evidence.install_native_receipt_evidence()
            with self.assertRaisesRegex(ProbeError, "source-bound"):
                core().megartx_owned_native_receipt({"client_purpose": PURPOSE, "client_evidence_source_sha256": "0" * 64})
            self.assertEqual(calls, [])


class SourceAndOperationalControls(unittest.TestCase):
    def test_cpu_imports_and_help_under_python_S(self):
        code = "import sys; from megartx import speculative_native_client, speculative_native_plan, speculative_native_evidence, speculative_native_compare; assert not any(k.split('.')[0] in ('torch','vllm','flashinfer') for k in sys.modules)"
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, "-S", "-c", code], env=env, check=True, capture_output=True)
        for script in ("speculative_native_receipt_client.py", "speculative_native_receipt_preflight.py", "speculative_native_receipt_compare.py"):
            subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / script), "--help"], env=env, check=True, capture_output=True)

    def test_actual_pinned_API_path_is_present_not_only_provider(self):
        source = (ROOT / "src/megartx/speculative_native_client.py").read_text()
        for call in ("EngineArgs(**plan", "EngineCoreClient.make_client(multiprocess_mode=True", "Executor.get_class(config)", "call_utility_async", "call_utility(RECEIPT"):
            self.assertIn(call, source)
        self.assertEqual(len(CLIENT_SOURCES), 3)

    def test_exact_engine_flags_and_M1_off_environment(self):
        kwargs = engine_kwargs()
        self.assertFalse(kwargs["async_scheduling"])
        self.assertFalse(kwargs["enable_prefix_caching"])
        self.assertIn("--no-async-scheduling", engine_argv())
        self.assertNotIn("--no-trust-remote-code", engine_argv())
        env = environment(ROOT, Path("/tmp/private-receipt"))
        self.assertEqual(env["MEGARTX_NATIVE_DIAGNOSTIC"], "1")
        self.assertEqual(env["VLLM_PLUGINS"], "megartx_scale_adapter")
        self.assertFalse(any(k.startswith("MEGARTX_M1_") for k in env))

    def test_authorization_cannot_override_collector_acquisition_blocker(self):
        plan = {"plan_sha256": "a" * 64, "source_head": REVIEWED_LIFECYCLE,
                "preflight_blockers": ["collector_snapshot_and_digest_preallocation_bound_unresolved"],
                "collector_materialization_guard": None}
        auth = {"schema": "megartx-native-receipt-authorization-v1", "purpose": PURPOSE,
                "plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
                "independent_review_clear": True, "exact_head_ci_green": True,
                "parent_source_protocol_accepted": True, "gpu_slot_assigned": True,
                "owned_lifecycle_verified": True, "parent_slot": "fixture", "parent_acceptance_reference": "fixture",
                "independent_review_reference": "fixture", "ci_reference": "fixture"}
        with self.assertRaisesRegex(ProbeError, "cannot override"):
            validate_authorization(auth, plan)
        auth["gpu_slot_assigned"] = 1
        with self.assertRaisesRegex(ProbeError, "authorization"):
            validate_authorization(auth, plan)

    def test_supervisor_and_child_gate_before_runtime_acquisition(self):
        spec = importlib.util.spec_from_file_location("owned_client_fixture", ROOT / "scripts/speculative_native_receipt_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = NS(plan=Path("fixture-plan"), authorization=Path("fixture-auth"), control_fd=99,
                  private_directory=Path("fixture-output"))
        with patch.object(module, "read_json", return_value={}), patch.object(module, "validate_plan", return_value={}), \
                patch.object(module, "validate_authorization", side_effect=ProbeError("source blocker")), \
                patch.object(module, "gpu_processes") as gpu, patch.object(module, "installed_preflight") as installed:
            with self.assertRaises(ProbeError):
                module.supervise(args)
            with self.assertRaises(ProbeError):
                module.child_entry(args)
            gpu.assert_not_called()
            installed.assert_not_called()

    def test_bounded_private_log_never_writes_excess(self):
        spec = importlib.util.spec_from_file_location("bounded_log_fixture", ROOT / "scripts/speculative_native_receipt_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(module.LIMITS, {"private_log_bytes": 100}):
            path, failures = Path(tmp) / "private.log", []
            module.bounded_log(io.BytesIO(b"x" * 101), path, failures.append)
            self.assertEqual(path.read_bytes(), b"")
            self.assertIn("private_log_limit", failures)

    def test_plan_and_authorization_schemas_match_source_contract(self):
        from megartx.speculative_native_plan import OWNED_FILES
        plan_schema = read_json(ROOT / "schemas/speculative-native-receipt-client-plan.schema.json", 256 << 10)
        self.assertEqual(set(plan_schema["properties"]["source_sha256"]["required"]), set(OWNED_FILES))
        self.assertEqual(plan_schema["properties"]["limits"]["const"], LIMITS)
        self.assertIs(plan_schema["properties"]["gpu_authorized"]["const"], False)
        self.assertIs(plan_schema["additionalProperties"], False)
        auth = read_json(ROOT / "schemas/speculative-native-receipt-authorization.schema.json", 64 << 10)
        self.assertIs(auth["properties"]["gpu_slot_assigned"]["const"], True)

    def test_cleanup_evidence_failure_preserves_primary(self):
        spec = importlib.util.spec_from_file_location("cleanup_fixture", ROOT / "scripts/speculative_native_receipt_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        primary, cleanup = TimeoutError("fixture primary"), OSError("fixture cleanup")
        self.assertIs(module.retain_primary(primary, cleanup, "evidence"), primary)
        self.assertIs(module.retain_primary(None, cleanup, "evidence"), cleanup)


def private_fixture():
    """Host metadata fault fixture, never native execution/ownership evidence."""
    row = scalar()
    row["allocation_lower_bound"] = allocation_lower_bound([64] * 30, 568, reject=False)
    row["decision"] = fit_decision(row["allocation_lower_bound"], allocated=1000, reserved=1000,
                                   gpu_free=3 << 30, host_free=10 << 30)
    names = ["fixture.layer." + str(i) for i in range(30)]
    layer_stride, block_stride = 132 * 64, 64
    layers = []
    for i, name in enumerate(names):
        base = i * layer_stride
        layers.append({"owner": name, "device": "cuda:0", "dtype": "torch.bfloat16",
            "storage_ptr": 4096, "storage_bytes": 30 * layer_stride, "storage_offset_bytes": base,
            "shape": [132, 32, 1, 1], "strides": [32, 1, 1, 1], "group": 0,
            "manager_block_tokens": 16, "kernel_block_tokens": 16,
            "placement": {"size": 30 * layer_stride, "offset": 0, "layer_stride": layer_stride,
                          "block_stride": block_stride, "layers": names},
            "owned_pages": [[p, base + p * block_stride, base + (p + 1) * block_stride] for p in range(1, 131)]})
    plan = {"plan_sha256": "b" * 64, "source_head": REVIEWED_LIFECYCLE,
            "source_sha256": {"src/megartx/" + name: "c" * 64 for name in ADAPTER_FILES}}
    plan["source_sha256"]["src/megartx/speculative_native_evidence.py"] = "d" * 64
    raw = {k: v for k, v in row.items() if k != "receipt_sha256"}
    raw.update(purpose="exclusive_native_verifier", checkpoint_revision=REVISION,
        lease_nonce="00000000-0000-4000-8000-000000000001", worker_pid=123, worker_start="456",
        source_sha256={k: v["sha256"] for k, v in source_manifest()["files"].items()},
        adapter_source_sha256={k: "c" * 64 for k in ADAPTER_FILES}, drained=True,
        allocator={"allocated_bytes": 1000, "reserved_bytes": 1000, "cached_slack_bytes": 0,
                   "snapshot_coverage": "unavailable"},
        external_gpu_workspace_bound_bytes=None, ffi_allocator={"argument_exchange_coverage_verified": False,
            "source_sha256": FFI_SOURCES, "managed_allocator_callback": None, "callback_binary_sha256": None},
        baseline_policy="model_cache_existing_startup_workspaces_separate_from_incremental_verifier",
        workspaces={"owners": [], "deduplicated_storage_bytes": 0, "deduplicated_GPU_storage_bytes": 0,
            "deduplicated_host_storage_bytes": 0, "required_size_M1_M2_M256_bytes": None},
        layers=layers)
    row["receipt_sha256"] = stream_digest(raw)
    identity = {"schema": "megartx-native-receipt-evidence-identity-v1", "purpose": PURPOSE,
        "plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"], "checkpoint_revision": REVISION,
        "receipt_sha256": row["receipt_sha256"], "lease_nonce": raw["lease_nonce"],
        "engine_pid": 100, "engine_start": "111", "worker_pid": 123, "worker_start": "456",
        "reserved_group_block_sizes": [16], "reserved_groups": [list(range(1, 131))],
        "client_evidence_source_sha256": "d" * 64}
    return raw, identity, row, plan


class PrivateComparisonControls(unittest.TestCase):
    def test_fixture_geometry_checks_and_unknown_coverage_no_fit_claim(self):
        raw, identity, row, plan = private_fixture()
        result = verify_private_receipt(raw, identity, row, plan)
        self.assertFalse(result["fit_admitted"])
        self.assertEqual(result["actual_cache_layer_count"], 30)
        self.assertEqual(result["snapshot_coverage"], "explicit_partial_or_unavailable")
        self.assertNotIn("worker_pid", result)
        self.assertNotIn("layers", result)

    def test_wrong_purpose_or_checkpoint_rejected(self):
        for key, value in (("purpose", "probe"), ("checkpoint_revision", "0" * 40)):
            raw, identity, row, plan = private_fixture()
            raw[key] = value
            row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
            with self.assertRaises(ProbeError):
                verify_private_receipt(raw, identity, row, plan)

    def test_wrong_loaded_source_rejected_after_matching_digest(self):
        raw, identity, row, plan = private_fixture()
        raw["source_sha256"]["vllm/v1/engine/core.py"] = "0" * 64
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        with self.assertRaisesRegex(ProbeError, "loaded receipt source"):
            verify_private_receipt(raw, identity, row, plan)

    def test_worker_owner_and_nonce_cannot_be_substituted(self):
        for key, value in (("worker_start", "999"), ("lease_nonce", "00000000-0000-4000-8000-000000000099")):
            raw, identity, row, plan = private_fixture()
            identity[key] = value
            with self.assertRaises(ProbeError):
                verify_private_receipt(raw, identity, row, plan)

    def test_padding_or_manager_kernel_split_rejected(self):
        raw, identity, row, plan = private_fixture()
        raw["layers"][0]["kernel_block_tokens"] = 32
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        with self.assertRaises(ProbeError):
            verify_private_receipt(raw, identity, row, plan)

    def test_private_public_scalar_difference_rejected(self):
        raw, identity, row, plan = private_fixture()
        raw["host_free_bytes"] += 1
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        with self.assertRaisesRegex(ProbeError, "scalar field differs"):
            verify_private_receipt(raw, identity, row, plan)

    def test_receipt_digest_substitution_rejected(self):
        raw, identity, row, plan = private_fixture()
        row["receipt_sha256"] = identity["receipt_sha256"] = "0" * 64
        with self.assertRaises(ProbeError):
            verify_private_receipt(raw, identity, row, plan)

    def test_coverage_field_does_not_publish_private_free_text(self):
        raw, identity, row, plan = private_fixture()
        raw["allocator"]["snapshot_coverage"] = {"private_pointer": 42}
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        result = verify_private_receipt(raw, identity, row, plan)
        self.assertNotIn("private_pointer", json.dumps(result))

    def test_measured_lower_bound_cannot_be_replaced(self):
        raw, identity, row, plan = private_fixture()
        raw["allocation_lower_bound"] = row["allocation_lower_bound"] = scalar()["allocation_lower_bound"]
        raw["decision"] = row["decision"] = scalar()["decision"]
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        with self.assertRaisesRegex(ProbeError, "lower bound differs"):
            verify_private_receipt(raw, identity, row, plan)

    def test_actual_reserved_page_identity_must_match_core(self):
        raw, identity, row, plan = private_fixture()
        identity["reserved_groups"][0][0] = 999
        with self.assertRaisesRegex(ProbeError, "page identities differ"):
            verify_private_receipt(raw, identity, row, plan)

    def test_workspace_baseline_not_substituted_into_incremental_cap(self):
        raw, identity, row, plan = private_fixture()
        raw["workspaces"]["deduplicated_GPU_storage_bytes"] = 1
        row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
        with self.assertRaisesRegex(ProbeError, "baseline storage ledger differs"):
            verify_private_receipt(raw, identity, row, plan)


if __name__ == "__main__":
    unittest.main()
