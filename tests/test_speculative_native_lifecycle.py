"""CPU fault controls only. Fixtures are not promoted as native runtime owners."""
import os
import queue
import ctypes
import ast
import hashlib
import io
import json
from contextlib import nullcontext
import tempfile
import types
from pathlib import Path
from concurrent.futures import Future
import subprocess
import sys
import unittest
import uuid
from types import SimpleNamespace as NS
from unittest.mock import patch

from megartx import speculative_native_lifecycle as lifecycle
from megartx import speculative_native_receipt as receipt
from megartx.speculative_native_probe import GPU_CAP, GPU_FREE, HOST_FREE, ProbeError, allocation_lower_bound


def classes():
    class Core:
        def __init__(self):
            self.input_queue = queue.Queue()
            self.batch_queue = None
            self.pending_add_requests = []
            pool = NS(get_new_blocks=lambda n: [NS(block_id=i + 1, ref_cnt=1, is_null=False) for i in range(n)],
                      free_blocks=lambda blocks: self.frees.append(list(blocks)))
            group = NS(kv_cache_spec=NS(block_size=16))
            self.scheduler = NS(has_requests=lambda: False, kv_cache_manager=NS(block_pool=pool),
                                kv_cache_config=NS(kv_cache_groups=[group]))
            self.vllm_config = NS(parallel_config=NS(world_size=1, enable_dbo=False, worker_extension_cls=lifecycle.WORKER_EXTENSION),
                scheduler_config=NS(async_scheduling=False), cache_config=NS(enable_prefix_caching=False),
                speculative_config=None, use_v2_model_runner=False, kv_transfer_config=None, ec_transfer_config=None)
            self.paused, self.frees, self.steps = False, [], 0
            self.model_executor = NS(collective_rpc=self.rpc)
        def rpc(self, method, **kwargs):
            if method == "megartx_native_release":
                return [{"drained": True, "revoked": True}]
            return [{"lease_nonce": kwargs["kwargs"]["ticket"]["nonce"], "drained": True, "diagnostic_target_forwards": 0}]
        def pause_scheduler(self, mode="abort", clear_cache=True):
            self.paused = True
        def is_scheduler_paused(self):
            return self.paused
        def resume_scheduler(self):
            self.paused = False
        def preprocess_add_request(self, *args):
            return args
        def add_request(self, *args):
            pass
        def step(self):
            self.steps += 1
            return {}, False
        step_with_batch_queue = step
    class Proc(Core):
        def _handle_client_request(self, kind, request):
            return getattr(self, request[2])(*request[3])
    lifecycle._register_classes(Core, Proc)
    return Core, Proc


class LeaseControls(unittest.TestCase):
    def setUp(self):
        self.identity = patch.object(lifecycle, "_process_start", return_value="bound_start")
        self.source = patch.object(lifecycle, "_class_source")
        self.admission = patch.object(lifecycle, "check_receipt_admission")
        for mock in (self.identity, self.source, self.admission):
            mock.start()
            self.addCleanup(mock.stop)
        self.base, self.proc = classes()
        self.core = self.proc()
        self.core.pause_scheduler("keep", False)
        self.core._megartx_native_state["utility"] = lifecycle.UTILITIES[0]
    def take_receipt(self):
        with patch.object(receipt, "sanitized_receipt", return_value={}), patch.object(receipt, "receipt_digest", return_value="private_digest"):
            return lifecycle.owned_receipt(self.core, {"deadline_monotonic": 1e20})
    def test_real_reservation_lifetime_until_explicit_drain_release(self):
        self.take_receipt()
        lease = self.core._megartx_native_lease
        self.assertEqual(len(lease.blocks), 130)
        self.assertEqual(lease.ticket["purpose"], lifecycle.PURPOSE)
        self.assertEqual(self.core.frees, [])
        self.core._megartx_native_state["utility"] = lifecycle.UTILITIES[2]
        lifecycle.owned_release(self.core)
        self.assertEqual(len(self.core.frees), 1)
        with self.assertRaisesRegex(ProbeError, "already released"):
            lifecycle.owned_release(self.core)
        self.assertTrue(self.core.paused)
    def test_unknown_drain_retains_objects_and_poison(self):
        self.take_receipt()
        self.core.model_executor.collective_rpc = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("unknown device drain"))
        with self.assertRaises(TimeoutError):
            lifecycle._release(self.core)
        self.assertIs(self.core._megartx_native_retained_blocks, self.core._megartx_native_lease.blocks)
        self.assertTrue(self.core._megartx_native_poisoned)
        self.assertEqual(self.core.frees, [])
        with self.assertRaisesRegex(ProbeError, "owned teardown"):
            lifecycle._release(self.core)
    def test_failed_rpc_and_failed_drain_retains_refs(self):
        self.core.model_executor.collective_rpc = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("RPC uncertain"))
        with self.assertRaisesRegex(TimeoutError, "RPC uncertain"):
            self.take_receipt()
        self.assertTrue(self.core._megartx_native_lease.blocks)
        self.assertEqual(self.core.frees, [])
    def test_failed_rpc_known_drain_releases_once(self):
        original = self.core.rpc
        def fault(method, **kwargs):
            if method == "megartx_native_receipt":
                raise RuntimeError("receipt collection failed")
            return original(method, **kwargs)
        self.core.model_executor.collective_rpc = fault
        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            self.take_receipt()
        self.assertEqual(len(self.core.frees), 1)
    def test_oversized_receipt_error_keeps_confirmed_drain_cleanup(self):
        original = self.core.rpc
        def fault(method, **kwargs):
            result = original(method, **kwargs)
            if method == "megartx_native_receipt":
                result[0]["oversized_metadata"] = "x" * (receipt.RECEIPT_LIMITS["string_chars"] + 1)
            return result
        self.core.model_executor.collective_rpc = fault
        with self.assertRaisesRegex(ProbeError, "oversized receipt scalar"):
            lifecycle.owned_receipt(self.core, {"deadline_monotonic": 1e20})
        self.assertEqual(len(self.core.frees), 1)
        self.assertTrue(self.core.paused)
    def test_wrong_process_and_unfinished_pause_before_reservation(self):
        self.core._megartx_native_state["identity"] = (os.getpid(), "old_start")
        with self.assertRaisesRegex(ProbeError, "Wrong process"):
            self.take_receipt()
        self.assertFalse(hasattr(self.core, "_megartx_native_lease"))
        self.core._megartx_native_state["identity"] = (os.getpid(), "bound_start")
        self.core._megartx_native_state["pause"] = None
        with self.assertRaisesRegex(ProbeError, "completed keep-pause"):
            self.take_receipt()
    def test_queued_request_and_prior_request_rejected_before_reservation(self):
        self.core.input_queue.put("queued_ADD")
        with self.assertRaisesRegex(ProbeError, "queued work"):
            self.take_receipt()
        self.core.input_queue.get()
        self.core.preprocess_add_request("prior request")
        with self.assertRaises(ProbeError):
            self.take_receipt()
        self.assertFalse(hasattr(self.core, "_megartx_native_lease"))
    def test_stays_paused_idle_step_and_new_ingress_closed(self):
        self.take_receipt()
        for action in (self.core.resume_scheduler, lambda: self.core.add_request("new"),
                       lambda: self.core.preprocess_add_request("new"), lambda: self.core.pause_scheduler("abort", True)):
            with self.assertRaises(ProbeError):
                action()
        self.assertEqual(self.core.step(), ({}, False))
        self.assertEqual(self.core.steps, 0)
    def test_receipt_cannot_grant_probe_capability(self):
        self.take_receipt()
        self.core._megartx_native_state["utility"] = lifecycle.UTILITIES[1]
        with patch.object(lifecycle, "check_admission"), patch.object(receipt, "receipt_digest", return_value="exact"):
            self.core._megartx_native_lease.receipt["decision"] = {"admitted": False}
            with self.assertRaisesRegex(ProbeError, "not fit admission"):
                lifecycle.owned_probe(self.core, (), {"zero_forward_receipt_sha256": "exact"})
        self.assertFalse(self.core._megartx_native_lease.probe_used)
    def test_direct_nonutility_entry_and_observation_ticket_rejected(self):
        self.core._megartx_native_state["utility"] = None
        with self.assertRaisesRegex(ProbeError, "synchronous utility"):
            self.take_receipt()
        with self.assertRaisesRegex(ProbeError, "Wrong-purpose"):
            lifecycle._worker_ticket({"purpose": "read_only_prefill_frame"})
        from megartx.speculative_native_probe import OwnedNativeProbe
        with patch("megartx.speculative_native_probe.check_admission"):
            with self.assertRaisesRegex(ProbeError, "worker-bound admitted mutation lease"):
                OwnedNativeProbe.from_runner(object(), {"purpose": "read_only_prefill_frame"}, {}, enabled=True)
    def test_source_failure_precedes_reservation(self):
        with patch.object(lifecycle, "_class_source", side_effect=ProbeError("source changed")):
            with self.assertRaisesRegex(ProbeError, "source changed"):
                self.take_receipt()
        self.assertFalse(hasattr(self.core, "_megartx_native_lease"))


class DefaultOffAndReceiptMath(unittest.TestCase):
    def test_birth_identity_failure_precedes_class_mutation(self):
        base, proc = classes()
        before_base, before_proc = dict(base.__dict__), dict(proc.__dict__)
        active = proc.__new__(proc)
        failure = OSError("birth identity unavailable")
        with patch.object(lifecycle, "_process_start", side_effect=failure):
            with self.assertRaises(OSError) as caught:
                lifecycle._register_classes(base, proc, active)
        self.assertIs(caught.exception, failure)
        self.assertEqual(before_base, dict(base.__dict__))
        self.assertEqual(before_proc, dict(proc.__dict__))
        self.assertFalse(hasattr(active, "_megartx_native_state"))

    def test_partial_hook_write_failure_rolls_back_only_owned_writes(self):
        base, proc = classes()
        before_base, before_proc = dict(base.__dict__), dict(proc.__dict__)
        calls = []
        original_setattr = setattr
        failure = OSError("hook write failed")
        def setter(cls, name, value):
            calls.append((cls, name))
            if len(calls) == 4:
                raise failure
            return original_setattr(cls, name, value)
        with patch.object(lifecycle, "setattr", setter, create=True):
            with self.assertRaises(OSError) as caught:
                lifecycle._register_classes(base, proc)
        self.assertIs(caught.exception, failure)
        self.assertEqual(before_base, dict(base.__dict__))
        self.assertEqual(before_proc, dict(proc.__dict__))

    def test_loaded_code_defaults_and_inherited_hooks_are_verified(self):
        # Compile-only identity control with a small explicit CPU source fixture.
        # Its synthetic source manifest is scoped to this test, never native admission.
        source = '''class EngineCore:
    def __init__(self, value=None): pass
    def preprocess_add_request(self, request): pass
    def add_request(self, request, request_wave=0): pass
    def resume_scheduler(self): pass
    def step(self): pass
    def step_with_batch_queue(self): pass
    def pause_scheduler(self, mode="abort", clear_cache=True): pass
class EngineCoreProc(EngineCore):
    def pause_scheduler(self, mode="abort", clear_cache=True): pass
    def _handle_client_request(self, kind, request): pass
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.py"
            path.write_text(source)
            module = types.ModuleType("vllm.v1.engine.core")
            exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
            base, proc = module.EngineCore, module.EngineCoreProc
            manifest = {"files": {"vllm/v1/engine/core.py": {"sha256": hashlib.sha256(source.encode()).hexdigest()}}}
            with patch.dict(sys.modules, {module.__name__: module}), \
                    patch("megartx.speculative_native_probe.source_manifest", return_value=manifest):
                lifecycle._verify_loaded_methods(base, proc, path)
                original = base.__init__
                def partial(self, *args, **kwargs):
                    return original(self, *args, **kwargs)
                partial.__module__, partial.__qualname__ = original.__module__, original.__qualname__
                base.__init__ = partial
                with self.assertRaisesRegex(ProbeError, "EngineCore.__init__"):
                    lifecycle._verify_loaded_methods(base, proc, path)
                base.__init__ = original
                original.__defaults__ = ("changed",)
                with self.assertRaisesRegex(ProbeError, "EngineCore.__init__"):
                    lifecycle._verify_loaded_methods(base, proc, path)
                original.__defaults__ = (None,)
                proc.step = base.step
                lifecycle._verify_loaded_methods(base, proc, path)  # exact same inherited callable remains safe
                proc.step = lambda self: None
                with self.assertRaisesRegex(ProbeError, "shadowed"):
                    lifecycle._verify_loaded_methods(base, proc, path)

    def test_registration_inside_constructor_observes_birth_before_executor(self):
        # Source-site lifecycle control only, not an actual vLLM engine receipt.
        observed = []
        class Birth:
            def __init__(self):
                observed.append(lifecycle._active_core_birth(Birth))
                self.model_executor = object()
                with self_test.assertRaisesRegex(ProbeError, "missed"):
                    lifecycle._active_core_birth(Birth)
        self_test = self
        core = Birth()
        self.assertIs(observed[0], core)

    def test_registration_source_failure_and_duplicate_have_no_mutation(self):
        # Import/source fault controls; not accepted as actual vLLM classes.
        base, proc = classes()
        module = NS(EngineCore=base, EngineCoreProc=proc)
        env = {"MEGARTX_NATIVE_DIAGNOSTIC": "1", "MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter"}
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"vllm.v1.engine.core": module}), \
                patch.object(lifecycle, "inspect_sources", side_effect=ProbeError("source identity failed")), \
                patch.object(lifecycle, "_register_classes") as mutate:
            with self.assertRaisesRegex(ProbeError, "source identity failed"):
                lifecycle.install_native_diagnostic()
            mutate.assert_not_called()
        base.__module__, base.__name__ = "vllm.v1.engine.core", "EngineCore"
        proc.__module__, proc.__name__ = "vllm.v1.engine.core", "EngineCoreProc"
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"vllm.v1.engine.core": module}), \
                patch.object(lifecycle.inspect, "getsourcefile", return_value="/site/vllm/v1/engine/core.py"), \
                patch.object(lifecycle, "inspect_sources"), patch.object(lifecycle, "_register_classes") as mutate:
            with self.assertRaisesRegex(ProbeError, "Duplicate"):
                lifecycle.install_native_diagnostic()
            mutate.assert_not_called()

    def test_failed_pause_future_never_becomes_verified(self):
        # The original proc pause Future can fail after the utility returns it.
        base, proc = classes()
        future = Future()
        # Exercise a fresh registration with an original Future-returning pause.
        class FreshBase(base):
            pass
        class FreshProc(FreshBase):
            def pause_scheduler(self, mode="abort", clear_cache=True):
                return future
            def _handle_client_request(self, kind, request):
                pass
        with patch.object(lifecycle, "_process_start", return_value="start"):
            lifecycle._register_classes(FreshBase, FreshProc)
            core = FreshProc()
            core.pause_scheduler("keep", False)
        with self.assertLogs("concurrent.futures", level="ERROR"):
            future.set_exception(RuntimeError("pause drain failed"))
        self.assertIsNone(core._megartx_native_state["pause"])

    def test_worker_double_release_and_failed_drain_do_not_revoke(self):
        worker = lifecycle.NativeDiagnosticWorkerExtension()
        ticket = {"purpose": lifecycle.PURPOSE, "nonce": str(uuid.uuid4()), "engine_pid": os.getpid(), "engine_start": "start"}
        worker.model_runner = NS(device="CPU_fault_control_only")
        worker._megartx_native_bound_runner = worker.model_runner
        worker._megartx_native_bound_ticket = ticket
        worker._megartx_native_lease_released = False
        cuda = NS(synchronize=lambda _: (_ for _ in ()).throw(TimeoutError("device drain unknown")))
        with patch.object(lifecycle, "_class_source"), patch.object(lifecycle, "_process_start", return_value="start"), \
                patch.dict(sys.modules, {"torch": NS(cuda=cuda)}):
            with self.assertRaises(TimeoutError):
                worker.megartx_native_release(ticket)
            self.assertFalse(worker._megartx_native_lease_released)
            cuda.synchronize = lambda _: None
            self.assertEqual(worker.megartx_native_release(ticket), {"drained": True, "revoked": True})
            with self.assertRaisesRegex(ProbeError, "already revoked"):
                worker.megartx_native_release(ticket)

    def test_existing_workspace_scan_deduplicates_without_lazy_calls(self):
        class Tensor:
            device, dtype, shape = "cuda:0", "CPU_metadata_only", (16,)
            def untyped_storage(self):
                return NS(data_ptr=lambda: 123, nbytes=lambda: 16)
            def storage_offset(self):
                return 0
            def element_size(self):
                return 1
            def stride(self):
                return (1,)
        tensor = Tensor()
        builder = NS(_workspace_buffer=tensor, _prefill_wrapper=NS(_float_workspace_buffer=tensor),
                     _decode_wrapper=None, _get_workspace_buffer=lambda: self.fail("lazy getter called"))
        runner = NS(attn_groups=[[NS(get_metadata_builder=lambda: builder)]])
        result = receipt.existing_workspaces(runner, NS(Tensor=Tensor))
        self.assertEqual(result["deduplicated_storage_bytes"], 16)
        self.assertIn("attention.0.0._decode_wrapper", result["missing_lazy_owners"])

    def test_FFI_capsule_reads_callback_without_invoking_or_claiming_coverage(self):
        class Prefix(ctypes.Structure):
            _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                        ("previous", ctypes.c_void_p), ("allocator", ctypes.c_void_p)]
        prefix = Prefix(1, 3, None, 1)  # invalid callable; inspection must not invoke it
        make = ctypes.pythonapi.PyCapsule_New
        make.argtypes, make.restype = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p], ctypes.py_object
        capsule = make(ctypes.addressof(prefix), b"dlpack_exchange_api", None)
        tensor_type = type("HostCapsuleFixture", (), {"__dlpack_c_exchange_api__": capsule})
        address = ctypes.addressof(prefix)
        maps = f"{address:x}-{address + ctypes.sizeof(prefix):x} rw-p 00000000 00:00 0"
        with patch.object(receipt, "digest", side_effect=lambda path: receipt.FFI_SOURCES[str(path).split("/site/")[-1]]), \
                patch.object(receipt.Path, "open", return_value=io.BytesIO(maps.encode())):
            from pathlib import Path
            result = receipt.ffi_allocator_identity(NS(Tensor=tensor_type), Path("/site"))
        self.assertEqual(result["managed_allocator_callback"], 1)
        self.assertFalse(result["argument_exchange_coverage_verified"])
        self.assertIsNone(result["callback_binary_sha256"])

    def test_default_off_import_and_registration(self):
        code = "import sys; from megartx.speculative_native_lifecycle import install_native_diagnostic; assert install_native_diagnostic() is False; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
        subprocess.run([sys.executable, "-S", "-c", code], env={"PYTHONPATH": "src"}, check=True, capture_output=True)
        with patch.dict(os.environ, {"MEGARTX_NATIVE_DIAGNOSTIC": "bad"}):
            with self.assertRaisesRegex(ProbeError, "configuration differs"):
                lifecycle.install_native_diagnostic()
    def test_historical_LBNHC_addresses_do_not_imply_current_fit(self):
        for shape, strides in (((3276, 8, 16, 512), (65536, 512, 4096, 1)),
                               ((3276, 2, 32, 1024), (65536, 1024, 2048, 1))):
            self.assertEqual(receipt.page_bytes(shape=shape, strides=strides, element_bytes=2,
                storage_offset_bytes=0, page=17, placement_offset=0, layer_ordinal=0,
                layer_stride=3276 * 131072, block_stride=131072, raw_bytes=2146959360),
                (17 * 131072, 18 * 131072))
        allocation = allocation_lower_bound([131072] * 30, 3152)
        self.assertEqual(allocation["known_bytes"], 4995152)
        decision = receipt.fit_decision(allocation, allocated=100, reserved=100,
            gpu_free=GPU_FREE, host_free=HOST_FREE)
        self.assertFalse(decision["admitted"])
        self.assertIsNone(decision["bounded_incremental_bytes"])
    def test_overlay_requires_disjoint_owned_page_union(self):
        receipt.disjoint_pages([(1, 0, 131072), (1, 131072, 262144)])
        with self.assertRaisesRegex(ProbeError, "alias"):
            receipt.disjoint_pages([(1, 0, 131072), (1, 1, 131073)])
    def test_bad_placement_or_page_bounds_rejected(self):
        values = dict(shape=(3276, 8, 16, 512), strides=(65536, 512, 4096, 1), element_bytes=2,
            storage_offset_bytes=0, page=1, placement_offset=0, layer_ordinal=0, layer_stride=131072,
            block_stride=131072, raw_bytes=2146959360)
        for key, value in (("placement_offset", 1), ("page", 3276), ("raw_bytes", 131072)):
            with self.assertRaises(ProbeError):
                receipt.page_bytes(**{**values, key: value})
        with self.assertRaisesRegex(ProbeError, "internally aliased"):
            receipt.page_bytes(**{**values, "shape": (3276, 2, 2, 2), "strides": (65536, 1, 1, 5)})
        begin, end = receipt.page_bytes(**{**values, "allocated_page_bytes": 131328,
            "strides": (65664, 512, 4096, 1), "block_stride": 131328})
        self.assertEqual(end - begin, 131328)  # manager padding remains owned/charged
        over = allocation_lower_bound([GPU_CAP], 1, reject=False)
        self.assertGreater(over["known_bytes"], GPU_CAP)  # receipt reports rejection before target calls
    def test_cached_slack_rejection_reports_required_resource_decision(self):
        allocation = allocation_lower_bound([131072] * 30, 3152)
        decision = receipt.fit_decision(allocation, allocated=100, reserved=100 + GPU_CAP,
            gpu_free=GPU_FREE, host_free=HOST_FREE)
        self.assertIn("known_buffers_plus_cached_slack_exceed_8MiB", decision["blockers"])
        self.assertEqual(decision["required_bytes_above_cap"], allocation["known_bytes"])
    def test_scalar_allowlist_hides_pointers_prompts_snapshots(self):
        keys = ("schema", "diagnostic_target_forwards", "existing_startup_target_forwards",
                "existing_forced_expert_fixture_pairs", "existing_forced_correction_rows", "original_native_head_dtype",
                "allocation_lower_bound", "gpu_free_bytes", "host_free_bytes", "decision")
        private = {k: 1 for k in keys}
        private.update(prompt=[7], layers=[{"storage_ptr": 123}], snapshot=[1], ffi_allocator={"pointer": 2})
        self.assertEqual(set(receipt.sanitized_receipt(private)), set(keys))


class BoundedReceiptControls(unittest.TestCase):
    """CPU primitives and fault fixtures; never native owners or GPU evidence."""
    def test_canonical_digest_matches_prior_encoding_with_unicode_and_tuples(self):
        value = {"z": (None, True, False, -0.0, 1.25, 2 ** 128),
            "a": {"unicode": "\u00e9\U0001f642\ud800", "escaped": "\n\t\"\\", "empty": []}}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(receipt.receipt_preflight(value), len(encoded))
        self.assertEqual(receipt.receipt_digest(value), hashlib.sha256(encoded).hexdigest())
        self.assertEqual(receipt.receipt_digest(dict(reversed(list(value.items())))), receipt.receipt_digest(value))

    def test_protocol_declares_actual_limits_and_unknown_counter_acquisition_bound(self):
        protocol = json.loads((Path(__file__).resolve().parents[1] / "docs/design/speculative-native-zero-forward-protocol.json").read_text())
        self.assertEqual(protocol["metadata_acquisition_limits"], receipt.RECEIPT_LIMITS)
        self.assertEqual(protocol["bounds"]["receipt_host_metadata_bytes"], receipt.RECEIPT_LIMITS["serialized_bytes"])
        self.assertIsNone(protocol["allocator_counter_scope"]["native_query_preallocation_bound_bytes"])
        self.assertEqual(protocol["allocator_counter_scope"]["installed_torch_cuda_memory_source_sha256"], receipt.TORCH_MEMORY_SOURCE_SHA256)
        self.assertFalse(protocol["gpu_authorized"])

    def test_malformed_deep_and_oversized_scalars_fail_before_hash_encoding(self):
        cycle = []
        cycle.append(cycle)
        deep = 1
        for _ in range(receipt.RECEIPT_LIMITS["depth"] + 1):
            deep = [deep]
        invalid = [cycle, deep, {1: "nonstring_key"}, object(), float("nan"), float("inf"),
            1 << receipt.RECEIPT_LIMITS["integer_bits"], "x" * (receipt.RECEIPT_LIMITS["string_chars"] + 1),
            [None] * (receipt.RECEIPT_LIMITS["container_items"] + 1)]
        with patch.object(json.JSONEncoder, "iterencode", side_effect=AssertionError("hashing began before rejection")):
            for value in invalid:
                with self.subTest(kind=type(value).__name__), self.assertRaises(ProbeError):
                    receipt.receipt_digest(value)

    def test_real_8MiB_serialized_limit_checked_before_hashing(self):
        # A reused small scalar expands via JSON escaping beyond the real cap;
        # this is no giant input string or global allocator snapshot.
        value = ["\0" * 4096] * 342
        with patch.object(json.JSONEncoder, "iterencode", side_effect=AssertionError("oversized data encoded")):
            with self.assertRaisesRegex(ProbeError, "8 MiB serialized"):
                receipt.receipt_digest(value)

    def test_value_budget_and_exact_serialized_boundary(self):
        with patch.dict(receipt.RECEIPT_LIMITS, {"values": 8}):
            with self.assertRaisesRegex(ProbeError, "value/depth"):
                receipt.receipt_digest([[0, 0, 0], [0, 0, 0]])
        with patch.dict(receipt.RECEIPT_LIMITS, {"serialized_bytes": 5}):
            self.assertEqual(receipt.receipt_digest([1, 2]), hashlib.sha256(b"[1,2]").hexdigest())
            with self.assertRaisesRegex(ProbeError, "serialized"):
                receipt.receipt_digest([1, 22])

    def test_workspace_field_limit_precedes_tensor_inspection(self):
        class Tensor:
            def untyped_storage(self):
                raise AssertionError("storage inspected before owner limit")
        builder = NS(**{str(i): Tensor() for i in range(receipt.RECEIPT_LIMITS["owner_fields"] + 1)})
        runner = NS(attn_groups=[[NS(get_metadata_builder=lambda: builder)]])
        with self.assertRaisesRegex(ProbeError, "field acquisition limit"):
            receipt.existing_workspaces(runner, NS(Tensor=Tensor))

    def test_workspace_record_and_tensor_dimension_limits_precede_storage_read(self):
        class Tensor:
            shape = (1,) * (receipt.RECEIPT_LIMITS["tensor_dimensions"] + 1)
            def stride(self):
                raise AssertionError("strides inspected before shape limit")
            def untyped_storage(self):
                raise AssertionError("storage inspected before shape limit")
        with self.assertRaisesRegex(ProbeError, "Tensor shape acquisition"):
            receipt._tensor_record(Tensor(), "CPU_fixture")
        builder = NS(a=Tensor())
        runner = NS(attn_groups=[[NS(get_metadata_builder=lambda: builder)]])
        with patch.dict(receipt.RECEIPT_LIMITS, {"workspace_records": 0}):
            with self.assertRaisesRegex(ProbeError, "Workspace record acquisition"):
                receipt.existing_workspaces(runner, NS(Tensor=Tensor))

    def test_owner_topology_limit_precedes_tensor_and_builder_access(self):
        group = NS(layer_names=["CPU_fixture"] * 31)
        config = NS(kv_cache_groups=[group], kv_cache_tensors=[])
        with self.assertRaisesRegex(ProbeError, "Cache layers acquisition"):
            receipt._owner_limits(NS(kv_cache_config=config), {"groups": [[1] * 130], "block_sizes": [16]})

    def test_process_maps_read_is_bounded_before_decoding(self):
        stream = io.BytesIO(b"x" * (receipt.RECEIPT_LIMITS["process_maps_bytes"] + 1))
        with patch.object(receipt.Path, "open", return_value=stream):
            with self.assertRaisesRegex(ProbeError, "mapping acquisition limit"):
                receipt._process_maps()

    def counters(self, pools=None):
        stats = {"allocated_bytes": {"all": {"current": 100, "peak": 150}},
            "reserved_bytes": {"all": {"current": 180, "peak": 220}}}
        if pools is not None:
            stats["reserved_bytes_by_private_pools"] = pools
        calls = []
        def query(device):
            calls.append(device)
            return stats
        def forbidden(*args, **kwargs):
            self.fail("snapshot, flattening or repeated scalar getter invoked")
        torch = NS(cuda=NS(get_allocator_backend=lambda: "native", memory_stats_as_nested_dict=query,
            memory_snapshot=forbidden, memory_stats=forbidden, memory_allocated=forbidden, memory_reserved=forbidden))
        return torch, stats, calls

    def test_one_nested_counter_query_without_snapshot_or_false_coverage(self):
        torch, _, calls = self.counters({(0, 1): {"not_traversed": object()}})
        with patch.object(receipt, "_host_free", side_effect=[HOST_FREE + 1024, HOST_FREE]):
            result = receipt.allocator_counters(torch, "CPU_counter_fixture")
        self.assertEqual(calls, ["CPU_counter_fixture"])
        self.assertEqual((result["allocated_bytes"], result["reserved_bytes"], result["cached_slack_bytes"]), (100, 180, 80))
        self.assertEqual((result["allocated_peak_bytes"], result["reserved_peak_bytes"]), (150, 220))
        self.assertIsNone(result["snapshot_segments"])
        self.assertIsNone(result["snapshot_sha256"])
        self.assertEqual(result["snapshot_status"], "not_collected_unbounded_global_snapshot")
        self.assertEqual(result["acquisition"]["private_pool_records_observed"], 1)
        self.assertIsNone(result["acquisition"]["native_query_preallocation_bound_bytes"])

    def test_host_reserve_precedes_counter_acquisition_and_is_checked_after(self):
        torch, _, calls = self.counters()
        with patch.object(receipt, "_host_free", return_value=HOST_FREE - 1):
            with self.assertRaisesRegex(ProbeError, "before allocator"):
                receipt.allocator_counters(torch, "CPU_counter_fixture")
        self.assertEqual(calls, [])
        with patch.object(receipt, "_host_free", side_effect=[HOST_FREE, HOST_FREE - 1]):
            with self.assertRaisesRegex(ProbeError, "after allocator"):
                receipt.allocator_counters(torch, "CPU_counter_fixture")
        self.assertEqual(len(calls), 1)

    def test_private_pool_guard_is_honestly_post_query_not_preallocation_bound(self):
        torch, _, calls = self.counters({(0, i): {} for i in range(65)})
        with patch.object(receipt, "_host_free", return_value=HOST_FREE):
            with self.assertRaisesRegex(ProbeError, "after native counter acquisition"):
                receipt.allocator_counters(torch, "CPU_counter_fixture")
        self.assertEqual(len(calls), 1)

    def test_missing_counter_and_private_pool_coverage_are_not_zero(self):
        torch, stats, _ = self.counters()
        with patch.object(receipt, "_host_free", return_value=HOST_FREE):
            result = receipt.allocator_counters(torch, "CPU_counter_fixture")
            self.assertIsNone(result["acquisition"]["private_pool_records_observed"])
            del stats["allocated_bytes"]["all"]["current"]
            with self.assertRaisesRegex(ProbeError, "Invalid integer"):
                receipt.allocator_counters(torch, "CPU_counter_fixture")

    def test_collector_fixture_preserves_false_fit_and_never_requests_snapshot(self):
        # Every native/source/version guard is explicitly mocked. This checks
        # orchestration only, not actual EngineCore/model/CUDA admission.
        torch, _, calls = self.counters({})
        class Tensor:
            shape, device, dtype = (256, 2, 16, 1), "cuda:0", "torch.bfloat16"
            def __init__(self, pointer):
                self.pointer = pointer
            def stride(self):
                return (32, 16, 1, 1)
            def untyped_storage(self):
                return NS(data_ptr=lambda: self.pointer, nbytes=lambda: 16384)
            def storage_offset(self):
                return 0
            def element_size(self):
                return 2
        torch.Tensor, torch.bfloat16, torch.float32 = Tensor, "torch.bfloat16", "torch.float32"
        torch.is_inference_mode_enabled = lambda: True
        torch.cuda.synchronize = lambda _: None
        torch.cuda.mem_get_info = lambda _: (GPU_FREE, GPU_FREE * 2)
        names = [f"CPU_fixture_layer_{i}" for i in range(30)]
        placements = [NS(layers=[name], size=16384, offset=0, layer_stride=16384,
            block_stride=64, host_resident=False) for name in names]
        context = {name: NS(kv_cache=Tensor(1000 + i), impl=NS()) for i, name in enumerate(names)}
        spec = NS(block_size=16, page_size_bytes=64)
        integration_report = {"forced_fixtures": [{"forced_correction_rows": 16} for _ in range(6)]}
        integration_layers = [NS(_megartx={"dispatch_calls": 1}) for _ in range(30)]
        class Model:
            def forward(self):
                raise AssertionError((integration_report, integration_layers, "model forward invoked"))
        model = Model()
        model.language_model = NS(logits_processor=NS(head_dtype=None, soft_cap=30.0),
            lm_head=NS(weight=NS(dtype=torch.bfloat16)))
        runner = NS(input_batch=NS(num_reqs=0), model_config=NS(enforce_eager=True, model="/CPU_fixture"),
            parallel_config=NS(world_size=1), cache_config=NS(cache_dtype="bfloat16"),
            kv_cache_config=NS(kv_cache_groups=[NS(layer_names=names, kv_cache_spec=spec)], kv_cache_tensors=placements),
            attn_groups=[[NS(get_metadata_builder=lambda: NS(page_size=16))]], _kernel_block_sizes=[16],
            compilation_config=NS(static_forward_context=context), get_model=lambda: model,
            device="CPU_counter_fixture", vllm_config=NS())
        ticket = {"groups": [list(range(1, 131))], "block_sizes": [16], "purpose": lifecycle.PURPOSE, "nonce": "CPU_fixture"}
        packages = {"vllm": "0.30.0", "flashinfer-python": "0.6.18.post1", "torch": "2.13.0+cu130"}
        def source(path):
            return receipt.TORCH_MEMORY_SOURCE_SHA256 if str(path).endswith("torch/cuda/memory.py") else receipt.CONFIG_HASH
        def startup(proof):
            proof.startup_forwards = 1
        fake_context = NS(set_forward_context=lambda *a, **k: nullcontext())
        with patch.object(receipt.sys, "version_info", (3, 12, 3)), patch.object(receipt, "_class_source"), \
                patch.object(receipt.inspect, "getsourcefile", return_value="/site/vllm/v1/worker/gpu_model_runner.py"), \
                patch.object(receipt, "inspect_sources", return_value={"CPU_fixture_only": "CPU_fixture"}), \
                patch.object(receipt, "digest", side_effect=source), \
                patch("importlib.metadata.version", side_effect=packages.__getitem__), \
                patch.object(receipt, "_host_free", return_value=HOST_FREE), \
                patch.object(receipt, "_process_start", return_value="CPU_fixture"), \
                patch.object(receipt, "ffi_allocator_identity", return_value={"argument_exchange_coverage_verified": False}), \
                patch("megartx.speculative_native_probe.OwnedNativeProbe._scale_binding", startup), \
                patch.dict(sys.modules, {"torch": torch, "vllm": types.ModuleType("vllm"), "vllm.forward_context": fake_context}), \
                patch.dict(os.environ, {"MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter"}, clear=True):
            result = receipt.collect_receipt(runner, ticket, {"adapter_source_sha256": {"CPU_fixture_only": "CPU_fixture"}})
        self.assertEqual(calls, ["CPU_counter_fixture"])
        self.assertEqual(len(result["layers"]), 30)
        self.assertEqual(sum(len(layer["owned_pages"]) for layer in result["layers"]), 3900)
        self.assertEqual(result["diagnostic_target_forwards"], 0)
        self.assertFalse(result["decision"]["admitted"])
        self.assertIsNone(result["external_gpu_workspace_bound_bytes"])
        self.assertIsNone(result["allocator"]["snapshot_segments"])
        self.assertIsNone(result["allocator"]["acquisition"]["native_query_preallocation_bound_bytes"])
        self.assertIn("FFI_argument_exchange_allocator_coverage_unverified", result["decision"]["blockers"])
        self.assertIn("external_CUDA_allocation_bound_unresolved", result["decision"]["blockers"])
        self.assertIn("M1_M2_M256_original_lane_temporary_bound_unresolved", result["decision"]["blockers"])
        self.assertLess(receipt.receipt_preflight(result), 8 << 20)
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(receipt.receipt_digest(result), hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
