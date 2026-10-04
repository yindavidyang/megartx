"""CPU fault controls only. Fixtures are not promoted as native runtime owners."""
import os
import queue
import ctypes
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
                patch.object(receipt.Path, "read_text", return_value=maps):
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


if __name__ == "__main__":
    unittest.main()
