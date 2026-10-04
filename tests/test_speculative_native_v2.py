"""CPU adversarial controls. Extracted methods are not native/GPU evidence."""
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from megartx import speculative_native_v2 as binding
from megartx import speculative_native_v2_lifecycle as lifecycle
from megartx import speculative_native_lifecycle as common
from megartx import speculative_native_receipt as receipt
from megartx.speculative_native_probe import ProbeError, HOST_FREE, GPU_FREE

FIXTURE = Path(__file__).parent / "fixtures/speculative_v2_installed_methods.json"


def extracted(path, cls, methods, destination):
    entries = json.loads(FIXTURE.read_text())["methods"]
    source = "from __future__ import annotations\nclass " + cls + ":\n"
    source += "".join(e["source"] for e in entries if e["path"] == path and e["method"] in methods)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    module = types.ModuleType(path[:-3].replace("/", "."))
    module.__file__ = str(destination)
    exec(compile(source, str(destination), "exec", dont_inherit=True), module.__dict__)
    return module, source


class ExtractedSourceControls(unittest.TestCase):
    def test_all_extractions_bind_exact_installed_file_and_method_bytes(self):
        from megartx.speculative_native_probe import source_manifest
        sources = {p: v["sha256"] for p, v in source_manifest()["files"].items()}
        sources.update(binding.source_binding()["additional_sources"])
        entries = json.loads(FIXTURE.read_text())["methods"]
        self.assertEqual(len(entries), 6)
        for entry in entries:
            self.assertEqual(entry["installed_source_sha256"], sources[entry["path"]])
            self.assertEqual(entry["extracted_source_sha256"], hashlib.sha256(entry["source"].encode()).hexdigest())
        root = os.environ.get("MEGARTX_V2_INSTALLED_SOURCE_ROOT")
        if root:
            binding.inspect_v2_sources(root)
            for entry in entries:
                lines = (Path(root)/entry["path"]).read_text().splitlines(keepends=True)
                self.assertEqual(entry["source"], "".join(lines[entry["first_line"]-1:entry["last_line"]]))

    def test_actual_extracted_getter_authenticates_loaded_code_origin_defaults(self):
        path = "vllm/v1/worker/gpu/model_runner.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, source = extracted(path, "GPUModelRunner", ["get_model"], root/path)
            obj = module.GPUModelRunner()
            obj.model = object()
            manifest = {"additional_sources": {path: hashlib.sha256(source.encode()).hexdigest()}}
            with patch.dict(sys.modules, {module.__name__: module}), patch.object(binding, "source_binding", return_value=manifest):
                binding.verify_method(obj, "get_model", root)
                self.assertIs(obj.get_model(), obj.model)
                obj.get_model = lambda: obj.model
                with self.assertRaisesRegex(ProbeError, "getter differs"):
                    binding.verify_method(obj, "get_model", root)
                del obj.get_model
                original = module.GPUModelRunner.get_model
                original.__defaults__ = (None,)
                with self.assertRaisesRegex(ProbeError, "getter differs"):
                    binding.verify_method(obj, "get_model", root)
                original.__defaults__ = None
                replacement = lambda self: self.model
                replacement.__module__, replacement.__qualname__ = original.__module__, original.__qualname__
                module.GPUModelRunner.get_model = replacement
                with self.assertRaisesRegex(ProbeError, "getter differs"):
                    binding.verify_method(obj, "get_model", root)
                module.GPUModelRunner.get_model = original
                other = type("GPUModelRunner", (), {"__module__": module.__name__})()
                with self.assertRaisesRegex(ProbeError, "identity differs"):
                    binding.class_source(other, module.__name__, "GPUModelRunner", root)
                module.__file__ = str(root/"foreign.py")
                with self.assertRaisesRegex(ProbeError, "identity differs"):
                    binding.class_source(obj, module.__name__, "GPUModelRunner", root)

    def test_actual_extracted_selector_and_V2_guard_reject_overrides(self):
        path = "vllm/config/vllm.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, source = extracted(path, "VllmConfig", ["use_v2_model_runner"], root/path)
            module.envs, module.HAS_TRITON = NS(VLLM_USE_V2_MODEL_RUNNER=None), True
            module.logger = NS(warning_once=lambda *a: None)
            platform = types.ModuleType("vllm.platforms")
            platform.current_platform = NS(is_rocm=lambda: False)
            config = module.VllmConfig()
            config.attention_config = NS(hisparse_config=None)
            config._get_v2_model_runner_unsupported_features = lambda: []
            config.model_config = NS(enforce_eager=True)
            config.parallel_config = NS(world_size=1, enable_dbo=False)
            config.cache_config = NS(enable_prefix_caching=False)
            config.scheduler_config = NS(async_scheduling=False)
            config.speculative_config = config.kv_transfer_config = config.ec_transfer_config = None
            config.is_mm_encoder_only = False
            manifest = {"additional_sources": {path: hashlib.sha256(source.encode()).hexdigest()}}
            with patch.dict(sys.modules, {module.__name__: module, platform.__name__: platform}), \
                    patch.object(binding, "source_binding", return_value=manifest), patch.dict(os.environ, {}, clear=True):
                binding.verify_method(config, "use_v2_model_runner", root)
                self.assertTrue(binding.require_config(config)["resolved_v2"])
                with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
                    with self.assertRaises(ProbeError): binding.require_config(config)
                module.envs.VLLM_USE_V2_MODEL_RUNNER = False
                with self.assertRaises(ProbeError): binding.require_config(config)
                module.envs.VLLM_USE_V2_MODEL_RUNNER = None
                module.HAS_TRITON = False
                with self.assertRaises(ProbeError): binding.require_config(config)
                module.HAS_TRITON = True
                config._get_v2_model_runner_unsupported_features = lambda: ["unsupported"]
                with self.assertRaises(ProbeError): binding.require_config(config)

    def test_actual_extracted_BlockPool_retains_refs_until_confirmed_drain(self):
        path = "vllm/v1/core/block_pool.py"
        with tempfile.TemporaryDirectory() as directory:
            module, _ = extracted(path, "BlockPool", ["get_new_blocks", "free_blocks"], Path(directory)/path)
            pool = module.BlockPool()
            blocks = [NS(block_id=i+1, ref_cnt=0, is_null=False, pool=pool, block_hash=None) for i in range(130)]
            free = list(blocks)
            def pop(n):
                result = free[:n]; del free[:n]; return result
            pool.free_block_queue = NS(popleft_n=pop, prepend_n=lambda b: free.__setitem__(slice(0, 0), b), append_n=free.extend)
            pool.get_num_free_blocks = lambda: len(free)
            pool._reuse_watchers, pool.enable_caching, pool.metrics_collector = {}, False, None
            core = NS(scheduler=NS(kv_cache_manager=NS(block_pool=pool),
                kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=16))])))
            with patch.object(common, "_class_source"), patch.object(common, "_process_start", return_value="fixture"):
                lease = common._reserve(core, purpose=lifecycle.PURPOSE)
                self.assertTrue(all(b.ref_cnt == 1 for b in blocks))
                self.assertEqual(free, [])
                core.model_executor = NS(collective_rpc=lambda *a, **k: [{"drained": False, "revoked": True}])
                with self.assertRaises(ProbeError): common._release(core)
                self.assertIs(core._megartx_native_retained_blocks, lease.blocks)
                self.assertTrue(all(b.ref_cnt == 1 for b in blocks))
                self.assertTrue(core._megartx_native_poisoned)
                with self.assertRaises(ProbeError): common._release(core)


def owner_fixture():
    names = [f"fixture.layers.{i}.attn" for i in range(30)]
    spec = NS(block_size=16, page_size_bytes=64)
    builder = NS(page_size=16)
    ag = NS(layer_names=names, metadata_builders=[builder], get_metadata_builder=lambda ubatch_id=0: builder)
    group = NS(layer_names=names, kv_cache_spec=spec, host_resident=False, is_eagle_group=False)
    placements = [NS(layers=[name], size=16384, offset=0, layer_stride=16384, block_stride=64, host_resident=False) for name in names]
    config = NS(use_v2_model_runner=True, is_mm_encoder_only=False, speculative_config=None,
        cache_config=NS(enable_prefix_caching=False, cache_dtype="bfloat16"),
        scheduler_config=NS(async_scheduling=False), model_config=NS(enforce_eager=True, model="/CPU_fixture"),
        parallel_config=NS(world_size=1, enable_dbo=False), kv_transfer_config=None, ec_transfer_config=None,
        compilation_config=NS(static_forward_context={name: NS(kv_cache=object(), impl=object()) for name in names}))
    Runner = type("GPUModelRunner", (), {"__module__": binding.RUNNER_MODULE, "get_model": lambda self: self.model})
    runner = Runner()
    runner.vllm_config, runner.model = config, object()
    runner.req_states = NS(req_id_to_index={}, index_to_req_id={})
    runner.model_state = NS(model=runner.model, rope_state=None)
    runner.block_tables = NS(block_sizes=[16], kernel_block_sizes=[16], num_kv_cache_groups=1, blocks_per_kv_block=[1])
    runner.kernel_block_sizes = [16]
    runner.kv_cache_config = NS(kv_cache_groups=[group], kv_cache_tensors=placements)
    runner.attn_groups = [[ag]]
    runner.model_config, runner.parallel_config, runner.cache_config = config.model_config, config.parallel_config, config.cache_config
    runner.compilation_config = config.compilation_config
    runner.device = "CPU_fixture"
    for name in ("execute_model_state", "speculator", "pcp_manager", "ubatch_runner", "batch_sharder", "fast_prefill"):
        setattr(runner, name, None)
    ticket = {"purpose": lifecycle.PURPOSE, "nonce": "00000000-0000-4000-8000-000000000001",
              "engine_pid": 1, "engine_start": "fixture", "groups": [list(range(1, 131))], "block_sizes": [16]}
    return runner, ticket


class OwnerControls(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(binding, "inspect_v2_sources", return_value={"CPU_fixture": "CPU_fixture"}))
        self.stack.enter_context(patch.object(binding, "class_source"))
        self.stack.enter_context(patch.object(binding, "verify_method"))
        self.stack.enter_context(patch.object(binding.inspect, "getsourcefile", return_value="/site/vllm/v1/worker/gpu/model_runner.py"))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def test_empty_V2_uses_request_state_without_persistent_InputBatch(self):
        runner, ticket = owner_fixture()
        self.assertFalse(hasattr(runner, "input_batch"))
        owner = binding.V2Owner(runner, ticket)
        owner.check()
        self.assertFalse(owner.private_identity()["mutable_verifier_lease_granted"])
        self.assertFalse(owner.private_identity()["input_batch_prepared"])

    def test_readonly_frame_V1_and_ticket_guesses_are_not_V2_owners(self):
        for runner in (NS(), {"runner_module": binding.RUNNER_MODULE}, object()):
            with self.assertRaises(ProbeError): binding.V2Owner(runner, owner_fixture()[1])
        runner, ticket = owner_fixture()
        ticket["groups"][0][1] = ticket["groups"][0][0]
        with self.assertRaisesRegex(ProbeError, "aliased"): binding.V2Owner(runner, ticket)

    def test_pending_or_replaced_runner_owners_and_topology_rejected(self):
        mutations = [lambda r: r.req_states.req_id_to_index.update(req=0),
            lambda r: r.req_states.index_to_req_id.update({0: "req"}),
            lambda r: setattr(r, "execute_model_state", object()), lambda r: setattr(r, "speculator", object()),
            lambda r: setattr(r, "block_tables", NS()), lambda r: setattr(r.model_state, "model", object()),
            lambda r: setattr(r, "fast_prefill", object()), lambda r: setattr(r, "pcp_manager", object()),
            lambda r: setattr(r, "ubatch_runner", object()), lambda r: setattr(r, "batch_sharder", object()),
            lambda r: r.block_tables.blocks_per_kv_block.__setitem__(0, 2),
            lambda r: r.kernel_block_sizes.__setitem__(0, 8),
            lambda r: setattr(r.kv_cache_config.kv_cache_groups[0], "host_resident", True),
            lambda r: setattr(r.kv_cache_config.kv_cache_tensors[0], "offset", 2),
            lambda r: r.attn_groups[0].__setitem__(0, object()),
            lambda r: setattr(next(iter(r.compilation_config.static_forward_context.values())), "kv_cache", object())]
        for mutation in mutations:
            runner, ticket = owner_fixture(); owner = binding.V2Owner(runner, ticket)
            mutation(runner)
            with self.assertRaises((ProbeError, AttributeError)): owner.check()

    def test_split_pages_host_and_draft_groups_rejected_before_collection(self):
        for field in ("host_resident", "is_eagle_group"):
            runner, ticket = owner_fixture(); setattr(runner.kv_cache_config.kv_cache_groups[0], field, True)
            with self.assertRaises(ProbeError): binding.V2Owner(runner, ticket)
        runner, ticket = owner_fixture(); runner.kernel_block_sizes[0] = 8
        with self.assertRaises(ProbeError): binding.V2Owner(runner, ticket)

    def test_ticket_and_changed_owner_acquisition_is_bounded(self):
        runner, ticket = owner_fixture()
        ticket["read_only_frame"] = object()
        with self.assertRaisesRegex(ProbeError, "ticket fields"): binding.V2Owner(runner, ticket)
        runner, ticket = owner_fixture(); owner = binding.V2Owner(runner, ticket)
        runner.block_tables.blocks_per_kv_block = [1] * 31
        with self.assertRaisesRegex(ProbeError, "acquisition limit"): owner.check()
        runner, ticket = owner_fixture(); owner = binding.V2Owner(runner, ticket)
        runner.kv_cache_config.kv_cache_tensors[0].layers = ["unused"] * 31
        with self.assertRaisesRegex(ProbeError, "acquisition limit"): owner.check()


class DefaultOffControls(unittest.TestCase):
    def test_no_flag_has_no_runtime_imports_and_all_probes_reject(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(lifecycle.install_native_v2_diagnostic())
        with self.assertRaises(ProbeError): lifecycle.owned_probe(None, None)
        worker = lifecycle.NativeV2DiagnosticWorkerExtension()
        with self.assertRaises(ProbeError): worker.megartx_native_probe({}, [], {"admitted": True})

    def test_registration_conflicting_flags_fail_before_runtime_import(self):
        env = {"MEGARTX_NATIVE_V2_DIAGNOSTIC": "1", "MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter"}
        for extra in ({"MEGARTX_NATIVE_DIAGNOSTIC": "1"}, {"VLLM_USE_V2_MODEL_RUNNER": "0"}, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
            with patch.dict(os.environ, {**env, **extra}, clear=True):
                with self.assertRaises(ProbeError): lifecycle.install_native_v2_diagnostic()


class V2LeaseControls(unittest.TestCase):
    def setUp(self):
        from test_speculative_native_lifecycle import classes
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name in ("_class_source", "_process_start"):
            self.stack.enter_context(patch.object(common, name, return_value="fixture"))
        self.stack.enter_context(patch.object(lifecycle, "check_receipt_admission"))
        self.stack.enter_context(patch.object(lifecycle, "verify_method"))
        self.stack.enter_context(patch.object(lifecycle.inspect, "getsourcefile", return_value="/site/vllm/v1/engine/core.py"))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        register = common._register_classes
        def v2_register(base, proc):
            register(base, proc, utilities=lifecycle.UTILITIES, receipt_callback=lifecycle.owned_receipt,
                     probe_callback=lifecycle.owned_probe, release_callback=lifecycle.owned_release)
        with patch.object(common, "_register_classes", side_effect=v2_register):
            self.base, self.proc = classes()
        self.new_core()
        self.stack.enter_context(patch.object(lifecycle, "sanitized_receipt", return_value={"fixture_only": True}))

    def new_core(self):
        self.core = self.proc()
        config = self.core.vllm_config
        config.use_v2_model_runner, config.is_mm_encoder_only = True, False
        config.model_config = NS(enforce_eager=True)
        config.parallel_config.worker_extension_cls = lifecycle.WORKER_EXTENSION
        self.core.pause_scheduler("keep", False)
        self.core._megartx_native_state["utility"] = lifecycle.UTILITIES[0]
        def rpc(method, **kwargs):
            if method == "megartx_native_release": return [{"drained": True, "revoked": True}]
            return [{"schema": "megartx-native-v2-zero-forward-receipt-v1", "purpose": lifecycle.PURPOSE,
                "lease_nonce": kwargs["kwargs"]["ticket"]["nonce"], "drained": True,
                "diagnostic_target_forwards": 0, "decision": {"admitted": False},
                "v2_owner": {"mutable_verifier_lease_granted": False, "drafter_loaded": False,
                             "metadata_built": False, "input_batch_prepared": False, "runner_policy": dict(binding.POLICY)}}]
        self.core.model_executor.collective_rpc = rpc

    def take(self): return lifecycle.owned_receipt(self.core, {"deadline_monotonic": 1e20})

    def test_single_use_V2_reservation_and_release_keep_engine_sealed(self):
        self.take()
        lease = self.core._megartx_native_lease
        self.assertEqual(lease.ticket["purpose"], lifecycle.PURPOSE)
        self.assertTrue(all(b.ref_cnt == 1 for b in lease.blocks))
        with self.assertRaises(ProbeError): self.take()
        with self.assertRaises(ProbeError): self.core.resume_scheduler()
        with self.assertRaises(ProbeError): self.core.add_request("req")
        with self.assertRaises(ProbeError): self.core.preprocess_add_request("req")
        with self.assertRaises(ProbeError): self.core.pause_scheduler("abort", True)
        self.assertEqual(self.core.step(), ({}, False))
        self.assertEqual(self.core.steps, 0)
        self.core._megartx_native_state["utility"] = lifecycle.UTILITIES[2]
        self.assertTrue(lifecycle.owned_release(self.core)["scheduler_stays_paused"])
        self.assertEqual(len(self.core.frees), 1)
        with self.assertRaises(ProbeError): lifecycle.owned_release(self.core)
        with self.assertRaises(ProbeError): self.core.resume_scheduler()

    def test_dirty_or_wrong_pause_V2_engine_never_reserves(self):
        mutations = [lambda c: setattr(c, "paused", False),
            lambda c: c.input_queue.put("queued"), lambda c: c.pending_add_requests.append("req"),
            lambda c: c._megartx_native_state.update(requests_seen=1),
            lambda c: c._megartx_native_state.update(pause=("keep", True, True)),
            lambda c: c._megartx_native_state.update(pause=("keep", False, False)),
            lambda c: setattr(c.vllm_config, "use_v2_model_runner", False)]
        for mutation in mutations:
            self.new_core()
            mutation(self.core)
            with self.assertRaises(ProbeError): self.take()
            self.assertFalse(hasattr(self.core, "_megartx_native_lease"))
            self.assertEqual(self.core.frees, [])

    def test_wrong_receipt_or_unknown_drain_never_grants_authority(self):
        original = self.core.model_executor.collective_rpc
        def rpc(method, **kwargs):
            if method == "megartx_native_release": raise TimeoutError("drain uncertain")
            result = original(method, **kwargs)
            result[0]["decision"]["admitted"] = True
            return result
        self.core.model_executor.collective_rpc = rpc
        with self.assertRaisesRegex(ProbeError, "later-stage authority"): self.take()
        self.assertTrue(self.core._megartx_native_lease.poisoned)
        self.assertIs(self.core._megartx_native_retained_blocks, self.core._megartx_native_lease.blocks)
        self.assertEqual(self.core.frees, [])
        with self.assertRaises(ProbeError): self.core.resume_scheduler()

    def test_loaded_config_source_failure_precedes_reservation(self):
        with patch.object(lifecycle, "verify_method", side_effect=ProbeError("config getter source drift")):
            with self.assertRaisesRegex(ProbeError, "source drift"): self.take()
        self.assertFalse(hasattr(self.core, "_megartx_native_lease"))
        self.assertEqual(self.core.frees, [])

    def test_worker_single_use_bound_runner_and_uncertain_drain(self):
        worker = lifecycle.NativeV2DiagnosticWorkerExtension()
        worker.model_runner = NS(device="CPU_fixture", vllm_config=NS())
        worker.device, worker.vllm_config = worker.model_runner.device, worker.model_runner.vllm_config
        owner = NS(root=Path("/site"), check=lambda: None)
        row = {"schema": "megartx-native-v2-zero-forward-receipt-v1", "purpose": lifecycle.PURPOSE,
               "decision": {"admitted": False}, "v2_owner": {"mutable_verifier_lease_granted": False, "drafter_loaded": False,
                             "metadata_built": False, "input_batch_prepared": False, "runner_policy": dict(binding.POLICY)}}
        runner, ticket = owner_fixture()
        torch = NS(inference_mode=nullcontext, cuda=NS(synchronize=lambda device: None))
        from megartx import speculative_native_v2_receipt as collector
        with patch.object(lifecycle, "V2Owner", return_value=owner), patch.object(lifecycle, "class_source"), \
                patch.object(collector, "collect_receipt", return_value=row), patch.dict(sys.modules, {"torch": torch}):
            worker.megartx_native_receipt(ticket, {})
            self.assertFalse(hasattr(worker.model_runner, "_megartx_native_mutation_lease"))
            with self.assertRaises(ProbeError): worker.megartx_native_receipt(ticket, {})
            changed = dict(ticket, nonce="00000000-0000-4000-8000-000000000002")
            with self.assertRaises(ProbeError): worker.megartx_native_release(changed)
            original = worker.model_runner
            worker.model_runner = runner
            with self.assertRaises(ProbeError): worker.megartx_native_release(ticket)
            worker.model_runner = original
            torch.cuda.synchronize = lambda device: (_ for _ in ()).throw(TimeoutError("unknown drain"))
            with self.assertRaises(TimeoutError): worker.megartx_native_release(ticket)
            with self.assertRaises(ProbeError): worker.megartx_native_release(ticket)


class V2CollectorControls(unittest.TestCase):
    setUp = OwnerControls.setUp
    def test_target_only_collection_preserves_unknown_bounds_and_private_identity(self):
        from megartx import speculative_native_v2_receipt as collector
        from test_speculative_native_lifecycle import BoundedReceiptControls
        from megartx.speculative_native_probe import TORCH_VERSION_SOURCE_SHA256
        torch, _, calls = BoundedReceiptControls().counters({})
        class Tensor:
            shape, device, dtype = (256, 2, 16, 1), "cuda:0", "torch.bfloat16"
            def __init__(self, pointer): self.pointer = pointer
            def stride(self): return (32, 16, 1, 1)
            def untyped_storage(self): return NS(data_ptr=lambda: self.pointer, nbytes=lambda: 16384)
            def storage_offset(self): return 0
            def element_size(self): return 2
        torch.Tensor, torch.bfloat16, torch.float32 = Tensor, "torch.bfloat16", "torch.float32"
        torch.__version__ = "2.13.0+cu130"
        torch.version = NS(cuda="13.0", git_version="cf30153c4c131c8164ee7798e5022d810682e2cb")
        torch.is_inference_mode_enabled = lambda: True
        torch.cuda.synchronize = lambda device: None
        torch.cuda.mem_get_info = lambda device: (GPU_FREE, GPU_FREE*2)
        runner, ticket = owner_fixture()
        integration_report = {"forced_fixtures": [{"forced_correction_rows": 16} for _ in range(6)]}
        integration_layers = [NS(_megartx={"dispatch_calls": 1}) for _ in range(30)]
        class Model:
            def forward(self): raise AssertionError((integration_report, integration_layers, "forward forbidden"))
        runner.model = Model(); runner.model_state.model = runner.model
        runner.model.language_model = NS(logits_processor=NS(head_dtype=None, soft_cap=30.0),
                                        lm_head=NS(weight=NS(dtype=torch.bfloat16)))
        for i, layer in enumerate(runner.compilation_config.static_forward_context.values()): layer.kv_cache = Tensor(1000+i*16384)
        owner = binding.V2Owner(runner, ticket)
        packages = {"vllm": "0.30.0", "flashinfer-python": "0.6.18.post1", "torch": "2.13.0"}
        def source(path):
            if str(path).endswith("torch/cuda/memory.py"): return receipt.TORCH_MEMORY_SOURCE_SHA256
            if str(path).endswith("torch/version.py"): return TORCH_VERSION_SOURCE_SHA256
            return collector.CONFIG_HASH
        def startup(proof): proof.startup_forwards = 1
        with ExitStack() as stack:
            stack.enter_context(patch.object(collector.sys, "version_info", (3, 12, 3)))
            stack.enter_context(patch.object(collector, "_class_source"))
            stack.enter_context(patch.object(collector, "digest", side_effect=source))
            stack.enter_context(patch("megartx.speculative_native_probe.digest", side_effect=source))
            stack.enter_context(patch("importlib.metadata.version", side_effect=packages.__getitem__))
            stack.enter_context(patch.object(collector, "_host_free", return_value=HOST_FREE))
            stack.enter_context(patch.object(receipt, "_host_free", return_value=HOST_FREE))
            stack.enter_context(patch.object(collector, "_process_start", return_value="fixture"))
            stack.enter_context(patch.object(collector, "ffi_allocator_identity", return_value={"argument_exchange_coverage_verified": False}))
            stack.enter_context(patch("megartx.speculative_native_probe.OwnedNativeProbe._scale_binding", startup))
            stack.enter_context(patch.dict(sys.modules, {"torch": torch,
                "vllm.forward_context": NS(set_forward_context=lambda *a, **k: nullcontext())}))
            stack.enter_context(patch.dict(os.environ, {"MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter"}, clear=True))
            result = collector.collect_receipt(owner, ticket, {"adapter_source_sha256": {"CPU_fixture": "CPU_fixture"}})
        self.assertEqual(result["diagnostic_target_forwards"], 0)
        self.assertEqual(calls, ["CPU_fixture"])
        self.assertIsNone(result["allocator"]["acquisition"]["native_query_preallocation_bound_bytes"])
        self.assertFalse(result["decision"]["admitted"])
        self.assertEqual(len(result["layers"]), 30)
        self.assertEqual(sum(len(layer["owned_pages"]) for layer in result["layers"]), 3900)
        summary = lifecycle.sanitized_receipt(result)
        self.assertNotIn("runner_identity", summary)
        self.assertNotIn("layers", summary)
        self.assertFalse(summary["mutable_verifier_lease_granted"])
        self.assertLess(receipt.receipt_preflight(result), 8 << 20)
        self.assertEqual(receipt.receipt_digest(result), hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        next(iter(runner.compilation_config.static_forward_context.values())).kv_cache.pointer += 1
        with self.assertRaisesRegex(ProbeError, "storage identity"): owner.check()


if __name__ == "__main__":
    unittest.main()
