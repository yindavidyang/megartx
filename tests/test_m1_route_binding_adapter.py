"""CPU adversarial controls for uninstalled actual-object binding readers."""
import ast
import builtins
from dataclasses import replace
import gc
import importlib.util
import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS, MethodType
import unittest
import weakref

import test_m1_route_observer as old
from megartx.m1_route_receipt import UNKNOWN, StreamSnapshot

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "probes"))
from m1_route_binding_adapter import (
    ActualBindingReaders, BindingRouteObserver, BoundaryObservation, LiveMetadata,
    ProducerOwners, SharedRouteDispatcher, SourceFileAttestation, TensorMetadataReader,
    _FunctionSeal,
)
spec = importlib.util.spec_from_file_location("route_binding_reference", ROOT / "tests/fixtures/route_binding_reference.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class Storage:
    def __init__(self, size):
        self.pointer, self.size = 4096, size
    def data_ptr(self):
        return self.pointer
    def nbytes(self):
        return self.size


class Tensor:
    """Actual metadata interfaces, deliberately no device/content APIs."""
    def __init__(self, shape, dtype="float32", device=7):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = NS(type="cuda", index=device)
        self.is_cuda = True
        self.storage = Storage(self.numel() * self.element_size())
        self.offset, self.contiguous_value = 0, True
        self.strides = (self.shape[1], 1) if len(shape) == 2 else (1,)
    def contiguous(self):
        return self
    def to(self, dtype):
        return self if self.dtype == dtype else Tensor(self.shape, dtype, self.device.index)
    def stride(self):
        return self.strides
    def is_contiguous(self):
        return self.contiguous_value
    def untyped_storage(self):
        return self.storage
    def data_ptr(self):
        return self.storage.pointer + self.offset * self.element_size()
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n
    def element_size(self):
        return 2 if self.dtype in ("bfloat16", "float16", "int16") else 4
    def storage_offset(self):
        return self.offset


class Parameter(Tensor):
    pass


class DispatchCache(dict):
    """CPU actual-get seam; no scan/replay/recomputed key."""
    def __init__(self, fixture):
        super().__init__(fixture.cache)
        self.fixture = fixture
    def get(self, key, default=None):
        selected = super().get(key, default)
        f = self.fixture
        f.cache_gets.append((self, key, selected))
        if f.observe_cache:
            event = dict(cache=self, key=key, selected=selected, target=f.target,
                         specialization=f.last_specialization, options=f.last_options)
            event.update(f.cache_event_changes)
            f.dispatcher.cache_selection_observed(**event)
        f.invoke("cache_selected")
        return selected


class BindingFixture(old.Fixture):
    def __init__(self, *, dispatcher=None, shared=None, install=True, observe_cache=True, candidate_seam=False, jit_builtins=None):
        super().__init__(install=False, observe_cache=observe_cache)
        if jit_builtins is not None:
            type(self.jit).run = old.reference.load("JITFunction.run", {
                **self.jit.run.__func__.__globals__, "__builtins__": jit_builtins})
        self.dispatcher = dispatcher or SharedRouteDispatcher()
        if shared is not None:
            for name in ("jit", "kernel", "launcher", "metadata", "cache", "target", "key", "utils",
                         "runtime", "compilation", "HookChain", "context"):
                setattr(self, name, getattr(shared, name))
            self.reference = shared.reference
            self.producer_stream, self.consumer_stream = shared.producer_stream, shared.consumer_stream
        else:
            self.cache = DispatchCache(self)
            old_entry = self.jit.device_caches[7]
            self.jit.device_caches[7] = (self.cache, *old_entry[1:])
            self.reference = replace(self.reference, cache=self.cache)
        if candidate_seam:
            from m1_route_cache_seam import build_cache_selection_candidate, SUPPORTED_JIT_RUN_SHA256
            fixture = self
            class Lookup(dict):
                def get(self, key, default=None):
                    selected = super().get(key, default)
                    fixture.cache_gets.append((self, key, selected))
                    return selected
            self.cache = Lookup(self.cache)
            entry = self.jit.device_caches[7]
            self.jit.device_caches[7] = (self.cache, *entry[1:])
            self.reference = replace(self.reference, cache=self.cache)
            self.candidate = build_cache_selection_candidate(old.reference.SOURCES["JITFunction.run"],
                expected_sha256=SUPPORTED_JIT_RUN_SHA256, namespace=dict(self.jit.run.__func__.__globals__),
                observe_selection=self.dispatcher.cache_selection_observed)
            type(self.jit).run = self.candidate.run
        self.frame, self.scale = {"forward_index": 9}, Parameter((128,))
        self.controller = NS(forward=self.frame, diagnostics=False, external_observer=None, external_observer_requested=False, route_controls=False)
        torch = NS(float32="float32", int32="int32", int16="int16",
                   empty=lambda *shape, dtype, device: Tensor(shape, dtype, device.index))
        routing = old.reference.load("gemma4_fused_routing_kernel_triton", {
            "torch": torch, "triton": NS(next_power_of_2=lambda x: 1 << (x-1).bit_length()),
            "_gemma4_routing_kernel": self.jit})
        self.closure_owner = NS(per_expert_scale=self.scale, experts=self.runner)
        self.platform = NS(is_cuda_alike=lambda: True, is_xpu=lambda: False)
        custom = source.load_routing_function(self.closure_owner, {
            "current_platform": self.platform, "gemma4_fused_routing_kernel_triton": routing,
            "gemma4_routing_function_torch": lambda *a: (_ for _ in ()).throw(AssertionError("CPU fallback"))})
        self.router.custom_routing_function = custom
        Prepare = type("Prepare", (), {
            "prepare": source.load("MoEPrepareAndFinalizeNoDPEPModular.prepare", {"_quantize_input": source.load("_quantize_input")}),
            "finalize": source.load("MoEPrepareAndFinalizeNoDPEPModular.finalize")})
        self.prepare_finalize = Prepare()
        Impl, Experts, Method = type("Impl", (), {}), type("Experts", (), {}), type("Method", (), {})
        impl, experts, method = Impl(), Experts(), Method()
        impl.prepare_finalize, impl.fused_experts = self.prepare_finalize, experts
        Kernel = type("MoeKernel", (), {
            "fused_experts": property(source.load("FusedMoEKernel.fused_experts")),
            "prepare_finalize": property(source.load("FusedMoEKernel.prepare_finalize"))})
        kernel = Kernel()
        kernel.impl = impl
        method.moe_kernel, method.is_monolithic = kernel, False
        Layer = type("Layer", (), {})
        layer = Layer()
        layer.quant_method, layer.layer_name = method, "layer." + str(id(self))
        fixture = self
        controlled = None  # Actual plugin stores os.environ.get, not a Boolean
        diagnostics = validation_profile = False
        controlled_observer = normal_observer = router_observer = route_audit = None
        m1 = self.controller
        def routed_adapter(layer, **kwargs):
            # Model the actual plugin closure guard cells, without running its
            # tensor-content/native correction body in a CPU binding fixture.
            (controlled, diagnostics, validation_profile, controlled_observer,
             normal_observer, router_observer, route_audit, m1)
            return fixture.forward_modular(**kwargs)
        layer.forward_modular = MethodType(routed_adapter, layer)
        self.experts = layer
        self.runner.routed_experts = layer
        del self.runner._quant_method
        type(self.runner)._quant_method = property(source.load("MoERunner._quant_method"))
        method.topk_indices_dtype = "int32"
        del self.runner._maybe_apply_shared_experts
        def shared_experts(runner, *args):
            return fixture.shared_before(*args)
        type(self.runner)._maybe_apply_shared_experts = shared_experts
        self.registry = {layer.layer_name: self.runner}
        layer._megartx = {"owner": method, "kernel": kernel, "registry": self.registry}
        self.forward_context = NS(no_compile_layers=self.registry)
        self.jit.hash = "already-observed-CPU-source-key"
        self.boundary = BoundaryObservation(threading.current_thread(), self.frame, 9,
            "none", False, False, False, False, None, "gemma_cuda", (self.producer_stream, self.consumer_stream))
        self.owners = ProducerOwners(self.router, custom, self.closure_owner, routing, self.jit,
            self.kernel, self.prepare_finalize, experts, Method, Impl, Experts)
        live = LiveMetadata(lambda: self.forward_context, lambda: self.boundary,
            self.runtime, self.compilation, self.HookChain, self.HookChain.__call__)
        self.tensors = TensorMetadataReader((Tensor, Parameter), Storage,
            ((torch.float32, "float32", 4), (torch.int32, "int32", 4), ("bfloat16", "bfloat16", 2), (torch.int16, "int16", 2)))
        self.extractor = ActualBindingReaders(layer, self.runner, routed_adapter, self.controller,
            self.owners, live, self.tensors, self.dispatcher)
        self.readers = self.extractor.readers
        self.expected = replace(self.readers.bindings(), cache_key=self.key,
            jit_run=shared.originals["jit"] if shared else self.jit.run,
            launcher=shared.launcher if shared else self.kernel._run)
        self.reference = replace(self.reference, extra_bindings=self.readers.extra_bindings())
        self.observer = BindingRouteObserver(self.expected, self.reference, self.readers)
        self.originals = dict(runner=self.runner._apply_quant_method, select=self.router.select_experts,
            jit=self.expected.jit_run, launch=self.expected.launcher)
        self.wrappers = self.dispatcher.register(self.observer)
        if install:
            self.install()

    def forward_modular(self, **kwargs):
        self.events.append("consumer")
        self.invoke("consumer")
        scope = self.dispatcher.active_scope
        self.events.append("existing_wait_succeeded")
        self.dispatcher.dependency_observed(scope, self.producer_stream, self.consumer_stream)
        self.receipt = self.dispatcher.checked_consumer(scope, kwargs["topk_ids"], self.frame, self.consumer_stream)
        self.events.extend(("route_D2H", "host_synchronization", "native_route_check"))
        self.invoke("after_consumer")
        return kwargs["topk_ids"]

    def bindings(self):
        return self.readers.bindings()
    def run(self, hidden=None, scores=None, **kwargs):
        return self.runner._apply_quant_method(hidden or Tensor((1, 64)), scores or Tensor((1, 128)), None, **kwargs)


class BindingAdapterTests(unittest.TestCase):
    def assert_checked(self, f, complete=False, checks=1):
        self.assertEqual(f.events.count("route_D2H"), checks)
        self.assertEqual(f.events.count("host_synchronization"), checks)
        self.assertEqual(f.events.count("native_route_check"), checks)
        self.assertEqual(f.observer.last_audit.complete, complete, f.observer.last_audit.reasons)
        self.assertIsNone(f.dispatcher.active_scope)
        self.assertIsNone(f.dispatcher.active_observer)
        self.assertFalse(f.observer.ledger.active)
        self.assertTrue(f.observer.last_audit.route_readback_required)
        with self.assertRaises(TypeError):
            bool(f.observer.last_audit)

    def test_actual_binding_reader_completes_cpu_chain_without_producer_replay(self):
        f = BindingFixture()
        ids = f.run()[1]
        self.assert_checked(f, True)
        self.assertIs(ids, f.bound[3])
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))
        self.assertIs(f.expected.closure_owner, f.closure_owner)
        self.assertIs(f.expected.experts, f.owners.experts)
        self.assertIs(f.readers.bindings().cache_key, UNKNOWN)
        self.assertEqual(f.compiles + f.allocations, [])

    def test_factories_install_nothing(self):
        f = BindingFixture(install=False)
        self.assertIs(f.kernel._run, f.launcher)
        self.assertIs(f.jit.run.__func__, f.originals["jit"].__func__)
        f.readers.bindings()
        f.readers.output(f.scale)
        self.assertEqual(f.events + f.binds + f.cache_gets + f.launches, [])

    def test_missing_selection_stays_unknown(self):
        f = BindingFixture(observe_cache=False)
        f.run()
        self.assert_checked(f)
        self.assertIs(f.observer.last_audit.cache_key, UNKNOWN)

    def test_source_attestation_cannot_supply_live_source_or_handles(self):
        f = BindingFixture()
        attestation = SourceFileAttestation("triton/runtime/jit.py", "5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77")
        self.assertEqual(len(attestation.sha256), 64)
        f.jit.hash = None
        type(f.jit).cache_key = property(lambda self: (_ for _ in ()).throw(AssertionError("lazy hash")))
        self.assertIs(f.readers.bindings().source_key, UNKNOWN)
        del f.kernel.module
        self.assertIs(f.readers.bindings().cuda_module, UNKNOWN)
        f.kernel.module = f.expected.cuda_module
        f.run()
        self.assert_checked(f)

    def test_registry_closure_prepare_and_bound_owners_reject_substitution(self):
        mutations = (
            lambda f: setattr(f.forward_context, "no_compile_layers", dict(f.registry)),
            lambda f: f.registry.update({f.experts.layer_name: old.Owner()}),
            lambda f: setattr(f.closure_owner, "experts", old.Owner()),
            lambda f: setattr(f.closure_owner, "per_expert_scale", Tensor((128,))),
            lambda f: setattr(f.experts.quant_method.moe_kernel.impl, "prepare_finalize", NS(prepare=lambda: None, finalize=lambda: None)),
            lambda f: f.owners.custom_routing.__globals__.update(gemma4_fused_routing_kernel_triton=lambda *a: None),
            lambda f: f.owners.routing_function.__globals__.update(_gemma4_routing_kernel=old.Owner()),
            lambda f: setattr(f.experts, "quant_method", old.Owner()),
        )
        for mutation in mutations:
            f = BindingFixture()
            mutation(f)
            with self.assertRaises(Exception):
                f.readers.bindings()

    def test_hook_chain_changes_unknown_guards_and_platform_replacement(self):
        mutations = (
            lambda f: f.runtime.launch_enter_hook.calls.append(lambda *a: None),
            lambda f: setattr(f.runtime, "kernel_load_end_hook", f.HookChain(reversed=True)),
            lambda f: setattr(f.runtime.launch_exit_hook, "reversed", False),
            lambda f: setattr(f.runtime, "jit_cache_hook", lambda *a: None),
            lambda f: setattr(f, "boundary", replace(f.boundary, external_observer=UNKNOWN)),
            lambda f: setattr(f, "boundary", replace(f.boundary, capture_state=UNKNOWN)),
            lambda f: setattr(f, "boundary", replace(f.boundary, simulation=True)),
            lambda f: setattr(f.router, "_convert_indices_dtype", f.router._convert_indices_dtype),
            lambda f: setattr(f.platform, "is_cuda_alike", lambda: True),
        )
        for mutate in mutations:
            f = BindingFixture()
            mutate(f)
            f.run()
            self.assert_checked(f)
            self.assertEqual(len(f.launches), 1)

    def test_frame_generation_and_stream_changes_reject(self):
        for mutate in (lambda f: f.frame.update(forward_index=10),
                       lambda f: setattr(f.controller, "forward", dict(f.frame)),
                       lambda f: setattr(f, "boundary", replace(f.boundary, streams=()))):
            f = BindingFixture()
            f.callbacks["binder"] = lambda: mutate(f)
            f.run()
            self.assert_checked(f)

    def test_invalid_argument_metadata_delegates_once(self):
        cases = (lambda f, score: setattr(score, "strides", (1, 128)),
            lambda f, score: setattr(score, "dtype", "bfloat16"),
            lambda f, score: setattr(score.storage, "size", 1),
            lambda f, score: setattr(score, "is_cuda", False),
            lambda f, score: setattr(score.device, "index", True),
            lambda f, score: setattr(f.scale, "dtype", "bfloat16"))
        for change in cases:
            f, score = BindingFixture(), Tensor((1, 128))
            change(f, score)
            f.run(scores=score)
            self.assert_checked(f)
            self.assertEqual((len(f.binds), len(f.launches)), (1, 1))

    def test_input_storage_retained_replacement_rejected_and_released(self):
        f = BindingFixture()
        retained = weakref.ref(f.scale.storage)
        def replace_storage():
            f.scale.storage = Storage(512)
            gc.collect()
            self.assertIsNotNone(retained())
        f.callbacks["launch"] = replace_storage
        f.run()
        self.assert_checked(f)
        gc.collect()
        self.assertIsNone(retained())

    def test_metadata_interface_override_rejected(self):
        f = BindingFixture()
        f.scale.untyped_storage = lambda: f.scale.storage
        with self.assertRaises(Exception):
            f.readers.output(f.scale)

    def test_primary_and_cleanup_errors_preserve_primary_identity(self):
        f, error = BindingFixture(), old.HostileError()
        f.callbacks["launch"] = lambda: (_ for _ in ()).throw(error)
        f.observer._finish = lambda scope: (_ for _ in ()).throw(RuntimeError("cleanup"))
        with self.assertRaises(old.HostileError) as raised:
            f.run()
        self.assertIs(raised.exception, error)
        self.assertEqual((len(f.binds), len(f.launches)), (1, 1))
        self.assertIsNone(f.dispatcher.active_observer)
        self.assertTrue(f.observer.poisoned)
        self.assertEqual(f.scopes[0].retained, [])

    def test_native_opaque_metadata_unsupported(self):
        f = BindingFixture(install=False)
        f.launcher.arg_annotations = (old.Owner(),)
        with self.assertRaises(Exception):
            BindingRouteObserver(f.expected, f.reference, f.readers)
        self.assertIs(f.kernel._run, f.launcher)

    def test_thirty_layers_share_exactly_one_jit_and_launcher_wrapper(self):
        dispatcher = SharedRouteDispatcher()
        first = BindingFixture(dispatcher=dispatcher)
        layers = [first] + [BindingFixture(dispatcher=dispatcher, shared=first) for _ in range(29)]
        self.assertEqual(len(dispatcher._shared), 2)
        self.assertEqual(len({id(f.wrappers["jit"]) for f in layers}), 1)
        self.assertEqual(len({id(f.wrappers["launch"]) for f in layers}), 1)
        for f in reversed(layers):
            f.run()
            self.assert_checked(f, True)
        self.assertEqual((len(first.binds), len(first.cache_gets), len(first.launches)), (30, 30, 30))
        for f in layers:
            f.run()
            self.assert_checked(f, True, checks=2)
        self.assertEqual(len(first.launches), 60)

    def test_nested_layers_reject_without_duplicate_delegation(self):
        f = BindingFixture()
        other = BindingFixture(dispatcher=f.dispatcher, shared=f)
        f.callbacks["runner"] = other.run
        f.run()
        self.assert_checked(f)
        self.assert_checked(other)
        self.assertEqual((len(f.binds), len(f.launches)), (2, 2))
        self.assertTrue(f.observer.poisoned)
        self.assertTrue(other.observer.poisoned)

    def test_foreign_thread_global_call_is_unobserved_and_delegates_once(self):
        f = BindingFixture()
        failures = []
        def other_thread():
            try:
                f.router.select_experts(Tensor((1, 64)), Tensor((1, 128)), "int32")
            except BaseException as e:
                failures.append(e)
        def invoke():
            t = threading.Thread(target=other_thread)
            t.start(); t.join(10)
            self.assertFalse(t.is_alive())
        f.callbacks["runner"] = invoke
        f.run()
        self.assertEqual(failures, [])
        self.assert_checked(f, True)
        self.assertEqual((len(f.binds), len(f.launches)), (2, 2))

    def test_concurrent_registered_runners_invalidate_both_scopes(self):
        f = BindingFixture()
        other = BindingFixture(dispatcher=f.dispatcher, shared=f)
        failures = []
        def call():
            try:
                other.boundary = replace(other.boundary, thread=threading.current_thread())
                other.run()
            except BaseException as e:
                failures.append(e)
        def invoke():
            t = threading.Thread(target=call)
            t.start(); t.join(10)
            self.assertFalse(t.is_alive())
        f.callbacks["runner"] = invoke
        f.run()
        self.assertEqual(failures, [])
        self.assert_checked(f)
        self.assert_checked(other)
        self.assertEqual(len(f.launches), 2)

    def test_shared_wrapper_replacement_is_not_hidden(self):
        for target, name in (("jit", "run"), ("kernel", "_run")):
            f = BindingFixture()
            original = getattr(getattr(f, target), name)
            setattr(getattr(f, target), name, lambda *a, **k: original(*a, **k))
            f.run()
            self.assert_checked(f)
            self.assertEqual(len(f.launches), 1)

    def test_cache_and_selected_kernel_substitution_rejected(self):
        for changes in (dict(cache={}), dict(key="posthoc-key"), dict(selected=old.Owner()), dict(target=old.Owner())):
            f = BindingFixture()
            f.cache_event_changes.update(changes)
            f.run()
            self.assert_checked(f)
            self.assertEqual((len(f.binds), len(f.launches)), (1, 1))

    def test_cold_and_new_specialization_stays_incomplete_without_observer_compile(self):
        f = BindingFixture()
        f.cache.clear()
        f.compile_result = f.kernel
        f.run()
        self.assert_checked(f)
        self.assertEqual((len(f.binds), len(f.compiles), len(f.launches)), (1, 1, 1))
        f = BindingFixture()
        f.specialization.append(("new", "cpu-only"))
        f.compile_result = f.kernel
        f.run()
        self.assert_checked(f)
        self.assertEqual((len(f.binds), len(f.compiles), len(f.launches)), (1, 1, 1))

    def test_actual_ast_lookup_seam_integrates_with_shared_dispatcher_on_cpu_only(self):
        f = BindingFixture(candidate_seam=True)
        f.run()
        self.assert_checked(f, True)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))
        self.assertEqual(f.candidate.callback_failures, 0)
        self.assertEqual(f.observer.last_audit.cache_key, f.key)
        self.assertFalse(f.candidate.proof.installed)
        self.assertFalse(f.candidate.proof.loaded_bindings_verified)

    def test_refreshed_boundary_cannot_hide_same_frame_generation_change(self):
        f = BindingFixture()
        def mutate():
            f.frame["forward_index"] += 1
            f.boundary = replace(f.boundary, frame_index=f.frame["forward_index"])
        f.callbacks["binder"] = mutate
        f.run()
        self.assert_checked(f)

    def test_stream_metadata_uses_actual_inner_argument_not_current_stream(self):
        f = BindingFixture()
        f.boundary = replace(f.boundary, streams=(f.producer_stream, f.consumer_stream,
            StreamSnapshot(7, f.context, f.outer_handle)))
        f.run()
        self.assert_checked(f, True)
        self.assertEqual(f.observer.last_audit.inner_stream, f.inner_handle)
        self.assertNotEqual(f.observer.last_audit.inner_stream, f.outer_handle)

    def test_duplicate_and_unobserved_inner_stream_records_reject(self):
        for streams in ((), (None,), ("inferred-default",)):
            f = BindingFixture()
            f.boundary = replace(f.boundary, streams=streams)
            f.run()
            self.assert_checked(f)
        f = BindingFixture()
        f.boundary = replace(f.boundary, streams=(f.producer_stream, f.producer_stream))
        f.run()
        self.assert_checked(f)

    def test_cleanup_failure_after_success_preserves_return_and_releases_arguments(self):
        f = BindingFixture()
        original_finish = f.observer._finish
        def failed_finish(scope):
            original_finish(scope)
            raise old.HostileError()
        f.observer._finish = failed_finish
        ids = f.run()[1]
        self.assertIs(ids, f.bound[3])
        self.assert_checked(f)
        self.assertTrue(f.observer.poisoned)
        self.assertEqual(f.scopes[0].retained, [])

    def test_cleanup_failure_before_context_exit_releases_open_ledger(self):
        for raise_primary in (False, True):
            f = BindingFixture()
            primary = old.HostileError()
            f.observer._finish = lambda scope: (_ for _ in ()).throw(old.HostileError())
            if raise_primary:
                f.callbacks["after_consumer"] = lambda: (_ for _ in ()).throw(primary)
                with self.assertRaises(old.HostileError) as raised:
                    f.run()
                self.assertIs(raised.exception, primary)
            else:
                ids = f.run()[1]
                self.assertIs(ids, f.bound[3])
            self.assert_checked(f)
            self.assertIsNone(f.scopes[0].ledger_context)
            self.assertEqual(f.scopes[0].retained, [])

    def test_routed_adapter_wrong_bound_self_rejected_by_extractor(self):
        f = BindingFixture()
        f.experts.forward_modular = MethodType(f.extractor.adapter, old.Owner())
        with self.assertRaises(Exception):
            f.readers.bindings()

    def test_inflight_overlap_blocks_new_scope_after_first_owner_returns(self):
        a = BindingFixture()
        b = BindingFixture(dispatcher=a.dispatcher, shared=a)
        c = BindingFixture(dispatcher=a.dispatcher, shared=a)
        entered, release = threading.Event(), threading.Event()
        errors = []
        def hold():
            entered.set()
            if not release.wait(10):
                raise AssertionError("test release timed out")
        b.callbacks["runner"] = hold
        def second():
            try:
                b.boundary = replace(b.boundary, thread=threading.current_thread())
                b.run()
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=second)
        def start():
            thread.start()
            self.assertTrue(entered.wait(10))
        a.callbacks["runner"] = start
        try:
            a.run()
            self.assertEqual(a.dispatcher._inflight, 1)
            c.run()
            self.assertFalse(c.observer.last_audit.complete)
            self.assertEqual(a.dispatcher._inflight, 1)
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(a.dispatcher._inflight, 0)
        self.assert_checked(a)
        self.assert_checked(b)
        self.assert_checked(c)
        d = BindingFixture(dispatcher=a.dispatcher, shared=a)
        d.run()
        self.assert_checked(d)

    def test_source_global_substitution_rejected_without_replaying_binder(self):
        f = BindingFixture()
        globals_ = f.originals["jit"].__func__.__globals__
        old_key = globals_["compute_cache_key"]
        globals_["compute_cache_key"] = lambda *a, **k: old_key(*a, **k)
        f.run()
        self.assert_checked(f)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))

    def test_absent_global_builtin_shadow_rejected_at_each_observation_boundary(self):
        for stage in (None, "runner", "binder", "consumer", "after_consumer"):
            with self.subTest(stage=stage):
                f = BindingFixture()
                fn = f.originals["jit"].__func__
                self.assertIn("len", fn.__code__.co_names)
                self.assertNotIn("len", fn.__globals__)
                calls = []
                def inject():
                    fn.__globals__["len"] = lambda value: (calls.append(value), builtins.len(value))[1]
                if stage is None:
                    inject()
                else:
                    f.callbacks[stage] = inject
                ids = f.run()[1]
                self.assertIs(ids, f.bound[3])
                self.assert_checked(f)
                self.assertIn("bound callable global changed", f.observer.last_audit.reasons)
                self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))
                self.assertEqual(len(calls), 0 if stage in ("consumer", "after_consumer") else 1)

    def test_effective_builtin_mutation_rejected_without_using_globals_builtin_entry(self):
        for stage in (None, "runner", "binder", "after_consumer"):
            with self.subTest(stage=stage):
                effective = dict(vars(builtins))
                f = BindingFixture(jit_builtins=effective)
                fn = f.originals["jit"].__func__
                self.assertIs(fn.__builtins__, effective)
                fn.__globals__["__builtins__"] = dict(vars(builtins))
                calls = []
                def inject():
                    effective["len"] = lambda value: (calls.append(value), builtins.len(value))[1]
                if stage is None:
                    inject()
                else:
                    f.callbacks[stage] = inject
                ids = f.run()[1]
                self.assertIs(ids, f.bound[3])
                self.assert_checked(f)
                self.assertIn("bound callable builtin changed", f.observer.last_audit.reasons)
                self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))
                self.assertEqual(len(calls), 0 if stage == "after_consumer" else 1)

    def test_rebinding_globals_builtin_entry_does_not_change_existing_function_resolution(self):
        f = BindingFixture(jit_builtins=dict(vars(builtins)))
        fn = f.originals["jit"].__func__
        fn.__globals__["__builtins__"] = {"len": lambda value: 0}
        f.run()
        self.assert_checked(f, True)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))

    def test_namespace_seal_distinguishes_missing_from_unknown_and_checks_removal(self):
        cases = (
            ({}, {}, "globals", UNKNOWN),
            ({}, {}, "builtins", UNKNOWN),
            ({"reference": UNKNOWN}, {}, "globals", None),
            ({}, {"reference": UNKNOWN}, "builtins", None),
            ({"reference": object()}, {}, "globals", object()),
            ({}, {"reference": object()}, "builtins", object()),
        )
        for globals_, builtins_, target, replacement in cases:
            with self.subTest(target=target, replacement=replacement):
                namespace = dict(globals_, __builtins__=builtins_)
                exec("def read(): return reference", namespace)
                seal = _FunctionSeal(namespace["read"])
                seal.check()
                mapping = namespace if target == "globals" else builtins_
                if replacement is None:
                    del mapping["reference"]
                else:
                    mapping["reference"] = replacement
                with self.assertRaisesRegex(Exception, "bound callable (global|builtin) changed"):
                    seal.check()

    def test_tensor_and_parameter_are_exact_admitted_types_not_arbitrary_subclasses(self):
        f = BindingFixture()
        self.assertIs(type(f.scale), Parameter)
        self.assertIs(f.readers.output(f.scale).owner, f.scale)
        self.assertIs(type(f.readers.output(Tensor((1, 8), "int32")).owner), Tensor)
        class Unadmitted(Tensor):
            pass
        with self.assertRaises(Exception):
            f.readers.output(Unadmitted((1, 8), "int32"))
        only_tensor = TensorMetadataReader(Tensor, Storage, (("float32", "float32", 4),))
        with self.assertRaises(Exception):
            only_tensor.view(f.scale)

    def test_actual_plugin_closure_and_controller_guards_cannot_be_hidden_by_boundary(self):
        for name in ("controlled_observer", "normal_observer", "router_observer", "route_audit"):
            f = BindingFixture()
            cells = dict(zip(f.extractor.adapter.__code__.co_freevars, f.extractor.adapter.__closure__))
            cells[name].cell_contents = old.Owner()
            f.run()
            self.assert_checked(f)
        for name, value in (("diagnostics", True), ("external_observer", old.Owner()),
                            ("external_observer_requested", True), ("route_controls", True)):
            f = BindingFixture()
            setattr(f.controller, name, value)
            f.run()
            self.assert_checked(f)
        f = BindingFixture()
        f.experts._megartx["fixture_mode"] = "native"
        f.run()
        self.assert_checked(f)

    def test_consumer_extraction_failure_still_runs_original_checks_once(self):
        f = BindingFixture()
        def replace_reader():
            def failed():
                raise old.HostileError()
            f.observer.readers = replace(f.observer.readers, bindings=failed)
        f.callbacks["consumer"] = replace_reader
        ids = f.run()[1]
        self.assertIs(ids, f.bound[3])
        self.assert_checked(f)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))

    def test_explicit_foreign_events_reject_without_releasing_owner_scope(self):
        f = BindingFixture()
        errors = []
        def foreign():
            scope = f.observer._active
            try:
                f.dispatcher.dependency_observed(scope, f.producer_stream, f.consumer_stream)
                f.dispatcher.checked_consumer(scope, f.bound[3], f.frame, f.consumer_stream)
                self.assertIs(f.observer._active, scope)
            except BaseException as error:
                errors.append(error)
        def invoke():
            t = threading.Thread(target=foreign)
            t.start(); t.join(10)
            self.assertFalse(t.is_alive())
        f.callbacks["consumer"] = invoke
        f.run()
        self.assertEqual(errors, [])
        self.assert_checked(f)

    def test_foreign_rejection_is_serialized_before_owner_terminal_audit(self):
        f = BindingFixture()
        parked, resume, cleanup_attempt, owner_done = (threading.Event() for _ in range(4))
        errors, scopes = [], []
        lines, start = inspect.getsourcelines(type(f.dispatcher)._event_observer)
        reject_line = start + next(i for i, line in enumerate(lines) if "active._reject(active_scope" in line)
        lines, start = inspect.getsourcelines(old.RouteObserver._runner)
        cleanup_line = start + next(i for i, line in enumerate(lines) if "if self._active is scope:" in line) - 1
        def foreign():
            def trace(frame, event, arg):
                if event == "line" and frame.f_code is type(f.dispatcher)._event_observer.__code__ and frame.f_lineno == reject_line:
                    parked.set()
                    if not resume.wait(10):
                        raise AssertionError("resume timeout")
                return trace
            sys.settrace(trace)
            try:
                f.dispatcher.dependency_observed(scopes[0], f.producer_stream, f.consumer_stream)
            except BaseException as error:
                errors.append(("foreign", type(error).__name__))
            finally:
                sys.settrace(None)
        foreign_thread = threading.Thread(target=foreign)
        def consumer():
            scopes.append(f.dispatcher.active_scope)
            foreign_thread.start()
            if not parked.wait(10):
                raise AssertionError("park timeout")
        f.callbacks["consumer"] = consumer
        def owner():
            def trace(frame, event, arg):
                if event == "line" and frame.f_code is old.RouteObserver._runner.__code__ and frame.f_lineno == cleanup_line:
                    cleanup_attempt.set()
                return trace
            sys.settrace(trace)
            try:
                f.boundary = replace(f.boundary, thread=threading.current_thread())
                f.run()
            except BaseException as error:
                errors.append(("owner", type(error).__name__))
            finally:
                sys.settrace(None)
                owner_done.set()
        owner_thread = threading.Thread(target=owner)
        owner_thread.start()
        try:
            self.assertTrue(parked.wait(10))
            self.assertTrue(cleanup_attempt.wait(10))
            self.assertFalse(owner_done.is_set())
        finally:
            resume.set()
            foreign_thread.join(10)
            owner_thread.join(10)
        self.assertFalse(foreign_thread.is_alive() or owner_thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_checked(f)
        self.assertTrue(f.observer.last_audit.poisoned and f.observer.poisoned)
        self.assertEqual(f.dispatcher._inflight, 0)

    def test_registered_overlap_rejection_holds_owner_terminal_lock_until_poison(self):
        a = BindingFixture()
        b = BindingFixture(dispatcher=a.dispatcher, shared=a)
        parked, resume, cleanup_attempt = (threading.Event() for _ in range(3))
        errors = []
        run_code = a.wrappers["runner"].__code__
        lines, start = inspect.getsourcelines(a.wrappers["runner"])
        reject_line = start + next(i for i, line in enumerate(lines) if "self._active._reject(active_scope" in line)
        lines, start = inspect.getsourcelines(old.RouteObserver._runner)
        cleanup_line = start + next(i for i, line in enumerate(lines) if "if self._active is scope:" in line) - 1
        def overlap():
            def trace(frame, event, arg):
                if event == "line" and frame.f_code is run_code and frame.f_lineno == reject_line:
                    parked.set()
                    if not resume.wait(10):
                        raise AssertionError("resume timeout")
                return trace
            sys.settrace(trace)
            try:
                b.boundary = replace(b.boundary, thread=threading.current_thread())
                b.run()
            except BaseException as error:
                errors.append(("overlap", type(error).__name__))
            finally:
                sys.settrace(None)
        overlap_thread = threading.Thread(target=overlap)
        def consumer():
            overlap_thread.start()
            if not parked.wait(10):
                raise AssertionError("park timeout")
        a.callbacks["consumer"] = consumer
        def owner():
            def trace(frame, event, arg):
                if event == "line" and frame.f_code is old.RouteObserver._runner.__code__ and frame.f_lineno == cleanup_line:
                    cleanup_attempt.set()
                return trace
            sys.settrace(trace)
            try:
                a.boundary = replace(a.boundary, thread=threading.current_thread())
                a.run()
            except BaseException as error:
                errors.append(("owner", type(error).__name__))
            finally:
                sys.settrace(None)
        owner_thread = threading.Thread(target=owner)
        owner_thread.start()
        try:
            self.assertTrue(parked.wait(10))
            self.assertTrue(cleanup_attempt.wait(10))
            # No sleep/scheduler assumption: accepted rejection must own the
            # same lock needed for terminal detach, before its poison write.
            acquired = a.observer._lock.acquire(blocking=False)
            if acquired:
                a.observer._lock.release()
            self.assertFalse(acquired)
            self.assertIsNone(a.observer.last_audit)
        finally:
            resume.set()
            overlap_thread.join(10)
            owner_thread.join(10)
        self.assertFalse(overlap_thread.is_alive() or owner_thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_checked(a)
        self.assert_checked(b)
        self.assertTrue(a.observer.last_audit.poisoned and a.observer.poisoned)
        self.assertTrue(b.observer.last_audit.poisoned and b.observer.poisoned)
        self.assertEqual(a.dispatcher._inflight, 0)
        self.assertEqual((len(a.binds), len(a.cache_gets), len(a.launches)), (2, 2, 2))

    def test_retired_scope_audit_stays_historical_while_overlap_poisons_future_calls(self):
        a = BindingFixture()
        b = BindingFixture(dispatcher=a.dispatcher, shared=a)
        retired, resume = threading.Event(), threading.Event()
        errors = []
        lines, start = inspect.getsourcelines(old.RouteObserver._runner)
        post_audit_line = start + next(i for i, line in enumerate(lines) if "scope.retained.clear()" in line)
        def owner():
            def trace(frame, event, arg):
                if event == "line" and frame.f_code is old.RouteObserver._runner.__code__ and frame.f_lineno == post_audit_line:
                    retired.set()
                    if not resume.wait(10):
                        raise AssertionError("resume timeout")
                return trace
            sys.settrace(trace)
            try:
                a.boundary = replace(a.boundary, thread=threading.current_thread())
                a.run()
            except BaseException as error:
                errors.append(type(error).__name__)
            finally:
                sys.settrace(None)
        thread = threading.Thread(target=owner)
        thread.start()
        try:
            self.assertTrue(retired.wait(10))
            self.assertIsNone(a.observer._active)
            audit = a.observer.last_audit
            self.assertTrue(audit.complete)
            self.assertFalse(audit.poisoned)
            b.run()
            self.assertIs(a.observer.last_audit, audit)
            self.assertTrue(a.observer.poisoned and b.observer.poisoned)
            self.assertFalse(b.observer.last_audit.complete)
            self.assertEqual(a.dispatcher._inflight, 1)
        finally:
            resume.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_checked(a, True)
        self.assert_checked(b)
        a.boundary = replace(a.boundary, thread=threading.current_thread())
        a.run()
        self.assert_checked(a, checks=2)
        self.assertEqual(a.dispatcher._inflight, 0)
        self.assertEqual((len(a.binds), len(a.cache_gets), len(a.launches)), (3, 3, 3))

    def test_fixture_sources_and_production_disconnection(self):
        self.assertTrue(source.validate_sources())
        for path in (ROOT / "src").rglob("*.py"):
            self.assertNotIn("m1_route_binding_adapter", path.read_text())
            self.assertNotIn("m1_route_cache_seam", path.read_text())
        imported = []
        for node in ast.walk(ast.parse((ROOT / "probes/m1_route_binding_adapter.py").read_text())):
            if isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(set(imported), {"dataclasses", "threading", "types", "m1_route_observer", "megartx.m1_route_receipt"})


if __name__ == "__main__":
    unittest.main()
