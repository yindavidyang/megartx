"""Disconnected CPU observer controls using exact-source host-path excerpts.

This fixture manually installs wrappers on newly made CPU fake instances only.
The optional Cache.get callback models an actual selection seam; it does not
recompute a binder/key, inspect a returned cache, or claim installed-source proof.
No device runtime, compilation, package installation, or production hook is used.
"""
import ast
from dataclasses import FrozenInstanceError, fields, replace
import gc
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
import weakref

_OBSERVER_PATH = Path(__file__).parents[1] / "probes" / "m1_route_observer.py"
_OBSERVER_SPEC = importlib.util.spec_from_file_location("m1_route_observer", _OBSERVER_PATH)
observer_module = importlib.util.module_from_spec(_OBSERVER_SPEC)
sys.modules[_OBSERVER_SPEC.name] = observer_module
_OBSERVER_SPEC.loader.exec_module(observer_module)
ArgumentSnapshot = observer_module.ArgumentSnapshot
KernelReference = observer_module.KernelReference
ObservationAudit = observer_module.ObservationAudit
ObservationReaders = observer_module.ObservationReaders
RouteObserver = observer_module.RouteObserver
freeze_values = observer_module.freeze_values
from megartx.m1_route_receipt import (
    BindingSnapshot, GuardSnapshot, OutputSnapshot, StreamSnapshot, UNKNOWN,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "route_observer_reference.py"
_SPEC = importlib.util.spec_from_file_location("route_observer_test_reference", _FIXTURE_PATH)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


class Owner:
    pass


class HostileError(BaseException):
    def __str__(self):
        raise AssertionError("must not format primary exception")

    def add_note(self, note):
        raise AssertionError("must not call primary exception override")


class Tensor:
    """Metadata-only fake; it has no CUDA, synchronization, or value-reading API."""
    def __init__(self, shape, dtype="float32", device=7):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.storage_owner = Owner()
        self.pointer = self.storage_pointer = 4096
        self.storage_bytes, self.offset_bytes = 32, 0
        self.strides = (shape[1], 1) if len(shape) == 2 else (1,)

    def contiguous(self):
        return self

    def to(self, dtype):
        return self if self.dtype == dtype else Tensor(self.shape, dtype, self.device)


class Cache(dict):
    """Observe the sole lookup made by the unchanged exact-source JIT body."""
    def __init__(self, fixture):
        super().__init__()
        self.fixture = fixture

    def get(self, key, default=None):
        f = self.fixture
        selected = super().get(key, default)
        f.cache_gets.append((self, key, selected))
        if f.observe_cache:
            scope = f.observer.active_scope
            observed = dict(cache=self, key=key, selected=selected, target=f.target,
                            specialization=f.last_specialization, options=f.last_options)
            observed.update(f.cache_event_changes)
            f.observer.cache_selection_observed(scope, **observed)
        f.invoke("cache_selected")
        return selected


class Fixture:
    def __init__(self, *, observe_cache=True, install=True):
        self.observe_cache = observe_cache
        self.callbacks = {}
        self.cache_event_changes = {}
        self.events, self.cache_gets, self.binds, self.launches = [], [], [], []
        self.allocations, self.compiles, self.scopes = [], [], []
        self.result = None
        self.frame, self.context, self.target = Owner(), Owner(), Owner()
        self.inner_handle, self.outer_handle = 71, 999
        self.producer_stream = StreamSnapshot(7, self.context, self.inner_handle)
        self.consumer_stream = StreamSnapshot(7, self.context, 83)
        self.specialization, self.options = [("argument", "cpu-reference")], {"num_warps": 1}
        self.guard_changes, self.binding_changes = {}, {}
        self.binding_reads = 0
        self.stream_reads = []
        self.HookChain = reference.load("HookChain")
        self.runtime = NS(debug=False,
            launch_enter_hook=self.HookChain(),
            launch_exit_hook=self.HookChain(reversed=True),
            kernel_load_start_hook=self.HookChain(),
            kernel_load_end_hook=self.HookChain(reversed=True),
            jit_cache_hook=None, jit_post_compile_hook=None,
            add_stages_inspection_hook=None)
        self.compilation = NS(instrumentation_mode="")
        knobs = NS(runtime=self.runtime, compilation=self.compilation)
        active = NS(get_current_device=lambda: 7,
                    get_current_stream=lambda device: self.inner_handle)
        self.utils = NS(launch=self.native_launch)
        active.utils = self.utils
        driver = NS(active=active)
        allocator = NS(get=self.allocator)
        CudaLauncher = reference.load("CudaLauncher", {
            "_allocation": NS(_allocator=allocator, _profile_allocator=allocator),
            "triton": NS(runtime=NS(driver=driver)),
            "expand_signature": lambda signature, meta: tuple(signature),
            "annotate_arguments": lambda signature: ("annotations", signature),
            "make_kernel_signature": lambda signature: ("signature", signature),
            "wrap_handle_tensordesc": lambda launch, signature, meta: launch,
        })
        self.metadata = NS(num_warps=1, num_ctas=1, global_scratch_size=0,
            global_scratch_align=1, profile_scratch_size=0, profile_scratch_align=1,
            launch_cooperative_grid=False, launch_pdl=False)
        self.launcher = CudaLauncher(NS(signature={i: "cpu-reference" for i in range(7)}),
                                     self.metadata)
        compiled_globals = {"driver": driver, "knobs": knobs,
            "LazyDict": reference.load("LazyDict"), "ASTSource": type("ASTSource", (), {})}
        CompiledKernel = type("CompiledKernel", (), {
            "run": property(reference.load("CompiledKernel.run")),
            "_init_handles": reference.load("CompiledKernel._init_handles", compiled_globals),
            "launch_metadata": reference.load("CompiledKernel.launch_metadata", compiled_globals),
        })
        self.kernel = CompiledKernel()
        self.kernel._run = self.launcher
        self.kernel.module, self.kernel.function = Owner(), Owner()
        self.kernel.packed_metadata, self.kernel.metadata = Owner(), self.metadata
        self.kernel.src, self.kernel.name = Owner(), "exact-source-cpu-only"
        key_function = reference.load("compute_cache_key", {
            "JITCallable": type("JITCallable", (), {}),
            "is_namedtuple": lambda obj: isinstance(obj, tuple) and hasattr(obj, "_fields")})
        key_cache = {}
        self.key = key_function(key_cache, self.specialization, self.options)
        JIT = type("JIT", (), {
            "__getitem__": reference.load("KernelInterface.__getitem__"),
            "run": reference.load("JITFunction.run", {
                "driver": driver, "knobs": knobs, "compute_cache_key": key_function}),
        })
        self.jit = JIT()
        self.jit.debug, self.jit.pre_run_hooks, self.jit.used_global_vals = False, [], {}
        self.jit.launch_metadata = None
        self.jit._pack_args = lambda *args: ({}, {}, {}, {})
        self.jit._do_compile = self.compile
        self.compile_result = None
        self.cache = Cache(self)
        self.cache[self.key] = self.kernel
        self.jit.device_caches = {7: (self.cache, key_cache, self.target, Owner(), self.binder)}
        torch = NS(float32="float32", int32="int32", int16="int16",
                   empty=lambda *shape, dtype, device: Tensor(shape, dtype, device))
        routing = reference.load("gemma4_fused_routing_kernel_triton", {
            "torch": torch, "triton": NS(next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
            "_gemma4_routing_kernel": self.jit})
        self.scale = Tensor((128,))
        scale = self.scale
        def custom_routing_function(**kwargs):
            return routing(kwargs["gating_output"], kwargs["topk"], scale)
        Router = type("Router", (), {
            "select_experts": reference.load("FusedMoERouter.select_experts", {"torch": torch}),
            "_select_experts": reference.load("BaseRouter._select_experts"),
            "_compute_routing": reference.load("CustomRoutingRouter._compute_routing", {"torch": torch}),
            "_validate_eplb_state": reference.load("BaseRouter._validate_eplb_state"),
            "_apply_eplb_mapping": reference.load("BaseRouter._apply_eplb_mapping"),
            "_convert_indices_dtype": reference.load("BaseRouter._convert_indices_dtype"),
        })
        self.router = Router()
        self.router.eplb_state = self.router.capture_fn = self.router._routing_replay_out = None
        self.router.top_k, self.router.renormalize = 8, True
        self.router.custom_routing_function = custom_routing_function
        self.prepare_finalize = NS(prepare=lambda *args: args, finalize=lambda *args: args)
        self.experts = NS(quant_method=NS(is_monolithic=False), forward_modular=self.forward_modular)
        Runner = type("Runner", (), {"_apply_quant_method": reference.load(
            "MoERunner._apply_quant_method", {"SharedExpertsOrder": NS(NO_OVERLAP=Owner())})})
        self.runner = Runner()
        self.runner.router, self.runner.routed_experts = self.router, self.experts
        self.runner._quant_method = NS(topk_indices_dtype="int32")
        self.runner._shared_experts = None
        self.runner._maybe_apply_shared_experts = self.shared_before
        self.originals = dict(runner=self.runner._apply_quant_method,
            select=self.router.select_experts, jit=self.jit.run, launch=self.launcher)
        self.expected = BindingSnapshot(runner=self.runner, layer=self.experts,
            router=self.router, select_experts=self.originals["select"],
            custom_routing=self.router.custom_routing_function, closure_owner=self.scale,
            prepare_finalize=self.prepare_finalize, prepare=self.prepare_finalize.prepare,
            experts=self.experts, jit_function=self.jit, jit_run=self.originals["jit"],
            compiled_kernel=self.kernel, cuda_module=self.kernel.module,
            cuda_function=self.kernel.function, launcher=self.launcher,
            source_key="exact-source-reference-not-installed-proof", cache_key=self.key)
        self.reference = KernelReference(cache=self.cache, target=self.target,
            specialization=freeze_values(self.specialization), options=freeze_values(self.options),
            metadata=self.metadata, metadata_fields=tuple(vars(self.metadata).items()),
            packed_metadata=self.kernel.packed_metadata, launcher_launch=self.launcher.launch,
            utils_launch=self.utils.launch, read_utils_launch=lambda: self.utils.launch,
            runtime=self.runtime, compilation=self.compilation, hookchain_type=self.HookChain,
            hookchain_call=self.HookChain.__call__,
            extra_bindings=(self.prepare_finalize.finalize,))
        self.readers = ObservationReaders(bindings=self.bindings, guards=self.guards,
            frame=lambda: self.frame, output=self.output, stream=self.stream,
            extra_bindings=lambda: (self.prepare_finalize.finalize,),
            argument=lambda owner: ArgumentSnapshot(owner, owner.device, owner.shape, owner.strides, owner.dtype))
        self.observer = RouteObserver(self.expected, self.reference, self.readers)
        self.wrappers = dict(
            runner=self.observer.wrap_runner(self.runner, self.originals["runner"]),
            select=self.observer.wrap_select(self.router, self.originals["select"]),
            jit=self.observer.wrap_jit(self.jit, self.originals["jit"]),
            launch=self.observer.wrap_launcher(self.kernel, self.originals["launch"]))
        if install:
            self.install()

    def install(self):
        self.runner._apply_quant_method = self.wrappers["runner"]
        self.router.select_experts = self.wrappers["select"]
        self.jit.run = self.wrappers["jit"]
        self.kernel._run = self.wrappers["launch"]

    def invoke(self, event):
        callback = self.callbacks.get(event)
        if callback is not None:
            callback()

    def allocator(self):
        self.allocations.append("requested")
        return lambda *args: self.allocations.append(args)

    def compile(self, *args):
        self.compiles.append(args)
        return self.compile_result

    def binder(self, *args, **kwargs):
        self.binds.append((args, kwargs))
        bound = dict(enumerate(args))
        self.last_specialization, self.last_options = list(self.specialization), dict(self.options)
        self.bound = bound
        self.invoke("binder")
        return bound, self.last_specialization, self.last_options

    def native_launch(self, *args):
        self.launches.append(args)
        self.events.append("native_launch")
        self.invoke("launch")
        return self.result

    def shared_before(self, *args):
        self.events.append("runner_enter")
        self.scopes.append(self.observer.active_scope)
        self.invoke("runner")

    def forward_modular(self, **kwargs):
        self.events.append("consumer")
        self.invoke("consumer")
        scope = self.observer.active_scope
        self.events.append("existing_wait_succeeded")
        self.observer.dependency_observed(scope, self.producer_stream, self.consumer_stream)
        self.receipt = self.observer.checked_consumer(scope, kwargs["topk_ids"],
            self.bindings(), self.frame, self.consumer_stream)
        # These stand for the production path's existing mandatory checks.
        self.events.extend(("route_D2H", "host_synchronization", "native_route_check"))
        self.invoke("after_consumer")
        return kwargs["topk_ids"]

    def bindings(self):
        self.binding_reads += 1
        current = dict(select_experts=self.router.select_experts,
            custom_routing=self.router.custom_routing_function,
            prepare=self.prepare_finalize.prepare, jit_run=self.jit.run,
            cuda_module=self.kernel.module, cuda_function=self.kernel.function,
            launcher=self.kernel._run, cache_key=UNKNOWN)
        current.update(self.binding_changes)
        return replace(self.expected, **current)

    def guards(self):
        return replace(GuardSnapshot(capture_state="none", capture_fn=self.router.capture_fn,
            replay_output=self.router._routing_replay_out, eplb_state=self.router.eplb_state,
            instance_overrides=self.observer.foreign_overrides(self.router,
                ("select_experts", "_select_experts", "_compute_routing")),
            pre_run_hooks=tuple(self.jit.pre_run_hooks), launch_hooks=(),
            simulation=False, diagnostics=False, controlled=False, forced=False,
            routing_variant="gemma_cuda"), **self.guard_changes)

    def output(self, ids):
        return OutputSnapshot(ids, ids.storage_owner, ids.pointer, ids.storage_pointer,
            ids.storage_bytes, ids.offset_bytes, ids.device, ids.shape, ids.strides, ids.dtype)

    def stream(self, handle):
        self.stream_reads.append(handle)
        return StreamSnapshot(7, self.context, handle)

    def run(self, hidden=None, scores=None, **kwargs):
        return self.runner._apply_quant_method(
            hidden or Tensor((1, 64)), scores or Tensor((1, 128)), None, **kwargs)


class ExactProfile:
    """Record exact extracted callable execution without wrapping its body."""
    def __init__(self, f):
        self.codes = {getattr(fn, "__func__", fn).__code__: name
            for name, fn in f.originals.items() if name != "launch"}
        self.codes[type(f.launcher).__call__.__code__] = "launch"
        self.calls, self.returns = {}, {}

    def profile(self, frame, event, result):
        name = self.codes.get(frame.f_code)
        if name:
            if event == "call":
                self.calls.setdefault(name, []).append(dict(frame.f_locals))
            elif event == "return":
                self.returns.setdefault(name, []).append(result)

    def __enter__(self):
        self.previous = sys.getprofile()
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(self.previous)


class RouteObserverTests(unittest.TestCase):
    def assert_checked(self, f, complete=False):
        self.assertEqual(f.events.count("route_D2H"), 1)
        self.assertEqual(f.events.count("host_synchronization"), 1)
        self.assertEqual(f.events.count("native_route_check"), 1)
        self.assertEqual(f.observer.last_audit.complete, complete, f.observer.last_audit.reasons)
        self.assertIsNone(f.observer.active_scope)
        self.assertFalse(f.observer.ledger.active)
        self.assertTrue(f.observer.last_audit.route_readback_required)
        self.assertTrue(f.observer.last_audit.synchronization_required)
        self.assertTrue(f.observer.last_audit.native_checks_required)

    def test_reference_slices_validate_and_load_without_runtime_imports(self):
        self.assertTrue(reference.validate_sources())
        module = _OBSERVER_PATH
        tree = ast.parse(module.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(set(imported), {"dataclasses", "threading", "types", "megartx.m1_route_receipt"})
        # No production source may depend on this disconnected observer.
        for path in (module.parents[1] / "src").rglob("*.py"):
            if path != module:
                self.assertNotIn("m1_route_observer", path.read_text(), str(path))

    def test_factories_install_nothing_and_only_owned_wrappers_are_recognized(self):
        f = Fixture(install=False)
        self.assertIs(f.kernel._run, f.launcher)
        self.assertIs(f.runner._apply_quant_method.__func__, f.originals["runner"].__func__)
        self.assertIs(f.router.select_experts.__func__, f.originals["select"].__func__)
        self.assertIs(f.jit.run.__func__, f.originals["jit"].__func__)
        self.assertEqual(f.events + f.binds + f.cache_gets + f.launches, [])
        f.install()
        self.assertTrue(f.observer.owns(f.router, "select_experts", f.router.select_experts))
        self.assertFalse(f.observer.owns(Owner(), "select_experts", f.router.select_experts))
        self.assertEqual(f.observer.foreign_overrides(f.router, ("select_experts",)), ())
        with self.assertRaises(ValueError):
            f.observer.wrap_select(f.router, f.originals["select"])

    def test_actual_selection_completes_exact_source_chain_once_with_original_objects(self):
        f = Fixture()
        hidden, scores, input_ids = Tensor((1, 64)), Tensor((1, 128)), Owner()
        with ExactProfile(f) as profile:
            result = f.run(hidden, scores, input_ids=input_ids)
        self.assert_checked(f, complete=True)
        self.assertEqual({key: len(value) for key, value in profile.calls.items()},
                         dict(runner=1, select=1, jit=1, launch=1))
        self.assertIs(profile.returns["runner"][0], result)
        self.assertIs(profile.returns["select"][0][1], result[1])
        self.assertIs(profile.returns["jit"][0], f.kernel)
        self.assertIsNone(profile.returns["launch"][0])
        self.assertIs(profile.calls["runner"][0]["hidden_states"], hidden)
        self.assertIs(profile.calls["runner"][0]["input_ids"], input_ids)
        self.assertIs(profile.calls["select"][0]["router_logits"], scores)
        actual = profile.calls["launch"][0]
        self.assertIs(actual["function"], f.expected.cuda_function)
        self.assertIs(actual["kernel_metadata"], f.kernel.packed_metadata)
        self.assertIs(actual["args"][0], scores)
        self.assertIs(actual["args"][1], f.scale)
        self.assertIs(actual["args"][3], result[1])
        self.assertEqual(actual["stream"], f.inner_handle)
        self.assertEqual(len(f.binds), 1)
        self.assertEqual(len(f.cache_gets), 1)
        self.assertIs(f.cache_gets[0][0], f.cache)
        self.assertIs(f.cache_gets[0][2], f.kernel)
        self.assertEqual(len(f.launches), 1)
        self.assertIs(f.launches[0][-1][3], result[1])
        self.assertEqual(f.launches[0][11:13], (None, None))
        self.assertEqual(f.allocations + f.compiles, [])
        self.assertEqual(f.observer.last_audit.cache_key, f.key)
        self.assertIs(f.observer.last_audit.receipt, f.receipt)

    def test_default_missing_actual_cache_seam_is_unknown_not_reference_key(self):
        f = Fixture(observe_cache=False)
        result = f.run()
        self.assert_checked(f)
        self.assertIs(f.observer.last_audit.cache_key, UNKNOWN)
        self.assertIn("cache selection unobserved", f.observer.last_audit.reasons)
        self.assertEqual((len(f.cache_gets), len(f.binds), len(f.launches)), (1, 1, 1))
        self.assertIs(result[1], f.launches[0][-1][3])
        self.assertFalse(f.observer.poisoned)

    def test_actual_inner_stream_used_instead_of_outer_current_stream(self):
        f = Fixture()
        f.run()
        self.assert_checked(f, complete=True)
        self.assertEqual(f.stream_reads, [f.inner_handle, f.inner_handle])
        self.assertNotIn(f.outer_handle, f.stream_reads)
        self.assertEqual(f.receipt.producer_stream, f.inner_handle)
        self.assertEqual(f.receipt.consumer_stream, f.consumer_stream.handle)
        self.assertEqual(f.observer.last_audit.inner_stream, f.inner_handle)

    def test_audit_is_immutable_non_boolean_and_cannot_disable_mandatory_checks(self):
        f = Fixture()
        f.run()
        audit = f.observer.last_audit
        with self.assertRaises(TypeError):
            bool(audit)
        for name in ("route_readback_required", "synchronization_required", "native_checks_required"):
            with self.subTest(name=name), self.assertRaises((FrozenInstanceError, AttributeError)):
                setattr(audit, name, False)
        with self.assertRaises(TypeError):
            ObservationAudit(1, True, (), f.key, 71, None, False, native_checks_required=False)

    def test_repeat_generations_reuse_no_old_scope_or_receipt(self):
        f = Fixture()
        f.run()
        first, old_scope = f.observer.last_audit, f.scopes[0]
        f.events.clear()
        f.run()
        self.assert_checked(f, complete=True)
        second = f.observer.last_audit
        self.assertEqual((first.generation, second.generation), (1, 2))
        self.assertIsNot(first.receipt, second.receipt)
        self.assertIsNot(old_scope, f.scopes[1])
        self.assertEqual(old_scope.retained, [])
        self.assertEqual(old_scope.launches, [])
        self.assertIs(old_scope.selected, UNKNOWN)
        self.assertIs(old_scope.frame, UNKNOWN)
        self.assertIsNone(old_scope.jit_args)

    def test_actual_runner_ownership_edges_cannot_be_hidden_by_stale_reader(self):
        for edge in ("router", "routed_experts"):
            f = Fixture()
            if edge == "router":
                replacement = NS(select_experts=f.router.select_experts)
            else:
                replacement = NS(quant_method=f.experts.quant_method,
                                 forward_modular=f.experts.forward_modular)
            f.callbacks["runner"] = lambda: setattr(f.runner, edge, replacement)
            with self.subTest(edge=edge):
                f.run()
                self.assert_checked(f)
                self.assertIn("actual runner ownership changed", f.observer.last_audit.reasons)
                self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))

    def test_compiled_entrypoint_source_and_descriptor_substitution_rejected(self):
        for mode in ("launch_metadata", "_init_handles", "run", "source", "source_fn"):
            f = Fixture()
            seen = []
            if mode in ("launch_metadata", "_init_handles"):
                original = getattr(f.kernel, mode)
                setattr(f.kernel, mode, lambda *args: (seen.append(mode), original(*args))[1])
            elif mode == "run":
                type(f.kernel).run = property(lambda kernel: kernel._run)
            elif mode == "source":
                f.kernel.src = Owner()
            else:
                f.kernel.src.fn = Owner()
            with self.subTest(mode=mode):
                f.run()
                self.assert_checked(f)
                self.assertEqual(len(f.launches), 1)
                if mode in ("launch_metadata", "_init_handles"):
                    self.assertEqual(seen, [mode])

    def test_foreign_router_helpers_instance_and_class_substitution_rejected(self):
        for helper in ("_select_experts", "_compute_routing", "_validate_eplb_state",
                       "_apply_eplb_mapping", "_convert_indices_dtype"):
            for location in ("instance", "class"):
                f = Fixture()
                calls = []
                original = getattr(f.router, helper)
                if location == "instance":
                    setattr(f.router, helper, lambda *a, **kw: (calls.append(helper), original(*a, **kw))[1])
                else:
                    setattr(type(f.router), helper,
                            lambda owner, *a, **kw: (calls.append(helper), original(*a, **kw))[1])
                with self.subTest(helper=helper, location=location):
                    f.run()
                    self.assert_checked(f)
                    self.assertEqual(calls, [helper])
                    self.assertIn("router/runner/consumer helper changed or unknown",
                                  f.observer.last_audit.reasons)

    def test_factory_rejects_wrong_current_original_before_registering_wrapper(self):
        f = Fixture(install=False)
        other = RouteObserver(f.expected, f.reference, f.readers)
        original = f.runner._apply_quant_method
        calls = []
        def foreign(*args, **kwargs):
            calls.append("foreign")
            return original(*args, **kwargs)
        with self.assertRaises(Exception):
            other.wrap_runner(f.runner, foreign)
        self.assertEqual(calls, [])
        self.assertEqual(other._slots, {})
        f.router.select_experts = lambda *args, **kwargs: None
        with self.assertRaises(Exception):
            other.wrap_select(f.router, f.originals["select"])
        self.assertEqual(other._slots, {})

    def test_each_guard_unknown_or_active_keeps_exact_production_checked(self):
        for info in fields(GuardSnapshot):
            f = Fixture()
            f.guard_changes[info.name] = UNKNOWN
            with self.subTest(guard=info.name):
                f.run()
                self.assert_checked(f)
                self.assertIn("guard/callback unknown or active", f.observer.last_audit.reasons)
                self.assertEqual(len(f.launches), 1)
        for name, value in (("simulation", True), ("diagnostics", True),
                            ("controlled", True), ("forced", True),
                            ("routing_variant", "cpu")):
            f = Fixture()
            f.guard_changes[name] = value
            with self.subTest(guard=name, active=True):
                f.run()
                self.assert_checked(f)

    def test_each_hook_chain_requires_exact_type_calls_direction_and_no_override(self):
        for name, reverse in (("launch_enter_hook", False), ("launch_exit_hook", True),
                              ("kernel_load_start_hook", False), ("kernel_load_end_hook", True)):
            for mutation in ("calls_tuple", "callback", "direction", "instance_call", "subclass"):
                f = Fixture()
                chain = getattr(f.runtime, name)
                if mutation == "calls_tuple":
                    chain.calls = ()
                elif mutation == "callback":
                    chain.calls.append(lambda *args: None)
                elif mutation == "direction":
                    chain.reversed = not reverse
                elif mutation == "instance_call":
                    chain.__call__ = lambda *args: None
                else:
                    setattr(f.runtime, name, type("ForeignChain", (f.HookChain,), {})(reversed=reverse))
                with self.subTest(hook=name, mutation=mutation):
                    f.run()
                    self.assert_checked(f)
                    self.assertEqual(len(f.launches), 1)

    def test_hook_call_implementation_and_optional_callbacks_are_rejected(self):
        f = Fixture()
        f.HookChain.__call__ = lambda *args: None
        f.run()
        self.assert_checked(f)
        self.assertIn("unknown hook chain", f.observer.last_audit.reasons)
        for name in ("jit_cache_hook", "jit_post_compile_hook", "add_stages_inspection_hook"):
            f = Fixture()
            setattr(f.runtime, name, lambda: ("stage", "hash"))
            with self.subTest(callback=name):
                f.run()
                self.assert_checked(f)
        for where, name in (("compilation", "instrumentation_mode"), ("jit", "launch_metadata")):
            f = Fixture()
            setattr(getattr(f, where), name, lambda: None)
            with self.subTest(callback=name):
                f.run()
                self.assert_checked(f)

    def test_prepare_finalize_and_custom_callable_substitution_before_or_after_launch(self):
        for phase in ("runner", "launch", "consumer"):
            for name in ("prepare", "finalize", "custom"):
                f = Fixture()
                def mutate(f=f, name=name):
                    if name == "custom":
                        original = f.router.custom_routing_function
                        f.router.custom_routing_function = lambda **kwargs: original(**kwargs)
                    else:
                        setattr(f.prepare_finalize, name, lambda *args: args)
                f.callbacks[phase] = mutate
                with self.subTest(phase=phase, binding=name):
                    f.run()
                    self.assert_checked(f)
                    self.assertEqual(len(f.launches), 1)

    def test_closure_and_original_callable_code_mutation_detected(self):
        f = Fixture()
        original = f.router.custom_routing_function
        scale_cell = next(cell for cell in original.__closure__ if cell.cell_contents is f.scale)
        scale_cell.cell_contents = Tensor((128,))
        f.run()
        self.assert_checked(f)
        self.assertIn("producer callable/closure changed", f.observer.last_audit.reasons)
        f = Fixture()
        original = f.originals["runner"].__func__
        original.__defaults__ = (Owner(), False)
        f.run()
        self.assert_checked(f)
        self.assertIn("original callable changed", f.observer.last_audit.reasons)

    def test_foreign_instance_override_is_not_erased_or_unwrapped(self):
        f = Fixture()
        original = f.router._compute_routing
        def foreign(*args, **kwargs):
            return original(*args, **kwargs)
        foreign.__wrapped__ = original
        f.router._compute_routing = foreign
        self.assertEqual(f.observer.foreign_overrides(f.router,
            ("select_experts", "_compute_routing")), ("_compute_routing",))
        f.run()
        self.assert_checked(f)
        self.assertIn("router/runner/consumer helper changed or unknown", f.observer.last_audit.reasons)

    def test_owned_wrapper_replacement_still_delegates_without_accepting_foreign_wrapper(self):
        for kind in ("select", "jit", "launch"):
            f = Fixture()
            owner, name = {"select": (f.router, "select_experts"),
                           "jit": (f.jit, "run"), "launch": (f.kernel, "_run")}[kind]
            original = getattr(owner, name)
            def foreign(*args, **kwargs):
                return original(*args, **kwargs)
            foreign.__wrapped__ = original
            setattr(owner, name, foreign)
            with self.subTest(kind=kind):
                f.run()
                self.assert_checked(f)
                self.assertEqual(len(f.launches), 1)
                self.assertIn("owned wrapper replaced or absent", f.observer.last_audit.reasons)

    def test_bound_owner_and_kernel_function_mutation_rejected(self):
        for change in ("input", "output", "function", "packed_metadata"):
            f = Fixture()
            def mutate(f=f, change=change):
                if change in ("input", "output"):
                    index = 0 if change == "input" else 3
                    f.bound[index] = Tensor((1, 128) if index == 0 else (1, 8),
                                            "float32" if index == 0 else "int32")
                elif change == "function":
                    f.kernel.function = Owner()
                else:
                    f.kernel.packed_metadata = Owner()
            f.callbacks["binder"] = mutate
            with self.subTest(change=change):
                f.run()
                self.assert_checked(f)
                self.assertEqual(len(f.launches), 1)

    def test_output_view_mutation_across_launch_or_at_consumer_is_rejected(self):
        for phase in ("launch", "consumer"):
            for field, value in (("pointer", 4100), ("shape", (2, 8)),
                                 ("dtype", "int64"), ("storage_owner", Owner())):
                f = Fixture()
                f.callbacks[phase] = lambda f=f, field=field, value=value: setattr(f.bound[3], field, value)
                with self.subTest(phase=phase, field=field):
                    f.run()
                    self.assert_checked(f)
                    self.assertTrue(f.observer.poisoned)

    def test_launcher_metadata_and_underlying_target_mutation_rejected(self):
        mutations = (("launcher", "launch_cooperative_grid", True),
            ("launcher", "launch_pdl", True), ("launcher", "num_ctas", 2),
            ("launcher", "global_scratch_align", 2), ("launcher", "profile_scratch_align", 2),
            ("launcher", "arg_annotations", Owner()), ("launcher", "kernel_signature", Owner()),
            ("metadata", "num_warps", 2), ("metadata", "num_ctas", 2),
            ("metadata", "global_scratch_size", 4), ("metadata", "profile_scratch_size", 4))
        for phase in ("runner", "launch"):
            for owner, name, value in mutations:
                f = Fixture()
                f.callbacks[phase] = lambda f=f, owner=owner, name=name, value=value: setattr(getattr(f, owner), name, value)
                with self.subTest(phase=phase, owner=owner, field=name):
                    f.run()
                    self.assert_checked(f)
        for owner in ("utils", "launcher"):
            f = Fixture()
            original = getattr(f, owner).launch
            setattr(getattr(f, owner), "launch", lambda *args: original(*args))
            with self.subTest(target=owner):
                f.run()
                self.assert_checked(f)

    def test_cold_cache_async_and_unqualified_returns_do_not_retry_or_warm(self):
        for mode in ("async", "cold_then_initialized", "unqualified"):
            f = Fixture()
            f.cache.clear()
            if mode == "cold_then_initialized":
                f.compile_result = f.kernel
            elif mode == "unqualified":
                foreign = NS(function=Owner(), packed_metadata=Owner(),
                    launch_metadata=lambda *args: None, run=lambda *args: f.events.append("foreign_launch"))
                f.cache[f.key] = foreign
            with self.subTest(mode=mode):
                f.run()
                self.assert_checked(f)
                self.assertEqual(len(f.binds), 1)
                self.assertEqual(len(f.cache_gets), 1)
                self.assertEqual(len(f.compiles), 0 if mode == "unqualified" else 1)
                self.assertEqual(len(f.launches), 1 if mode == "cold_then_initialized" else 0)
                self.assertEqual(f.observer.last_audit.cache_key, f.key)
                self.assertIn("cold/unqualified cache kernel", f.observer.last_audit.reasons)

    def test_new_specialization_or_cache_key_never_reuses_expected_key(self):
        f = Fixture()
        f.specialization.append(("new", "specialization"))
        f.run()
        self.assert_checked(f)
        self.assertIs(f.observer.last_audit.cache_key, UNKNOWN)
        self.assertEqual(len(f.cache_gets), 1)
        self.assertEqual(len(f.binds), 1)
        self.assertIn("unqualified cache selection", f.observer.last_audit.reasons)

    def test_duplicate_cache_event_is_rejected_without_second_lookup(self):
        f = Fixture()
        f.callbacks["cache_selected"] = lambda: f.observer.cache_selection_observed(
            f.observer.active_scope, f.cache, f.key, f.kernel, f.target,
            f.specialization, f.options)
        f.run()
        self.assert_checked(f)
        self.assertEqual(len(f.cache_gets), 1)
        self.assertEqual(len(f.launches), 1)
        self.assertIn("cache selection absent/duplicate/out of order", f.observer.last_audit.reasons)

    def test_missing_inner_launch_is_incomplete_even_with_selected_kernel(self):
        f = Fixture()
        type(f.kernel).run = property(lambda kernel: lambda *args: f.events.append("unobserved_launch"))
        f.run()
        self.assert_checked(f)
        self.assertEqual(f.events.count("unobserved_launch"), 1)
        self.assertEqual(len(f.launches), 0)
        self.assertTrue(set(f.observer.last_audit.reasons) &
                        {"missing or duplicate inner launch", "compiled run descriptor changed or unknown"})

    def test_duplicate_inner_launch_delegates_each_call_once_but_poisons(self):
        f = Fixture()
        metadata = f.kernel.launch_metadata
        def launch_during_metadata(grid, stream, *args):
            result = metadata(grid, stream, *args)
            f.kernel.run(1, 1, 1, stream, f.kernel.function, f.kernel.packed_metadata,
                         result, f.runtime.launch_enter_hook, f.runtime.launch_exit_hook, *args)
            return result
        f.kernel.launch_metadata = launch_during_metadata
        f.run()
        self.assert_checked(f)
        self.assertEqual(len(f.launches), 2)
        self.assertIn("duplicate inner launch", f.observer.last_audit.reasons)
        self.assertTrue(f.observer.poisoned)

    def test_unknown_frame_and_reader_failure_cannot_interrupt_delegation(self):
        for frame in (None, UNKNOWN):
            f = Fixture()
            f.frame = frame
            with self.subTest(frame=frame):
                f.run()
                self.assert_checked(f)
                self.assertIn("unknown frame", f.observer.last_audit.reasons)
        for reader in ("bindings", "guards", "frame", "output", "stream", "extra_bindings", "argument"):
            f = Fixture()
            def failure(*args):
                raise HostileError()
            f.observer.readers = replace(f.readers, **{reader: failure})
            with self.subTest(reader=reader):
                f.run()
                self.assert_checked(f)
                self.assertTrue(f.observer.poisoned)
                self.assertEqual(len(f.launches), 1)

    def test_unrelated_global_jit_and_launcher_calls_delegate_outside_scope(self):
        f = Fixture()
        inputs = (Tensor((1, 128)), f.scale, Tensor((1, 8)), Tensor((1, 8), "int32"), 128, 8, 128)
        result = f.jit.run(*inputs, grid=(1,), warmup=False, num_warps=1)
        self.assertIs(result, f.kernel)
        self.assertEqual(len(f.launches), 1)
        self.assertIsNone(f.observer.last_audit)
        self.assertFalse(f.observer.poisoned)
        self.assertIsNone(f.observer.active_scope)

    def test_nested_runner_poison_does_not_hide_or_repeat_real_work(self):
        f = Fixture()
        def nested():
            del f.callbacks["runner"]
            f.run()
        f.callbacks["runner"] = nested
        f.run()
        self.assertEqual(f.events.count("runner_enter"), 2)
        self.assertEqual(len(f.launches), 2)
        self.assertEqual(f.events.count("native_route_check"), 2)
        self.assertFalse(f.observer.last_audit.complete)
        self.assertTrue(f.observer.poisoned)
        self.assertIn("nested or concurrent runner", f.observer.last_audit.reasons)
        self.assertIsNone(f.observer.active_scope)
        self.assertFalse(f.observer.ledger.active)

    def test_foreign_thread_event_poison_preserves_owner_scope_until_cleanup(self):
        f = Fixture()
        failures = []
        def foreign_event():
            scope = f.observer.active_scope
            def work():
                try:
                    self.assertIsNone(f.observer.active_scope)
                    f.observer.dependency_observed(scope, f.producer_stream, f.consumer_stream)
                except BaseException as error:
                    failures.append(error)
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertIs(f.observer.active_scope, scope)
            self.assertTrue(scope.retained)
        f.callbacks["consumer"] = foreign_event
        f.run()
        self.assertEqual(failures, [])
        self.assert_checked(f)
        self.assertTrue(f.observer.poisoned)
        self.assertIn("foreign/stale event scope", f.observer.last_audit.reasons)

    def test_stale_scope_is_rejected_in_new_scope_but_idle_events_are_harmless(self):
        f = Fixture()
        f.run()
        old_scope = f.scopes[0]
        f.observer.dependency_observed(old_scope, f.producer_stream, f.consumer_stream)
        self.assertFalse(f.observer.poisoned)
        f.events.clear()
        f.callbacks["runner"] = lambda: f.observer.cache_selection_observed(old_scope,
            f.cache, f.key, f.kernel, f.target, f.specialization, f.options)
        f.run()
        self.assert_checked(f)
        self.assertIn("foreign/stale event scope", f.observer.last_audit.reasons)

    def test_original_exception_identity_survives_observer_and_cleanup_failures(self):
        for phase in ("runner", "binder", "launch", "consumer", "after_consumer"):
            f = Fixture()
            primary = HostileError()
            def fail():
                raise primary
            f.callbacks[phase] = fail
            def cleanup(*args):
                raise HostileError()
            f.observer.ledger._cleanup = cleanup
            with self.subTest(phase=phase), self.assertRaises(HostileError) as raised:
                f.run()
            self.assertIs(raised.exception, primary)
            self.assertTrue(f.observer.poisoned)
            self.assertIsNone(f.observer.active_scope)
            self.assertFalse(f.observer.ledger.active)
            self.assertIsNone(f.scopes[0].primary)
            self.assertEqual(f.scopes[0].retained, [])
            self.assertLessEqual(len(f.launches), 1)

    def test_cleanup_failure_preserves_successful_return_but_poison_is_terminal(self):
        f = Fixture()
        def cleanup(*args):
            raise HostileError()
        f.observer.ledger._cleanup = cleanup
        result = f.run()
        self.assertIs(result[1], f.launches[0][-1][3])
        self.assert_checked(f)
        self.assertTrue(f.observer.poisoned)
        f.events.clear()
        f.run()
        self.assert_checked(f)
        self.assertIn("observer poisoned", f.observer.last_audit.reasons)
        self.assertEqual(len(f.launches), 2)

    def test_output_and_frame_owners_held_through_runner_cleanup_then_released(self):
        f = Fixture()
        refs = {}
        def consumer():
            scope = f.observer.active_scope
            refs["output"] = weakref.ref(f.bound[3])
            refs["storage"] = weakref.ref(f.bound[3].storage_owner)
            refs["frame"] = weakref.ref(f.frame)
            refs["scope"] = scope
            f.observer.readers = replace(f.observer.readers, frame=lambda: refs["frame"]())
        def after():
            # Remove fixture-side references while observer scope is still live.
            f.binds.clear()
            f.launches.clear()
            f.bound.clear()
            f.frame = None
            gc.collect()
            self.assertTrue(all(refs[name]() is not None for name in ("output", "storage", "frame")))
        f.callbacks["consumer"] = consumer
        f.callbacks["after_consumer"] = after
        result = f.run()
        self.assert_checked(f, complete=True)
        self.assertEqual(refs["scope"].retained, [])
        self.assertEqual(refs["scope"].launches, [])
        del result
        gc.collect()
        self.assertIsNone(refs["output"]())
        self.assertIsNone(refs["storage"]())
        self.assertIsNone(refs["frame"]())

    def test_every_current_binding_missing_or_substituted_rejects(self):
        for info in fields(BindingSnapshot):
            for value in (None if info.name == "cache_key" else UNKNOWN,
                          "unobserved-key" if info.name.endswith("key") else Owner()):
                f = Fixture()
                f.binding_changes[info.name] = value
                with self.subTest(binding=info.name, value=type(value).__name__):
                    f.run()
                    self.assert_checked(f)
                    self.assertEqual(len(f.launches), 1)

    def test_reader_cannot_claim_reference_cache_key_without_actual_selection(self):
        f = Fixture(observe_cache=False)
        f.binding_changes["cache_key"] = f.key
        f.run()
        self.assert_checked(f)
        self.assertIs(f.observer.last_audit.cache_key, UNKNOWN)
        self.assertIn("binding reader must leave cache selection unknown", f.observer.last_audit.reasons)
        self.assertEqual((len(f.cache_gets), len(f.binds)), (1, 1))

    def test_exact_empty_hookchain_replacement_still_rejects_before_and_after(self):
        for phase in ("runner", "launch", "consumer"):
            for name, reverse in (("launch_enter_hook", False), ("launch_exit_hook", True),
                                  ("kernel_load_start_hook", False), ("kernel_load_end_hook", True)):
                f = Fixture()
                f.callbacks[phase] = lambda f=f, name=name, reverse=reverse: setattr(
                    f.runtime, name, f.HookChain(reversed=reverse))
                with self.subTest(phase=phase, name=name):
                    f.run()
                    self.assert_checked(f)
                    self.assertIn("hook chain replaced", f.observer.last_audit.reasons)
                    self.assertEqual(len(f.launches), 1)

    def test_warmup_callable_grid_unknown_geometry_and_options_delegate_once(self):
        cases = (("warmup", True), ("grid", lambda bound: (1,)),
                 ("grid", [1]), ("grid", (True,)), ("grid", (2,)),
                 ("num_warps", 2), ("num_warps", True), ("debug", False))
        for name, value in cases:
            f = Fixture()
            def getitem(jit, grid, name=name, value=value):
                def call(*args, **kwargs):
                    options = dict(grid=grid, warmup=False, **kwargs)
                    options[name] = value
                    return jit.run(*args, **options)
                return call
            type(f.jit).__getitem__ = getitem
            with self.subTest(name=name, value=type(value).__name__):
                with ExactProfile(f) as profile:
                    f.run()
                self.assert_checked(f)
                self.assertEqual(len(profile.calls["jit"]), 1)
                self.assertEqual(len(f.cache_gets), 1)
                self.assertEqual(len(f.binds), 1)
                self.assertEqual(len(f.launches), 0 if name == "warmup" else 1)
                self.assertEqual(f.compiles, [])

    def test_all_input_views_must_be_known_and_match_before_and_after(self):
        for owner_index in range(4):
            for field, value in (("device", True), ("shape", (1,)),
                                 ("strides", (9,)), ("dtype", "unknown")):
                for phase in ("binder", "launch"):
                    f = Fixture()
                    f.callbacks[phase] = lambda f=f, i=owner_index, field=field, value=value: setattr(
                        f.bound[i], field, value)
                    with self.subTest(owner=owner_index, field=field, phase=phase):
                        f.run()
                        self.assert_checked(f)
                        self.assertEqual(len(f.launches), 1)
                        if phase == "launch":
                            self.assertTrue(f.observer.poisoned)

    def test_actual_stream_reader_must_describe_actual_inner_handle_and_device(self):
        for changed in (dict(handle=999), dict(device=8), dict(context=UNKNOWN), dict(handle=True)):
            f = Fixture()
            f.observer.readers = replace(f.readers,
                stream=lambda handle: replace(f.producer_stream, **changed))
            with self.subTest(changed=changed):
                f.run()
                self.assert_checked(f)
                self.assertEqual(f.launches[0][3], f.inner_handle)

    def test_scratch_is_never_allocated_by_observer_but_original_path_is_unchanged(self):
        for name in ("global_scratch_size", "profile_scratch_size"):
            f = Fixture()
            setattr(f.launcher, name, 8)
            with self.subTest(name=name):
                f.run()
                self.assert_checked(f)
                self.assertEqual(f.allocations, ["requested", (8, 1, f.inner_handle)])
                self.assertIn("scratch allocator not qualified", f.observer.last_audit.reasons)
                self.assertEqual(len(f.launches), 1)

    def test_instrumentation_none_is_unknown_and_empty_string_is_source_default(self):
        for value in (None, False, UNKNOWN, "profile"):
            f = Fixture()
            f.compilation.instrumentation_mode = value
            with self.subTest(value=type(value).__name__):
                f.run()
                self.assert_checked(f)
                self.assertIn("instrumentation not absent", f.observer.last_audit.reasons)

    def test_concurrent_runner_delegates_both_and_keeps_owner_thread_scope(self):
        f = Fixture()
        entered, release = threading.Event(), threading.Event()
        errors, outcomes = [], []
        original_thread = threading.current_thread()
        def pause_owner():
            if threading.current_thread() is not original_thread:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test release was not signaled")
        f.callbacks["runner"] = pause_owner
        def first():
            try:
                outcomes.append(f.run())
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=first)
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            owner_scope = f.scopes[0]
            self.assertIsNone(f.observer.active_scope)
            outcomes.append(f.run())
            self.assertIs(f.observer._active, owner_scope)
            self.assertTrue(owner_scope.retained)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(outcomes), 2)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (2, 2, 2))
        self.assertEqual(f.events.count("native_route_check"), 2)
        self.assertTrue(f.observer.poisoned)
        self.assertFalse(f.observer.last_audit.complete)
        self.assertIsNone(f.observer._active)
        self.assertEqual(owner_scope.retained, [])

    def test_unrelated_thread_jit_does_not_attach_to_active_owner_scope(self):
        f = Fixture()
        errors = []
        def external_jit():
            owner_scope = f.observer.active_scope
            def work():
                try:
                    result = f.jit.run(Tensor((1, 128)), f.scale, Tensor((1, 8)),
                        Tensor((1, 8), "int32"), 128, 8, 128,
                        grid=(1,), warmup=False, num_warps=1)
                    self.assertIs(result, f.kernel)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertIs(f.observer.active_scope, owner_scope)
            self.assertEqual(owner_scope.cache_count, 0)
            self.assertEqual(owner_scope.jit_count, 0)
            self.assertEqual(owner_scope.launches, [])
        f.callbacks["runner"] = external_jit
        f.run()
        self.assertEqual(errors, [])
        self.assert_checked(f, complete=True)
        self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (2, 2, 2))

    def test_dependency_before_producer_and_duplicate_consumer_never_replace_checks(self):
        f = Fixture()
        f.callbacks["launch"] = lambda: f.observer.dependency_observed(
            f.observer.active_scope, f.producer_stream, f.consumer_stream)
        f.run()
        self.assert_checked(f)
        self.assertIn("dependency before producer return", f.observer.last_audit.reasons)
        f = Fixture()
        f.callbacks["after_consumer"] = lambda: f.observer.checked_consumer(
            f.observer.active_scope, f.bound[3], f.bindings(), f.frame, f.consumer_stream)
        f.run()
        self.assert_checked(f)
        self.assertIn("duplicate checked consumer", f.observer.last_audit.reasons)
        self.assertTrue(f.observer.poisoned)

    def test_checked_consumer_cannot_substitute_stale_bindings_owner_or_frame(self):
        for mode in ("bindings", "owner", "frame", "unknown"):
            f = Fixture()
            original = f.observer.checked_consumer
            def incorrect(scope, ids, bindings, frame, stream, mode=mode):
                if mode == "bindings":
                    bindings = f.expected
                elif mode == "owner":
                    ids = Tensor((1, 8), "int32")
                elif mode == "frame":
                    frame = Owner()
                else:
                    bindings = UNKNOWN
                return original(scope, ids, bindings, frame, stream)
            f.observer.checked_consumer = incorrect
            with self.subTest(mode=mode):
                f.run()
                self.assert_checked(f)
                self.assertTrue(f.observer.poisoned)

    def test_missing_dependency_consumer_and_changed_current_frame_abort_transaction(self):
        for mode in ("dependency", "consumer", "frame"):
            f = Fixture()
            if mode == "dependency":
                f.observer.dependency_observed = lambda *args: None
            elif mode == "consumer":
                f.observer.checked_consumer = lambda *args: None
            else:
                f.callbacks["consumer"] = lambda: setattr(f, "frame", Owner())
            with self.subTest(mode=mode):
                f.run()
                self.assert_checked(f)
                self.assertTrue(f.observer.poisoned)

    def test_freeze_values_copies_mutable_metadata_without_opaque_equality(self):
        original = {"z": [1, True, None], "a": {"nested": "value"}}
        frozen = freeze_values(original)
        original["z"].append(99)
        self.assertEqual(frozen, (("a", (("nested", "value"),)), ("z", (1, True, None))))
        class Opaque:
            __hash__ = object.__hash__

            def __eq__(self, other):
                raise AssertionError("opaque equality must not run")
        with self.assertRaises(Exception):
            freeze_values(Opaque())
        with self.assertRaises(Exception):
            freeze_values({Opaque(): 1})

    def test_actual_cache_observation_requires_exact_cache_target_key_and_metadata(self):
        changes = (dict(cache=Owner()), dict(target=Owner()), dict(key="other"),
            dict(key=UNKNOWN), dict(selected=Owner()), dict(selected=None),
            dict(specialization=UNKNOWN), dict(specialization=(("argument", "cpu-reference"),)),
            dict(options={"num_warps": True}), dict(options={"num_warps": 2}),
            dict(options=(("num_warps", 1),)), dict(options=UNKNOWN))
        for change in changes:
            f = Fixture()
            f.cache_event_changes = change
            with self.subTest(change=change):
                f.run()
                self.assert_checked(f)
                self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))
                self.assertEqual(f.compiles, [])

    def test_nested_inner_launch_is_terminal_duplicate_even_before_first_return(self):
        f = Fixture()
        def nested():
            del f.callbacks["launch"]
            # Re-enter the exact wrapped launcher with its observed call arguments.
            original = f.launches[0]
            f.kernel.run(*original[:5], original[7], original[8], original[9], original[10],
                         *original[-1])
        f.callbacks["launch"] = nested
        f.run()
        self.assert_checked(f)
        self.assertEqual(len(f.launches), 2)
        self.assertTrue(f.observer.poisoned)


    def test_borrowed_scope_cache_and_consumer_events_from_foreign_thread_poison(self):
        for event in ("cache", "consumer"):
            f = Fixture()
            errors = []
            def borrowed():
                scope = f.observer.active_scope
                before = len(scope.retained)
                def work():
                    try:
                        if event == "cache":
                            f.observer.cache_selection_observed(scope, f.cache, f.key, f.kernel,
                                f.target, f.specialization, f.options)
                        else:
                            f.observer.checked_consumer(scope, f.bound[3], f.bindings(),
                                f.frame, f.consumer_stream)
                    except BaseException as error:
                        errors.append(error)
                thread = threading.Thread(target=work)
                thread.start()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertIs(f.observer.active_scope, scope)
                self.assertEqual(len(scope.retained), before)
                self.assertIsNotNone(scope.token)
            f.callbacks["consumer"] = borrowed
            with self.subTest(event=event):
                f.run()
                self.assertEqual(errors, [])
                self.assert_checked(f)
                self.assertTrue(f.observer.poisoned)
                self.assertIn("foreign/stale event scope", f.observer.last_audit.reasons)

    def test_thread_identity_number_reuse_does_not_admit_different_thread_object(self):
        f = Fixture()
        first, reused = NS(ident=17), NS(ident=17)
        current = [first]
        actual_threading = observer_module.threading
        observer_module.threading = NS(current_thread=lambda: current[0])
        def swap():
            scope = f.observer.active_scope
            self.assertIs(scope.thread, first)
            current[0] = reused
            self.assertIsNone(f.observer.active_scope)
            f.observer.dependency_observed(scope, f.producer_stream, f.consumer_stream)
            current[0] = first
            self.assertIs(f.observer.active_scope, scope)
        f.callbacks["consumer"] = swap
        try:
            f.run()
            self.assert_checked(f)
            self.assertTrue(f.observer.poisoned)
            self.assertIn("foreign/stale event scope", f.observer.last_audit.reasons)
        finally:
            observer_module.threading = actual_threading

    def test_router_returned_ids_must_be_the_exact_inner_bound_output(self):
        f = Fixture()
        f.callbacks["launch"] = lambda: setattr(type(f.router), "_convert_indices_dtype",
            lambda self, ids, dtype: Tensor((1, 8), "int32"))
        result = f.run()
        self.assert_checked(f)
        self.assertIsNot(result[1], f.launches[0][-1][3])
        self.assertIn("router/runner/consumer helper changed or unknown", f.observer.last_audit.reasons)
        self.assertTrue(f.observer.poisoned)

    def test_arguments_and_binding_readers_returning_wrong_snapshot_types_fail_closed(self):
        for reader in ("argument", "output", "stream", "bindings", "guards", "extra_bindings"):
            f = Fixture()
            f.observer.readers = replace(f.readers, **{reader: lambda *args: UNKNOWN})
            with self.subTest(reader=reader):
                f.run()
                self.assert_checked(f)
                self.assertEqual((len(f.binds), len(f.cache_gets), len(f.launches)), (1, 1, 1))



if __name__ == "__main__":
    unittest.main()
