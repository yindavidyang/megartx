"""Disconnected actual-object readers and shared CPU observer dispatcher.

Installs nothing and imports no device runtime. Source-file hashes are not
loaded-object evidence. Original readback/synchronization/native checks remain
required. Existing observer/ledger and opaque-native-metadata limits unchanged.
"""
from dataclasses import dataclass
import threading
import types

from m1_route_observer import (
    ArgumentSnapshot, ObservationReaders, RouteObserver, _Slot, _fingerprint,
    _require, _same_callable, _same_refs,
)
from megartx.m1_route_receipt import (
    BindingSnapshot, GuardSnapshot, OutputSnapshot, StreamSnapshot, UNKNOWN,
)


@dataclass(frozen=True)
class SourceFileAttestation:
    """Source-only inventory; never fills a loaded binding or cache selection."""
    relative_path: str
    sha256: str


@dataclass(frozen=True, eq=False)
class ProducerOwners:
    router: object
    custom_routing: object
    closure_owner: object
    routing_function: object
    jit: object
    compiled_kernel: object
    prepare_finalize: object
    experts: object
    modular_method_type: type
    modular_impl_type: type
    experts_type: type


@dataclass(frozen=True, eq=False)
class BoundaryObservation:
    """Already observed owner-thread boundary from future integration.

    Missing observations stay UNKNOWN. No field is inferred from environment
    absence or source files, and no helper here performs a CUDA query.
    """
    thread: object = UNKNOWN
    frame: object = UNKNOWN
    frame_index: object = UNKNOWN
    capture_state: object = UNKNOWN
    simulation: object = UNKNOWN
    diagnostics: object = UNKNOWN
    controlled: object = UNKNOWN
    forced: object = UNKNOWN
    external_observer: object = UNKNOWN
    routing_variant: object = UNKNOWN
    streams: tuple = ()


@dataclass(frozen=True)
class LiveMetadata:
    current_context: object
    current_boundary: object
    runtime: object
    compilation: object
    hookchain_type: type
    hookchain_call: object


class _FunctionSeal:
    """Identity/code/default/closure/global references, not source attestation."""
    def __init__(self, function):
        self.function = function
        self.fingerprint = _fingerprint(function)
        fn = function.__func__ if type(function) is types.MethodType else function
        if type(fn) is not types.FunctionType:
            fn = type(fn).__call__
        self.fn = fn
        self.globals = tuple((name, fn.__globals__[name]) for name in fn.__code__.co_names
                             if name in fn.__globals__) if type(fn) is types.FunctionType else ()

    def check(self):
        _require(_same_refs(self.fingerprint, _fingerprint(self.function)),
                 "bound callable code/default/closure changed")
        for name, value in self.globals:
            _require(self.fn.__globals__.get(name, UNKNOWN) is value, "bound callable global changed")


class TensorMetadataReader:
    """Read actual tensor_view metadata interfaces without reading contents.

    Exact tensor/storage classes and dtype objects come from a separately
    inspected runtime or CPU fixture. This does not import or attest Torch.
    Strong storage owner must be stable across two reads; missing stays unknown.
    """
    def __init__(self, tensor_type, storage_type, dtypes):
        self.tensor_types = tensor_type if type(tensor_type) is tuple else (tensor_type,)
        _require(bool(self.tensor_types) and all(isinstance(cls, type) for cls in self.tensor_types),
                 "explicit tensor owner types required")
        self.storage_type = storage_type
        self.dtypes = tuple(dtypes)
        tensor_names = ("is_cuda", "device", "dtype", "shape", "stride", "is_contiguous",
                        "untyped_storage", "data_ptr", "numel", "element_size", "storage_offset")
        interfaces = tuple((cls, tensor_names) for cls in self.tensor_types) + ((storage_type, ("data_ptr", "nbytes")),)
        self.entries = tuple((cls, name, getattr(cls, name, UNKNOWN))
                             for cls, names in interfaces for name in names)
        self.seals = tuple(_FunctionSeal(entry) for _, _, entry in self.entries if callable(entry))

    def view(self, tensor):
        _require(any(type(tensor) is cls for cls in self.tensor_types), "unknown tensor owner type")
        for cls, name, entry in self.entries:
            _require(getattr(cls, name, UNKNOWN) is entry, "tensor/storage metadata interface changed")
        for seal in self.seals:
            seal.check()
        for name in ("stride", "is_contiguous", "untyped_storage", "data_ptr", "numel", "element_size", "storage_offset"):
            _require(name not in vars(tensor), "tensor metadata method overridden")
        _require(tensor.is_cuda is True and tensor.is_contiguous() is True,
                 "non-CUDA or noncontiguous tensor view")
        dtype = tensor.dtype
        matches = [(name, size) for owner, name, size in self.dtypes if dtype is owner]
        _require(len(matches) == 1, "unknown tensor dtype")
        name, size = matches[0]
        _require(type(name) is str and type(size) is int and size > 0, "invalid dtype interface")
        device = tensor.device
        _require(type(getattr(device, "type", UNKNOWN)) is str and device.type == "cuda" and
                 type(getattr(device, "index", UNKNOWN)) is int and device.index >= 0, "unknown tensor device")
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        _require(len(shape) == len(strides) and len(shape) > 0 and
                 all(type(x) is int and x >= 0 for x in (*shape, *strides)), "invalid tensor shape/stride")
        elements, actual_size, offset = tensor.numel(), tensor.element_size(), tensor.storage_offset()
        _require(type(elements) is int and elements > 0 and type(actual_size) is int and actual_size == size and
                 type(offset) is int and offset >= 0, "invalid tensor extent")
        count = 1
        for dim in shape:
            count *= dim
        _require(count == elements, "shape/element count mismatch")
        storage = tensor.untyped_storage()
        _require(type(storage) is self.storage_type and tensor.untyped_storage() is storage,
                 "unknown or unstable storage owner")
        _require(not any(name in vars(storage) for name in ("data_ptr", "nbytes")), "storage metadata method overridden")
        pointer, storage_pointer, storage_bytes = tensor.data_ptr(), storage.data_ptr(), storage.nbytes()
        _require(all(type(x) is int and x > 0 for x in (pointer, storage_pointer, storage_bytes)), "invalid tensor/storage pointer")
        view_bytes, offset_bytes = elements * size, offset * size
        _require(pointer == storage_pointer + offset_bytes and pointer % size == 0 and
                 offset_bytes + view_bytes <= storage_bytes, "tensor/storage extent mismatch")
        return OutputSnapshot(tensor, storage, pointer, storage_pointer, storage_bytes,
            offset_bytes, device.index, shape, strides, name, view_bytes)

    def argument(self, tensor):
        view = self.view(tensor)
        return ArgumentSnapshot(view.owner, view.device, view.shape, view.strides, view.dtype)


class ActualBindingReaders:
    """Retained actual owners; extraction runs neither producer nor binder.

    Reuses nvfp4_integration.binding's identity contract without its imports or
    modifying that production function. current_context is its supplied getter.
    jit.hash is read from __dict__ only; lazy jit.cache_key is never invoked.
    That already-observed source key is not a specialization-selection key.
    """
    def __init__(self, layer, runner, routed_adapter, controller, producer_owners, live_metadata, tensors, wrappers):
        self.layer, self.runner, self.adapter = layer, runner, routed_adapter
        self.controller, self.owners, self.live = controller, producer_owners, live_metadata
        self.tensors, self.wrappers = tensors, wrappers
        self.registry = vars(layer).get("_megartx", {}).get("registry", UNKNOWN)
        self.quant_method = layer.quant_method
        self.moe_kernel = self.quant_method.moe_kernel
        self.impl = self.moe_kernel.impl
        self.scale = getattr(producer_owners.closure_owner, "per_expert_scale", UNKNOWN)
        _require(self.scale is not UNKNOWN, "missing actual closure scale")
        self.platform = producer_owners.custom_routing.__globals__.get("current_platform", UNKNOWN)
        self._extra = (producer_owners.prepare_finalize.finalize, producer_owners.routing_function,
            self.quant_method, self.moe_kernel, self.impl, self.scale, self.platform,
            getattr(self.platform, "is_cuda_alike", UNKNOWN), getattr(self.platform, "is_xpu", UNKNOWN))
        self.seals = tuple(_FunctionSeal(fn) for fn in (
            producer_owners.custom_routing, producer_owners.routing_function,
            producer_owners.prepare_finalize.prepare, producer_owners.prepare_finalize.finalize,
            routed_adapter, live_metadata.current_context, live_metadata.current_boundary, *self._extra[-2:]))
        self.hooks = tuple(getattr(live_metadata.runtime, name, UNKNOWN) for name in
            ("launch_enter_hook", "launch_exit_hook", "kernel_load_start_hook", "kernel_load_end_hook"))
        self.readers = ObservationReaders(self.bindings, self.guards, self.frame,
            tensors.view, self.stream, self.extra_bindings, tensors.argument)

    def _boundary(self):
        value = self.live.current_boundary()
        _require(type(value) is BoundaryObservation and value.thread is threading.current_thread(), "unknown/foreign boundary observation")
        frame = vars(self.controller).get("forward", UNKNOWN)
        _require(frame is value.frame and type(frame) is dict and type(frame.get("forward_index")) is int and
                 type(value.frame_index) is int and frame["forward_index"] == value.frame_index,
                 "current frame/generation changed or unknown")
        return value

    def frame(self):
        return self._boundary().frame

    def bindings(self):
        owners = self.owners
        for seal in self.seals:
            seal.check()
        data = vars(self.layer).get("_megartx", UNKNOWN)
        context = self.live.current_context()
        _require(type(data) is dict and type(self.registry) is dict and data.get("registry") is self.registry and
                 getattr(context, "no_compile_layers", UNKNOWN) is self.registry, "actual registered context changed")
        name = getattr(self.layer, "layer_name", UNKNOWN)
        _require(type(name) is str and self.registry.get(name) is self.runner and self.runner.routed_experts is self.layer and
                 self.runner._quant_method is self.layer.quant_method is self.quant_method, "actual runner/layer/method binding changed")
        owner, method = data.get("owner", UNKNOWN), self.quant_method
        _require(owner is not UNKNOWN and (method is owner or
                 (type(method) is owners.modular_method_type and method.old_quant_method is owner)), "captured quantization owner changed")
        _require(method.moe_kernel is self.moe_kernel is data.get("kernel") and method.is_monolithic is False and
                 self.moe_kernel.impl is self.impl and type(self.impl) is owners.modular_impl_type and
                 self.moe_kernel.fused_experts is owners.experts and type(owners.experts) is owners.experts_type,
                 "actual modular/expert kernel changed")
        _require(type(self.layer.forward_modular) is types.MethodType and self.layer.forward_modular.__func__ is self.adapter and self.layer.forward_modular.__self__ is self.layer,
                 "actual routed adapter changed")
        _require(self.runner.router is owners.router and owners.router.custom_routing_function is owners.custom_routing,
                 "actual router/custom callable changed")
        fn = owners.custom_routing
        _require(type(fn) is types.FunctionType and "self" in fn.__code__.co_freevars, "routing callable has no actual self closure")
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        _require(cells["self"].cell_contents is owners.closure_owner and owners.closure_owner.experts is self.runner and
                 owners.closure_owner.per_expert_scale is self.scale, "actual Gemma closure owner/scale changed")
        _require(fn.__globals__.get("gemma4_fused_routing_kernel_triton", UNKNOWN) is owners.routing_function and
                 type(owners.routing_function) is types.FunctionType and
                 owners.routing_function.__globals__.get("_gemma4_routing_kernel", UNKNOWN) is owners.jit, "actual routing global/JIT changed")
        _require(self.impl.prepare_finalize is owners.prepare_finalize, "actual prepare/finalize owner changed")
        kernel = owners.compiled_kernel
        source_key = vars(owners.jit).get("hash", UNKNOWN)
        if type(source_key) is not str or not source_key:
            source_key = UNKNOWN
        return BindingSnapshot(self.runner, self.layer, owners.router, owners.router.select_experts, fn,
            owners.closure_owner, owners.prepare_finalize, owners.prepare_finalize.prepare, owners.experts,
            owners.jit, owners.jit.run, kernel, vars(kernel).get("module", UNKNOWN),
            vars(kernel).get("function", UNKNOWN), vars(kernel).get("_run", UNKNOWN), source_key, UNKNOWN)

    def extra_bindings(self):
        o = self.owners
        return (self.impl.prepare_finalize.finalize, o.routing_function, self.layer.quant_method,
            self.layer.quant_method.moe_kernel, self.layer.quant_method.moe_kernel.impl, o.closure_owner.per_expert_scale,
            o.custom_routing.__globals__.get("current_platform", UNKNOWN),
            getattr(self.platform, "is_cuda_alike", UNKNOWN), getattr(self.platform, "is_xpu", UNKNOWN))

    def _hooks(self):
        live = self.live
        for name, expected, reverse in zip(
                ("launch_enter_hook", "launch_exit_hook", "kernel_load_start_hook", "kernel_load_end_hook"),
                self.hooks, (False, True, False, True)):
            chain = getattr(live.runtime, name, UNKNOWN)
            _require(chain is expected and type(chain) is live.hookchain_type and type(chain).__call__ is live.hookchain_call,
                     "unknown/replaced HookChain")
            values, calls = vars(chain), vars(chain).get("calls", UNKNOWN)
            _require(type(calls) is list and values.get("reversed", UNKNOWN) is reverse and
                     "__call__" not in values and not tuple(calls), "active/unknown HookChain")
        for name in ("jit_cache_hook", "jit_post_compile_hook", "add_stages_inspection_hook"):
            _require(getattr(live.runtime, name, UNKNOWN) is None, "active/unknown compile hook")
        _require(type(getattr(live.compilation, "instrumentation_mode", UNKNOWN)) is str and
                 live.compilation.instrumentation_mode == "", "active/unknown instrumentation")
        return ()

    def _plugin_guards(self):
        # Read the actual captured state of the routed adapter. A provider's
        # booleans or absent environment flags cannot hide closure callbacks.
        fn = self.adapter
        _require(type(fn) is types.FunctionType, "unknown plugin adapter function")
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        required = ("controlled", "diagnostics", "validation_profile", "controlled_observer",
                    "normal_observer", "router_observer", "route_audit", "m1")
        _require(all(name in cells for name in required), "plugin closure guard seam unobserved")
        values = {name: cells[name].cell_contents for name in required}
        _require(values["m1"] is self.controller, "plugin controller closure changed")
        _require(values["controlled"] is None and
                 all(values[name] is False for name in ("diagnostics", "validation_profile")) and
                 all(values[name] is None for name in ("controlled_observer", "normal_observer", "router_observer", "route_audit")),
                 "plugin closure observer/diagnostic active or unknown")
        _require(vars(self.controller).get("diagnostics", UNKNOWN) is False and
                 vars(self.controller).get("external_observer", UNKNOWN) is None and
                 vars(self.controller).get("external_observer_requested", UNKNOWN) is False and
                 vars(self.controller).get("route_controls", UNKNOWN) is False,
                 "controller diagnostic/external/forced guard active or unknown")
        _require("fixture_mode" not in vars(self.layer).get("_megartx", {}), "actual forced fixture active")

    def guards(self):
        boundary, o = self._boundary(), self.owners
        self._plugin_guards()
        overrides = []
        for owner, names in ((self.runner, ("_apply_quant_method", "_maybe_apply_shared_experts")),
                (o.router, ("select_experts", "_select_experts", "_compute_routing", "_validate_eplb_state",
                            "_apply_eplb_mapping", "_convert_indices_dtype")),
                (o.jit, ("run",)), (o.compiled_kernel, ("run", "_run"))):
            overrides.extend(self.wrappers.foreign_overrides(owner, names))
        pre = vars(o.jit).get("pre_run_hooks", UNKNOWN)
        pre = tuple(pre) if type(pre) is list else UNKNOWN
        diagnostics = boundary.diagnostics
        if boundary.external_observer is not None:
            diagnostics = UNKNOWN if boundary.external_observer is UNKNOWN else True
        return GuardSnapshot(boundary.capture_state, getattr(o.router, "capture_fn", UNKNOWN),
            getattr(o.router, "_routing_replay_out", UNKNOWN), getattr(o.router, "eplb_state", UNKNOWN),
            tuple(overrides), pre, self._hooks(), boundary.simulation, diagnostics, boundary.controlled,
            boundary.forced, boundary.routing_variant)

    def stream(self, actual_inner_handle):
        _require(type(actual_inner_handle) is int and actual_inner_handle >= 0, "unknown actual stream handle")
        matches = tuple(stream for stream in self._boundary().streams if type(stream) is StreamSnapshot and
                        type(stream.handle) is int and stream.handle == actual_inner_handle)
        _require(len(matches) == 1, "actual stream ownership unobserved or ambiguous")
        return matches[0]


def make_readers(layer, runner, routed_adapter, controller, producer_owners, live_metadata, tensors, wrappers):
    """Build disconnected readers; bound callbacks retain the extractor."""
    return ActualBindingReaders(layer, runner, routed_adapter, controller,
        producer_owners, live_metadata, tensors, wrappers).readers


class BindingRouteObserver(RouteObserver):
    """Existing CPU protocol plus strong storage views for all launch arguments.

    Existing f32 CPU specialization is never inferred as installed scale dtype.
    Unknown native/opaque metadata cannot create a profile.
    """
    def __init__(self, expected, reference, readers):
        super().__init__(expected, reference, readers)
        entries = (expected.select_experts, expected.jit_run, expected.launcher,
                   reference.launcher_launch, reference.utils_launch,
                   expected.runner._apply_quant_method, self._run_getter,
                   *(value for _, value in self._kernel_entries),
                   *(value for _, _, value in self._host_entries))
        self._binding_function_seals = tuple(_FunctionSeal(value) for value in entries if callable(value))

    def _runner(self, slot, args, kwargs):
        # Keep the owner-thread scope independently of replaceable audit cleanup.
        # The base already preserves the producer result/error and clears its
        # ordinary retained owners. If _finish itself raises before entering
        # ledger cleanup, finish that same observation context here, never rerun
        # any producer/binder/launcher or clean a different thread's context.
        stack = getattr(self._local, "binding_scopes", None)
        if stack is None:
            stack = self._local.binding_scopes = []
        holder = [None]
        stack.append(holder)
        try:
            return super()._runner(slot, args, kwargs)
        finally:
            stack.pop()
            scope = holder[0]
            if scope is not None and scope.ledger_context is not None:
                self._safe(scope, lambda: RouteObserver._finish(self, scope))

    def _state(self, scope):
        stack = getattr(self._local, "binding_scopes", ())
        if stack:
            stack[-1][0] = scope
        for seal in self._binding_function_seals:
            seal.check()
        result = super()._state(scope)
        frame = scope.frame
        _require(type(frame) is dict and type(frame.get("forward_index")) is int,
                 "unknown frame generation")
        previous = getattr(scope, "binding_frame_index", UNKNOWN)
        if previous is UNKNOWN:
            scope.binding_frame_index = frame["forward_index"]
        else:
            _require(previous == frame["forward_index"], "frame generation changed within scope")
        return result

    def _arguments(self, owners):
        values = super()._arguments(owners)
        scope = self.active_scope
        _require(scope is not None, "argument extraction outside owner scope")
        views = tuple(self.readers.output(owner) for owner in owners)
        previous = next((entry[1] for entry in scope.retained
                         if type(entry) is tuple and len(entry) == 2 and entry[0] is self), None)
        if previous is None:
            scope.retained.append((self, views))
        else:
            _require(len(views) == len(previous), "argument view count changed")
            for old, new in zip(previous, views):
                _require(old.owner is new.owner and old.storage_owner is new.storage_owner and
                    all(getattr(old, name) == getattr(new, name) for name in
                        ("pointer", "storage_pointer", "storage_bytes", "offset_bytes", "device",
                         "shape", "strides", "dtype", "view_bytes")), "argument storage/view changed")
        return values


class SharedRouteDispatcher:
    """One wrapper per exact shared JIT/initialized launch slot; many runners.

    Factories retain and return wrappers; no automatic installation. Unrelated
    calls delegate once. Nested/concurrent runners invalidate observations.
    """
    def __init__(self):
        self._shared, self._profiles = {}, []
        self._active = self._thread = None
        self._inflight, self._poison = 0, None
        self._local, self._lock = threading.local(), threading.RLock()

    @property
    def active_observer(self):
        if self._thread is threading.current_thread() and not getattr(self._local, "blocked", 0):
            return self._active
        return None

    @property
    def active_scope(self):
        observer = self.active_observer
        return observer.active_scope if observer is not None else None

    def owns(self, owner, name, value):
        return any(slot.owner is owner and slot.name == name and slot.wrapper is value
                   for observer in self._profiles for slot in observer._slots.values())

    def foreign_overrides(self, owner, names):
        values = vars(owner)
        return tuple(name for name in names if name in values and not self.owns(owner, name, values[name]))

    def _shared_slot(self, kind, owner, name, original):
        key = (id(owner), name)
        if key in self._shared:
            slot = self._shared[key]
            _require(slot.owner is owner and _same_callable(slot.original, original) and
                     (getattr(owner, name, UNKNOWN) is slot.wrapper or _same_callable(getattr(owner, name, UNKNOWN), original)),
                     "shared original/wrapper changed")
            return slot
        _require(callable(original) and _same_callable(getattr(owner, name, UNKNOWN), original), "shared original is not currently bound")
        slot = _Slot(owner, name, original, _fingerprint(original))
        def wrapper(*args, **kwargs):
            observer = self.active_observer
            if observer is None:
                return slot.original(*args, **kwargs)
            expected = observer.expected.jit_function if kind == "jit" else observer.expected.compiled_kernel
            if expected is not owner:
                if observer.active_scope is not None:
                    observer._reject(observer.active_scope, "foreign shared producer", poison=True)
                else:
                    observer._poison = "foreign shared producer"
                return slot.original(*args, **kwargs)
            return (observer._jit if kind == "jit" else observer._launch)(slot, args, kwargs)
        slot.wrapper = wrapper
        self._shared[key] = slot
        return slot

    def register(self, observer):
        _require(isinstance(observer, BindingRouteObserver) and not observer._slots, "fresh binding observer required")
        expected = observer.expected
        _require(not any(x.expected.runner is expected.runner or x.expected.router is expected.router
                         for x in self._profiles), "runner/router already registered")
        jit = self._shared_slot("jit", expected.jit_function, "run", expected.jit_run)
        launch = self._shared_slot("launch", expected.compiled_kernel, "_run", expected.launcher)
        original = expected.runner._apply_quant_method
        runner = _Slot(expected.runner, "_apply_quant_method", original, _fingerprint(original))
        def run(*args, **kwargs):
            with self._lock:
                busy = self._inflight != 0
                self._inflight += 1
                if busy:
                    self._poison = "nested/concurrent registered runner"
                    active_scope = self._active._active
                    if active_scope is not None:
                        self._active._reject(active_scope, "nested/concurrent registered runner", poison=True)
                    else:
                        self._active._poison = "nested/concurrent registered runner"
                    observer._poison = "nested/concurrent registered runner"
                    self._local.blocked = getattr(self._local, "blocked", 0) + 1
                else:
                    self._active, self._thread = observer, threading.current_thread()
                if self._poison is not None:
                    observer._poison = self._poison
            try:
                return observer._runner(runner, args, kwargs)
            finally:
                with self._lock:
                    if busy:
                        self._local.blocked -= 1
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._active = self._thread = None
                    elif not busy:
                        # The initial owner returned, but an overlapping call
                        # still runs. No later call can acquire observation scope.
                        self._thread = None
        runner.wrapper = run
        original = expected.select_experts
        _require(_same_callable(expected.router.select_experts, original), "router selection is not currently bound")
        select = _Slot(expected.router, "select_experts", original, _fingerprint(original))
        def selection(*args, **kwargs):
            current = self.active_observer
            if current is observer:
                return observer._select(select, args, kwargs)
            if current is not None:
                if current.active_scope is not None:
                    current._reject(current.active_scope, "foreign registered router", poison=True)
                else:
                    current._poison = "foreign registered router"
            return select.original(*args, **kwargs)
        select.wrapper = selection
        observer._slots.update(runner=runner, select=select, jit=jit, launch=launch)
        self._profiles.append(observer)
        return {kind: slot.wrapper for kind, slot in observer._slots.items()}

    def _event_observer(self, scope):
        observer = self.active_observer
        if observer is not None and scope is observer.active_scope and scope is not None:
            return observer
        # Explicit stale/foreign dependency/consumer events differ from an
        # unrelated global launch. They can never validate another owner scope.
        active = self._active
        if active is not None:
            # The base runner clears _active under this same lock before
            # constructing its terminal audit. Reject while the owner scope is
            # still live, or observe that cleanup already won and leave it alone.
            with active._lock:
                active_scope = active._active
                if active_scope is not None:
                    active._reject(active_scope, "foreign/stale dispatcher event", poison=True)
        return None

    def dependency_observed(self, scope, producer, consumer):
        """Notify only after the existing wait; performs no wait."""
        observer = self._event_observer(scope)
        if observer is not None:
            observer.dependency_observed(scope, producer, consumer)

    def checked_consumer(self, scope, actual_ids, frame, consumer_stream):
        """Failure-contained extraction at the unchanged checked consumer.

        Caller must still execute original route D2H/fence/native checks and
        cleanup regardless of this metadata return. No receipt enters native.
        """
        observer = self._event_observer(scope)
        if observer is None:
            return None
        bindings = observer._safe(scope, observer.readers.bindings)
        return observer.checked_consumer(scope, actual_ids, bindings, frame, consumer_stream)

    def cache_selection_observed(self, **event):
        observer = self.active_observer
        if observer is not None:
            observer.cache_selection_observed(observer.active_scope, **event)
