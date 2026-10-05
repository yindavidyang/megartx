"""Disconnected CPU/reference route wrappers; never dispatch qualification.

Factories return wrappers but install nothing. Every wrapped call delegates once
with the original argument objects, result and exception. Observation failures
only invalidate/poison the audit, never trigger retries, device work or fallback.
Readers are an explicit source-bound adapter seam, NOT installed-source proof.
No Torch, Triton, CUDA, plugin or launcher module is imported here.
"""
from dataclasses import dataclass, field, fields, replace
import threading
import types

from megartx.m1_route_receipt import (
    BindingSnapshot, GuardSnapshot, ProducerSnapshot, RouteObservationLedger,
    StreamSnapshot, UNKNOWN, _guards_valid, _same_binding, _same_output,
    _stream_valid,
)


@dataclass(frozen=True)
class ObservationReaders:
    """Read current metadata only; never query contents, compile, warm or wait.

    bindings() must leave cache_key UNKNOWN: only the real selection seam may
    fill it. extra_bindings() covers finalize and other source-bound callables
    absent from the unchanged ledger. stream(handle) describes that actual inner
    handle's already-known device/context. It must not read a 'current stream'.
    """
    bindings: object
    guards: object
    frame: object
    output: object
    stream: object
    extra_bindings: object
    argument: object


@dataclass(frozen=True, eq=False)
class ArgumentSnapshot:
    owner: object
    device: int
    shape: tuple
    strides: tuple
    dtype: str


@dataclass(frozen=True)
class KernelReference:
    """Explicit test/reference expectations, never an installed kernel pin.

    cache_key in the associated BindingSnapshot is an EXPECTATION. It is never
    used as an observation. Source-only installed intake and later live review
    remain separate prerequisites, even if these reference tests complete.
    """
    cache: object
    target: object
    specialization: tuple
    options: tuple
    metadata: object
    metadata_fields: tuple
    packed_metadata: object
    launcher_launch: object
    utils_launch: object
    read_utils_launch: object
    runtime: object
    compilation: object
    hookchain_type: type
    hookchain_call: object
    extra_bindings: tuple


@dataclass(frozen=True)
class ObservationAudit:
    generation: int
    complete: bool
    reasons: tuple
    cache_key: object
    inner_stream: object
    receipt: object
    poisoned: bool

    @property
    def route_readback_required(self):
        return True

    @property
    def synchronization_required(self):
        return True

    @property
    def native_checks_required(self):
        return True

    def __bool__(self):
        raise TypeError("route audit is not dispatch eligibility")


@dataclass(eq=False)
class _Slot:
    owner: object
    name: str
    original: object
    fingerprint: object
    wrapper: object = None


@dataclass(eq=False)
class _Scope:
    generation: int
    thread: threading.Thread
    frame: object = UNKNOWN
    reasons: list = field(default_factory=list)
    selecting: bool = False
    jitting: bool = False
    selected: object = UNKNOWN
    jit_args: object = None
    argument_views: object = None
    launches: list = field(default_factory=list)
    launch_count: int = 0
    selection_count: int = 0
    jit_count: int = 0
    cache_count: int = 0
    key: object = UNKNOWN
    stream: object = UNKNOWN
    ledger_context: object = None
    token: object = None
    receipt: object = None
    primary: object = None
    retained: list = field(default_factory=list)
    consumer_count: int = 0


class _Incomplete(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _Incomplete(reason)


def _same_callable(left, right):
    return (left is right or (type(left) is types.MethodType and
            type(right) is types.MethodType and left.__self__ is right.__self__
            and left.__func__ is right.__func__))


def _fingerprint(value):
    """Retain code/default/closure identities; never use user-defined equality."""
    fn = value.__func__ if type(value) is types.MethodType else value
    if type(fn) is not types.FunctionType:
        fn = type(fn).__call__
    if type(fn) is not types.FunctionType:
        return (fn,)
    closure = tuple(cell.cell_contents for cell in (fn.__closure__ or ()))
    defaults = fn.__defaults__ or ()
    keywords = fn.__kwdefaults__ or {}
    return (fn, fn.__code__, *defaults, *tuple(keywords), *keywords.values(), *closure)


def _same_refs(left, right):
    return type(right) is tuple and len(left) == len(right) and all(
        a is b for a, b in zip(left, right))


def freeze_values(value):
    """Copy source-seam scalar metadata without invoking opaque __eq__."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (tuple, list):
        return tuple(freeze_values(item) for item in value)
    if type(value) is dict and all(type(key) is str for key in value):
        return tuple(sorted((key, freeze_values(item)) for key, item in value.items()))
    raise _Incomplete("unknown specialization/options metadata")


def _same_values(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(_same_values(a, b) for a, b in zip(left, right))
    return left == right


class RouteObserver:
    """One retained reference profile and one non-reentrant observation scope.

    A scope starts at the runner. The unchanged ledger opens only after an
    actually observed cache selection and inner launch supply its required
    fields; a missing seam cannot be papered over with reference expectations.
    A terminal audit failure never changes the checked production call's result.
    """
    def __init__(self, expected, reference, readers):
        self.expected = expected
        self.reference = reference
        self.readers = readers
        self.ledger = RouteObservationLedger(expected)
        self._slots = {}
        self._active = None
        self._generation = 0
        self._poison = None
        self._lock = threading.RLock()
        self._local = threading.local()
        self.last_audit = None
        kernel = expected.compiled_kernel
        self._kernel_type = type(kernel)
        self._kernel_source = vars(kernel).get("src", UNKNOWN)
        self._kernel_source_type = type(self._kernel_source)
        self._kernel_source_fn = getattr(self._kernel_source, "fn", UNKNOWN)
        self._kernel_entries = tuple((name, getattr(kernel, name, UNKNOWN))
            for name in ("launch_metadata", "_init_handles"))
        self._run_descriptor = vars(type(kernel)).get("run", UNKNOWN)
        self._run_getter = (self._run_descriptor.fget
                            if type(self._run_descriptor) is property else UNKNOWN)
        self._router_type = type(expected.router)
        self._host_entries = tuple((expected.router, name, getattr(expected.router, name, UNKNOWN))
            for name in ("_select_experts", "_compute_routing", "_validate_eplb_state",
                         "_apply_eplb_mapping", "_convert_indices_dtype")) + (
            (expected.runner, "_maybe_apply_shared_experts",
             getattr(expected.runner, "_maybe_apply_shared_experts", UNKNOWN)),
            (expected.layer, "forward_modular", getattr(expected.layer, "forward_modular", UNKNOWN)),
        )
        self._expected_functions = tuple((value, _fingerprint(value)) for value in
            (expected.custom_routing, expected.prepare, reference.hookchain_call,
             reference.launcher_launch, reference.utils_launch, self._run_getter,
             *(entry for _, entry in self._kernel_entries),
             *(entry for _, _, entry in self._host_entries), *reference.extra_bindings)
            if callable(value))
        self._hook_objects = tuple(getattr(reference.runtime, name, UNKNOWN) for name in
            ("launch_enter_hook", "launch_exit_hook", "kernel_load_start_hook", "kernel_load_end_hook"))
        self._launcher_values = tuple((name, freeze_values(getattr(expected.launcher, name, UNKNOWN)))
            for name in ("num_ctas", "global_scratch_align", "profile_scratch_align",
                         "launch_cooperative_grid", "launch_pdl", "arg_annotations", "kernel_signature"))

    @property
    def poisoned(self):
        return self._poison is not None or self.ledger.poisoned

    @property
    def active_scope(self):
        current = self._active
        if (current is not None and current.thread is threading.current_thread()
                and not getattr(self._local, "blocked", 0)):
            return current
        return None

    def _reject(self, scope, reason, poison=False):
        if reason not in scope.reasons:
            scope.reasons.append(reason)
        if poison and self._poison is None:
            self._poison = reason

    def _safe(self, scope, operation):
        try:
            return operation()
        except _Incomplete as error:
            self._reject(scope, error.args[0])
        except BaseException:
            # No str(error), add_note, logging callback or retry can mask the
            # primary producer/consumer exception or interrupt delegation.
            self._reject(scope, "observation operation failed", poison=True)
        return None

    def _slot(self, kind, owner, name, original, body):
        if kind in self._slots or not callable(original):
            raise ValueError("one exact callable per wrapper kind is required")
        _require(_same_callable(getattr(owner, name, UNKNOWN), original),
                 "wrapper original differs from current callable")
        slot = _Slot(owner, name, original, _fingerprint(original))
        def wrapper(*args, **kwargs):
            return body(slot, args, kwargs)
        slot.wrapper = wrapper
        self._slots[kind] = slot
        return wrapper

    def owns(self, owner, name, value):
        return any(slot.owner is owner and slot.name == name and slot.wrapper is value
                   for slot in self._slots.values())

    def foreign_overrides(self, owner, names):
        """Do not erase arbitrary instance overrides or trust __wrapped__."""
        values = vars(owner)
        return tuple(name for name in names if name in values and
                     not self.owns(owner, name, values[name]))

    def _bindings(self, scope):
        # These ownership edges are present in the inspected runner/router
        # bodies; do not let an otherwise correct reader snapshot hide swaps.
        _require(getattr(self.expected.runner, "router", UNKNOWN) is self.expected.router and
                 getattr(self.expected.runner, "routed_experts", UNKNOWN) is self.expected.layer,
                 "actual runner ownership changed")
        _require(_same_callable(getattr(self.expected.router, "custom_routing_function", UNKNOWN),
                                self.expected.custom_routing), "actual custom routing changed")
        _require(type(self.expected.router) is self._router_type, "router class changed")
        for owner, name, expected_entry in self._host_entries:
            _require(callable(expected_entry) and
                     _same_callable(getattr(owner, name, UNKNOWN), expected_entry),
                     "router/runner/consumer helper changed or unknown")
        value = self.readers.bindings()
        _require(type(value) is BindingSnapshot and value.cache_key is UNKNOWN,
                 "binding reader must leave cache selection unknown")
        for slot in self._slots.values():
            _require(getattr(slot.owner, slot.name, UNKNOWN) is slot.wrapper,
                     "owned wrapper replaced or absent")
            _require(_same_refs(slot.fingerprint, _fingerprint(slot.original)),
                     "original callable changed")
        _require(len(self._slots) == 4, "missing owned wrapper")
        changes = {"cache_key": scope.key}
        for name in ("select_experts", "jit_run", "launcher"):
            kind = {"select_experts": "select", "jit_run": "jit", "launcher": "launch"}[name]
            slot = self._slots[kind]
            _require(getattr(value, name) is slot.wrapper, "foreign binding wrapper")
            changes[name] = slot.original
        # Fresh bound-method objects denote the same exact function and owner.
        for info in fields(BindingSnapshot):
            name = info.name
            if name not in changes and _same_callable(getattr(value, name), getattr(self.expected, name)):
                changes[name] = getattr(self.expected, name)
        value = replace(value, **changes)
        # Before selection, compare all other fields without inventing an
        # observed key; this temporary comparison never escapes or reaches ledger.
        compare = replace(value, cache_key=self.expected.cache_key) if scope.key is UNKNOWN else value
        _require(_same_binding(self.expected, compare), "binding changed or unknown")
        for original, fingerprint in self._expected_functions:
            _require(_same_refs(fingerprint, _fingerprint(original)), "producer callable/closure changed")
        extra = self.readers.extra_bindings()
        _require(type(extra) is tuple and len(extra) == len(self.reference.extra_bindings)
                 and all(_same_callable(a, b) for a, b in zip(extra, self.reference.extra_bindings)),
                 "prepare/finalize or extra binding changed")
        return value

    def _hooks(self):
        ref = self.reference
        observed = []
        for name, reverse in (("launch_enter_hook", False), ("launch_exit_hook", True),
                              ("kernel_load_start_hook", False), ("kernel_load_end_hook", True)):
            chain = getattr(ref.runtime, name, UNKNOWN)
            _require(type(chain) is ref.hookchain_type and
                     type(chain).__call__ is ref.hookchain_call, "unknown hook chain")
            attrs = vars(chain)
            calls = attrs.get("calls", UNKNOWN)
            _require(type(calls) is list and attrs.get("reversed", UNKNOWN) is reverse,
                     "unknown hook calls/direction")
            snapshot = tuple(calls)
            _require(not snapshot and "__call__" not in attrs, "foreign hook callback")
            _require(chain is self._hook_objects[len(observed)], "hook chain replaced")
            observed.append(chain)
        for name in ("jit_cache_hook", "jit_post_compile_hook", "add_stages_inspection_hook"):
            _require(getattr(ref.runtime, name, UNKNOWN) is None, "foreign compile callback")
        _require(type(getattr(ref.compilation, "instrumentation_mode", UNKNOWN)) is str and
                 ref.compilation.instrumentation_mode == "",
                 "instrumentation not absent")
        return tuple(observed)

    def _state(self, scope):
        _require(self.readers.frame() is scope.frame, "current frame changed")
        bindings = self._bindings(scope)
        guards = self.readers.guards()
        _require(_guards_valid(guards), "guard/callback unknown or active")
        self._hooks()
        kernel = self.expected.compiled_kernel
        launcher = self._slots["launch"].original
        ref = self.reference
        _require(type(kernel) is self._kernel_type and
                 vars(type(kernel)).get("run", UNKNOWN) is self._run_descriptor and
                 type(self._run_descriptor) is property and "run" not in vars(kernel),
                 "compiled run descriptor changed or unknown")
        for name, expected_entry in self._kernel_entries:
            _require(callable(expected_entry) and
                     _same_callable(getattr(kernel, name, UNKNOWN), expected_entry),
                     "compiled kernel callback/entrypoint changed")
        source = vars(kernel).get("src", UNKNOWN)
        _require(source is self._kernel_source and source is not UNKNOWN and
                 type(source) is self._kernel_source_type and
                 getattr(source, "fn", UNKNOWN) is self._kernel_source_fn and
                 self._kernel_source_fn is self.expected.jit_function,
                 "compiled source/function owner changed or unknown")
        _require(vars(kernel).get("module", UNKNOWN) is self.expected.cuda_module and
                 vars(kernel).get("function", UNKNOWN) is self.expected.cuda_function and
                 vars(kernel).get("metadata", UNKNOWN) is ref.metadata and
                 vars(kernel).get("packed_metadata", UNKNOWN) is ref.packed_metadata,
                 "kernel uninitialized or metadata/handle changed")
        _require(getattr(launcher, "launch", UNKNOWN) is ref.launcher_launch and
                 ref.read_utils_launch() is ref.utils_launch, "underlying launch target changed")
        for name in ("global_scratch_size", "profile_scratch_size"):
            _require(type(getattr(launcher, name, None)) is int and getattr(launcher, name) == 0,
                     "scratch allocator not qualified")
        _require({"num_warps", "num_ctas", "global_scratch_size", "profile_scratch_size"}.issubset(
                 {name for name, _ in ref.metadata_fields}), "missing launch metadata expectations")
        for name, wanted in self._launcher_values:
            _require(_same_values(freeze_values(getattr(launcher, name, UNKNOWN)), wanted),
                     "launcher configuration changed")
        for name, wanted in ref.metadata_fields:
            current = getattr(ref.metadata, name, UNKNOWN)
            _require(type(current) is type(wanted) and _same_values(freeze_values(current), wanted),
                     "kernel launch metadata changed")
        _require(getattr(self.expected.jit_function, "launch_metadata", UNKNOWN) is None,
                 "foreign launch metadata callback")
        return bindings, guards

    def wrap_runner(self, runner, original):
        _require(runner is self.expected.runner, "unexpected runner")
        return self._slot("runner", runner, "_apply_quant_method", original, self._runner)

    def wrap_select(self, router, original):
        _require(router is self.expected.router and _same_callable(original, self.expected.select_experts),
                 "unexpected router/select")
        return self._slot("select", router, "select_experts", original, self._select)

    def wrap_jit(self, jit, original):
        _require(jit is self.expected.jit_function and _same_callable(original, self.expected.jit_run),
                 "unexpected JIT instance/run")
        return self._slot("jit", jit, "run", original, self._jit)

    def wrap_launcher(self, kernel, original):
        _require(kernel is self.expected.compiled_kernel and original is self.expected.launcher and
                 vars(kernel).get("_run") is original and
                 vars(kernel).get("module") is self.expected.cuda_module and
                 vars(kernel).get("function") is self.expected.cuda_function,
                 "launcher must already be initialized")
        return self._slot("launch", kernel, "_run", original, self._launch)

    def _runner(self, slot, args, kwargs):
        with self._lock:
            self._generation += 1
            scope = _Scope(self._generation, threading.current_thread())
            busy = self._active is not None
            if busy:
                self._reject(self._active, "nested or concurrent runner", poison=True)
                self._reject(scope, "nested or concurrent runner", poison=True)
            else:
                self._active = scope
            if self.poisoned:
                self._reject(scope, "observer poisoned")
        if busy:
            self._local.blocked = getattr(self._local, "blocked", 0) + 1
        else:
            def before():
                scope.frame = self.readers.frame()
                _require(scope.frame is not None and scope.frame is not UNKNOWN, "unknown frame")
                scope.retained.extend((args, kwargs, scope.frame))
                self._state(scope)
            self._safe(scope, before)
        try:
            return slot.original(*args, **kwargs)
        except BaseException as error:
            scope.primary = error
            self._reject(scope, "wrapped execution raised", poison=True)
            raise
        finally:
            if busy:
                self._local.blocked -= 1
            else:
                self._safe(scope, lambda: self._state(scope))
                self._safe(scope, lambda: self._finish(scope))
                # Always drop per-invocation owners, including broken cleanup.
                with self._lock:
                    if self._active is scope:
                        self._active = None
            self.last_audit = ObservationAudit(scope.generation,
                not scope.reasons and scope.receipt is not None and not self.poisoned,
                tuple(scope.reasons), scope.key,
                scope.stream.handle if type(scope.stream) is StreamSnapshot else UNKNOWN,
                scope.receipt, self.poisoned)
            scope.retained.clear()
            scope.launches.clear()
            scope.jit_args = None
            scope.argument_views = None
            scope.selected = UNKNOWN
            scope.frame = UNKNOWN
            scope.primary = None

    def _finish(self, scope):
        if scope.receipt is None:
            self._reject(scope, "incomplete observation")
        context = scope.ledger_context
        scope.ledger_context = None
        if context is not None:
            # Deliberately pass an observer failure to abort/poison an already
            # opened transaction while keeping the original call unaffected.
            error = scope.primary
            if error is None and scope.reasons:
                error = _Incomplete("observation invalidated")
            context.__exit__(type(error) if error is not None else None, error, None)

    def _select(self, slot, args, kwargs):
        scope = self.active_scope
        if scope is None:
            return slot.original(*args, **kwargs)
        scope.selection_count += 1
        if scope.selecting or scope.selection_count != 1:
            self._reject(scope, "duplicate/nested selection", poison=True)
        scope.selecting = True
        self._safe(scope, lambda: self._state(scope))
        try:
            result = slot.original(*args, **kwargs)
        except BaseException:
            self._reject(scope, "router raised", poison=True)
            raise
        else:
            def after():
                self._state(scope)
                _require(type(result) is tuple and len(result) == 2, "unknown router result")
                scope.retained.append(result)
                scope.selected = result[1]
                _require(len(scope.launches) == 1 and result[1] is scope.launches[0][0].output.owner,
                         "selected IDs are not observed bound output")
            self._safe(scope, after)
            return result
        finally:
            scope.selecting = False

    def _jit(self, slot, args, kwargs):
        scope = self.active_scope
        if scope is None:
            return slot.original(*args, **kwargs)
        scope.jit_count += 1
        def before():
            _require(scope.selecting and not scope.jitting and scope.jit_count == 1,
                     "JIT outside one selection")
            self._state(scope)
            _require(type(kwargs.get("grid")) is tuple and kwargs["grid"] == (1,)
                     and all(type(x) is int for x in kwargs["grid"])
                     and kwargs.get("warmup", UNKNOWN) is False,
                     "warmup/callable/unknown grid")
            _require(set(kwargs) == {"grid", "warmup", "num_warps"} and
                     type(kwargs["num_warps"]) is int and kwargs["num_warps"] == 1 and
                     len(args) == 7 and all(type(x) is int for x in args[4:]) and
                     args[4:] == (128, 8, 128), "unknown Gemma specialization arguments")
            scope.argument_views = self._arguments(args[:4])
        self._safe(scope, before)
        scope.jitting = True
        scope.jit_args = args
        scope.retained.extend((args, kwargs))
        try:
            result = slot.original(*args, **kwargs)
        except BaseException:
            self._reject(scope, "JIT raised", poison=True)
            raise
        else:
            def after():
                self._state(scope)
                _require(result is self.expected.compiled_kernel, "unqualified/async/cold kernel return")
                _require(scope.cache_count == 1 and scope.key is not UNKNOWN,
                         "cache selection unobserved")
                _require(len(scope.launches) == 1, "missing or duplicate inner launch")
                if scope.reasons or self.poisoned:
                    return
                before, after = scope.launches[0]
                scope.ledger_context = self.ledger.invocation(before.bindings, before.guards,
                                                              before.stream, scope.frame)
                scope.token = scope.ledger_context.__enter__()
                self.ledger.observe_production(scope.token, before, after)
            self._safe(scope, after)
            return result
        finally:
            scope.jitting = False

    def cache_selection_observed(self, scope, cache, key, selected, target, specialization, options):
        """Only at the ACTUAL selection seam, not a second lookup/binder call.

        No wrapper installs such a seam. Without an independently source-bound
        observer, this remains uncalled and the resulting audit is incomplete.
        A cold cache reports selected=None; later compilation cannot undo it.
        """
        if not self._owns_scope(scope):
            return
        scope.cache_count += 1
        def observe():
            _require(scope.jitting and scope.launch_count == 0 and scope.cache_count == 1,
                     "cache selection absent/duplicate/out of order")
            ref = self.reference
            _require(cache is ref.cache and target is ref.target and type(key) is str and
                     bool(key) and key == self.expected.cache_key, "unqualified cache selection")
            # Actual observed key may be retained even when the selected kernel
            # is cold or unqualified. Never manufacture a launch for that case.
            scope.key = key
            _require(selected is self.expected.compiled_kernel, "cold/unqualified cache kernel")
            _require(type(specialization) is list and type(options) is dict and
                     _same_values(freeze_values(specialization), ref.specialization) and
                     _same_values(freeze_values(options), ref.options), "new/unknown specialization")
            scope.retained.extend((cache, selected, target, specialization, options))
        self._safe(scope, observe)

    def _arguments(self, owners):
        result = tuple(self.readers.argument(owner) for owner in owners)
        for view, owner, shape, strides, dtype in zip(result, owners,
                ((1, 128), (128,), (1, 8), (1, 8)),
                ((128, 1), (1,), (8, 1), (8, 1)),
                ("float32", "float32", "float32", "int32")):
            _require(type(view) is ArgumentSnapshot and view.owner is owner and
                     type(view.device) is int and view.device >= 0 and
                     type(view.shape) is tuple and type(view.strides) is tuple and
                     all(type(x) is int for x in (*view.shape, *view.strides)) and
                     view.shape == shape and view.strides == strides and
                     type(view.dtype) is str and view.dtype == dtype,
                     "unknown input/score/output argument view")
        _require(all(view.device == result[0].device for view in result), "argument device mismatch")
        return result

    def _launch(self, slot, args, kwargs):
        scope = self.active_scope
        if scope is None:
            return slot.original(*args, **kwargs)
        scope.launch_count += 1
        if scope.launch_count != 1:
            self._reject(scope, "duplicate inner launch", poison=True)
        before = None
        def snapshot():
            _require(scope.jitting and scope.selecting, "inner launch outside JIT/selection")
            _require(not kwargs and len(args) == 16, "unknown launcher signature")
            _require(all(type(x) is int for x in args[:4]) and args[:3] == (1, 1, 1),
                     "unknown inner grid/stream")
            _require(args[4] is self.expected.cuda_function and
                     args[5] is self.reference.packed_metadata, "actual function/metadata mismatch")
            hooks = self._hooks()
            _require(args[7] is hooks[0] and args[8] is hooks[1], "actual launch hook mismatch")
            bound = args[9:]
            incoming = scope.jit_args
            _require(incoming is not None and len(incoming) == 7 and
                     all(a is b for a, b in zip(bound[:4], incoming[:4])) and
                     all(type(x) is int for x in bound[4:]) and bound[4:] == incoming[4:],
                     "actual bound arguments mismatch")
            views = self._arguments(bound[:4])
            _require(scope.argument_views is not None and all(
                     a.owner is b.owner and a.device == b.device and a.shape == b.shape and
                     a.strides == b.strides and a.dtype == b.dtype
                     for a, b in zip(views, scope.argument_views)), "argument views changed")
            bindings, guards = self._state(scope)
            stream = self.readers.stream(args[3])
            _require(_stream_valid(stream) and stream.handle == args[3], "actual stream not described")
            output = self.readers.output(bound[3])
            _require(output.owner is bound[3] and stream.device == views[0].device,
                     "output owner or stream device mismatch")
            scope.stream = stream
            scope.retained.extend((args, kwargs, output, stream))
            return ProducerSnapshot(bindings, guards, stream, output)
        before = self._safe(scope, snapshot)
        try:
            result = slot.original(*args, **kwargs)
        except BaseException:
            self._reject(scope, "launcher raised", poison=True)
            raise
        else:
            after = self._safe(scope, snapshot)
            if before is not None and after is not None:
                if not _same_output(before.output, after.output):
                    self._reject(scope, "output changed across launch", poison=True)
                scope.launches.append((before, after))
            else:
                self._reject(scope, "inner launch incomplete", poison=before is not None)
            return result

    def _owns_scope(self, scope):
        if scope is None:
            return False
        if scope is not self.active_scope:
            # Explicit foreign/stale event calls differ from unrelated global
            # JIT invocations: reject them, without releasing anyone's owners.
            if self._active is not None:
                self._reject(self._active, "foreign/stale event scope", poison=True)
            return False
        return True

    def dependency_observed(self, scope, producer, consumer):
        """Call only AFTER the existing wait succeeds; adds no wait."""
        if not self._owns_scope(scope):
            return
        def observe():
            _require(not scope.selecting and not scope.jitting, "dependency before producer return")
            self._state(scope)
            if scope.token is None or scope.reasons or self.poisoned:
                self._reject(scope, "dependency without complete production")
                return
            self.ledger.observe_dependency(scope.token, producer, consumer)
        self._safe(scope, observe)

    def checked_consumer(self, scope, actual_ids, current_bindings, frame, consumer_stream):
        """Audit just before the UNCHANGED D2H/fence/native checks, never instead.

        The return value is metadata only and must never enter native admission.
        The caller's existing route check, synchronization, and exception/finally
        behavior must execute regardless of this return value.
        """
        if not self._owns_scope(scope):
            return None
        scope.consumer_count += 1
        def observe():
            _require(scope.consumer_count == 1, "duplicate checked consumer")
            bindings, guards = self._state(scope)
            _require(current_bindings is not UNKNOWN and type(current_bindings) is BindingSnapshot,
                     "consumer bindings unknown")
            # Verify the supplied CURRENT snapshot against a fresh extraction;
            # it cannot substitute for that extraction or fill the cache key.
            fresh = self.readers.bindings()
            _require(type(fresh) is BindingSnapshot, "consumer binding extraction failed")
            for info in fields(BindingSnapshot):
                name = info.name
                observed = getattr(fresh, name)
                supplied = getattr(current_bindings, name)
                _require(_same_callable(observed, supplied) or
                         (type(observed) is str and type(supplied) is str and observed == supplied),
                         "consumer supplied bindings differ")
            _require(frame is scope.frame and actual_ids is scope.selected,
                     "consumer frame/selected owner mismatch")
            output = self.readers.output(actual_ids)
            if scope.token is None or scope.reasons or self.poisoned:
                self._reject(scope, "consumer without complete production")
                return None
            scope.receipt = self.ledger.consume(scope.token, bindings, guards, output,
                                                consumer_stream, frame)
            return scope.receipt
        return self._safe(scope, observe)
