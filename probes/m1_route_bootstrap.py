"""Uninstalled bootstrap at the reviewed lookup and original launcher call.

This only discovers a naturally selected, host-returned kernel for a later
BindingRouteObserver profile. It supplies no native qualification, receipt or
route-check exemption. Factories return wrappers; nothing assigns runtime
attributes, changes a module namespace, imports a device runtime or queries a
device. Shared-file installation and live source admission remain separate.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import threading
import types

from m1_route_binding_adapter import _FunctionSeal
from m1_route_observer import _Incomplete, _require, _same_callable, freeze_values
from megartx.m1_route_receipt import UNKNOWN


@dataclass(frozen=True, eq=False)
class CapturedKernel:
    """Actual stable owners only, never invocation tensor/route arguments.

    Host launch return does not establish device completion. This record is
    not accepted by the receipt ledger or native admission. Missing context,
    capture state, ABI interpretation and loaded-source association stay open.
    """
    owner: object
    jit: object
    cache: object
    key: str
    selected: object
    target: object
    specialization: object
    options: object
    launcher: object
    cuda_module: object
    cuda_function: object
    packed_metadata: object
    inner_stream_handle: int

    @property
    def qualification_missing(self):
        return ("loaded source association", "producer context/capture",
                "stream context ownership", "native launcher ABI metadata",
                "tensor dtype/storage profile", "current guards/frame")

    def __bool__(self):
        raise TypeError("captured kernel is not dispatch eligibility")


def build_bootstrap(*, enabled=False, **configuration):
    """Disabled means zero owner/namespace/callback inspection."""
    if type(enabled) is not bool:
        raise ValueError("explicit Boolean observation enable required")
    return NaturalSelectionBootstrap(**configuration) if enabled else None


class NaturalSelectionBootstrap:
    """One bounded bootstrap for the 30 natural runners and one shared JIT.

    ``invocation`` surrounds an original registered runner, without replacing
    its arguments or result. ``wrap_jit`` and ``wrap_launcher`` return exactly
    one shared wrapper each. The latter wraps CudaLauncher.__call__, not a lazy
    CompiledKernel.run read. The existing source-bound lookup helper calls
    cache_selection_observed before any later BindingRouteObserver registers.
    """
    def __init__(self, *, owners, jit, jit_run, launcher_call, kernel_type,
                 launcher_type, module_functions, callback_failures, limit=30):
        _require(type(owners) is tuple and len(owners) == 30 and
                 len({id(owner) for owner in owners}) == 30, "30 distinct actual runners required")
        _require(type(limit) is int and 0 < limit <= 30, "bounded natural invocation limit required")
        _require(type(jit_run) is types.MethodType and jit_run.__self__ is jit,
                 "actual bound shared JIT run required")
        _require(type(launcher_call) is types.FunctionType and
                 isinstance(kernel_type, type) and isinstance(launcher_type, type),
                 "explicit Python launcher and exact owner types required")
        _require(type(module_functions) is tuple and module_functions,
                 "loaded Python module/function ownership required")
        self._modules = tuple((module, function, _FunctionSeal(function))
                              for module, function in module_functions)
        functions = tuple(function for _, function, _ in self._modules)
        _require(any(fn is jit_run.__func__ for fn in functions) and
                 any(fn is launcher_call for fn in functions), "both original callable namespaces required")
        self.owners, self.jit = owners, jit
        self.jit_run, self.launcher_call = jit_run, launcher_call
        self.kernel_type, self.launcher_type = kernel_type, launcher_type
        self.callback_failures, self.limit = callback_failures, limit
        self._failure_seal = _FunctionSeal(callback_failures)
        self._lock = threading.RLock()
        self._active = None
        self._inflight = self._started = 0
        self._poisoned = self._closed = False
        self._captures = []
        self.rejected = 0
        self._namespaces()
        self.jit_wrapper = self._jit
        def launcher_wrapper(launcher, *args, **kwargs):
            return self._launcher(launcher, *args, **kwargs)
        self.launcher_wrapper = launcher_wrapper

    def _namespaces(self):
        for module, fn, seal in self._modules:
            _require(type(module) is types.ModuleType and
                     sys.modules.get(module.__name__) is module and
                     type(fn) is types.FunctionType and fn.__globals__ is vars(module),
                     "loaded module/function namespace changed")
            seal.check()
        self._failure_seal.check()
        failures = self.callback_failures()
        _require(type(failures) is int and failures == 0, "lookup callback failure invalidates bootstrap")

    # Retirement may clear the captured dict before a foreign thread checks
    # membership. A retired record is outside scope, never a pass-through error.
    def _scope(self):
        scope = self._active
        return scope if scope is not None and scope.get("thread") is threading.current_thread() else None

    def _safe(self, scope, operation):
        try:
            return operation()
        except _Incomplete:
            # A naturally cold or unsupported call does not authorize a retry,
            # and does not prevent observing a later ordinary warm invocation.
            with self._lock:
                scope["invalid"] = True
        except BaseException:
            # Keep no exception, traceback, formatted message or tensor owner.
            with self._lock:
                scope["invalid"] = True
                self._poisoned = True
        return None

    @contextmanager
    def invocation(self, owner):
        """Ordinary execution must run once even when observation is disabled."""
        with self._lock:
            busy = self._inflight != 0
            self._inflight += 1
            if busy:
                self._poisoned = True
                if self._active is not None:
                    self._active["invalid"] = True
            observe = (not busy and not self._closed and not self._poisoned and
                       self._started < self.limit and any(owner is item for item in self.owners))
            scope = None
            if observe:
                self._started += 1
                scope = dict(owner=owner, thread=threading.current_thread(), invalid=False,
                             jitting=False, jit_count=0, cache_count=0, launch_count=0,
                             args=None, selection=None, captured=None)
                self._active = scope
        if scope is not None:
            self._safe(scope, self._namespaces)
        try:
            yield
        except BaseException:
            with self._lock:
                self._poisoned = True
                if scope is not None:
                    scope["invalid"] = True
            raise
        finally:
            if scope is not None:
                self._safe(scope, self._namespaces)
            # Accepted overlap, retirement and publication use the same lock.
            with self._lock:
                if scope is not None:
                    if not scope["invalid"] and not self._poisoned and scope["captured"] is not None:
                        self._captures.append(scope["captured"])
                    else:
                        self.rejected += 1
                    self._active = None
                    scope.clear()
                self._inflight -= 1

    def wrap_jit(self, original):
        _require(_same_callable(original, self.jit_run), "one exact shared JIT wrapper required")
        return self.jit_wrapper

    def wrap_launcher(self, original):
        _require(original is self.launcher_call, "one exact shared launcher wrapper required")
        return self.launcher_wrapper

    def _jit(self, *args, **kwargs):
        scope = self._scope()
        if scope is None:
            return self.jit_run(*args, **kwargs)
        def before():
            self._namespaces()
            scope["jit_count"] += 1
            _require(not scope["jitting"] and scope["jit_count"] == 1, "duplicate/nested shared JIT")
            scope["jitting"], scope["args"] = True, args
        self._safe(scope, before)
        try:
            result = self.jit_run(*args, **kwargs)
            self._safe(scope, lambda: _require(scope["selection"] is not None and
                       result is scope["selection"][2], "cold/async/changed JIT return"))
            return result
        finally:
            scope["jitting"], scope["args"] = False, None

    def cache_selection_observed(self, *, cache, key, selected, target, specialization, options):
        scope = self._scope()
        if scope is None:
            return
        def observe():
            self._namespaces()
            scope["cache_count"] += 1
            _require(scope["jitting"] and scope["cache_count"] == 1 and not scope["launch_count"],
                     "lookup outside exact original selection order")
            _require(type(key) is str and key and type(selected) is self.kernel_type,
                     "cold/unknown selected kernel")
            scope["selection"] = (cache, key, selected, target,
                                    freeze_values(specialization), freeze_values(options))
        self._safe(scope, observe)

    def _launcher(self, launcher, *args, **kwargs):
        scope = self._scope()
        snapshot = None
        if scope is not None:
            def before():
                self._namespaces()
                scope["launch_count"] += 1
                _require(scope["jitting"] and scope["launch_count"] == 1 and scope["selection"] is not None,
                         "launch without one natural selection")
                _require(type(launcher) is self.launcher_type and not kwargs and len(args) == 16,
                         "unknown actual launcher/signature")
                selected = scope["selection"][2]
                values = vars(selected)
                _require(values.get("_run", UNKNOWN) is launcher and
                         values.get("function", UNKNOWN) is args[4] and
                         values.get("packed_metadata", UNKNOWN) is args[5], "selected launch owners differ")
                incoming, bound = scope["args"], args[9:]
                _require(incoming is not None and len(incoming) == 7 and
                         all(left is right for left, right in zip(incoming[:4], bound[:4])) and
                         all(type(value) is int for value in (*incoming[4:], *bound[4:])) and
                         incoming[4:] == bound[4:] and type(args[3]) is int and args[3] >= 0,
                         "actual bound arguments/inner stream differ")
                return (values.get("module", UNKNOWN), args[4], args[5], args[3],
                        vars(launcher).get("launch", UNKNOWN))
            snapshot = self._safe(scope, before)
        # This is the only original launcher call, including every failure path.
        result = self.launcher_call(launcher, *args, **kwargs)
        if scope is not None and snapshot is not None:
            def after():
                self._namespaces()
                cache, key, selected, target, specialization, options = scope["selection"]
                values = vars(selected)
                _require(values.get("_run", UNKNOWN) is launcher and
                         values.get("module", UNKNOWN) is snapshot[0] and
                         values.get("function", UNKNOWN) is snapshot[1] and
                         values.get("packed_metadata", UNKNOWN) is snapshot[2] and
                         vars(launcher).get("launch", UNKNOWN) is snapshot[4], "launch owners changed across call")
                scope["captured"] = CapturedKernel(scope["owner"], self.jit, cache, key, selected, target,
                    specialization, options, launcher, snapshot[0], snapshot[1], snapshot[2], snapshot[3])
            self._safe(scope, after)
        return result

    def take_captures(self):
        """Drain once, only after original runner scopes retire; never register here."""
        with self._lock:
            _require(self._inflight == 0, "original runner scopes have not drained")
            result = () if self._poisoned else tuple(self._captures)
            self._captures.clear()
            return result

    def close(self):
        """Disable observation without waiting on or disturbing ordinary calls."""
        with self._lock:
            self._closed = True
            self._captures.clear()
            if self._inflight:
                self._poisoned = True
                if self._active is not None:
                    self._active["invalid"] = True
