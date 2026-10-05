"""CPU-only route provenance bookkeeping, deliberately disconnected from dispatch.

This module imports no device runtime and installs no hooks. Supplied snapshots
are observations, not installed-source, compiled-kernel, stream or tensor-content
attestations. Even successful consumption ALWAYS requires the existing route
readback and synchronization. No receipt is a native admission capability.
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields
import threading


class RouteObservationError(RuntimeError):
    """Unsupported metadata or a broken observation lifecycle."""


class _Unknown:
    __slots__ = ()


UNKNOWN = _Unknown()


@dataclass(frozen=True, eq=False)
class BindingSnapshot:
    runner: object
    layer: object
    router: object
    select_experts: object
    custom_routing: object
    closure_owner: object
    prepare_finalize: object
    prepare: object
    experts: object
    jit_function: object
    jit_run: object
    compiled_kernel: object
    cuda_module: object
    cuda_function: object
    launcher: object
    source_key: str
    cache_key: str


@dataclass(frozen=True, eq=False)
class GuardSnapshot:
    # Unknown must never silently become absent/false. Future hooks must observe
    # each current value; these declarations do not inspect a live producer.
    capture_state: object = UNKNOWN
    capture_fn: object = UNKNOWN
    replay_output: object = UNKNOWN
    eplb_state: object = UNKNOWN
    instance_overrides: object = UNKNOWN
    pre_run_hooks: object = UNKNOWN
    launch_hooks: object = UNKNOWN
    simulation: object = UNKNOWN
    diagnostics: object = UNKNOWN
    controlled: object = UNKNOWN
    forced: object = UNKNOWN
    routing_variant: object = UNKNOWN


@dataclass(frozen=True, eq=False)
class StreamSnapshot:
    device: int
    context: object
    handle: int


@dataclass(frozen=True, eq=False)
class OutputSnapshot:
    owner: object
    storage_owner: object
    pointer: int
    storage_pointer: int
    storage_bytes: int
    offset_bytes: int
    device: int
    shape: tuple = (1, 8)
    strides: tuple = (8, 1)
    dtype: str = "int32"
    view_bytes: int = 32


@dataclass(frozen=True, eq=False)
class ProducerSnapshot:
    bindings: BindingSnapshot
    guards: GuardSnapshot
    stream: StreamSnapshot
    output: OutputSnapshot
    experts: int = 128
    top_k: int = 8
    block_experts: int = 128
    num_warps: int = 1
    grid: tuple = (1,)


@dataclass(frozen=True)
class RouteObservationReceipt:
    """An audit record of matched metadata; never a truthy admission token."""
    generation: int
    device: int
    producer_stream: int
    consumer_stream: int
    output_pointer: int

    @property
    def route_readback_required(self):
        return True

    def __bool__(self):
        raise TypeError("route observation is not dispatch eligibility")


@dataclass(eq=False)
class _Active:
    token: object
    generation: int
    thread: threading.Thread
    frame: object
    stream: StreamSnapshot
    phase: str = "open"
    output: OutputSnapshot | None = None
    consumer: StreamSnapshot | None = None


_REFERENCE_FIELDS = tuple(f.name for f in fields(BindingSnapshot)
                          if f.name not in ("source_key", "cache_key"))


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _binding_valid(value):
    return (type(value) is BindingSnapshot
            and all(getattr(value, name) is not None and
                    getattr(value, name) is not UNKNOWN for name in _REFERENCE_FIELDS)
            and type(value.source_key) is str and bool(value.source_key)
            and type(value.cache_key) is str and bool(value.cache_key))


def _same_binding(left, right):
    return (_binding_valid(right)
            and all(getattr(left, name) is getattr(right, name) for name in _REFERENCE_FIELDS)
            and left.source_key == right.source_key and left.cache_key == right.cache_key)


def _guards_valid(value):
    return (type(value) is GuardSnapshot
            and type(value.capture_state) is str and value.capture_state == "none"
            and value.capture_fn is None and value.replay_output is None
            and value.eplb_state is None
            and all(type(getattr(value, name)) is tuple and not getattr(value, name)
                    for name in ("instance_overrides", "pre_run_hooks", "launch_hooks"))
            and all(getattr(value, name) is False
                    for name in ("simulation", "diagnostics", "controlled", "forced"))
            and type(value.routing_variant) is str and value.routing_variant == "gemma_cuda")


def _stream_valid(value):
    return (type(value) is StreamSnapshot and _integer(value.device)
            and value.context is not None and value.context is not UNKNOWN
            and _integer(value.handle))


def _same_stream(left, right):
    return (_stream_valid(right) and left.device == right.device
            and left.context is right.context and left.handle == right.handle)


def _output_valid(value):
    return (type(value) is OutputSnapshot
            and value.owner is not None and value.owner is not UNKNOWN
            and value.storage_owner is not None and value.storage_owner is not UNKNOWN
            and _integer(value.pointer, 1) and _integer(value.storage_pointer, 1)
            and _integer(value.storage_bytes, 32) and _integer(value.offset_bytes)
            and _integer(value.device) and type(value.view_bytes) is int and value.view_bytes == 32
            and type(value.shape) is tuple and value.shape == (1, 8)
            and all(type(x) is int for x in value.shape)
            and type(value.strides) is tuple and value.strides == (8, 1)
            and all(type(x) is int for x in value.strides)
            and type(value.dtype) is str and value.dtype == "int32"
            and value.pointer % 4 == 0 and value.offset_bytes % 4 == 0
            and value.pointer == value.storage_pointer + value.offset_bytes
            and value.offset_bytes + value.view_bytes <= value.storage_bytes)


def _same_output(left, right):
    return (_output_valid(right) and left.owner is right.owner
            and left.storage_owner is right.storage_owner
            and all(getattr(left, name) == getattr(right, name)
                    for name in ("pointer", "storage_pointer", "storage_bytes", "offset_bytes",
                                 "device", "shape", "strides", "dtype", "view_bytes")))


class RouteObservationLedger:
    """One non-reentrant observer with terminal poisoning, not a trust issuer.

    The expected identities are retained for this ledger's lifetime. Per-call
    frame/output owners are retained only until owner-thread cleanup. No reset
    method exists: replacing a poisoned ledger requires a separately controlled
    lifecycle decision, and cannot legitimize partial device work.
    """
    def __init__(self, expected):
        if not _binding_valid(expected):
            raise RouteObservationError("expected binding is incomplete")
        self._expected = expected
        self._active = None
        self._generation = 0
        self._poison = None
        self._lock = threading.RLock()

    @property
    def poisoned(self):
        with self._lock:
            return self._poison is not None

    @property
    def active(self):
        with self._lock:
            return self._active is not None

    def _fail(self, message):
        if self._poison is None:
            self._poison = message
        raise RouteObservationError(message)

    def _current(self, token, phase):
        if self._poison is not None:
            raise RouteObservationError("observation ledger is poisoned: " + self._poison)
        current = self._active
        if current is None or token is not current.token:
            self._fail("foreign, stale or absent invocation token")
        if threading.current_thread() is not current.thread:
            self._fail("observation used outside its originating thread")
        if current.phase != phase:
            self._fail("out-of-order or duplicate observation")
        return current

    def _begin(self, bindings, guards, stream, frame):
        with self._lock:
            if self._poison is not None:
                raise RouteObservationError("observation ledger is poisoned: " + self._poison)
            if self._active is not None:
                self._fail("nested or concurrent invocation")
            # Unsupported before any active transaction is harmless. A caller
            # continues its ordinary checked path; no receipt was issued.
            if (not _same_binding(self._expected, bindings) or not _guards_valid(guards)
                    or not _stream_valid(stream) or frame is None or frame is UNKNOWN):
                raise RouteObservationError("unsupported or unobserved initial provenance")
            self._generation += 1
            token = object()
            self._active = _Active(token, self._generation, threading.current_thread(), frame, stream)
            return token

    def _cleanup(self, token, primary):
        with self._lock:
            current = self._active
            if current is None:
                return  # Idempotent after owner cleanup, including poison.
            if token is not current.token or threading.current_thread() is not current.thread:
                self._fail("only the originating scope/thread may clean up")
            if primary is not None and self._poison is None:
                self._poison = "exception inside observation transaction"
            failure = self._poison
            if primary is None and current.phase != "consumed":
                failure = failure or "scope exited without one complete observation"
                self._poison = failure
            self._active = None  # Release output/frame owners even after poison.
            if primary is None and failure is not None:
                raise RouteObservationError("observation ledger is poisoned: " + failure)

    @contextmanager
    def invocation(self, bindings, guards, stream, frame):
        token = self._begin(bindings, guards, stream, frame)
        primary = None
        try:
            yield token
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._cleanup(token, primary)
            except BaseException as cleanup:
                # Even an unexpected cleanup implementation failure is terminal.
                # Never release a different scope or another thread's owners.
                with self._lock:
                    if self._poison is None:
                        self._poison = "observation cleanup failed"
                    current = self._active
                    if (current is not None and token is current.token
                            and threading.current_thread() is current.thread):
                        self._active = None
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note("Route observation cleanup failed: " + str(cleanup))

    def _validate_producer(self, current, snapshot):
        if (type(snapshot) is not ProducerSnapshot
                or not _same_binding(self._expected, snapshot.bindings)
                or not _guards_valid(snapshot.guards)
                or not _same_stream(current.stream, snapshot.stream)
                or not _output_valid(snapshot.output)
                or snapshot.output.device != current.stream.device
                or any(type(getattr(snapshot, name)) is not int or getattr(snapshot, name) != expected
                       for name, expected in (("experts", 128), ("top_k", 8),
                                              ("block_experts", 128), ("num_warps", 1)))
                or type(snapshot.grid) is not tuple or snapshot.grid != (1,)
                or any(type(x) is not int for x in snapshot.grid)):
            self._fail("producer metadata changed, unsupported or unobserved")

    def observe_production(self, token, before, after):
        """Record supplied launch-boundary snapshots, not GPU completion.

        Future extraction must snapshot initialized identities at both actual
        launch boundaries; reading a mutable kernel object only after return is
        not sufficient. No live launch interception is implemented here.
        """
        with self._lock:
            current = self._current(token, "open")
            self._validate_producer(current, before)
            self._validate_producer(current, after)
            if not _same_output(before.output, after.output):
                self._fail("producer output owner/view changed across launch")
            current.output = after.output
            current.phase = "produced"

    def observe_dependency(self, token, producer, consumer):
        """Record metadata only AFTER an actual future wait succeeds.

        This method performs no wait and does not certify an event. Keep all
        current waits, record_stream calls and host readbacks unchanged.
        """
        with self._lock:
            current = self._current(token, "produced")
            if (not _same_stream(current.stream, producer) or not _stream_valid(consumer)
                    or consumer.handle == 0 or consumer.device != producer.device
                    or consumer.context is not producer.context):
                self._fail("dependency stream/device/context mismatch")
            current.consumer = consumer
            current.phase = "dependency_observed"

    def consume(self, token, bindings, guards, output, consumer, frame):
        """Consume once for audit; result always requires route readback."""
        with self._lock:
            current = self._current(token, "dependency_observed")
            if (not _same_binding(self._expected, bindings) or not _guards_valid(guards)
                    or not _same_output(current.output, output)
                    or not _same_stream(current.consumer, consumer) or frame is not current.frame):
                self._fail("consumer provenance/owner/frame changed")
            current.phase = "consumed"
            return RouteObservationReceipt(current.generation, current.stream.device,
                current.stream.handle, current.consumer.handle, current.output.pointer)
