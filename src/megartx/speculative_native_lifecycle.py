"""Default-off EngineCore utility registration and exclusive diagnostic lease.

No runtime imports until explicit plugin registration or an admitted worker RPC.
The normal utility loop owns reservations. An observation frame is not a lease.
"""

from dataclasses import dataclass
import copy
from functools import wraps
import inspect
import os
from pathlib import Path
import threading
import time
import uuid

from .speculative_native_probe import (ADAPTER_FILES, GPU_CAP, P, REVISION,
    NativeProbeWorkerExtension, ProbeError, _class_source, _process_start,
    check_admission, digest, inspect_sources)

PURPOSE = "exclusive_native_verifier"
WORKER_EXTENSION = "megartx.speculative_native_lifecycle.NativeDiagnosticWorkerExtension"
UTILITIES = ("megartx_owned_native_receipt", "megartx_owned_native_probe", "megartx_owned_native_release")


def implementation_binding():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in ("speculative_native_lifecycle.py",
            "speculative_native_receipt.py", "speculative_native_probe.py")}


def check_receipt_admission(admission):
    required = {"phase": "zero_forward_receipt", "independent_review_clear": True,
        "parent_source_protocol_accepted": True, "owned_lifecycle_verified": True,
        "max_concurrent_gpu_jobs": 1, "max_extra_gpu_bytes": GPU_CAP,
        "compiler_memory_bytes": 2 << 30, "compiler_timeout_seconds": 300,
        "compiler_scope": "shared_host", "checkpoint_revision": REVISION,
        "checkpoint_manifest_verified": True, "tokenizer_template_verified": True,
        "implementation_binding": implementation_binding(),
        "zero_protocol_sha256": digest(Path(__file__).resolve().parents[2] /
                                     "docs/design/speculative-native-zero-forward-protocol.json"),
        "adapter_source_sha256": {name: digest(Path(__file__).parent / name) for name in ADAPTER_FILES}}
    if not isinstance(admission, dict) or any(admission.get(k) != v for k, v in required.items()):
        raise ProbeError("Missing exact receipt source/protocol/owned startup admission")
    deadline = admission.get("deadline_monotonic")
    if (type(deadline) not in (int, float) or not time.monotonic() < deadline <= time.monotonic() + 1800
            or not isinstance(admission.get("parent_slot"), str) or not admission["parent_slot"]):
        raise ProbeError("Missing bounded receipt deadline or sole-owner parent slot")
    return admission


@dataclass
class Lease:
    pool: object
    blocks: list
    ticket: dict
    receipt: dict = None
    released: bool = False
    probe_used: bool = False
    ownership_verified: bool = False
    poisoned: bool = False


def _require_core(core, utility):
    _class_source(core, "vllm.v1.engine.core", "EngineCoreProc")
    state = getattr(core, "_megartx_native_state", None)
    if (not state or state["identity"] != (os.getpid(), _process_start(os.getpid()))
            or state.get("utility") != utility or not state["initialized"]):
        raise ProbeError("Wrong process or unregistered synchronous utility owner")
    if (state["requests_seen"] or state["pause"] != ("keep", False, True)
            or not core.is_scheduler_paused() or core.scheduler.has_requests()
            or core.batch_queue or getattr(core, "pending_add_requests", ())
            or not core.input_queue.empty()):
        raise ProbeError("Fresh empty completed keep-pause required; queued work rejected")
    config = core.vllm_config
    if (config.parallel_config.world_size != 1 or config.parallel_config.enable_dbo
            or config.parallel_config.worker_extension_cls != WORKER_EXTENSION
            or config.cache_config.enable_prefix_caching or config.speculative_config is not None
            or config.use_v2_model_runner or config.kv_transfer_config is not None
            or config.ec_transfer_config is not None or config.scheduler_config.async_scheduling is not False):
        raise ProbeError("Incompatible actual native lifecycle configuration")
    return state


def _reserve(core):
    manager = core.scheduler.kv_cache_manager
    _class_source(manager, "vllm.v1.core.kv_cache_manager", "KVCacheManager")
    pool = manager.block_pool
    _class_source(pool, "vllm.v1.core.block_pool", "BlockPool")
    sizes = [g.kv_cache_spec.block_size for g in core.scheduler.kv_cache_config.kv_cache_groups]
    if not sizes or any(type(b) is not int or b not in (16, 32, 64) for b in sizes):
        raise ProbeError("Unreviewed actual manager block size")
    counts = [(P + 4 + b - 1) // b + 1 for b in sizes]
    ticket = {"purpose": PURPOSE, "nonce": str(uuid.uuid4()), "engine_pid": os.getpid(),
              "engine_start": _process_start(os.getpid()), "block_sizes": sizes, "groups": []}
    blocks = pool.get_new_blocks(sum(counts))
    # Retain objects immediately, before any validation that could throw.
    lease = Lease(pool, blocks, ticket)
    core._megartx_native_lease = lease
    if (len(blocks) != sum(counts) or any(b.ref_cnt != 1 or b.is_null for b in blocks)
            or len({b.block_id for b in blocks}) != len(blocks)):
        raise ProbeError("BlockPool did not grant exclusive refcounted objects")
    cursor = 0
    for count in counts:
        ticket["groups"].append([b.block_id for b in blocks[cursor:cursor + count]])
        cursor += count
    lease.ownership_verified = True
    return lease


def _release(core):
    lease = getattr(core, "_megartx_native_lease", None)
    if lease is None or lease.released:
        raise ProbeError("Native lease absent or already released")
    if lease.poisoned:
        raise ProbeError("Poisoned lease retains refs until owned teardown")
    # Always retain objects on an uncertain drain; poisoning never permits resume.
    core._megartx_native_poisoned = True
    try:
        if not lease.ownership_verified:
            raise ProbeError("Uncertain reservation ownership retained for teardown")
        results = core.model_executor.collective_rpc("megartx_native_release", timeout=30,
            kwargs={"ticket": lease.ticket})
        if len(results) != 1 or results[0] != {"drained": True, "revoked": True}:
            raise ProbeError("Worker did not confirm drain and lease revocation")
        lease.pool.free_blocks(lease.blocks)
        lease.released = True
    except BaseException:
        lease.poisoned = True
        core._megartx_native_retained_blocks = lease.blocks
        raise
    return {"drained": True, "released": True, "scheduler_stays_paused": True}


def owned_receipt(core, admission):
    check_receipt_admission(admission)
    state = _require_core(core, UTILITIES[0])
    with state["lock"]:
        _require_core(core, UTILITIES[0])  # close the ingress race before sealing
        if state["exclusive"]:
            raise ProbeError("Receipt capability is single-use")
        # Admission/freshness checks precede mutation. Never resume after this point.
        state["exclusive"] = True
        core._megartx_native_poisoned = True
        try:
            lease = _reserve(core)
            results = core.model_executor.collective_rpc("megartx_native_receipt",
                timeout=min(900, admission["deadline_monotonic"] - time.monotonic()),
                kwargs={"ticket": lease.ticket, "admission": admission})
            if (len(results) != 1 or results[0].get("drained") is not True
                    or results[0].get("diagnostic_target_forwards") != 0
                    or results[0].get("lease_nonce") != lease.ticket["nonce"]):
                raise ProbeError("Unbound or nonzero-forward receipt")
            from .speculative_native_receipt import receipt_digest, sanitized_receipt
            lease.receipt = results[0]
            return {"receipt_sha256": receipt_digest(lease.receipt), **sanitized_receipt(lease.receipt)}
        except BaseException as primary:
            if getattr(core, "_megartx_native_lease", None) is not None:
                try:
                    _release(core)
                except BaseException as cleanup:
                    if hasattr(primary, "add_note"):
                        primary.add_note("Drain unknown; refs retained for owned teardown: " + str(cleanup))
            raise


def owned_probe(core, prompt, admission):
    check_admission(admission)
    _require_core(core, UTILITIES[1])
    lease = getattr(core, "_megartx_native_lease", None)
    from .speculative_native_receipt import receipt_digest
    if (lease is None or lease.released or lease.poisoned or lease.probe_used or lease.receipt is None
            or admission.get("zero_forward_receipt_sha256") != receipt_digest(lease.receipt)
            or lease.receipt["decision"]["admitted"] is not True):
        raise ProbeError("No live separately admitted verifier capability; receipt is not fit admission")
    lease.probe_used = True
    primary = None
    try:
        results = core.model_executor.collective_rpc("megartx_native_probe",
            timeout=min(900, admission["deadline_monotonic"] - time.monotonic()),
            kwargs={"ticket": lease.ticket, "prompt": list(prompt), "admission": admission})
        if len(results) != 1 or results[0].get("drained") is not True:
            raise ProbeError("Worker probe drainage unknown")
        return results[0]
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _release(core)
        except BaseException as cleanup:
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note("Drain unknown; refs retained: " + str(cleanup))


def owned_release(core):
    _require_core(core, UTILITIES[2])
    return _release(core)


def _birth_state():
    return {"identity": (os.getpid(), _process_start(os.getpid())), "initialized": False,
            "requests_seen": 0, "pause": None, "exclusive": False, "utility": None,
            "birth_observed": True, "lock": threading.RLock()}


def _active_core_birth(base):
    """General plugins run INSIDE pinned EngineCore.__init__ before its executor.

    Observe that exact source frame, never adopt an already loaded engine.
    Future instances instead enter the registered __init__ wrapper.
    """
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code is base.__init__.__code__:
                core = frame.f_locals.get("self")
                if core is None or hasattr(core, "model_executor") or hasattr(core, "scheduler"):
                    raise ProbeError("Native registration missed the source-bound core birth")
                return core
            frame = frame.f_back
        return None
    finally:
        del frame


def _register_classes(base, proc, active_core=None):
    """Internal mutation step; caller verifies ALL identities first."""
    old_init = base.__init__
    @wraps(old_init)
    def init(self, *args, **kwargs):
        self._megartx_native_state = _birth_state()
        old_init(self, *args, **kwargs)
        self._megartx_native_state["initialized"] = True
    patches = [(base, "__init__", init)]
    for name in ("preprocess_add_request", "add_request", "resume_scheduler", "step", "step_with_batch_queue"):
        old = getattr(base, name)
        def guard(self, *args, _name=name, _old=old, **kwargs):
            state = self._megartx_native_state
            with state["lock"]:
                if state["exclusive"]:
                    if _name in ("step", "step_with_batch_queue"):
                        # Busy loop still calls step_fn after each utility. Return
                        # its idle tuple without entering the original runner.
                        return {}, False
                    raise ProbeError("Exclusive diagnostic engine stays paused until shutdown")
                if _name in ("preprocess_add_request", "add_request"):
                    state["requests_seen"] += 1
                if _name == "resume_scheduler":
                    state["pause"] = None
                return _old(self, *args, **kwargs)
        patches.append((base, name, wraps(old)(guard)))
    for cls in (base, proc):
        old = cls.pause_scheduler
        def pause(self, mode="abort", clear_cache=True, _old=old):
            state = self._megartx_native_state
            if state["exclusive"]:
                raise ProbeError("Diagnostic engine cannot change pause or cache state")
            state["pause"] = None
            result = _old(self, mode=mode, clear_cache=clear_cache)
            def complete(future=None):
                if future is not None:
                    future.result()  # failure leaves pause unverified
                state["pause"] = (mode, clear_cache, True)
            if result is None:
                complete()
            else:
                result.add_done_callback(complete)
            return result
        patches.append((cls, "pause_scheduler", wraps(old)(pause)))
    old_handle = proc._handle_client_request
    @wraps(old_handle)
    def handle(self, kind, request):
        state = self._megartx_native_state
        # The observed current constructor cannot be wrapped retroactively.
        # Its first synchronous loop dispatch proves that constructor completed.
        if state["birth_observed"] and not state["initialized"]:
            state["initialized"] = True
        name = request[2] if kind.name == "UTILITY" else None
        if state["exclusive"] and kind.name not in ("WAKEUP",) and name not in UTILITIES[1:]:
            raise ProbeError("Exclusive native utility owner rejects serving and unrelated utilities")
        state["utility"] = name
        try:
            return old_handle(self, kind, request)
        finally:
            state["utility"] = None
    patches.extend([(proc, "_handle_client_request", handle), (proc, UTILITIES[0], owned_receipt),
                    (proc, UTILITIES[1], owned_probe), (proc, UTILITIES[2], owned_release)])
    for cls, name, callback in patches:
        setattr(cls, name, callback)
    if active_core is not None:
        active_core._megartx_native_state = _birth_state()


def install_native_diagnostic():
    """Call from general plugin before EngineCore construction; no installed edits."""
    flag = os.environ.get("MEGARTX_NATIVE_DIAGNOSTIC")
    if flag is None:
        return False  # before vLLM/Torch imports or monkeypatches
    if (flag != "1" or os.environ.get("MEGARTX_SCALE_MODE") != "native"
            or os.environ.get("VLLM_PLUGINS") != "megartx_scale_adapter"
            or os.environ.get("MEGARTX_M1_PREPARATION")):
        raise ProbeError("Explicit default-off native registration configuration differs")
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    # Complete source/duplicate checks before changing any class.
    root = Path(inspect.getsourcefile(EngineCore)).resolve().parents[3]
    inspect_sources(root)
    for cls in (EngineCore, EngineCoreProc):
        if cls.__module__ != "vllm.v1.engine.core" or cls.__name__ not in ("EngineCore", "EngineCoreProc"):
            raise ProbeError("Unexpected actual EngineCore registration owner")
        if any(hasattr(cls, name) for name in UTILITIES):
            raise ProbeError("Duplicate or conflicting native registration")
    active_core = _active_core_birth(EngineCore)
    _register_classes(EngineCore, EngineCoreProc, active_core)
    return True


def _worker_ticket(ticket):
    if (not isinstance(ticket, dict) or ticket.get("purpose") != PURPOSE
            or str(uuid.UUID(ticket["nonce"])) != ticket["nonce"]
            or _process_start(ticket["engine_pid"]) != ticket["engine_start"]):
        raise ProbeError("Wrong-purpose or stale EngineCore lease")


class NativeDiagnosticWorkerExtension(NativeProbeWorkerExtension):
    def megartx_native_receipt(self, ticket, admission):
        check_receipt_admission(admission)
        _class_source(self, "vllm.v1.worker.gpu_worker", "Worker")
        _worker_ticket(ticket)
        if getattr(self, "_megartx_native_receipt_used", False):
            raise ProbeError("Worker receipt capability is single-use")
        self._megartx_native_receipt_used = True
        self._megartx_native_bound_ticket = copy.deepcopy(ticket)
        self._megartx_native_bound_runner = self.model_runner
        self._megartx_native_lease_released = False
        import torch
        from .speculative_native_receipt import collect_receipt, receipt_digest
        with torch.inference_mode():
            receipt = collect_receipt(self.model_runner, ticket, admission)
        if self.model_runner is not self._megartx_native_bound_runner:
            raise ProbeError("Worker model runner changed during receipt collection")
        self._megartx_native_receipt = receipt
        self._megartx_native_receipt_hash = receipt_digest(receipt)
        self.model_runner._megartx_native_mutation_lease = {
            "ticket": copy.deepcopy(ticket), "receipt_sha256": self._megartx_native_receipt_hash,
            "admitted": receipt["decision"]["admitted"] is True}
        return receipt

    def megartx_native_probe(self, ticket, prompt, admission):
        _worker_ticket(ticket)
        if (ticket != getattr(self, "_megartx_native_bound_ticket", None)
                or self.model_runner is not getattr(self, "_megartx_native_bound_runner", None)
                or getattr(self, "_megartx_native_lease_released", True)
                or admission.get("zero_forward_receipt_sha256") != getattr(self, "_megartx_native_receipt_hash", None)
                or self._megartx_native_receipt["decision"]["admitted"] is not True):
            raise ProbeError("Worker requires bound live mutation lease and separate fit admission")
        return super().megartx_native_probe(ticket, prompt, admission)

    def megartx_native_release(self, ticket):
        _class_source(self, "vllm.v1.worker.gpu_worker", "Worker")
        _worker_ticket(ticket)
        if (ticket != getattr(self, "_megartx_native_bound_ticket", None) or self._megartx_native_lease_released
                or self.model_runner is not self._megartx_native_bound_runner):
            raise ProbeError("Unknown or already revoked worker lease")
        import torch
        torch.cuda.synchronize(self.model_runner.device)
        if hasattr(self.model_runner, "_megartx_native_mutation_lease"):
            del self.model_runner._megartx_native_mutation_lease
        self._megartx_native_lease_released = True
        return {"drained": True, "revoked": True}
