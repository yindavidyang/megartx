"""One-shot sync/async orchestration against the actual pinned utility API.

The frontend transport can be async while EngineCore scheduling stays sync.
Fixtures test ordering only; the runtime path always creates EngineCoreClient.
"""
import asyncio
from concurrent.futures import Future
from enum import Enum
import inspect
import math
import time

from .speculative_native_probe import ProbeError
from .speculative_native_plan import LIMITS, PURPOSE, SHA, COMMIT

RECEIPT = "megartx_owned_native_receipt"
RELEASE = "megartx_owned_native_release"
EXPECTED_RELEASE = {"drained": True, "released": True, "scheduler_stays_paused": True}


class State(Enum):
    FRESH = "fresh"
    PAUSING = "pausing"
    PAUSED = "paused"
    RECEIPT_ENTERED = "receipt_entered"
    RECEIPT = "receipt"
    RELEASING = "releasing"
    RELEASED = "released"
    UNCERTAIN = "uncertain"
    CLOSED = "closed"


def validate_summary(value):
    from .speculative_native_compare import validate_scalar_receipt
    return validate_scalar_receipt(value)


class ReceiptSession:
    def __init__(self, client, *, deadline, resources):
        if not callable(resources):
            raise ProbeError("Owned resource gate is required")
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ProbeError("Finite owned monotonic deadline required")
        self.client, self.deadline, self.resources = client, deadline, resources
        self.state, self.events = State.FRESH, []
        self.release_attempted = self.shutdown_attempted = False

    def gate(self):
        self.resources()
        if time.monotonic() >= self.deadline - LIMITS["cleanup_seconds"]:
            raise TimeoutError("Receipt deadline retains owned cleanup reserve")

    def mark(self, state):
        self.state = state
        self.events.append({"state": state.value, "monotonic": time.monotonic()})

    def _begin(self, admission):
        if self.state is not State.FRESH:
            raise ProbeError("Receipt session is single-use; uncertain state cannot be retried")
        if (type(admission) is not dict or admission.get("phase") != "zero_forward_receipt"
                or admission.get("client_purpose") != PURPOSE
                or not SHA.fullmatch(str(admission.get("client_plan_sha256", "")))
                or not COMMIT.fullmatch(str(admission.get("source_head", "")))
                or admission.get("deadline_monotonic") != self.deadline):
            raise ProbeError("Wrong-purpose or malformed source-bound client admission")
        self.gate()
        self.mark(State.PAUSING)

    def _complete_pause(self, result):
        if result is not None:
            raise ProbeError("Pinned pause utility must complete with None")
        self.mark(State.PAUSED)
        self.gate()

    def _complete_receipt(self, result):
        validate_summary(result)
        self.mark(State.RECEIPT)
        self.gate()
        return result

    def _complete_release(self, result):
        if (type(result) is not dict or set(result) != set(EXPECTED_RELEASE)
                or any(result[k] is not True for k in EXPECTED_RELEASE)):
            raise ProbeError("Worker drain/revocation/free result remains uncertain")
        self.mark(State.RELEASED)

    def _shutdown(self, primary):
        if self.shutdown_attempted:
            return
        self.shutdown_attempted = True
        try:
            self.client.shutdown(timeout=max(0.1, min(10, self.deadline - time.monotonic())))
        except BaseException as cleanup:
            self.mark(State.UNCERTAIN)
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                # Error strings can contain runtime data. Persist only error type.
                primary.add_note("Owned client shutdown failed: " + type(cleanup).__name__)
        else:
            self.mark(State.CLOSED)

    def run_sync(self, admission):
        if self.state is not State.FRESH:
            raise ProbeError("Receipt session is single-use; uncertain state cannot be retried")
        primary = None
        try:
            self._begin(admission)
            pause = self.client.call_utility("pause_scheduler", "keep", False)
            # Actual SyncMPClient waits internally. A Future is accepted only as
            # a compatibility completion object; it must finish before receipt.
            if isinstance(pause, Future):
                pause = pause.result(timeout=max(.1, self.deadline - time.monotonic() - LIMITS["cleanup_seconds"]))
            elif inspect.isawaitable(pause):
                raise ProbeError("Async transport passed to synchronous receipt path")
            self._complete_pause(pause)
            self.mark(State.RECEIPT_ENTERED)
            result = self._complete_receipt(self.client.call_utility(RECEIPT, admission))
            self.release_attempted = True
            self.mark(State.RELEASING)
            self._complete_release(self.client.call_utility(RELEASE))
            return result
        except BaseException as error:
            primary = error
            self.mark(State.UNCERTAIN)
            # Failed/partial receipt RPC may already have released in EngineCore,
            # or still be running. Never enqueue a second uncertain release.
            raise
        finally:
            self._shutdown(primary)

    async def run_async(self, admission):
        if self.state is not State.FRESH:
            raise ProbeError("Receipt session is single-use; uncertain state cannot be retried")
        primary = None
        async def utility(name, *args):
            self.gate()
            timeout = max(.1, self.deadline - time.monotonic() - LIMITS["cleanup_seconds"])
            return await asyncio.wait_for(self.client.call_utility_async(name, *args), timeout)
        try:
            self._begin(admission)
            self._complete_pause(await utility("pause_scheduler", "keep", False))
            self.mark(State.RECEIPT_ENTERED)
            result = self._complete_receipt(await utility(RECEIPT, admission))
            self.release_attempted = True
            self.mark(State.RELEASING)
            self._complete_release(await utility(RELEASE))
            return result
        except BaseException as error:
            primary = error
            self.mark(State.UNCERTAIN)
            raise
        finally:
            # Pinned AsyncMPClient.shutdown is a synchronous method. Keep it
            # bounded and do not leave a to_thread task alive past teardown.
            self._shutdown(primary)


def validate_actual_config(config):
    """Observe resolved selection; never select/override another runner."""
    selected = config.use_v2_model_runner
    if selected is not False:
        raise ProbeError("Resolved V2/unknown runner is outside the pinned V1 receipt; no runner override applied")
    if (config.scheduler_config.async_scheduling is not False
            or config.parallel_config.world_size != 1
            or config.cache_config.enable_prefix_caching
            or config.speculative_config is not None):
        raise ProbeError("Resolved actual EngineCore configuration differs")
    return {"use_v2_model_runner": False,
            "expected_runner_class": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            "actual_loaded_runner_identity": None}


def make_actual_client(plan):
    """Runtime-only. The supervisor must admit resources BEFORE this call."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.engine.core_client import EngineCoreClient, SyncMPClient, AsyncMPClient
    config = EngineArgs(**plan["engine_kwargs"]).create_engine_config()
    validate_actual_config(config)
    client = EngineCoreClient.make_client(multiprocess_mode=True,
        asyncio_mode=plan["client_mode"] == "async", vllm_config=config,
        executor_class=Executor.get_class(config), log_stats=False, renderer=None)
    expected = AsyncMPClient if plan["client_mode"] == "async" else SyncMPClient
    if type(client) is not expected:
        client.shutdown(timeout=10)
        raise ProbeError("Actual owned utility transport differs")
    return client
