"""Bounded prefill measurement collector. No device imports or dispatch.

Callbacks describe observed work, never create routes or modify K/V. Completion
checks are accounting evidence; independent native comparisons remain required.
"""

import copy
import math
import time
from contextlib import contextmanager
from functools import wraps

from . import prefill_runner as runner

GLOBAL_LAYERS = runner.GLOBAL_LAYERS
CHUNKS = {2048: (255, 256, 512, 1024, 2048),
          8192: (255, 256, 512, 1024, 2048, 8192)}
MAX_BYTES = 8 * 2**20
LOGIT_SITE = "Gemma4ForCausalLM.compute_logits"
LOGIT_SITE_SHA256 = runner.digest({"module": "vllm.model_executor.models.gemma4",
    "qualname": LOGIT_SITE, "file_sha256": "16ac0a67dcf5dd695a59edb177f20e1e69d9c3ec45c5883744470c7c06c516a3"})


def required_kv(start, end, layer):
    runner.integer(start, 0, 8191, "absolute start")
    runner.integer(end, start + 1, 8192, "absolute end")
    runner.integer(layer, 0, 29, "layer")
    return (0 if layer in GLOBAL_LAYERS else max(0, start - 1023), end)


def _bounded(value):
    import json
    size = len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_BYTES:
        raise ValueError("Collector exceeds 8 MiB; select a smaller cell")
    return size


def _poison_on_failure(method):
    """A failed callback burns its request, including interruptions in finalization."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.abort()
            raise
    return guarded


class CacheLedger:
    """I03 read-lifetime observations, separate from I06 tentative transactions.

    The provider must inspect real writer/block-table/cache state. This ledger
    cannot prove that a supplied digest corresponds to native tensor bytes.
    """

    def __init__(self, prompt_tokens, chunk_tokens, layout_sha256):
        if prompt_tokens not in CHUNKS or chunk_tokens not in CHUNKS[prompt_tokens]:
            raise ValueError("Only reviewed 2K/8K chunk sweep supported")
        runner.sha(layout_sha256, "cache layout")
        self.prompt_tokens = prompt_tokens
        self.spans = runner.plan_contract.chunk_spans(prompt_tokens, chunk_tokens)
        self.layout = layout_sha256
        self.index, self.active, self.poisoned = 0, None, False
        self.owners, self.frames = {}, []

    @_poison_on_failure
    def begin(self, start, end):
        if self.poisoned or self.active is not None or self.index >= len(self.spans):
            self.abort()
            raise ValueError("Nested, poisoned or excessive cache frame")
        runner.integer(start, 0, 8191, "start")
        runner.integer(end, start + 1, 8192, "end")
        if (start, end) != tuple(self.spans[self.index]):
            self.abort()
            raise ValueError("Cache frame differs from exact absolute schedule")
        self.active = {"start": start, "end": end, "layers": {}}

    @_poison_on_failure
    def observe(self, value):
        try:
            if self.active is None or self.poisoned:
                raise ValueError("Cache observation outside an active frame")
            runner.keys(value, {"layer", "kind", "kv_dtype", "required_start", "required_end",
                                "retained_start", "retained_end", "k_storage_id", "v_storage_id",
                                "processed_k_sha256", "processed_v_sha256", "layout_sha256",
                                "queries_complete"}, "cache observation")
            layer = value["layer"]
            runner.integer(layer, 0, 29, "cache layer")
            if layer in self.active["layers"]:
                raise ValueError("Duplicate cache layer observation")
            start, end = required_kv(self.active["start"], self.active["end"], layer)
            for key, expected in {"kind": "global" if layer in GLOBAL_LAYERS else "local",
                                  "kv_dtype": "bfloat16", "required_start": start,
                                  "required_end": end, "layout_sha256": self.layout,
                                  "queries_complete": True}.items():
                runner.equal(value[key], expected, "cache." + key)
            runner.integer(value["retained_start"], 0, start, "retained start")
            runner.integer(value["retained_end"], end, self.prompt_tokens + 256, "retained end")
            for key in ("k_storage_id", "v_storage_id"):
                runner._text(value[key], key)
            if value["k_storage_id"] == value["v_storage_id"]:
                raise ValueError("Processed K/V regions must remain distinct")
            owner = (value["k_storage_id"], value["v_storage_id"])
            # IDs denote regions, not allocator base pointers; K/V may share an allocation.
            if layer in self.owners and self.owners[layer] != owner:
                raise ValueError("Cache region owner changed during prompt")
            if any(set(owner) & set(v) for i, v in self.owners.items() if i != layer):
                raise ValueError("Cross-layer processed cache alias")
            for key in ("processed_k_sha256", "processed_v_sha256"):
                runner.sha(value[key], key)
            self.owners[layer] = owner
            self.active["layers"][layer] = copy.deepcopy(value)
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def end(self):
        if self.active is None or self.poisoned or set(self.active["layers"]) != set(range(30)):
            self.abort()
            raise ValueError("All thirty query/cache completions required before recycling")
        frame = self.active
        # These are safe lower bounds after completion, never cache mutation commands.
        release = {i: (0 if i in GLOBAL_LAYERS else max(0, frame["end"] - 1024))
                   for i in range(30)}
        self.frames.append(frame)
        self.active = None
        self.index += 1
        return release

    @_poison_on_failure
    def handoff(self, value, job):
        if self.poisoned or self.active is not None or self.index != len(self.spans):
            self.abort()
            raise ValueError("Partial/poisoned prompt cannot hand off to decode")
        try:
            runner._handoff(value, job, {"layout_sha256": self.layout})
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise
        return copy.deepcopy(value)

    def abort(self):
        self.poisoned = True
        self.active = None


class PrefillCollector:
    """One exact PR19 request; instrumentation and timing are separate jobs."""

    def __init__(self, job, tokens, layout_sha256, clock=time.monotonic_ns):
        runner.keys(job, {"run_id", "request_id", "workload", "prompt_tokens", "prompt_token_ids_sha256",
                          "chunk_tokens", "spans", "capacity_tokens", "mode", "region", "state", "observer_enabled"}, "collector job")
        if job["mode"] not in {"initialization", "correctness", "profile", "memory", "timing"}:
            raise ValueError("Unknown collector mode")
        if job["region"] not in (runner.PROFILE_COMPONENTS if job["mode"] == "profile" else {None}):
            raise ValueError("Collector region crossed mode boundary")
        if job["state"] not in {"cold", "warm"}:
            raise ValueError("Unknown collector state")
        if job["mode"] == "initialization":
            raise ValueError("Initialization belongs to the lifecycle provider")
        runner.request_payload(job, tokens)
        expected = runner.plan_contract.chunk_spans(job["prompt_tokens"], job["chunk_tokens"])
        if job["spans"] != [list(s) for s in expected] or job["capacity_tokens"] != job["prompt_tokens"] + 256:
            raise ValueError("Collector schedule/output reserve changed")
        runner.equal(job["observer_enabled"], job["mode"] in {"correctness", "profile", "memory"},
                     "observer policy")
        self.job, self.tokens, self.clock = copy.deepcopy(job), tuple(tokens), clock
        self.cache = CacheLedger(job["prompt_tokens"], job["chunk_tokens"], layout_sha256)
        self.active, self.failed, self.index = None, False, 0
        self.forwards, self.routes, self.events, self.timestamps = [], [], [], {}
        self.forward_times = []
        self.decode_rows = 0
        self.decode_input_hashes = []
        self.last_ns = -1
        self.handoff_record = None
        self.bootstrap_record = None
        self._writer = None

    @_poison_on_failure
    def attach_writer(self, writer):
        """The one native writer ledger shares this request's poison lifetime."""
        if self.failed or self._writer is not None:
            writer.abort()
            raise ValueError("Cannot attach a writer to a poisoned/already bound request")
        self._writer = writer
        if writer.poisoned:
            raise ValueError("Cannot attach an already poisoned native writer")

    @_poison_on_failure
    def _now(self):
        value = self.clock()
        runner.integer(value, 0, 2**63 - 1, "host monotonic time")
        if value < self.last_ns:
            self.abort()
            raise ValueError("Host clock moved backwards")
        self.last_ns = value
        return value

    @_poison_on_failure
    def mark(self, boundary):
        sequence = ("request_accept", "input_ready", "prompt_begin", "prompt_complete", "kv_ready", "first_token")
        if self.failed or boundary not in sequence or boundary in self.timestamps:
            self.abort()
            raise ValueError("Duplicate/unknown/poisoned timing boundary")
        if list(self.timestamps) != list(sequence[:sequence.index(boundary)]):
            self.abort()
            raise ValueError("Timing boundaries out of dependency order")
        if boundary == "prompt_complete" and (self.active is not None or self.index != len(self.job["spans"])):
            self.abort()
            raise ValueError("Whole prompt boundary before all prompt rows complete")
        if boundary == "kv_ready" and self.handoff_record is None:
            self.abort()
            raise ValueError("KV readiness requires all-layer handoff")
        self.timestamps[boundary] = self._now()

    @_poison_on_failure
    def begin_forward(self, tokens, positions, phase="prompt"):
        try:
            if self.failed or self.active is not None:
                raise ValueError("Nested or poisoned model forward")
            if type(tokens) is not list or type(positions) is not list:
                raise ValueError("Flat observed token/position lists required")
            if any(type(p) is not int for p in positions):
                raise ValueError("Absolute positions must be integers")
            if any(type(t) is not int or not 0 <= t < 262144 for t in tokens):
                raise ValueError("Observed token IDs out of range")
            if phase == "prompt":
                if self.index >= len(self.job["spans"]):
                    raise ValueError("Excess prompt forward")
                start, end = self.job["spans"][self.index]
                if positions != list(range(start, end)) or tokens != list(self.tokens[start:end]):
                    raise ValueError("Actual scheduler positions/token slice differ")
                self.cache.begin(start, end)
            elif phase == "decode":
                if self.handoff_record is None or self.decode_rows >= 255:
                    raise ValueError("Decode requires committed handoff and fixed 256-output budget")
                # The head on the final prompt row generates output #1; 255 new inputs follow.
                if self.bootstrap_record is None:
                    raise ValueError("Decode requires final-logit/uncached-anchor bootstrap")
                if self.decode_rows == 0 and runner.digest(tokens) != self.bootstrap_record["pending_anchor_sha256"]:
                    raise ValueError("First decode input differs from emitted uncached anchor")
                start = self.job["prompt_tokens"] + self.decode_rows
                end = start + 1
                if positions != [start] or len(tokens) != 1:
                    raise ValueError("Decode continuation must be one row at next absolute position")
            else:
                raise ValueError("Unsupported phase; verifier transactions need I06")
            self.active = {"phase": phase, "start": start, "end": end, "token_hash": runner.digest(tokens), "layers": set(), "routes": set(),
                           "begin_ns": self._now()}
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def layer_complete(self, layer, cache_observation=None):
        try:
            if self.failed or self.active is None:
                raise ValueError("Layer completion outside active forward")
            runner.integer(layer, 0, 29, "completed layer")
            if layer in self.active["layers"]:
                raise ValueError("Repeated layer completion")
            if self.active["phase"] == "prompt":
                if cache_observation is None or cache_observation.get("layer") != layer:
                    raise ValueError("Prompt completion requires matching native cache observation")
                self.cache.observe(cache_observation)
            elif cache_observation is not None:
                raise ValueError("Prompt KV ledger cannot ingest decode transaction")
            self.active["layers"].add(layer)
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def record_routes(self, layer, ids, weights, scheduled_m, correction_hits, suppressed_stock_rows):
        """Consume observed natural IDs, weights and provider-measured scheduled M.

        Instrumented expert/correctness lanes only. Never derive padded/scheduled
        execution from selected M; provider must report actual dispatch geometry.
        """
        try:
            if (self.failed or self.active is None or self.active["phase"] != "prompt"
                    or not self.job["observer_enabled"] or self.job["mode"] not in {"correctness", "profile"}
                    or self.job["mode"] == "profile" and self.job["region"] != "expert"):
                raise ValueError("Route capture forbidden in timing/non-expert lane")
            runner.integer(layer, 0, 29, "route layer")
            if layer in self.active["routes"]:
                raise ValueError("Repeated routed layer")
            m = self.active["end"] - self.active["start"]
            runner._list(ids, m, m, "route rows")
            runner._list(weights, m, m, "route weight rows")
            selected, positive = [0] * 128, [0] * 128
            for row, factors in zip(ids, weights):
                runner._list(row, 8, 8, "top8 IDs")
                runner._list(factors, 8, 8, "top8 weights")
                if len(set(row)) != 8:
                    raise ValueError("Distinct unchanged top8 IDs required")
                for expert, weight in zip(row, factors):
                    runner.integer(expert, 0, 127, "expert")
                    if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
                        raise ValueError("Finite nonnegative natural route weights required")
                    selected[expert] += 1
                    positive[expert] += weight > 0
            runner._list(scheduled_m, 128, 128, "actual scheduled M")
            for p, count in zip(positive, scheduled_m):
                runner.integer(count, p, 2**31 - 1, "scheduled M must cover positive routes")
            for count in (correction_hits, suppressed_stock_rows):
                runner.integer(count, 0, m * 8, "actual adapter work count")
            self.routes.append({"forward_index": self.index, "layer": layer, "selected_m": selected,
                                "positive_m": positive, "scheduled_m": list(scheduled_m),
                                "route_sha256": runner.digest({"ids": ids, "weights": weights}),
                                "correction_hits": correction_hits, "suppressed_stock_rows": suppressed_stock_rows})
            self.active["routes"].add(layer)
            _bounded(self.routes)
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def end_forward(self):
        if self.failed or self.active is None or self.active["layers"] != set(range(30)):
            self.abort()
            raise ValueError("Incomplete all-layer model forward")
        frame = self.active
        if frame["phase"] == "prompt":
            needs_routes = self.job["mode"] == "correctness" or self.job["mode"] == "profile" and self.job["region"] == "expert"
            if needs_routes and frame["routes"] != set(range(30)):
                self.abort()
                raise ValueError("Missing all-layer actual natural routing ledger")
            release = self.cache.end()
            self.forwards.append({"forward_index": self.index, "start": frame["start"], "end": frame["end"],
                                  "rows": frame["end"] - frame["start"], "output_rows": 0, "layer_count": 30,
                                  "token_ids_sha256": runner.digest(list(self.tokens[frame["start"]:frame["end"]])),
                                  "complete": True})
            self.index += 1
        else:
            release = None
            self.decode_rows += 1
            self.decode_input_hashes.append(frame["token_hash"])
        self.forward_times.append({"phase": frame["phase"], "start": frame["start"], "end": frame["end"],
                                   "host_begin_ns": frame["begin_ns"], "host_end_ns": self._now()})
        self.active = None
        return release

    @_poison_on_failure
    def handoff(self, value):
        try:
            if self.failed or self.active is not None or self.handoff_record is not None:
                raise ValueError("Nested, repeated or poisoned handoff")
            self.handoff_record = self.cache.handoff(value, self.job)
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def bootstrap(self, value):
        """Bind final prompt logits and the first emitted, still-uncached anchor.

        The independent comparison receipt authenticates private logits/tokens.
        Hashes alone do not establish arithmetic or sampling correctness. The
        verifier receives cached_length=P and pending anchor at absolute P;
        its first row predicts P+1. Output #1 already consumes the 256 budget.
        """
        try:
            if self.failed or self.handoff_record is None or self.bootstrap_record is not None:
                raise ValueError("Bootstrap requires unique complete prompt handoff")
            runner.keys(value, {"request_id", "logit_position", "predicts_position", "input_token_sha256",
                                "logit_rows", "vocab_size", "logit_bits_sha256", "source_site", "source_sha256",
                                "comparison_receipt_sha256", "sampling_policy", "emitted_tokens",
                                "pending_anchor_sha256", "pending_anchor_position", "anchor_kv_written",
                                "cached_length", "remaining_output_capacity"}, "final logit bootstrap")
            p = self.job["prompt_tokens"]
            for key, expected in {"request_id": self.job["request_id"], "logit_position": p - 1,
                                  "predicts_position": p, "input_token_sha256": runner.digest([self.tokens[-1]]),
                                  "logit_rows": 1, "vocab_size": 262144, "sampling_policy": "greedy_seed_1234",
                                  "emitted_tokens": 1, "pending_anchor_position": p, "anchor_kv_written": False,
                                  "cached_length": p, "remaining_output_capacity": 255, "source_site": LOGIT_SITE,
                                  "source_sha256": LOGIT_SITE_SHA256}.items():
                runner.equal(value[key], expected, "bootstrap." + key)
            for key in ("logit_bits_sha256", "source_sha256", "comparison_receipt_sha256", "pending_anchor_sha256"):
                runner.sha(value[key], key)
            runner._text(value["source_site"], "logit source site")
            self.bootstrap_record = copy.deepcopy(value)
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @contextmanager
    def stage(self, component, layer, source_site, source_sha256, shape, device_clock):
        """Synchronized event ranges, not kernel geometry or CUPTI attribution.

        EventClock records the actual current stream; all device event resolution
        is deferred to finish. Nested inclusive ranges must not be summed as a
        critical path. The pinned external profiler supplies kernel attribution.
        """
        try:
            if (self.failed or not self.job["observer_enabled"]
                    or self.job["mode"] != "profile"
                    or self.active is None and self.job["region"] != "head"
                    or self.active is not None and self.active["phase"] != "prompt"
                    or component not in runner.PROFILE_COMPONENTS[self.job["region"]]):
                self.abort()
                raise ValueError("Stage observation crossed profiling/request boundary")
            if len(self.events) >= 16384:
                self.abort()
                raise ValueError("Stage event budget exhausted; narrow the profiling scope")
            if type(shape) is not tuple or len(shape) != 3:
                raise ValueError("Observed M/N/K required")
            if self.job["region"] == "head":
                if self.index != len(self.job["spans"]) or self.active is not None or layer != 29 or shape[0] != 1:
                    self.abort()
                    raise ValueError("Head profile requires completed final prompt, last-row M=1")
            runner.integer(layer, 0, 29, "stage layer")
            runner._text(source_site, "source site")
            runner.sha(source_sha256, "observer source")
            for value in shape:
                runner.integer(value, 1, 2**31 - 1, "stage shape")
            if component in {"attention", "qkv", "o_projection", "shared_mlp"}:
                runner.equal(shape[0], self.active["end"] - self.active["start"], "stage actual M")
            span = {"component": component, "layer": layer, "forward_index": min(self.index, len(self.job["spans"]) - 1),
                    "source_site": source_site, "source_sha256": source_sha256,
                    "shape": list(shape), "host_begin_ns": self._now()}
            ticket = device_clock.begin()
            yield
            span["host_end_ns"] = self._now()
            span["ticket"] = device_clock.end(ticket)
            self.events.append(span)
        except BaseException:
            self.abort()
            raise

    @_poison_on_failure
    def finish_events(self, device_clock):
        if self.failed or self.active is not None or self.index != len(self.job["spans"]):
            raise ValueError("Cannot resolve events for incomplete/poisoned prompt")
        expected = {(len(self.job["spans"]) - 1, 29)} if self.job["region"] == "head" else {
            (i, layer) for i in range(len(self.job["spans"])) for layer in range(30)}
        if {(x["forward_index"], x["layer"]) for x in self.events} != expected:
            self.abort()
            raise ValueError("Incomplete actual callable event coverage")
        result = []
        device_clock.synchronize()
        for event in self.events:
            duration, stream = device_clock.resolve(event["ticket"])
            if type(duration) not in {float, int} or not math.isfinite(duration) or duration < 0:
                self.abort()
                raise ValueError("Finite nonnegative synchronized GPU event duration required")
            runner._text(stream, "runtime stream identity")
            result.append({**{k: v for k, v in event.items() if k != "ticket"},
                           "device_elapsed_ns": duration, "runtime_stream_key": stream,
                           "scope": "inclusive_event_range", "cupti_correlated": False})
        _bounded(result)
        return result

    def abort(self):
        self.failed = True
        self.active = None
        self.cache.abort()
        if self._writer is not None:
            self._writer.abort()


class CudaEventClock:
    """Dormant event implementation; constructing it requires an admitted slot.

    CUDA is imported only by the separately reviewed launcher provider, supplied
    here as an already imported module. This module never imports Torch/CUDA.
    """

    def __init__(self, torch_module):
        self.torch = torch_module
        self.streams = {}

    def begin(self):
        stream = self.torch.cuda.current_stream()
        key = str(stream.cuda_stream)
        self.streams[key] = stream
        begin = self.torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        return begin, stream, key

    def end(self, ticket):
        begin, stream, key = ticket
        current = self.torch.cuda.current_stream()
        if current.cuda_stream != stream.cuda_stream:
            raise ValueError("Stage changed streams; dependency-aware provider required")
        end = self.torch.cuda.Event(enable_timing=True)
        end.record(stream)
        return begin, end, key

    def synchronize(self):
        # Synchronize each observed stream before reading events, without mixing clocks.
        for stream in self.streams.values():
            stream.synchronize()

    def resolve(self, ticket):
        begin, end, key = ticket
        return begin.elapsed_time(end) * 10**6, key
