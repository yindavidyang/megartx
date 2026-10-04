"""Default-off native cache/handoff diagnostic, never a correctness oracle.

Hooks are purpose-specific and run on the ordinary owned scheduler. No cache
mutation, verifier lease, scheduler bypass, raw tensor publication or timings.
"""
from dataclasses import asdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

from .controlled_kv_capture import gather_writer_rows
from .loaded_engine_access import LoadedEngineAccess
from .prefill_diagnostic_plan import (Evidence, load_plan, digest, remaining, checkpoint_identity,
                                     verify_adapter_sources)
from .prefill_kv import PrefillKVObserver


class RequestLedger:
    """CPU accounting state; only native hooks supply its observations."""
    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        if len(tokens) != 2048:
            raise ValueError("Frozen 2048-row prompt required")
        self.end, self.frames, self.outputs = 0, 0, []
        self.pending, self.failed, self.request_id = None, False, None
        self.positions = {i: {} for i in range(30)}
        self.identities = None
        self.awaiting_sample = False

    def begin(self, tokens, positions, slots, identities, request_id):
        if self.failed or self.pending is not None or self.awaiting_sample or self.end >= 2303:
            raise ValueError("Nested, excessive or poisoned native frame")
        count = 256 if self.end < 2048 else 1
        if positions != list(range(self.end, self.end+count)):
            raise ValueError("Actual scheduler positions differ from chunk256/one-row continuation")
        expected = list(self.tokens[self.end:self.end+count]) if self.end < 2048 else self.outputs[-1:]
        if tokens != expected or (self.end >= 2048 and len(self.outputs) != self.end-2048+1):
            raise ValueError("Actual input differs from prompt or prior emitted uncached sample")
        if self.request_id is not None and self.request_id != request_id:
            raise ValueError("Actual engine request identity changed")
        if set(slots) != set(range(30)) or set(identities) != set(range(30)):
            raise ValueError("All thirty writer/cache owners required")
        if self.identities is not None and self.identities != identities:
            raise ValueError("Native cache allocation/view identity changed")
        for layer in range(30):
            values = slots[layer]
            if len(values) != count or len(set(values)) != count or set(values) & set(self.positions[layer].values()):
                raise ValueError("Writer overwrites a retained full-context row")
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Negative/noninteger writer slot")
        self.request_id, self.identities = request_id, identities
        self.pending = (self.end, self.end+count, slots)

    def complete(self):
        if self.failed or self.pending is None:
            raise ValueError("Native query completion lacks a prepared frame")
        start, end, slots = self.pending
        for layer in range(30):
            self.positions[layer].update(zip(range(start, end), slots[layer]))
        self.end, self.pending, self.frames = end, None, self.frames+1
        self.awaiting_sample = True

    def sampled(self, token, discarded):
        if self.failed or self.pending is not None or not self.awaiting_sample:
            raise ValueError("Sample lacks unique completed native frame")
        if type(token) is not int or not 0 <= token < 262144 or type(discarded) is not bool:
            raise ValueError("Actual bounded sampler output required")
        if discarded != (self.end < 2048):
            raise ValueError("Actual sampler discard mask differs from prompt completion")
        if not discarded:
            if len(self.outputs) >= 256:
                raise ValueError("Native output budget exceeded")
            self.outputs.append(token)
        self.awaiting_sample = False

    @property
    def complete_request(self):
        return (not self.failed and self.end == 2303 and self.frames == 263
                and len(self.outputs) == 256 and self.pending is None and not self.awaiting_sample)

    def abort(self):
        self.failed, self.pending, self.awaiting_sample = True, None, False


class GpuScratch:
    """Serialized observer phases; incumbent allocations are baseline resources.

    Managed tensor reservations count simultaneous lifetimes before allocation.
    The CUDA allocator peak also includes framework temporaries in each phase.
    Its counters are reset only around observer callbacks, never model work;
    this diagnostic consequently cannot supply a performance baseline.
    """
    def __init__(self, torch, limit):
        self.torch, self.limit = torch, limit
        self.active, self.live, self.managed_peak, self.cuda_peak, self.phases = False, 0, 0, 0, 0

    @contextmanager
    def scope(self, phase):
        if self.active:
            raise RuntimeError("Concurrent/nested observer GPU scratch phase")
        cuda = self.torch.cuda
        cuda.synchronize()
        baseline = cuda.memory_allocated()
        cuda.reset_peak_memory_stats()
        self.active, primary = True, None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            self.active = False
            try:
                cuda.synchronize()
                peak = max(0, cuda.max_memory_allocated()-baseline)
                self.cuda_peak, self.phases = max(self.cuda_peak, peak), self.phases+1
                if self.live or peak > self.limit:
                    raise RuntimeError("Aggregate observer GPU scratch cap breached: " + phase)
            except BaseException as error:
                if primary is None:
                    raise
                if hasattr(primary, 'add_note'):
                    primary.add_note("Observer GPU scratch measurement/cleanup failed: " + repr(error))

    @contextmanager
    def allocation(self, size):
        if not self.active or type(size) is not int or size < 0 or self.live+size > self.limit:
            raise RuntimeError("Aggregate observer GPU scratch rejected before allocation")
        self.live += size
        self.managed_peak = max(self.managed_peak, self.live)
        try:
            yield
        finally:
            self.live -= size

    def receipt(self):
        return {"domain": "incremental_gpu_allocator_bytes", "cap_bytes": self.limit,
                "managed_tensor_simultaneous_peak_bytes": self.managed_peak,
                "measured_phase_allocator_increment_peak_bytes": self.cuda_peak,
                "measured_phases": self.phases, "allocator_peak_counters_reset_per_observer_phase": True,
                "incumbent_model_cache_and_hidden_allocations_are_baseline": True,
                "host_heap_excluded": True}


def retained_host_bytes(value, seen=None):
    """Actual Python object sizes, separately charged to host RAM headroom."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(retained_host_bytes(k, seen)+retained_host_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(retained_host_bytes(v, seen) for v in value)
    return size


def memory_sample(torch, phase, scratch, ledger):
    free, total = torch.cuda.mem_get_info()
    host_free = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines()
                         if row.startswith('MemAvailable:')))*1024
    if free < 2 << 30 or host_free < 8 << 30:
        raise RuntimeError("Native memory reserve breached")
    return {"schema": "megartx-prefill-native-memory-v2", "phase": phase,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(), "device_used_bytes": total-free,
            "device_free_bytes": free, "host_available_bytes": host_free,
            "observer_gpu_scratch": scratch.receipt(),
            "retained_position_ledger_host_bytes": retained_host_bytes(ledger.positions),
            "host_heap_budget": "host_available_reserve_separate_from_gpu_scratch",
            "startup_reference_fp64_projection_floor_bytes": 15859712,
            "startup_reference_separately_accounted": True}


class NativeProvider(PrefillKVObserver):
    """Reuse exact native dtype/owner checks, with a separate observation ledger.

    No comparison receipt is invented. Full-context storage retains all written
    positions; per-frame K/V hashes cover actual new rows and retained boundaries.
    Those observations cannot establish native numerical equivalence.
    """
    def __init__(self, runner, plan, directory, torch):
        self.torch, self.plan = torch, plan
        self.adapter_sources = verify_adapter_sources(plan)
        self.access = LoadedEngineAccess.from_runner(runner, torch)
        if checkpoint_identity(os.environ['MEGARTX_CHECKPOINT_PATH']) != plan['checkpoint_identity']:
            raise RuntimeError('Loaded checkpoint config/index/shard stat identity changed')
        self.bind_owners(self.access.model)
        if set(self.access.groups) != {entry[0] for entry in self.layers.values()}:
            raise RuntimeError("All thirty actual cache-group owners required")
        self.directory, self.evidence = Path(directory), Evidence(directory)
        self.ledger = RequestLedger(plan['tokens'])
        self.scratch = GpuScratch(torch, plan['bounds']['max_observer_gpu_scratch_bytes'])
        self.frame, self.hidden, self.logits_indices = None, None, None
        self.logit_seen = False
        self.failed, self.started, self.completed = False, False, False
        self.deadline = float(os.environ['MEGARTX_PREFILL_NATIVE_DEADLINE'])
        remaining(self.deadline)
        self.evidence.write('loaded.json', {"schema": "megartx-loaded-model-observation-v1",
                            "identity": asdict(self.access.identity), "plan_sha256": plan['plan_sha256'],
                            "executing_adapter_sources": self.adapter_sources,
                            "mutable_lease_granted": False,
                            "memory": memory_sample(torch, 'cache_initialized', self.scratch, self.ledger)})

    def active(self):
        remaining(self.deadline)
        if self.failed:
            raise RuntimeError("Poisoned native provider cannot be reused")
        marker = self.directory / 'request.json'
        if not marker.exists():
            if self.started and not self.completed:
                raise RuntimeError("Native request marker disappeared before completion")
            return False
        if self.failed or self.completed or marker.is_symlink() or marker.stat().st_size > 65536:
            raise RuntimeError("Stale/poisoned/repeated native request marker")
        value = json.loads(marker.read_text())
        if value != {'schema': 'megartx-prefill-native-request-v1', 'plan_sha256': self.plan['plan_sha256']}:
            raise RuntimeError("Native request marker differs from frozen plan")
        if verify_adapter_sources(self.plan) != self.adapter_sources:
            raise RuntimeError("Executing native adapter identity changed before frame admission")
        self.started = True
        return True

    def prepare_inputs(self, result):
        if self.failed:
            raise RuntimeError("Poisoned native provider cannot prepare another frame")
        self.access.common.clear()
        self.logits_indices = result[0]

    def begin(self, model, tokens, positions):
        if not self.active():
            return None
        if model is not self.access.model or self.frame is not None:
            raise RuntimeError("Unexpected/nested native model owner")
        slots, capacities, identities, bindings = self.read_frame(positions, tokens)
        if any(c < 2304 for c in capacities.values()):
            raise RuntimeError("Actual full-context KV omits P+256 output reserve")
        from vllm.forward_context import get_forward_context
        frame = self.access.bind_frame(self, self.ledger.frames, get_forward_context(),
                                       tokens, positions, slots, identities, self.ledger.positions)
        self.ledger.begin(tokens.tolist(), positions.tolist(), slots, identities, frame.request_id)
        indices = self.logits_indices
        if (indices is None or tuple(indices.shape) != (1,)
                or indices.cpu().tolist() != [len(positions)-1]):
            raise RuntimeError("Actual runner logits selection is not the last input row")
        self.boundaries = {}
        if self.ledger.end:
            for layer in range(30):
                low = 0 if layer % 6 == 5 else max(0, self.ledger.end-1023)
                self.boundaries[layer] = self.hash_rows(layer, self.ledger.positions[layer], [low, self.ledger.end-1])
        self.frame, self.logit_seen = frame, False
        if frame.sequence == 0:
            self.evidence.write('geometry.json', {'layers': bindings,
                'physical_policy': 'full_context', 'capacity_tokens': 2304,
                'layer_allocation_placements_verified': True,
                'actual_owned_page_ranges_disjoint': True,
                'compute_windows_verified': True,
                'layout_sha256': digest(bindings)})
        return frame

    def hash_rows(self, layer, mapping, positions):
        key, value = hashlib.sha256(), hashlib.sha256()
        cache, dim = self.layers[layer][2].kv_cache, self.descriptors[layer]['head_dim']
        # Bounded two-row copies. No full-cache clone or broadcast matching.
        positions = sorted(set(positions))
        for offset in range(0, len(positions), 2):
            selected = positions[offset:offset+2]
            # Only stack(rows) owns a GPU tensor here. NumPy/bytes copies live
            # on the host. Reservation covers the complete GPU tensor lifetime;
            # all layers/batches share one aggregate phase allocation ledger.
            with self.scratch.allocation(len(selected)*8*2*dim*2):
                k, v = gather_writer_rows(cache, [mapping[p] for p in selected], dim, self.torch)
            for index, absolute in enumerate(selected):
                tag = struct.pack('<Q', absolute)
                key.update(tag); key.update(k[index].tobytes())
                value.update(tag); value.update(v[index].tobytes())
        return {'processed_k_sha256': key.hexdigest(), 'processed_v_sha256': value.hexdigest()}

    def finish(self, ticket, result):
        if ticket is not self.frame or ticket.owner is not self or ticket.sequence != self.ledger.frames:
            raise RuntimeError("Stale/reused native OwnedFrame")
        self.torch.cuda.synchronize()
        slots, _, identities, _ = self.read_frame(ticket.positions, ticket.input_ids)
        from vllm.forward_context import get_forward_context
        if get_forward_context() is not ticket.context:
            raise RuntimeError("Native query completion escaped its actual ForwardContext")
        current = self.access.bind_frame(self, ticket.sequence, ticket.context, ticket.input_ids,
                                         ticket.positions, slots, identities, self.ledger.positions)
        if identities != ticket.identities or slots != ticket.slots or current.block_tables != ticket.block_tables:
            raise RuntimeError("Actual native cache/slots/owned blocks changed during queries")
        start, end, _ = self.ledger.pending
        records = []
        for layer in range(30):
            if layer in self.boundaries:
                low = 0 if layer % 6 == 5 else max(0, start-1023)
                actual = self.hash_rows(layer, self.ledger.positions[layer], [low, start-1])
                if actual != self.boundaries[layer]:
                    raise RuntimeError("Retained native K/V boundary changed during chunk queries")
            mapping = {**self.ledger.positions[layer], **dict(zip(range(start, end), slots[layer]))}
            records.append({'layer': layer, 'written_start': start, 'written_end': end,
                'required_start': 0 if layer % 6 == 5 else max(0, start-1023),
                'required_end': end, **self.hash_rows(layer, mapping, list(range(start, end)))})
        if (not isinstance(result, self.torch.Tensor) or tuple(result.shape) != (end-start, 2816)
                or result.dtype != self.torch.bfloat16):
            raise RuntimeError("Actual final hidden rows changed")
        self.hidden = result  # Borrow incumbent output; no candidate clone.
        self.ledger.complete()
        self.evidence.write('frames.jsonl', {'sequence': ticket.sequence, 'request_id': ticket.request_id,
            'input_ids_sha256': digest(ticket.input_ids.tolist()), 'start': start, 'end': end,
            'queries_complete': True, 'layer_rows': records,
            'coverage': 'new_written_rows_and_retained_boundaries',
            'memory': memory_sample(self.torch, 'chunk' if start < 2048 else 'decode',
                                    self.scratch, self.ledger)}, append=True)
        self.frame = None

    def head(self, model, hidden, logits):
        if not self.started or self.completed:
            return
        if self.failed or model is not self.access.model or self.frame is not None or self.hidden is None or self.logit_seen:
            raise RuntimeError("Native head lacks a unique completed forward")
        torch = self.torch
        if (tuple(hidden.shape) != (1, 2816) or logits is None or tuple(logits.shape) != (1, 262144)
                or logits.dtype != torch.float32):
            raise RuntimeError("Actual head shape differs")
        # Indices were captured from actual _prepare_inputs, not inferred from
        # token equality. Check selected raw hidden bytes without whole-view copies.
        indices = self.logits_indices.cpu().tolist()
        if indices != [len(self.hidden)-1]:
            raise RuntimeError("Actual runner logits index changed after queries")
        expected = self.hidden[indices[0]:indices[0]+1]  # Basic slice borrows the incumbent row.
        raw = lambda tensor: tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        if raw(hidden) != raw(expected):
            raise RuntimeError("Actual compute_logits row differs from runner selection")
        bits = raw(logits)  # One CPU row; no device isfinite/broadcast/index allocation.
        if any(word & 0x7F800000 == 0x7F800000 for word, in struct.iter_unpack('<I', bits)):
            raise RuntimeError("Nonfinite actual native logits")
        self.evidence.write('heads.jsonl', {'sequence': self.ledger.frames-1,
            'logit_position': self.ledger.end-1, 'predicts_position': self.ledger.end,
            'row_identity': 'actual_runner_logits_indices_and_hidden_bits',
            'logit_bits_sha256': hashlib.sha256(bits).hexdigest(),
            'source_site': 'Gemma4ForConditionalGeneration.compute_logits'}, append=True)
        self.logit_seen, self.hidden = True, None

    def sampled(self, result):
        if not self.started or self.completed:
            return
        if self.failed or not self.logit_seen or tuple(result.sampled_token_ids.shape) != (1, 1):
            raise RuntimeError("Native sample lacks actual unique logit row")
        runner = self.access._runner
        token = int(result.sampled_token_ids.item())
        discarded = bool(runner.discard_request_mask.np[0])
        self.ledger.sampled(token, discarded)
        if not discarded:
            self.evidence.write('samples.jsonl', {'output_index': len(self.ledger.outputs)-1,
                'sample_sha256': digest([token]), 'cached_length': self.ledger.end,
                'pending_anchor_position': self.ledger.end, 'anchor_kv_written': False}, append=True)
        if self.ledger.complete_request:
            self.evidence.write('observer.json', {'schema': 'megartx-prefill-native-observation-v1',
                'status': 'request_observed', 'plan_sha256': self.plan['plan_sha256'],
                'engine_request_id': self.ledger.request_id, 'prompt_frames': 8,
                'decode_input_rows': 255, 'emitted_outputs': 256,
                'output_ids_sha256': digest(self.ledger.outputs), 'committed_length': 2303,
                'bootstrap_cached_length': 2048, 'bootstrap_anchor_position': 2048,
                'bootstrap_remaining_outputs': 255,
                'sample_and_input_chain_verified': True, 'all30_actual_cache_writers_bound': True,
                'observed_fit': True, 'independent_comparison_pending': True,
                'observer_gpu_scratch': self.scratch.receipt(),
                'retained_position_ledger_host_bytes': retained_host_bytes(self.ledger.positions),
                'numerical_qualified': False, 'performance_qualified': False})
            self.completed = True

    def abort(self):
        self.failed, self.frame, self.hidden, self.logits_indices = True, None, None, None
        self.access.common.clear()
        self.ledger.abort()


def install_native_observer(torch, model_cls):
    """Install only after historical adapter registration; no general hooks."""
    plan_path = os.environ.get('MEGARTX_PREFILL_NATIVE_PLAN')
    directory = os.environ.get('MEGARTX_PREFILL_NATIVE_DIR')
    if not plan_path and not directory:
        return
    if (not plan_path or not directory or os.environ.get('MEGARTX_SCALE_MODE') != 'native'
            or any(os.environ.get(k) for k in ('MEGARTX_M1_PREPARATION', 'MEGARTX_CONTROLLED_DIR',
                   'MEGARTX_LOGITS_DIR', 'MEGARTX_NATIVE_DIAGNOSTIC'))):
        raise RuntimeError("Native prefill requires its exclusive observer-only opt-in")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    plan = load_plan(plan_path, os.environ.get("MEGARTX_PREFILL_NATIVE_SOURCE_ROOT"))
    verify_adapter_sources(plan)  # Before installing any new observation hooks.
    providers = {}
    old_init, old_prepare, old_sample = (GPUModelRunner.initialize_kv_cache,
                                        GPUModelRunner._prepare_inputs, GPUModelRunner._sample)
    old_build = FlashInferMetadataBuilder.build
    old_forward, old_head = model_cls.forward, model_cls.compute_logits

    def poison(provider, primary):
        if provider is None:
            return
        provider.failed = True
        try:
            provider.abort()
        except BaseException as error:
            if hasattr(primary, 'add_note'):
                primary.add_note("Native provider abort failed: " + repr(error))

    def require_live(provider):
        if provider is not None and provider.failed:
            raise RuntimeError("Poisoned native provider cannot be reused")

    def observed(provider, name, *args):
        require_live(provider)
        with provider.scratch.scope(name):
            return getattr(provider, name)(*args)

    @wraps(old_init)
    def initialized(runner, *args, **kwargs):
        provider = providers.get(id(runner))
        try:
            require_live(provider)
            if provider is not None:
                raise RuntimeError("Repeated native runner cache initialization")
            result = old_init(runner, *args, **kwargs)
            providers[id(runner)] = NativeProvider(runner, plan, directory, torch)
            return result
        except BaseException as error:
            poison(provider, error)
            raise

    @wraps(old_prepare)
    def prepared(runner, *args, **kwargs):
        provider = providers.get(id(runner))
        try:
            require_live(provider)
            result = old_prepare(runner, *args, **kwargs)
            if provider is not None:
                provider.prepare_inputs(result)
            return result
        except BaseException as error:
            poison(provider, error)
            raise

    @wraps(old_build)
    def built(builder, *args, **kwargs):
        owners = [p for p in providers.values() if id(builder) in p.access.builders]
        try:
            for provider in owners:
                require_live(provider)
            result = old_build(builder, *args, **kwargs)
            for provider in owners:
                common = kwargs.get('common_attn_metadata', args[1] if len(args) > 1 else None)
                if common is None:
                    raise RuntimeError("Actual builder common metadata missing")
                provider.access.record_metadata(builder, common, result)
            return result
        except BaseException as error:
            for provider in owners:
                poison(provider, error)
            raise

    def for_model(model):
        matches = [p for p in providers.values() if p.access.model is model]
        if len(matches) > 1:
            raise RuntimeError("Ambiguous loaded native model owner")
        return matches[0] if matches else None

    @wraps(old_forward)
    def forward(model, input_ids, positions, *args, **kwargs):
        provider, ticket = for_model(model), None
        try:
            if provider is not None:
                ticket = observed(provider, 'begin', model, input_ids, positions)
            result = old_forward(model, input_ids, positions, *args, **kwargs)
            if ticket is not None:
                observed(provider, 'finish', ticket, result)
            return result
        except BaseException as error:
            poison(provider, error)
            raise

    @wraps(old_head)
    def head(model, hidden, *args, **kwargs):
        provider = for_model(model)
        try:
            require_live(provider)
            result = old_head(model, hidden, *args, **kwargs)
            if provider is not None:
                observed(provider, 'head', model, hidden, result)
            return result
        except BaseException as error:
            poison(provider, error)
            raise

    @wraps(old_sample)
    def sampled(runner, *args, **kwargs):
        provider = providers.get(id(runner))
        try:
            require_live(provider)
            result = old_sample(runner, *args, **kwargs)
            if provider is not None:
                provider.sampled(result)
            return result
        except BaseException as error:
            poison(provider, error)
            raise

    GPUModelRunner.initialize_kv_cache = initialized
    GPUModelRunner._prepare_inputs = prepared
    FlashInferMetadataBuilder.build = built
    GPUModelRunner._sample = sampled
    model_cls.forward, model_cls.compute_logits = forward, head
