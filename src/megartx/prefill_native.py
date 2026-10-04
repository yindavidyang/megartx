"""Default-off native cache/handoff diagnostic, never a correctness oracle.

Hooks are purpose-specific and run on the ordinary owned scheduler. No cache
mutation, verifier lease, scheduler bypass, raw tensor publication or timings.
"""
from dataclasses import asdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from .controlled_kv_capture import gather_writer_rows
from .loaded_engine_access import LoadedEngineAccess
from .prefill_diagnostic_plan import Evidence, load_plan, digest, remaining, checkpoint_identity
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


def memory_sample(torch, phase):
    free, total = torch.cuda.mem_get_info()
    host_free = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines()
                         if row.startswith('MemAvailable:')))*1024
    if free < 2 << 30 or host_free < 8 << 30:
        raise RuntimeError("Native memory reserve breached")
    return {"phase": phase, "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(), "device_used_bytes": total-free,
            "device_free_bytes": free, "host_available_bytes": host_free,
            "observer_scratch_upper_bound_bytes": 4 << 20,
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
        self.access = LoadedEngineAccess.from_runner(runner, torch)
        if checkpoint_identity(os.environ['MEGARTX_CHECKPOINT_PATH']) != plan['checkpoint_identity']:
            raise RuntimeError('Loaded checkpoint config/index/shard stat identity changed')
        self.bind_owners(self.access.model)
        if set(self.access.groups) != {entry[0] for entry in self.layers.values()}:
            raise RuntimeError("All thirty actual cache-group owners required")
        self.directory, self.evidence = Path(directory), Evidence(directory)
        self.ledger = RequestLedger(plan['tokens'])
        self.frame, self.hidden, self.logits_indices = None, None, None
        self.logit_seen = False
        self.failed, self.started, self.completed = False, False, False
        self.deadline = float(os.environ['MEGARTX_PREFILL_NATIVE_DEADLINE'])
        remaining(self.deadline)
        self.evidence.write('loaded.json', {"schema": "megartx-loaded-model-observation-v1",
                            "identity": asdict(self.access.identity), "plan_sha256": plan['plan_sha256'],
                            "mutable_lease_granted": False,
                            "memory": memory_sample(torch, 'cache_initialized')})

    def active(self):
        remaining(self.deadline)
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
        self.started = True
        return True

    def prepare_inputs(self, result):
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
                                       tokens, positions, slots, identities)
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
        if 8*dim*2*2*8 > self.plan['bounds']['max_observer_scratch_bytes']:
            raise RuntimeError("Observer scratch bound rejected before row copy")
        positions = sorted(set(positions))
        for offset in range(0, len(positions), 2):
            selected = positions[offset:offset+2]
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
                                         ticket.positions, slots, identities)
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
            'memory': memory_sample(self.torch, 'chunk' if start < 2048 else 'decode')}, append=True)
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
        expected = self.hidden[self.logits_indices]
        if not torch.equal(hidden.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
            raise RuntimeError("Actual compute_logits row differs from runner selection")
        if logits.numel()*logits.element_size()*3 > self.plan['bounds']['max_observer_scratch_bytes']:
            raise RuntimeError("Observer scratch bound rejected before logits copy")
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("Nonfinite actual native logits")
        bits = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
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
                'numerical_qualified': False, 'performance_qualified': False})
            self.completed = True

    def abort(self):
        self.failed, self.frame, self.hidden = True, None, None
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
    providers = {}
    old_init, old_prepare, old_sample = (GPUModelRunner.initialize_kv_cache,
                                        GPUModelRunner._prepare_inputs, GPUModelRunner._sample)
    old_build = FlashInferMetadataBuilder.build
    old_forward, old_head = model_cls.forward, model_cls.compute_logits

    @wraps(old_init)
    def initialized(runner, *args, **kwargs):
        result = old_init(runner, *args, **kwargs)
        if id(runner) in providers:
            raise RuntimeError("Repeated native runner cache initialization")
        providers[id(runner)] = NativeProvider(runner, plan, directory, torch)
        return result

    @wraps(old_prepare)
    def prepared(runner, *args, **kwargs):
        result = old_prepare(runner, *args, **kwargs)
        provider = providers.get(id(runner))
        if provider is not None:
            provider.prepare_inputs(result)
        return result

    @wraps(old_build)
    def built(builder, *args, **kwargs):
        result = old_build(builder, *args, **kwargs)
        for provider in providers.values():
            if id(builder) in provider.access.builders:
                common = kwargs.get('common_attn_metadata', args[1] if len(args) > 1 else None)
                if common is None:
                    raise RuntimeError("Actual builder common metadata missing")
                provider.access.record_metadata(builder, common, result)
        return result

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
                ticket = provider.begin(model, input_ids, positions)
            result = old_forward(model, input_ids, positions, *args, **kwargs)
            if ticket is not None:
                provider.finish(ticket, result)
            return result
        except BaseException:
            if provider is not None:
                provider.abort()
            raise

    @wraps(old_head)
    def head(model, hidden, *args, **kwargs):
        provider = for_model(model)
        try:
            result = old_head(model, hidden, *args, **kwargs)
            if provider is not None:
                provider.head(model, hidden, result)
            return result
        except BaseException:
            if provider is not None:
                provider.abort()
            raise

    @wraps(old_sample)
    def sampled(runner, *args, **kwargs):
        provider = providers.get(id(runner))
        try:
            result = old_sample(runner, *args, **kwargs)
            if provider is not None:
                provider.sampled(result)
            return result
        except BaseException:
            if provider is not None:
                provider.abort()
            raise

    GPUModelRunner.initialize_kv_cache = initialized
    GPUModelRunner._prepare_inputs = prepared
    FlashInferMetadataBuilder.build = built
    GPUModelRunner._sample = sampled
    model_cls.forward, model_cls.compute_logits = forward, head
