"""Dormant native BHNC cache observer with bounded two-row hashing.

Reuses the pinned source/owner/layout checks, not the controlled 33-token plan.
No cache allocation/mutation or GPU imports at module import. Native observation
is deliberately perturbing and forbidden in timing jobs.
"""

import hashlib
import inspect
from pathlib import Path
import struct

from . import prefill_runner as runner
from .prefill_collect import required_kv, _poison_on_failure
from .controlled_kv_capture import (ControlledKV, gather_writer_rows, SOURCE_HASHES,
                                    _digest, validate_context, validate_metadata)


class WriterLedger:
    """Actual absolute-position/slot association before writes are dispatched."""

    def __init__(self):
        self.positions = {i: {} for i in range(30)}
        self.end, self.pending, self.poisoned = 0, None, False

    @_poison_on_failure
    def prepare(self, start, end, slots, capacities):
        try:
            if self.poisoned or self.pending is not None or start != self.end:
                raise ValueError("Nested/stale/poisoned writer frame")
            runner.integer(start, 0, 8191, "writer start")
            runner.integer(end, start + 1, 8192, "writer end")
            runner.keys(slots, set(range(30)), "all-layer writer slots")
            runner.keys(capacities, set(range(30)), "all-layer cache capacities")
            if any(type(k) is not int for k in slots) or any(type(k) is not int for k in capacities):
                raise ValueError("Writer layer identities must be exact integers")
            pending = {}
            for layer in range(30):
                low, _ = required_kv(start, end, layer)
                values, capacity = slots[layer], capacities[layer]
                runner.integer(capacity, end - low, 2**31 - 1, "physical union capacity")
                runner._list(values, end - start, end - start, "actual writer rows")
                for value in values: runner.integer(value, 0, capacity - 1, "physical writer slot")
                if len(set(values)) != len(values):
                    raise ValueError("Aliased current writer slots")
                live = {p: slot for p, slot in self.positions[layer].items() if p >= low}
                if set(values) & set(live.values()):
                    raise ValueError("Writer overwrites K/V still needed by chunk queries")
                if set(live) != set(range(low, start)):
                    raise ValueError("Missing retained prior K/V positions")
                pending[layer] = {**live, **dict(zip(range(start, end), values))}
            self.pending = {"start": start, "end": end, "positions": pending}
        except (ValueError, TypeError, KeyError):
            self.abort()
            raise

    @_poison_on_failure
    def complete(self):
        if self.poisoned or self.pending is None:
            raise ValueError("Incomplete/poisoned writer state")
        end = self.pending["end"]
        for layer, mapping in self.pending["positions"].items():
            low = 0 if layer in runner.GLOBAL_LAYERS else max(0, end - 1024)
            self.positions[layer] = {p: slot for p, slot in mapping.items() if p >= low}
        self.end, self.pending = end, None

    def abort(self):
        self.poisoned, self.pending = True, None


class PrefillKVObserver(ControlledKV):
    """Concrete installed-source cache extraction for the future owned provider.

    prepare() runs inside the exact ForwardContext before model work; finish()
    runs after query streams complete and before recycling. The bound runtime's
    BF16 BHNC cache is hashed two rows at a time, never copied wholesale. Native
    comparison receipts and equality tolerances are external reviewed evidence.
    """

    def __init__(self, model, collector, torch_module, physical_policy="full_context"):
        if not collector.job["observer_enabled"] or collector.job["mode"] not in {"correctness", "profile", "memory"}:
            raise ValueError("Native K/V observation forbidden in timing")
        if physical_policy != "full_context":
            raise ValueError("Native observer initially requires full-context physical capacity; rings need review")
        self.collector, self.torch = collector, torch_module
        self.bind_owners(model)  # Exact original Gemma/Attention/FlashInfer writer classes and file hashes.
        self.writer, self.storage_identities = WriterLedger(), None
        collector.attach_writer(self.writer)
        self.native_bindings = None

    @_poison_on_failure
    def read_frame(self, positions, tokens):
        """Prefill-local dtype repair; historical controlled helper stays byte-unchanged.

        All source/context/metadata/writer/cache checks below retain the original
        helper semantics. Real tensors and exact dtype objects are required;
        string aliases are not a substitute for Torch I64 tensor identity.
        """
        torch = self.torch
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        path = Path(inspect.getsourcefile(get_forward_context))
        if _digest(path) != SOURCE_HASHES["vllm.forward_context"]:
            raise RuntimeError("K/V ForwardContext source changed")
        validate_context(context)
        if (not isinstance(positions, torch.Tensor) or not isinstance(tokens, torch.Tensor)
                or positions.dtype != torch.int64 or tokens.dtype != torch.int64
                or positions.ndim != 1 or tokens.ndim != 1):
            raise RuntimeError("K/V actual position/token context must retain I64 row identity")
        positions, tokens = [int(p) for p in positions.tolist()], [int(t) for t in tokens.tolist()]
        slots, capacities, identities, bindings = {}, {}, {}, []
        for ordinal, descriptor in enumerate(self.descriptors):
            name, parent, attn, impl = self.layers[ordinal]
            if parent.attn is not attn or attn.impl is not impl or context.no_compile_layers.get(name) is not attn or name not in context.slot_mapping or name not in context.attn_metadata:
                raise RuntimeError("K/V actual registry/module/backend owner changed")
            meta, mapping, cache = context.attn_metadata[name], context.slot_mapping[name], attn.kv_cache
            validate_metadata(meta, len(positions))
            if not isinstance(mapping, torch.Tensor) or mapping.dtype != torch.int64 or tuple(mapping.shape) != (len(positions),) or not torch.equal(mapping, meta.slot_mapping):
                raise RuntimeError("K/V metadata differs from the actual writer slot map")
            if context.is_padding is not None and (context.is_padding.shape != mapping.shape or bool(context.is_padding.any().item())):
                raise RuntimeError("K/V controlled forward has padded rows")
            heads, dim = descriptor["kv_heads"], descriptor["head_dim"]
            if attn.num_kv_heads != heads or attn.head_size != dim or attn.head_size_v != dim or attn.sliding_window != descriptor["window_size"] or impl.num_kv_heads != heads or impl.head_size != dim or impl.window_left != (1023 if descriptor["window_size"] else -1):
                raise RuntimeError("K/V actual head/window geometry differs from the frozen config")
            if attn.kv_cache_dtype not in {"auto", "bfloat16"} or impl.cache_dtype != attn.kv_cache_dtype or impl.is_kvcache_nvfp4 or cache.dtype != torch.bfloat16 or cache.device.type != "cuda":
                raise RuntimeError("K/V capture requires unchanged BF16 CUDA storage")
            shape, strides = tuple(cache.shape), tuple(cache.stride())
            if len(shape) != 4 or shape[0] < 1 or shape[1] != heads or shape[2] < 1 or shape[3] != 2 * dim or any(s < 1 for s in strides) or strides[-1] != 1:
                raise RuntimeError("K/V view is not the pinned logical BHNC representation")
            storage, offset = cache.untyped_storage(), cache.storage_offset()
            extent = offset + sum((n - 1) * s for n, s in zip(shape, strides)) + 1
            if offset < 0 or extent * 2 > storage.nbytes():
                raise RuntimeError("K/V view exceeds its actual allocation")
            identities[ordinal] = (id(cache), storage.data_ptr(), storage.nbytes(), offset, shape, strides, str(cache.device))
            slots[ordinal] = [int(s) for s in mapping.detach().cpu().tolist()]
            capacities[ordinal] = shape[0] * shape[2]
            bindings.append({"layer": ordinal, "layer_name": name, "cache_view_shape": list(shape), "cache_view_strides": list(strides),
                             "resolved_layout": impl.kv_cache_layout.name, "writer": "FlashInferImpl.do_kv_cache_update",
                             "registry_owner_verified": True, "metadata_writer_slots_equal": True, "dtype": "bf16"})
        return slots, capacities, identities, bindings

    def prepare(self, positions, input_ids):
        try:
            if self.collector.active is None or self.collector.active["phase"] != "prompt":
                raise ValueError("Native writer preparation outside exact prompt forward")
            # Pinned helper validates eager single-sequence metadata, all writer
            # slots, registry owner, BF16 BHNC strides/extent and local/global geometry.
            slots, capacities, identities, bindings = self.read_frame(positions, input_ids)
            first, last = self.collector.active["start"], self.collector.active["end"]
            if positions.tolist() != list(range(first, last)) or input_ids.tolist() != list(self.collector.tokens[first:last]):
                raise ValueError("Native observed token/position tensors differ from exact prompt")
            if any(capacity < self.collector.job["capacity_tokens"] for capacity in capacities.values()):
                raise ValueError("Physical full-context KV omits P+256 output reserve")
            if self.storage_identities is not None and identities != self.storage_identities:
                raise ValueError("Native cache allocation/view identity changed")
            # Reject overlapping layer views in one allocation; unfamiliar strided
            # interleaving is unsupported rather than assumed safe.
            regions = []
            for layer, identity in identities.items():
                _, pointer, _, offset, shape, strides, _ = identity
                upper = offset + sum((n-1)*s for n,s in zip(shape, strides)) + 1
                if any(pointer == p and offset < high and low < upper for p,low,high in regions):
                    raise ValueError("Native cross-layer cache region alias")
                regions.append((pointer, offset, upper))
            start, end = self.collector.active["start"], self.collector.active["end"]
            self.writer.prepare(start, end, slots, capacities)
            self.storage_identities, self.native_bindings = identities, bindings
            self.prepared_tensors = (positions, input_ids)
            self.prepared_slots = slots
        except BaseException:
            self.abort()
            raise

    def _hash_positions(self, layer, mapping, low, end):
        key, value = hashlib.sha256(), hashlib.sha256()
        cache, dim = self.layers[layer][2].kv_cache, self.descriptors[layer]["head_dim"]
        for position in range(low, end, 2):
            selected = list(range(position, min(end, position + 2)))
            k, v = gather_writer_rows(cache, [mapping[p] for p in selected], dim, self.torch)
            for index, absolute in enumerate(selected):
                tag = struct.pack("<Q", absolute)
                key.update(tag); key.update(k[index].tobytes())
                value.update(tag); value.update(v[index].tobytes())
        return key.hexdigest(), value.hexdigest()

    def finish(self):
        try:
            pending = self.writer.pending
            if pending is None or self.collector.active is None:
                raise ValueError("Native cache completion lacks prepared writer frame")
            # This observer lane intentionally synchronizes; it makes no latency claim.
            self.torch.cuda.synchronize()
            slots, _, identities, _ = self.read_frame(*self.prepared_tensors)
            if identities != self.storage_identities or slots != self.prepared_slots:
                raise ValueError("Native writer/cache registry or physical slots changed during queries")
            for layer in range(30):
                low, end = required_kv(pending["start"], pending["end"], layer)
                mapping = pending["positions"][layer]
                k, v = self._hash_positions(layer, mapping, low, end)
                self.collector.layer_complete(layer, {"layer": layer,
                    "kind": "global" if layer in runner.GLOBAL_LAYERS else "local", "kv_dtype": "bfloat16",
                    "required_start": low, "required_end": end, "retained_start": min(mapping), "retained_end": end,
                    "k_storage_id": f"layer-{layer}-k-region", "v_storage_id": f"layer-{layer}-v-region",
                    "processed_k_sha256": k, "processed_v_sha256": v,
                    "layout_sha256": self.collector.cache.layout, "queries_complete": True})
            self.writer.complete()
            self.prepared_tensors = self.prepared_slots = None
        except BaseException:
            self.abort()
            raise

    @_poison_on_failure
    def final_handoff(self, comparison_receipt_sha256):
        if self.writer.poisoned or self.writer.pending is not None or self.writer.end != self.collector.job["prompt_tokens"]:
            raise ValueError("Native cache is not a complete committed prompt")
        layers, p = [], self.writer.end
        for layer in range(30):
            low = 0 if layer in runner.GLOBAL_LAYERS else max(0, p - 1024)
            k, v = self._hash_positions(layer, self.writer.positions[layer], low, p)
            layers.append({"layer": layer, "kind": "global" if layer in runner.GLOBAL_LAYERS else "local",
                "positions_start": low, "positions_end": p, "processed_k_sha256": k,
                "processed_v_sha256": v, "layout_sha256": self.collector.cache.layout})
        result = {"complete": True, "poisoned": False, "committed_length": p, "next_position": p,
                  "capacity": self.collector.job["capacity_tokens"], "kv_dtype": "bfloat16", "layers": layers,
                  "comparison_receipt_sha256": comparison_receipt_sha256}
        self.collector.handoff(result)
        return result

    def abort(self):
        self.writer.abort()
        self.collector.abort()
