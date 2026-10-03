"""Dormant native BHNC cache observer with bounded two-row hashing.

Reuses the pinned source/owner/layout checks, not the controlled 33-token plan.
No cache allocation/mutation or GPU imports at module import. Native observation
is deliberately perturbing and forbidden in timing jobs.
"""

import hashlib
import struct

from . import prefill_runner as runner
from .prefill_collect import required_kv
from .controlled_kv_capture import ControlledKV, gather_writer_rows


class WriterLedger:
    """Actual absolute-position/slot association before writes are dispatched."""

    def __init__(self):
        self.positions = {i: {} for i in range(30)}
        self.end, self.pending, self.poisoned = 0, None, False

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
        self.native_bindings = None

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
