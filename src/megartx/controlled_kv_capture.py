"""Bounded observational K/V rows for the pinned controlled 33-input case.

No model/cache mutation, global hook, server launch or whole-cache copy. Call
capture after the real model forward while its ForwardContext is still active.
Only the owner of that forward may call this helper. Raw snapshots stay private.
"""

import hashlib
import inspect
import json
from pathlib import Path
import re
import struct


ORIGIN = {"route_origin": "controlled", "routing_intervention": True,
          "routing_unchanged": False, "scope": "controlled_routing_fixture"}
CONFIG_SHA256 = "4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98"
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
SOURCE_HASHES = {
    "vllm.model_executor.models.gemma4": "16ac0a67dcf5dd695a59edb177f20e1e69d9c3ec45c5883744470c7c06c516a3",
    "vllm.model_executor.layers.attention.attention": "bc897e452f3aea603e353ca42a365d3be8836e9485d78cbf8cc4677baca2fb42",
    "vllm.v1.attention.backends.flashinfer": "8ee541fde43ed92a417b3f01ea1e6fc64af8ab2eb54cbfc28308a49aaca4ed7f",
    "vllm.forward_context": "cc1473477dcc7b762c3901b25cc7bf35c06a2c42f193136fed553abf6fe33f0e",
}


def _origin(record):
    if any(record.get(k) != v or type(record.get(k)) is not type(v)
           for k, v in ORIGIN.items()):
        raise RuntimeError("K/V capture requires explicit controlled origin")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(cls, module, name, checked):
    if cls.__module__ != module or cls.__name__ != name:
        raise RuntimeError("K/V owner/backend class differs from the pinned source")
    if cls not in checked:
        path = Path(inspect.getsourcefile(cls))
        if path.stat().st_size > 512 * 1024 or _digest(path) != SOURCE_HASHES[module]:
            raise RuntimeError("K/V owner/backend source bytes changed")
        checked.add(cls)


def nominal_layers():
    return [{"layer": i, "attention_type": "full_attention" if i % 6 == 5 else "sliding_attention",
             "kv_heads": 2 if i % 6 == 5 else 8,
             "head_dim": 512 if i % 6 == 5 else 256,
             "window_size": None if i % 6 == 5 else 1024,
             "logical_layout": "position,kv_head,head_dim", "dtype": "bf16"}
            for i in range(30)]


class SlotLedger:
    """Pure CPU write-association ledger; fresh short case, no slot eviction.

    It binds the actual writer slot map to absolute positions. It does not
    independently derive the scheduler's block table or certify attention.
    """

    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        if len(self.tokens) != 33 or any(type(t) is not int or not 0 <= t < 262144 for t in self.tokens):
            raise RuntimeError("K/V ledger requires the declared 33 exact token IDs")
        self.slots = {i: {} for i in range(30)}
        self.identities = None
        self.seen = set()
        self.calls = 0

    def record(self, positions, tokens, slots, capacities, identities):
        positions, tokens = tuple(positions), tuple(tokens)
        if self.calls >= 4 or not 1 <= len(positions) <= 33 or len(tokens) != len(positions):
            raise RuntimeError("K/V frame exceeds the bounded minimal forward count/rows")
        if any(type(p) is not int for p in positions) or sorted(positions) != list(range(len(self.seen), len(self.seen) + len(positions))) or len(self.seen) + len(positions) > 33:
            raise RuntimeError("K/V positions are duplicate, stale, padded or noncontiguous in the case")
        if any(type(t) is not int or t != self.tokens[p] for p, t in zip(positions, tokens)):
            raise RuntimeError("K/V input tokens differ at the actual absolute positions")
        if any(set(value) != set(range(30)) for value in (slots, capacities, identities)):
            raise RuntimeError("K/V ledger is missing exact layer ownership")
        if self.identities is not None and identities != self.identities:
            raise RuntimeError("K/V tensor/storage/shape/stride owner changed during the case")
        pending = {}
        for layer in range(30):
            values, capacity = tuple(slots[layer]), capacities[layer]
            if type(capacity) is not int or capacity < 1 or len(values) != len(positions) or any(type(s) is not int or not 0 <= s < capacity for s in values):
                raise RuntimeError("K/V writer slot is negative, padded or outside its actual view")
            if len(set(values)) != len(values) or set(values) & set(self.slots[layer].values()):
                raise RuntimeError("K/V writer reused a logical slot in the fresh short case")
            pending[layer] = dict(zip(positions, values))
        # Commit only after every layer is checked, preserving a failed frame.
        if self.identities is None:
            self.identities = identities.copy()
        for layer in range(30):
            self.slots[layer].update(pending[layer])
        self.seen.update(positions)
        self.calls += 1

    @property
    def complete(self):
        return self.seen == set(range(33))

    def selected(self, layer):
        if not self.complete:
            raise RuntimeError("Both handoff slots require committed length 33")
        return self.slots[layer][31], self.slots[layer][32]


def gather_rows(cache, slots, head_dim, torch):
    if len(slots) != 2:
        raise RuntimeError("K/V gather requires two distinct populated writer slots")
    return gather_writer_rows(cache, slots, head_dim, torch)


def gather_writer_rows(cache, slots, head_dim, torch):
    """Index logical BHNC axes, respecting strides; copy at most two BF16 rows."""
    if cache.dtype != torch.bfloat16 or len(cache.shape) != 4 or cache.shape[3] != 2 * head_dim or cache.stride()[-1] != 1:
        raise RuntimeError("K/V gather requires the pinned BF16 BHNC content layout")
    if not 1 <= len(slots) <= 2 or len(set(slots)) != len(slots) or any(type(s) is not int or not 0 <= s < cache.shape[0] * cache.shape[2] for s in slots):
        raise RuntimeError("K/V gather requires two distinct populated writer slots")
    rows = [cache[slot // cache.shape[2], :, slot % cache.shape[2], :] for slot in slots]
    # stack copies only selected rows; never call cpu/contiguous on the cache.
    bits = torch.stack(rows).detach().contiguous().view(torch.int16).cpu().numpy().view("uint16").copy()
    if ((bits & 0x7F80) == 0x7F80).any():
        raise RuntimeError("Captured logical K/V contains nonfinite BF16 bits")
    return bits[:, :, :head_dim].copy(), bits[:, :, head_dim:].copy()


def validate_context(context):
    if not isinstance(context.slot_mapping, dict) or not isinstance(context.attn_metadata, dict) or not isinstance(context.no_compile_layers, dict) or context.ubatch_slices is not None or context.cudagraph_runtime_mode.name != "NONE":
        raise RuntimeError("K/V capture requires eager, unsplit dictionary ForwardContext")


def validate_metadata(meta, rows):
    counts = (meta.num_actual_tokens, meta.num_decodes, meta.num_prefills,
              meta.num_decode_tokens, meta.num_prefill_tokens)
    if any(type(value) is not int or value < 0 for value in counts) or meta.num_actual_tokens != rows or meta.use_cascade is not False or meta.causal is not True or meta.num_decodes + meta.num_prefills != 1 or meta.num_decode_tokens + meta.num_prefill_tokens != rows:
        raise RuntimeError("K/V metadata is not one causal unpadded controlled sequence")
    if meta.num_decodes and (meta.num_decodes != 1 or meta.num_decode_tokens != 1 or rows != 1):
        raise RuntimeError("K/V decode metadata is not the single nonspec input")


class ControlledKV:
    def __init__(self, model, plan, mode):
        _origin(plan)
        if mode not in {"native", "paired_reference", "gate_only_negative_control"}:
            raise RuntimeError("Unknown controlled K/V arithmetic lane")
        self.tokens = tuple(int(t) for t in plan["tokens"].tolist())
        SlotLedger(self.tokens)
        self.token_sha256 = hashlib.sha256(struct.pack("<33q", *self.tokens)).hexdigest()
        if self.token_sha256 != plan["token_sha256"]:
            raise RuntimeError("K/V canonical token digest differs")
        self.schedule_sha256, self.mode = plan["schedule_sha256"], mode
        self.bind_owners(model)

    def bind_owners(self, model):
        """Bind the pinned writer/cache owners without changing routing or cache."""
        checked, self.layers = set(), {}
        for path, module in model.named_modules():
            if type(module).__module__ != "vllm.model_executor.models.gemma4" or type(module).__name__ != "Gemma4Attention":
                continue
            match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn$", path)
            if match is None or int(match[1]) in self.layers:
                raise RuntimeError("K/V module does not have a unique literal layer identity")
            ordinal, attn = int(match[1]), module.attn
            _source(type(module), "vllm.model_executor.models.gemma4", "Gemma4Attention", checked)
            _source(type(attn), "vllm.model_executor.layers.attention.attention", "Attention", checked)
            _source(type(attn.impl), "vllm.v1.attention.backends.flashinfer", "FlashInferImpl", checked)
            _source(attn.attn_backend, "vllm.v1.attention.backends.flashinfer", "FlashInferBackend", checked)
            method = attn.impl.do_kv_cache_update
            if method.__func__ is not type(attn.impl).do_kv_cache_update or method.__func__.__qualname__ != "FlashInferImpl.do_kv_cache_update" or attn.attn_backend.forward_includes_kv_cache_update is not False:
                raise RuntimeError("K/V writer dispatch differs from the pinned FlashInfer implementation")
            if module.is_kv_shared_layer or attn.kv_sharing_target_layer_name is not None or attn.impl.kv_sharing_target_layer_name is not None:
                raise RuntimeError("This K/V fixture forbids shared cache owners")
            if not re.search(rf"(?:^|\.)layers\.{ordinal}\.self_attn\.attn$", attn.layer_name):
                raise RuntimeError("K/V registry name differs from its literal model layer")
            self.layers[ordinal] = (attn.layer_name, module, attn, attn.impl)
        if set(self.layers) != set(range(30)):
            raise RuntimeError("K/V capture requires all thirty actual Gemma4 attention owners")
        self.descriptors = nominal_layers()
        self.directory = None

    def begin_case(self, directory):
        if self.directory is not None:
            raise RuntimeError("Prior K/V case was not finalized")
        self.directory = Path(directory)
        if not self.directory.is_dir() or self.directory.name not in {"full", "cached", "chunked"}:
            raise RuntimeError("K/V output must be the fresh declared controlled case directory")
        self.ledger, self.snapshots, self.bindings = SlotLedger(self.tokens), None, None

    def capture(self, positions, tokens):
        import torch
        from vllm.forward_context import get_forward_context

        if self.directory is None or self.snapshots is not None:
            raise RuntimeError("K/V capture has no active incomplete case")
        slots, capacities, identities, bindings = self.read_frame(positions, tokens)
        positions, tokens = [int(p) for p in positions.tolist()], [int(t) for t in tokens.tolist()]
        self.ledger.record(positions, tokens, slots, capacities, identities)
        self.bindings = bindings
        if self.ledger.complete:
            snapshots = []
            for ordinal in range(30):
                keys, values = gather_rows(self.layers[ordinal][2].kv_cache, self.ledger.selected(ordinal), self.descriptors[ordinal]["head_dim"], torch)
                snapshots.append((keys, values))
            self.snapshots = snapshots

    def read_frame(self, positions, tokens):
        """Validate actual eager writer metadata; return only bounded row mappings."""
        import torch
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        path = Path(inspect.getsourcefile(get_forward_context))
        if _digest(path) != SOURCE_HASHES["vllm.forward_context"]:
            raise RuntimeError("K/V ForwardContext source changed")
        validate_context(context)
        if str(positions.dtype) != "int64" or str(tokens.dtype) != "int64" or positions.ndim != 1 or tokens.ndim != 1:
            raise RuntimeError("K/V actual position/token context must retain I64 row identity")
        positions, tokens = [int(p) for p in positions.tolist()], [int(t) for t in tokens.tolist()]
        slots, capacities, identities, bindings = {}, {}, {}, []
        for ordinal, descriptor in enumerate(self.descriptors):
            name, parent, attn, impl = self.layers[ordinal]
            if parent.attn is not attn or attn.impl is not impl or context.no_compile_layers.get(name) is not attn or name not in context.slot_mapping or name not in context.attn_metadata:
                raise RuntimeError("K/V actual registry/module/backend owner changed")
            meta, mapping, cache = context.attn_metadata[name], context.slot_mapping[name], attn.kv_cache
            validate_metadata(meta, len(positions))
            if mapping.dtype != torch.int64 or tuple(mapping.shape) != (len(positions),) or not torch.equal(mapping, meta.slot_mapping):
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

    def finish_case(self):
        import numpy as np

        if self.directory is None or not self.ledger.complete or self.snapshots is None:
            raise RuntimeError("K/V case lacks committed logical positions 31/32 at all thirty layers")
        source_sha = hashlib.sha256(json.dumps(SOURCE_HASHES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        contract = {"checkpoint_revision": REVISION, "provenance": {"scope": "installed_runtime_confirmed", "config_sha256": CONFIG_SHA256,
                    "source_sha256": source_sha, "kv_owner_layout_confirmed": True}, "layers": self.descriptors}
        contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        records = []
        for ordinal, (keys, values) in enumerate(self.snapshots):
            filename = f"kv-{ordinal:02d}.npz"
            path = self.directory / filename
            with path.open("xb") as stream:
                np.savez_compressed(stream, logical_positions=np.asarray([31, 32], dtype=np.int64), key_bits=keys, value_bits=values)
            descriptor = self.descriptors[ordinal]
            records.append({**ORIGIN, "file": filename, "sha256": _digest(path), "mode": self.mode,
                            "token_sha256": self.token_sha256, "schedule_sha256": self.schedule_sha256,
                            "cache_contract_sha256": contract["sha256"], "layer": ordinal, "committed_length": 33,
                            "attention_type": descriptor["attention_type"], "kv_heads": descriptor["kv_heads"], "head_dim": descriptor["head_dim"],
                            "mask_window_start": 0, "logical_mapping_verified": True,
                            "writer_slots": list(self.ledger.selected(ordinal)), "same_cache_owner_across_forwards": True,
                            "logical_mapping_scope": "actual per-layer writer slot maps within one fresh short request"})
        for filename, payload in (("cache-contract.json", contract), ("kv-records.json", records),
                                  ("kv-binding.json", {**ORIGIN, "source_sha256": SOURCE_HASHES, "layers": self.bindings,
                                                       "forward_calls": self.ledger.calls, "independent_cache_correctness_qualified": False,
                                                       "scheduler_block_table_independently_reconstructed": False})):
            with (self.directory / filename).open("x") as stream:
                json.dump(payload, stream, indent=2)
        self.directory = None
        return records
