"""Opt-in, observational layer-0 boundaries for the frozen full/cached pair.

Selected operator rows 31/32, layer-0's 33-row cache prefix and one bounded
K-weight partition are copied. Hooks never
return replacements. The writer wrapper belongs to one live backend instance;
it calls the original writer once, with its original arguments and return value.
This diagnostic changes synchronization and cannot supply timing evidence.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import MethodType

import numpy as np

from .controlled_capture import ORIGIN, bits, digest, save_npz


SOURCE_HASHES = {
    "vllm.model_executor.layers.layernorm": "4126258bf85aa3af54c78cfbf2a0f32000491e42be37661f2d1ac9f0beea00f3",
    "vllm.model_executor.layers.linear": "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d",
    "vllm.model_executor.layers.rotary_embedding.base": "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5",
    "vllm.ir.op": "fd0355b398c0d96235bb6aba0c9c08f73c199429bd8ec2aed227178c297c1eb2",
    "vllm.model_executor.layers.utils": "88dee156d427709feb8a560f61eba8e0ebfd1dc10885978ccdce3a952c6afb10",
}
STAGES = (
    "input_norm_in", "input_norm_out", "qkv_in", "qkv_out",
    "q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out",
    "rope_q_in", "rope_k_in", "rope_q_out", "rope_k_out",
    "v_norm_in", "v_norm_out", "attention_q", "attention_k", "attention_v",
    "writer_k", "writer_v", "stored_k", "stored_v",
    "attention_out", "o_proj_in", "o_proj_out", "post_attention_norm_in",
    "post_attention_norm_out", "pre_ff2_norm_in", "pre_ff2_norm_out",
)
PREFIX_FIELDS = {"writer_all_k", "writer_all_v", "stored_prefix_k", "stored_prefix_v", "cache_positions", "cache_slots"}
DEFERRED_STAGES = {"attention_out", "o_proj_in", "o_proj_out", "post_attention_norm_in",
                   "post_attention_norm_out", "pre_ff2_norm_in", "pre_ff2_norm_out"}


def selected_rows(positions, tokens, canonical):
    """Select by absolute position, rejecting ambiguous row/token association."""
    positions, tokens = tuple(positions), tuple(tokens)
    if len(canonical) != 33 or len(positions) != len(tokens) or not 1 <= len(positions) <= 33:
        raise RuntimeError("Layer-0 capture requires the bounded canonical row context")
    if any(type(p) is not int or not 0 <= p < 33 for p in positions) or len(set(positions)) != len(positions):
        raise RuntimeError("Layer-0 positions are invalid or duplicated")
    if any(type(t) is not int or t != canonical[p] for p, t in zip(positions, tokens)):
        raise RuntimeError("Layer-0 actual tokens differ at their absolute positions")
    rows = [i for i, p in enumerate(positions) if p in (31, 32)]
    if not rows:
        raise RuntimeError("Layer-0 diagnostic only supports the full/cached boundary pair")
    return rows


def block_table_slots(positions, block_table, block_size):
    """Independent CPU address calculation from actual attention block IDs.

    This checks block-table/writer consistency, not scheduler correctness.
    """
    if type(block_size) is not int or block_size < 1 or not positions:
        raise RuntimeError("Invalid layer-0 block geometry")
    table = tuple(block_table)
    if any(type(b) is not int or b < 0 for b in table):
        raise RuntimeError("Invalid layer-0 block IDs")
    if any(type(p) is not int or p < 0 or p // block_size >= len(table) for p in positions):
        raise RuntimeError("Layer-0 block table does not cover actual positions")
    return [table[p // block_size] * block_size + p % block_size for p in positions]


def extend_prefix_slots(existing, positions, slots, capacity):
    """Bind only the frozen prefix to actual writer addresses; forbid aliases.

    This validates ownership/copy association, not independent scheduling.
    """
    if type(capacity) is not int or capacity < 33 or len(positions) != len(slots):
        raise RuntimeError("Invalid frozen prefix slot geometry")
    result = dict(existing)
    for position, slot in zip(positions, slots):
        if type(position) is not int or not 0 <= position < 33 or position in result or type(slot) is not int or not 0 <= slot < capacity:
            raise RuntimeError("Frozen prefix position/slot identity changed")
        result[position] = slot
    if sorted(result) != list(range(len(result))) or len(set(result.values())) != len(result):
        raise RuntimeError("Frozen prefix is incomplete or cache slots alias")
    return result


def callable_record(fn):
    fn = getattr(fn, "__func__", fn)
    record = {"module": getattr(fn, "__module__", type(fn).__module__),
              "qualname": getattr(fn, "__qualname__", type(fn).__qualname__)}
    try:
        source = Path(inspect.getsourcefile(fn))
        record["source_sha256"] = digest(source)
    except (TypeError, OSError):
        record["source_sha256"] = None
    return record


class ControlledLayer0:
    def __init__(self, model, owner):
        import torch
        import vllm.ir.op
        import vllm.model_executor.layers.utils
        from vllm import envs

        if owner.mode != "native":
            raise RuntimeError("Layer-0 boundary capture requires the bounded native pair")
        self.owner, self.torch = owner, torch
        self.capture_policy = os.environ.get("MEGARTX_LAYER0_CAPTURE_POLICY", "synchronous")
        if self.capture_policy not in {"synchronous", "deferred_downstream"}:
            raise RuntimeError("Unknown bounded layer-0 observation policy")
        matches = [(p, m) for p, m in model.named_modules()
                   if type(m).__name__ == "Gemma4DecoderLayer" and re.search(r"(?:^|\.)layers\.0$", p)]
        if len(matches) != 1:
            raise RuntimeError("Layer-0 diagnostic requires one literal decoder owner")
        self.path, self.decoder = matches[0]
        self.attention = self.decoder.self_attn
        name, parent, self.attn, self.impl = owner.kv.layers[0]
        if parent is not self.attention or self.attention.q_size != 4096 or self.attention.kv_size != 2048:
            raise RuntimeError("Layer-0 diagnostic owner or projection geometry changed")
        self.registry_name = name
        self.norms = {"input_norm": self.decoder.input_layernorm, "q_norm": parent.q_norm,
                      "k_norm": parent.k_norm, "v_norm": parent.v_norm,
                      "post_attention_norm": self.decoder.post_attention_layernorm,
                      "pre_ff2_norm": self.decoder.pre_feedforward_layernorm_2}
        if self.norms["pre_ff2_norm"] is None:
            raise RuntimeError("Layer-0 frozen MoE input norm is absent")
        sources = [type(n) for n in self.norms.values()] + [type(parent.qkv_proj), type(parent.rotary_emb), vllm.ir.op, vllm.model_executor.layers.utils]
        for obj in sources:
            module = obj.__module__ if isinstance(obj, type) else obj.__name__
            if module not in SOURCE_HASHES or digest(inspect.getsourcefile(obj)) != SOURCE_HASHES[module]:
                raise RuntimeError("Layer-0 preprocessing source differs from the pinned runtime")
        method = parent.qkv_proj.quant_method
        if type(method).__name__ != "UnquantizedLinearMethod" or not hasattr(method, "_gemm_impl") or parent.qkv_proj.bias is not None:
            raise RuntimeError("Layer-0 QKV is not the pinned unbiased unquantized projection")
        if envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError("Layer-0 diagnostic must preserve the original non-invariant runtime")
        o_method = parent.o_proj.quant_method
        if type(o_method).__name__ != "UnquantizedLinearMethod" or not hasattr(o_method, "_gemm_impl") or parent.o_proj.bias is not None:
            raise RuntimeError("Layer-0 O is not the pinned unbiased unquantized projection")
        self.binding = {**ORIGIN, "schema": 2, "capture_policy": self.capture_policy,
                        "prefix_observation_boundary": "writer_post" if self.capture_policy == "synchronous" else "decoder0_post",
                        "layer": 0, "layer_name": self.path,
                        "sources": SOURCE_HASHES, "qkv_quant_method": callable_record(type(method)),
                        "qkv_gemm": callable_record(method._gemm_impl),
                        "qkv_forward": callable_record(parent.qkv_proj.forward),
                        "o_proj_quant_method": callable_record(type(o_method)),
                        "o_proj_gemm": callable_record(o_method._gemm_impl),
                        "norms": {n: {"forward": callable_record(m.forward),
                                     "epsilon": m.variance_epsilon,
                                     "variance_size": m.variance_size_override,
                                     "pass_weight": m.pass_weight, "weight_dtype": str(m.weight.dtype)}
                                  for n, m in self.norms.items()},
                        "rope": {"forward": callable_record(parent.rotary_emb.forward),
                                 "use_flashinfer": parent.rotary_emb.use_flashinfer,
                                 "is_neox_style": parent.rotary_emb.is_neox_style,
                                 "head_size": parent.rotary_emb.head_size,
                                 "rotary_dim": parent.rotary_emb.rotary_dim},
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "batch_invariant": False, "timing_qualified": False}
        self.handles, self.frames = [], []
        self.spans, self.prefix_slots = {}, {}
        self.pending_tensors, self.prefix_pending, self.flushing = {}, None, False
        self.frame = None
        self.in_layer0 = False
        # get_rope caches module objects shared by other local layers. Scope
        # those hooks to this literal decoder invocation, not module identity.
        self.handles.append(self.decoder.register_forward_pre_hook(self._decoder_pre))
        self.handles.append(self.decoder.register_forward_hook(self._decoder_post))
        for label, module in self.norms.items():
            self.handles.append(module.register_forward_pre_hook(self._norm_pre(label)))
            self.handles.append(module.register_forward_hook(self._norm_post(label)))
        self.handles.append(parent.qkv_proj.register_forward_pre_hook(self._qkv_pre))
        self.handles.append(parent.qkv_proj.register_forward_hook(self._qkv_post))
        self.handles.append(parent.rotary_emb.register_forward_pre_hook(self._rope_pre))
        self.handles.append(parent.rotary_emb.register_forward_hook(self._rope_post))
        self.handles.append(self.attn.register_forward_pre_hook(self._attention_pre))
        self.handles.append(self.attn.register_forward_hook(self._attention_post))
        self.handles.append(parent.o_proj.register_forward_pre_hook(self._o_pre))
        self.handles.append(parent.o_proj.register_forward_hook(self._o_post))
        self.original_writer = self.impl.do_kv_cache_update
        if "do_kv_cache_update" in self.impl.__dict__:
            raise RuntimeError("Layer-0 writer already has an instance override")
        self.writer_wrapper = MethodType(self._writer, self.impl)
        self.impl.do_kv_cache_update = self.writer_wrapper

    def _current(self):
        if not self.owner.active or not self.in_layer0:
            return False
        if self.owner.case not in {"full", "cached"}:
            raise RuntimeError("Layer-0 capture forbids chunked or unrelated requests")
        if self.frame is None:
            context = self.owner.context
            positions = [int(p) for p in context["positions"]]
            tokens = [int(t) for t in context["tokens"]]
            self.indices = selected_rows(positions, tokens, self.owner.kv.tokens)
            self.frame = {"forward": self.owner.forward_counter, "source_rows": len(positions),
                          "positions": [positions[i] for i in self.indices],
                          "tokens": [tokens[i] for i in self.indices], "events": [], "dispatch": {}, "arrays": {}}
            if not self.frames:
                self._constants()
        return True

    def _decoder_pre(self, module, args):
        if self.in_layer0:
            raise RuntimeError("Layer-0 decoder invocation was nested or repeated")
        self.in_layer0 = self.owner.active

    def _decoder_post(self, module, args, result):
        if self.in_layer0 and self.capture_policy == "deferred_downstream":
            # Keep original tensor objects; copying occurs after this decoder's
            # numerical work. Lifetime/allocation effects still preclude a
            # claim of observer-free execution or timing.
            self.flushing = True
            try:
                for name, tensor in self.pending_tensors.items():
                    self._record(name, tensor)
                if self.prefix_pending is None:
                    raise RuntimeError("Deferred layer-0 prefix observation was missed")
                self._observe_prefix(*self.prefix_pending)
            finally:
                self.flushing = False
                self.pending_tensors, self.prefix_pending = {}, None
        self.in_layer0 = False

    def _constants(self):
        torch = self.torch
        qkv = self.attention.qkv_proj
        if qkv.weight.dtype != torch.bfloat16 or tuple(qkv.weight.shape) != (8192, 2816):
            raise RuntimeError("Layer-0 loaded QKV weight representation changed")
        k = bits(qkv.weight[4096:6144]).copy()
        v = bits(qkv.weight[6144:8192]).copy()
        arrays = {"k_weight_bits": k}
        self.binding["loaded_q_sha256"] = hashlib.sha256(bits(qkv.weight[:4096]).tobytes()).hexdigest()
        self.binding["loaded_k_sha256"] = hashlib.sha256(k.tobytes()).hexdigest()
        self.binding["loaded_v_sha256"] = hashlib.sha256(v.tobytes()).hexdigest()
        self.binding["loaded_k_v_bits_equal"] = bool(np.array_equal(k, v))
        for label, module in self.norms.items():
            if module.pass_weight:
                arrays[label + "_weight_bits"] = bits(module.weight).copy()
        path = self.owner.directory / "layer0-constants.npz"
        self.binding["constants_file"], self.binding["constants_sha256"] = path.name, save_npz(path, arrays)

    def _record(self, name, tensor, *, selected=False):
        if not self._current():
            return
        torch = self.torch
        if name in self.frame["arrays"] or tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
            raise RuntimeError(f"Layer-0 boundary {name} was repeated or lost BF16 CUDA identity: {tensor.dtype}, {tensor.device}")
        count = len(self.indices) if selected else self.frame["source_rows"]
        if tensor.ndim < 2 or tensor.shape[0] != count:
            raise RuntimeError("Layer-0 tensor rows differ from the actual forward context")
        if self.capture_policy == "deferred_downstream" and name in DEFERRED_STAGES and not self.flushing:
            if name in self.pending_tensors or selected:
                raise RuntimeError("Deferred layer-0 boundary was repeated or selected early")
            self.pending_tensors[name] = tensor
            return
        rows = tensor if selected else tensor[self.indices]
        value = bits(rows).reshape(len(self.indices), -1).copy()
        if ((value & 0x7F80) == 0x7F80).any():
            raise RuntimeError("Layer-0 boundary contains nonfinite BF16")
        self.frame["arrays"][name] = value
        self.frame["events"].append({"stage": name, "shape": list(tensor.shape), "strides": list(tensor.stride())})

    def _norm_pre(self, label):
        def pre(module, args):
            if not self._current():
                return
            if len(args) != 1:
                raise RuntimeError("Layer-0 norm unexpectedly receives a residual")
            from vllm import ir
            impl = ir.ops.rms_norm.dispatch(args[0], module.weight.data if module.pass_weight else None,
                                            module.variance_epsilon, module.variance_size_override)
            self.frame["dispatch"][label] = {"provider": impl.provider, "implementation": callable_record(impl.impl_fn),
                                               "forward_method": callable_record(module._forward_method)}
            self._record(label + "_in", args[0])
        return pre

    def _norm_post(self, label):
        def post(module, args, result):
            self._record(label + "_out", result)
        return post

    def _qkv_pre(self, module, args):
        self._record("qkv_in", args[0])
        self._span_start("qkv")

    def _qkv_post(self, module, args, result):
        self._span_end("qkv")
        self._record("qkv_out", result[0])

    def _span_start(self, name):
        if self._current():
            if name in self.spans:
                raise RuntimeError("Layer-0 profiler span was nested")
            span = self.torch.profiler.record_function("megartx.layer0." + name)
            span.__enter__()
            self.spans[name] = span

    def _span_end(self, name):
        span = self.spans.pop(name, None)
        if span is not None:
            span.__exit__(None, None, None)

    def _o_pre(self, module, args):
        self._record("o_proj_in", args[0])
        self._span_start("o_proj")

    def _o_post(self, module, args, result):
        self._span_end("o_proj")
        self._record("o_proj_out", result[0])

    def _rope_pre(self, module, args):
        if not self._current():
            return
        positions, q, k = args
        if positions.detach().cpu().tolist() != self.owner.context["positions"].tolist():
            raise RuntimeError("Layer-0 RoPE positions differ from the actual context")
        self._record("rope_q_in", q)
        self._record("rope_k_in", k)  # CPU copy completes before the in-place op.

    def _rope_post(self, module, args, result):
        if not self._current():
            return
        self._record("rope_q_out", result[0])
        self._record("rope_k_out", result[1])
        self.frame["arrays"]["rope_cache_bits"] = bits(module.cos_sin_cache[self.frame["positions"]]).copy()
        self.frame["rope_cache_dtype"] = str(module.cos_sin_cache.dtype)

    def _attention_pre(self, module, args):
        for name, tensor in zip(("attention_q", "attention_k", "attention_v"), args[:3]):
            self._record(name, tensor)
        self._span_start("attention")

    def _attention_post(self, module, args, result):
        self._span_end("attention")
        self._record("attention_out", result)

    def _writer(self, impl, layer, key, value, cache, slots):
        if not self._current():
            return self.original_writer(layer, key, value, cache, slots)
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        meta = context.attn_metadata[self.registry_name]
        torch = self.torch
        if impl is not self.impl or layer is not self.attn or cache is not self.attn.kv_cache or context.no_compile_layers[self.registry_name] is not layer:
            raise RuntimeError("Layer-0 writer owner changed")
        if slots.dtype != torch.int64 or tuple(slots.shape) != (self.frame["source_rows"],) or not torch.equal(slots, context.slot_mapping[self.registry_name]) or not torch.equal(slots, meta.slot_mapping):
            raise RuntimeError("Layer-0 writer slots differ from actual metadata")
        actual = [int(s) for s in slots[self.indices].detach().cpu().tolist()]
        if cache.dtype != torch.bfloat16 or tuple(cache.shape[1:]) != (8, 16, 512) or any(s < 0 or s >= cache.shape[0] * 16 for s in actual):
            raise RuntimeError("Layer-0 writer cache geometry or slot bounds changed")
        self.frame["writer_slots"] = actual
        metadata = meta.decode if meta.num_decodes else meta.prefill
        if hasattr(metadata, "block_tables"):
            blocks = metadata.block_tables
            if blocks.ndim != 2 or blocks.shape[0] != 1:
                raise RuntimeError("Layer-0 block table is not one sequence")
            count = max(self.frame["positions"]) // 16 + 1
            table = [int(b) for b in blocks[0, :count].detach().cpu().tolist()]
            calculated = block_table_slots(self.frame["positions"], table, 16)
            if calculated != actual:
                raise RuntimeError("Layer-0 block-table addresses differ from writer slots")
            self.frame["block_table"] = table
            self.frame["block_table_slots"] = calculated
        self.frame["metadata_class"] = type(metadata).__name__
        self.frame["cache_shape"], self.frame["cache_strides"] = list(cache.shape), list(cache.stride())
        self._record("writer_k", key)
        self._record("writer_v", value)
        result = self.original_writer(layer, key, value, cache, slots)
        rows = torch.stack([cache[s // 16, :, s % 16, :] for s in actual])
        self._record("stored_k", rows[..., :256], selected=True)
        self._record("stored_v", rows[..., 256:], selected=True)
        for field in ("k", "v"):
            if not np.array_equal(self.frame["arrays"]["writer_" + field], self.frame["arrays"]["stored_" + field]):
                raise RuntimeError("Layer-0 writer input differs from its stored cache row")
        if self.capture_policy == "deferred_downstream":
            if self.prefix_pending is not None:
                raise RuntimeError("Deferred layer-0 writer was repeated")
            self.prefix_pending = (cache, key, value, slots)
        else:
            self._observe_prefix(cache, key, value, slots)
        return result

    def _observe_prefix(self, cache, key, value, slots):
        torch = self.torch
        all_slots = [int(s) for s in slots.detach().cpu().tolist()]
        self.prefix_slots = extend_prefix_slots(self.prefix_slots, [int(p) for p in self.owner.context["positions"]], all_slots, cache.shape[0] * 16)
        for label, tensor in (("k", key), ("v", value)):
            if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda" or tuple(tensor.shape) != (self.frame["source_rows"], 8, 256):
                raise RuntimeError("Layer-0 full writer row representation changed")
            self.frame["arrays"]["writer_all_" + label] = bits(tensor).reshape(self.frame["source_rows"], 2048).copy()
        positions = sorted(self.prefix_slots)
        prefix = torch.stack([cache[self.prefix_slots[p] // 16, :, self.prefix_slots[p] % 16, :] for p in positions])
        for field, section in (("k", slice(0, 256)), ("v", slice(256, 512))):
            stored = bits(prefix[..., section]).reshape(len(positions), 2048).copy()
            self.frame["arrays"]["stored_prefix_" + field] = stored
            writer = self.frame["arrays"]["writer_all_" + field]
            if not np.array_equal(stored[[int(p) for p in self.owner.context["positions"]]], writer):
                raise RuntimeError("Layer-0 prefix writer input differs from its observed stored rows")
            if self.frames:
                old = self.frames[-1]["prefix_payload_sha256"][field]
                if hashlib.sha256(stored[:32].tobytes()).hexdigest() != old:
                    raise RuntimeError("Layer-0 decode write changed the retained 32-row prefix")
        self.frame["arrays"]["cache_positions"] = np.asarray(positions, dtype=np.int64)
        self.frame["arrays"]["cache_slots"] = np.asarray([self.prefix_slots[p] for p in positions], dtype=np.int64)
        self.frame["prefix_payload_sha256"] = {field: hashlib.sha256(self.frame["arrays"]["stored_prefix_" + field].tobytes()).hexdigest() for field in ("k", "v")}

    def end_forward(self):
        if self.frame is None or set(self.frame["arrays"]) != set(STAGES) | PREFIX_FIELDS | {"rope_cache_bits"} or self.spans or self.pending_tensors or self.prefix_pending is not None:
            raise RuntimeError("Layer-0 forward missed a required observational boundary")
        frame = self.frame
        arrays = frame.pop("arrays")
        arrays.update(positions=np.asarray(frame["positions"], dtype=np.int64),
                      tokens=np.asarray(frame["tokens"], dtype=np.int64))
        path = self.owner.directory / f"layer0-frame-{frame['forward']:02d}.npz"
        frame["file"], frame["sha256"] = path.name, save_npz(path, arrays)
        self.frames.append(frame)
        self.frame = None

    def finish(self):
        positions = [p for frame in self.frames for p in frame["positions"]]
        if sorted(positions) != [31, 32] or len(self.frames) != (1 if self.owner.case == "full" else 2):
            raise RuntimeError("Layer-0 capture is missing or duplicating the bounded pair")
        payload = {**self.binding, "path": self.owner.case, "mode": self.owner.mode,
                   "token_sha256": self.owner.plan["token_sha256"],
                   "schedule_sha256": self.owner.plan["schedule_sha256"], "frames": self.frames,
                   "writer_storage_bits_equal": True, "independent_scheduler_qualified": False,
                   "full_cached_handoff_accepted": False, "quality_gate_passed": False}
        with (self.owner.directory / "layer0-boundaries.json").open("x") as stream:
            json.dump(payload, stream, indent=2)
        self.close()

    def close(self):
        self.pending_tensors, self.prefix_pending = {}, None
        for name in list(self.spans):
            self._span_end(name)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.impl.__dict__.get("do_kv_cache_update") is self.writer_wrapper:
            del self.impl.do_kv_cache_update
