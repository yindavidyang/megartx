"""Independent CPU localization and conditional BF16 linear diagnostics.

No Torch/runtime imports, server, fitted tolerance, or model acceptance. Exact
row/copy associations are mandatory. Gamma-K intervals explicitly assume F32
RNE accumulation without reduced-precision partial sums; an enabled backend
preference does not prove that assumption for the actual kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import compare_controlled_capture as evidence
from router_reference import bf16_decode, bf16_rne_bits, gamma


PREFIX = "model.language_model.layers.0."
PROJECTIONS = {"q": (4096, 2816), "k": (2048, 2816), "v": (2048, 2816)}
WIDTHS = {"input_norm_in": 2816, "input_norm_out": 2816, "qkv_in": 2816, "qkv_out": 8192,
          **{name: 4096 for name in ("q_norm_in", "q_norm_out", "rope_q_in", "rope_q_out", "attention_q")},
          **{name: 2048 for name in ("k_norm_in", "k_norm_out", "rope_k_in", "rope_k_out", "v_norm_in", "v_norm_out",
                                    "attention_k", "attention_v", "writer_k", "writer_v", "stored_k", "stored_v")}}
ORDER = ("input_norm_in", "input_norm_out", "qkv_in", "qkv_out", "q_norm_in", "q_norm_out",
         "k_norm_in", "k_norm_out", "rope_q_in", "rope_k_in", "rope_q_out", "rope_k_out",
         "v_norm_in", "v_norm_out", "attention_q", "attention_k", "attention_v", "writer_k", "writer_v", "stored_k", "stored_v")
CONSTANT_FIELDS = {"k_weight_bits", "input_norm_weight_bits", "q_norm_weight_bits", "k_norm_weight_bits"}
EXTRA_WIDTHS = {"attention_out": 4096, "o_proj_in": 4096,
                **{name: 2816 for name in ("o_proj_out", "post_attention_norm_in", "post_attention_norm_out", "pre_ff2_norm_in", "pre_ff2_norm_out")}}
EXTRA_ORDER = ("attention_out", "o_proj_in", "o_proj_out", "post_attention_norm_in", "post_attention_norm_out", "pre_ff2_norm_in", "pre_ff2_norm_out")
PREFIX_FIELDS = {"writer_all_k", "writer_all_v", "stored_prefix_k", "stored_prefix_v", "cache_positions", "cache_slots"}
PINNED_SOURCES = {
    "vllm.model_executor.layers.layernorm": "4126258bf85aa3af54c78cfbf2a0f32000491e42be37661f2d1ac9f0beea00f3",
    "vllm.model_executor.layers.linear": "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d",
    "vllm.model_executor.layers.rotary_embedding.base": "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5",
    "vllm.ir.op": "fd0355b398c0d96235bb6aba0c9c08f73c199429bd8ec2aed227178c297c1eb2",
    "vllm.model_executor.layers.utils": "88dee156d427709feb8a560f61eba8e0ebfd1dc10885978ccdce3a952c6afb10",
}


def bit_difference(left, right):
    if left.dtype != np.uint16 or right.dtype != np.uint16 or left.shape != right.shape:
        raise ValueError("BF16 comparison needs identical uint16 geometry")
    a, b = bf16_decode(left).astype(np.float64), bf16_decode(right).astype(np.float64)
    return {"raw_bits_equal": bool(np.array_equal(left, right)),
            "different_elements": int(np.count_nonzero(left != right)), "elements": int(left.size),
            "max_absolute_difference": float(np.abs(a - b).max(initial=0)),
            "signed_zero_differences": int(np.count_nonzero((a == 0) & (b == 0) & (left != right)))}


def original_bf16_projection(checkpoint, projection):
    """Read exactly one original layer-0 Q/K/V tensor, at most 32 MiB."""
    if projection not in PROJECTIONS:
        raise ValueError("Only the bounded layer-0 Q/K/V tensors are permitted")
    reader = evidence.formats.CheckpointReader(Path(checkpoint))
    name = PREFIX + "self_attn." + projection + "_proj.weight"
    path, offset, entry = reader.entries[name]
    start, end = entry["data_offsets"]
    shape = PROJECTIONS[projection]
    size = math.prod(shape) * 2
    if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape or end - start != size or size > 32 << 20 or start < 0 or offset + end > path.stat().st_size:
        raise ValueError("Original layer-0 projection representation or extent changed")
    with path.open("rb") as stream:
        stream.seek(offset + start)
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Original layer-0 projection payload was truncated")
    value = np.frombuffer(raw, dtype="<u2").reshape(shape)
    bf16_decode(value)
    return value, hashlib.sha256(raw).hexdigest()


def original_layer0_inputs(checkpoint, tokens):
    """Read only the frozen 33 embedding rows and the 2816 input-norm weights.

    The large embedding table is never materialized. Row order and repeated
    token IDs are preserved, so the M33 replay uses actual prefix operands.
    """
    if tokens.dtype != np.int64 or tokens.shape != (33,) or np.any(tokens < 0) or np.any(tokens >= 262144):
        raise ValueError("Only the canonical 33 token IDs may be read")
    reader = evidence.formats.CheckpointReader(Path(checkpoint))
    arrays, hashes = {}, {}
    for label, name, shape in (
        ("embedding", "model.language_model.embed_tokens.weight", (262144, 2816)),
        ("input_norm", PREFIX + "input_layernorm.weight", (2816,)),
    ):
        path, offset, entry = reader.entries[name]
        start, end = entry["data_offsets"]
        if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape or end - start != math.prod(shape) * 2 or start < 0 or offset + end > path.stat().st_size:
            raise ValueError("Original embedding/norm representation or extent changed")
        with path.open("rb") as stream:
            if label == "embedding":
                chunks = []
                for token in tokens:
                    stream.seek(offset + start + int(token) * 2816 * 2)
                    chunk = stream.read(2816 * 2)
                    if len(chunk) != 2816 * 2:
                        raise ValueError("Original selected embedding row was truncated")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                output_shape = (33, 2816)
            else:
                stream.seek(offset + start)
                raw = stream.read(2816 * 2)
                output_shape = shape
        if len(raw) != math.prod(output_shape) * 2:
            raise ValueError("Original bounded layer-0 input was truncated")
        arrays[label] = np.frombuffer(raw, dtype="<u2").reshape(output_shape)
        bf16_decode(arrays[label])
        hashes[label] = hashlib.sha256(raw).hexdigest()
    return arrays, hashes


def projection_diagnostic(input_bits, weight_bits, candidate_bits):
    x, w = bf16_decode(input_bits).astype(np.float64), bf16_decode(weight_bits).astype(np.float64)
    observed = bf16_decode(candidate_bits).astype(np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1] or observed.shape != (len(x), len(w)):
        raise ValueError("BF16 projection operands/output have incompatible shapes")
    k = x.shape[1]
    g32, g64 = gamma(k, 2.0 ** -24), gamma(k, 2.0 ** -53)
    dot = x @ w.T
    sumabs = np.abs(x) @ np.abs(w).T
    upper_sumabs = np.nextafter(sumabs / (1.0 - g64), np.inf)
    bound = np.nextafter((g32 + g64) * upper_sumabs, np.inf)
    low, high = dot - bound, dot + bound
    # Outward F32 conversion precedes monotone BF16 RNE. Never shrink an
    # analytic interval by rounding its real endpoints toward its interior.
    low32, high32 = low.astype(np.float32), high.astype(np.float32)
    low32 = np.where(low32.astype(np.float64) > low, np.nextafter(low32, np.float32(-np.inf)), low32)
    high32 = np.where(high32.astype(np.float64) < high, np.nextafter(high32, np.float32(np.inf)), high32)
    lower = bf16_decode(bf16_rne_bits(low32)).astype(np.float64)
    upper = bf16_decode(bf16_rne_bits(high32)).astype(np.float64)
    ideal = bf16_rne_bits(dot.astype(np.float32))
    product_min, product_max = math.inf, 0.0
    for row in x:
        products = np.abs(w * row[None, :])
        nz = products[products != 0]
        if nz.size:
            product_min = min(product_min, float(nz.min()))
        product_max = max(product_max, float(products.max(initial=0)))
    f32 = np.finfo(np.float32)
    usable = product_min >= float(f32.tiny) and product_max <= float(f32.max) and bool(np.all(upper_sumabs < f32.max))
    outside = (observed < lower) | (observed > upper)
    report = {"outputs": int(observed.size), "ideal_fp64_dot_then_f32_bf16": bit_difference(candidate_bits, ideal),
              "conditional_f32_interval_usable": usable,
              "conditional_f32_interval_pass": bool(not outside.any()) if usable else None,
              "outside_conditional_interval": int(outside.sum()) if usable else None,
              "largest_accumulator_bound": float(bound.max(initial=0)), "gamma_k_f32": g32,
              "all_nonzero_products_normal_f32": product_min >= float(f32.tiny),
              "native_accumulation_qualified": False,
              "assumptions": ["Full BF16 products accumulated with F32 RNE; no BF16 partial reduction",
                              "No overflow or FTZ-sensitive nonzero products/intermediates",
                              "FP64 dot/sumabs uncertainty included analytically; no observed-error fitting"],
              "acceptance_scope": "Conditional arithmetic diagnostic only; never full/cache or native-kernel acceptance"}
    return report, {"dot": dot, "bound": bound, "ideal_bits": ideal}


def load_case(root, path, plan):
    root = Path(root).resolve()
    if path not in {"full", "cached"} or evidence.read_bytes(root / "run.exit", 32).strip() != b"0":
        raise ValueError("Layer-0 comparison requires a successful owned full/cached run")
    if evidence.read_json(root / "status.json")["phase"] != "cleanup_complete":
        raise ValueError("Owned server cleanup is not confirmed")
    directory = root / "controlled" / path
    if (directory / "INVALIDATED.json").exists():
        raise ValueError("Layer-0 capture was invalidated")
    binding = evidence.read_json(directory / "layer0-boundaries.json")
    evidence.reference._controlled_origin(binding)
    if binding["mode"] != "native" or binding["path"] != path or any(binding[k] != plan[k] for k in ("token_sha256", "schedule_sha256")):
        raise ValueError("Layer-0 binding differs from the frozen native fixture")
    if binding["sources"] != PINNED_SOURCES or binding["batch_invariant"] is not False:
        raise ValueError("Layer-0 recorded source or numerical lane changed")
    schema = binding.get("schema", 1)
    if type(schema) is not int or schema not in {1, 2}:
        raise ValueError("Unsupported layer-0 observational schema")
    if binding.get("capture_policy", "synchronous") not in {"synchronous", "deferred_downstream"}:
        raise ValueError("Unsupported layer-0 observation policy")
    widths = {**WIDTHS, **EXTRA_WIDTHS} if schema == 2 else WIDTHS
    constant_fields = CONSTANT_FIELDS | {"post_attention_norm_weight_bits", "pre_ff2_norm_weight_bits"} if schema == 2 else CONSTANT_FIELDS
    constants = evidence.read_npz(directory, binding["constants_file"], binding["constants_sha256"], constant_fields)
    if constants["k_weight_bits"].dtype != np.uint16 or constants["k_weight_bits"].shape != PROJECTIONS["k"] or hashlib.sha256(constants["k_weight_bits"].tobytes()).hexdigest() != binding["loaded_k_sha256"]:
        raise ValueError("Loaded layer-0 K partition differs from its recorded hash/geometry")
    norm_shapes = {field: (256,) if field in {"q_norm_weight_bits", "k_norm_weight_bits"} else (2816,) for field in constant_fields - {"k_weight_bits"}}
    for field, shape in norm_shapes.items():
        if constants[field].dtype != np.uint16 or constants[field].shape != shape:
            raise ValueError("Layer-0 norm weight representation changed")
        bf16_decode(constants[field])
    expected_frames = [(33, [31, 32])] if path == "full" else [(32, [31]), (1, [32])]
    if len(binding["frames"]) != len(expected_frames):
        raise ValueError("Layer-0 frame count differs from the bounded schedule")
    rows, prefixes = {}, []
    for ordinal, (frame, (count, positions)) in enumerate(zip(binding["frames"], expected_frames)):
        if type(frame["forward"]) is not int or frame["forward"] != ordinal or type(frame["source_rows"]) is not int or frame["source_rows"] != count or frame["positions"] != positions:
            raise ValueError("Layer-0 actual forward shape/position identity changed")
        arrays = evidence.read_npz(directory, frame["file"], frame["sha256"], set(widths) | {"positions", "tokens", "rope_cache_bits"} | (PREFIX_FIELDS if schema == 2 else set()))
        if arrays["positions"].dtype != np.int64 or arrays["tokens"].dtype != np.int64 or not np.array_equal(arrays["positions"], positions) or not np.array_equal(arrays["tokens"], plan["tokens"][positions]):
            raise ValueError("Layer-0 captured tokens/positions differ from the actual fixture")
        for name, width in widths.items():
            value = arrays[name]
            if value.dtype != np.uint16 or value.shape != (len(positions), width):
                raise ValueError("Layer-0 boundary dtype/shape differs: " + name)
            bf16_decode(value)
        cache = arrays["rope_cache_bits"]
        if cache.shape != (len(positions), 256) or cache.dtype not in (np.uint16, np.uint32):
            raise ValueError("Layer-0 selected RoPE cache representation changed")
        if cache.dtype == np.uint16:
            bf16_decode(cache)
        elif not np.isfinite(cache.view(np.float32)).all():
            raise ValueError("Layer-0 RoPE cache contains nonfinite F32")
        links = [("input_norm_out", "qkv_in"), ("q_norm_out", "rope_q_in"), ("k_norm_out", "rope_k_in"),
                 ("rope_q_out", "attention_q"), ("rope_k_out", "attention_k"), ("v_norm_out", "attention_v"),
                 ("attention_k", "writer_k"), ("attention_v", "writer_v"), ("writer_k", "stored_k"), ("writer_v", "stored_v")]
        if any(not np.array_equal(arrays[a], arrays[b]) for a, b in links):
            raise ValueError("Layer-0 observed operator/copy association failed")
        if schema == 2:
            if not np.array_equal(arrays["attention_out"], arrays["o_proj_in"]) or not np.array_equal(arrays["o_proj_out"], arrays["post_attention_norm_in"]):
                raise ValueError("Layer-0 attention/output-projection association failed")
            residual_bits = bf16_rne_bits(bf16_decode(arrays["post_attention_norm_out"]) + bf16_decode(arrays["input_norm_in"]))
            if not np.array_equal(residual_bits, arrays["pre_ff2_norm_in"]):
                raise ValueError("Layer-0 BF16 attention-residual addition association failed")
            prefix_count = 33 if path == "full" or ordinal else 32
            current_positions = [32] if path == "cached" and ordinal else list(range(count))
            shape = frame["cache_shape"]
            if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape) or shape[1:] != [8, 16, 512]:
                raise ValueError("Layer-0 cache geometry changed")
            if arrays["cache_positions"].dtype != np.int64 or not np.array_equal(arrays["cache_positions"], range(prefix_count)) or arrays["cache_slots"].dtype != np.int64 or arrays["cache_slots"].shape != (prefix_count,) or len(set(arrays["cache_slots"].tolist())) != prefix_count or np.any(arrays["cache_slots"] < 0) or np.any(arrays["cache_slots"] >= shape[0] * 16):
                raise ValueError("Layer-0 actual frozen prefix address association failed")
            if arrays["cache_slots"].tolist()[positions[0]:positions[-1]+1] != frame["writer_slots"]:
                raise ValueError("Layer-0 selected writer slots differ from prefix slots")
            for field in ("k", "v"):
                writer, stored = arrays["writer_all_" + field], arrays["stored_prefix_" + field]
                if writer.dtype != np.uint16 or writer.shape != (count, 2048) or stored.dtype != np.uint16 or stored.shape != (prefix_count, 2048):
                    raise ValueError("Layer-0 bounded full-prefix cache representation changed")
                bf16_decode(writer); bf16_decode(stored)
                if not np.array_equal(writer, stored[current_positions]) or not np.array_equal(stored[positions], arrays["stored_" + field]):
                    raise ValueError("Layer-0 prefix writer/storage bits differ")
                if hashlib.sha256(stored.tobytes()).hexdigest() != frame["prefix_payload_sha256"][field]:
                    raise ValueError("Layer-0 prefix payload hash changed")
                if prefixes and not np.array_equal(prefixes[0]["stored_prefix_" + field], stored[:32]):
                    raise ValueError("Layer-0 decode changed the cached 32-row prefix")
            if prefixes and not np.array_equal(prefixes[0]["cache_slots"], arrays["cache_slots"][:32]):
                raise ValueError("Layer-0 prefix ownership changed between forwards")
            prefixes.append({key: arrays[key] for key in PREFIX_FIELDS})
        for field, section in (("q", slice(0, 4096)), ("k", slice(4096, 6144)), ("v", slice(6144, 8192))):
            if not np.array_equal(arrays["qkv_out"][:, section], arrays[field + "_norm_in"]):
                raise ValueError("Layer-0 QKV split association failed")
        for i, position in enumerate(positions):
            rows[position] = {k: v[i].copy() for k, v in arrays.items() if k in set(widths) | {"rope_cache_bits"}}
    # Bind immediate post-write observations to the existing after-forward
    # collector through independently read raw arrays, not a manifest boolean.
    kv_records = evidence.read_json(directory / "kv-records.json")
    records = [r for r in kv_records if r["layer"] == 0]
    if len(records) != 1:
        raise ValueError("Layer-0 final K/V record is absent or ambiguous")
    record = records[0]
    kv = evidence.read_npz(directory, record["file"], record["sha256"], evidence.KV_FIELDS)
    if not np.array_equal(kv["logical_positions"], [31, 32]):
        raise ValueError("Layer-0 final stored rows lost position identity")
    for i, position in enumerate((31, 32)):
        for field in ("k", "v"):
            key = "key_bits" if field == "k" else "value_bits"
            if kv[key].dtype != np.uint16 or kv[key].shape != (2, 8, 256) or not np.array_equal(kv[key][i].reshape(-1), rows[position]["stored_" + field]):
                raise ValueError("Immediate layer-0 writer read differs from after-forward cache capture")
    if schema == 2:
        controlled = evidence.read_json(directory / "controlled-manifest.json")
        records = [r for r in controlled["executed_interventions"] if r["layer"] == 0]
        if len(records) != 2:
            raise ValueError("Layer-0 corrected expert input is missing or ambiguous")
        seen = []
        for record in records:
            stage = evidence.read_npz(directory, record["file"], record["stage_capture_sha256"], evidence.STAGE_FIELDS)
            positions = stage["positions"].tolist()
            if len(positions) != 1 or positions[0] not in {31, 32} or stage["input_bits"].dtype != np.uint16 or not np.array_equal(stage["input_bits"], rows[positions[0]]["pre_ff2_norm_out"][None, :]) or not np.array_equal(stage["token_ids"], plan["tokens"][positions]):
                raise ValueError("Layer-0 incoming corrected expert row differs from its observed norm output")
            seen += positions
        if sorted(seen) != [31, 32]:
            raise ValueError("Layer-0 corrected expert row identity is duplicated or incomplete")
    return {"binding": binding, "constants": constants, "rows": rows, "prefixes": prefixes,
            "binding_sha256": evidence.digest_file(directory / "layer0-boundaries.json", evidence.MAX_JSON)}


def compare_pair(plan_directory, full, cached, checkpoint):
    plan = evidence.load_plan(plan_directory)
    if evidence.digest_file(Path(checkpoint) / "config.json", evidence.MAX_JSON) != evidence.reference.CONFIG_SHA256:
        raise ValueError("Layer-0 original checkpoint config changed")
    cases = [load_case(root, path, plan) for root, path in ((full, "full"), (cached, "cached"))]
    if cases[0]["binding"].get("schema", 1) != cases[1]["binding"].get("schema", 1):
        raise ValueError("Layer-0 paired observational schemas differ")
    schema = cases[0]["binding"].get("schema", 1)
    if schema == 2 and any("loaded_q_sha256" not in case["binding"] for case in cases):
        raise ValueError("Layer-0 schema-2 Q provenance is missing")
    for key in ("sources", "qkv_quant_method", "qkv_gemm", "norms", "rope", "loaded_k_sha256", "loaded_v_sha256",
                "allow_bf16_reduced_precision_reduction", "batch_invariant") + (("loaded_q_sha256",) if schema == 2 else ()):
        if cases[0]["binding"][key] != cases[1]["binding"][key]:
            raise ValueError("Layer-0 positive path binding changed: " + key)
    for key in cases[0]["constants"]:
        if not np.array_equal(cases[0]["constants"][key], cases[1]["constants"][key]):
            raise ValueError("Layer-0 loaded constants changed across paths")
    order = ORDER + EXTRA_ORDER if cases[0]["binding"].get("schema", 1) == 2 else ORDER
    comparisons = {str(p): {stage: bit_difference(cases[0]["rows"][p][stage], cases[1]["rows"][p][stage]) for stage in order} for p in (31, 32)}
    first = {str(p): next((s for s in order if not comparisons[str(p)][s]["raw_bits_equal"]), None) for p in (31, 32)}
    report = {**evidence.reference.CONTROLLED_ORIGIN, "checkpoint_revision": evidence.REVISION,
              "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
              "binding_sha256": [c["binding_sha256"] for c in cases], "first_differing_stage": first,
              "stage_comparisons": comparisons, "projections": {},
              "writer_input_storage_and_final_capture_exact": True,
              "full_cached_handoff_accepted": False, "native_accumulation_qualified": False,
              "independent_scheduler_qualified": False, "quality_gate_passed": False, "timing_qualified": False,
              "numerical_contract": "Exact provenance, row identity and writer/copy bits; conditional unfitted operator diagnostics; whole-model acceptance remains closed"}
    report["selected_rope_cache_bits_equal"] = all(np.array_equal(cases[0]["rows"][p]["rope_cache_bits"], cases[1]["rows"][p]["rope_cache_bits"]) for p in (31, 32))
    for field, section in (("q", slice(0, 4096)), ("k", slice(4096, 6144)), ("v", slice(6144, 8192))):
        weights, source_hash = original_bf16_projection(checkpoint, field)
        key = "loaded_" + field + "_sha256"
        provenance = all(key in case["binding"] for case in cases)
        for case in cases:
            if key in case["binding"] and source_hash != case["binding"][key]:
                raise ValueError("Loaded layer-0 projection bytes differ from original checkpoint: " + field + " / " + case["binding"]["path"])
        projected, diagnostics = [], []
        for case in cases:
            inputs = np.stack([case["rows"][p]["qkv_in"] for p in (31, 32)])
            outputs = np.stack([case["rows"][p]["qkv_out"][section] for p in (31, 32)])
            item, raw = projection_diagnostic(inputs, weights, outputs)
            projected.append(outputs); diagnostics.append(raw)
            item["path"] = case["binding"]["path"]
            report["projections"].setdefault(field, {"original_weight_sha256": source_hash,
                "loaded_original_weight_hash_verified": provenance, "paths": []})["paths"].append(item)
        coordinates = np.flatnonzero(projected[0][1] != projected[1][1])
        report["projections"][field]["position32_differing_coordinates"] = [
            {"coordinate": int(j), "full_bits": int(projected[0][1, j]), "cached_bits": int(projected[1][1, j]),
             "fp64_dot": float(diagnostics[0]["dot"][1, j]), "ideal_bits": int(diagnostics[0]["ideal_bits"][1, j]),
             "conditional_accumulator_bound": float(diagnostics[0]["bound"][1, j])} for j in coordinates]
    report["runtime"] = {k: cases[0]["binding"][k] for k in ("qkv_gemm", "allow_bf16_reduced_precision_reduction", "torch_version", "cuda_version")}
    report["limits"] = ["Actual normalization provider is recorded; source/cast/reduction qualification remains separate",
                        "Q loaded-original hash is verified only for schema 2; historical schema 1 Q dots remain diagnostic",
                        "K and V use distinct original layer-0 tensors; the global projection-sharing config flag does not merge local weights",
                        "FIPrefill/FIDecode did not expose the scheduler block table in this observer",
                        "QKV batching is localized; actual native accumulation precision/order is not independently proved",
                        "No empirical whole-model tolerance, natural coverage, quantizer equivalence or benchmark acceptance"]
    return report


def compare_repeats(plan_directory, first, second, *, observation_ablation=False):
    """Strict CPU comparison of two successful cached observations.

    Raw prefix arrays and operator/copy associations are rechecked by load_case;
    a recorded boolean never substitutes for inspecting the bounded arrays.
    """
    plan = evidence.load_plan(plan_directory)
    cases = [load_case(root, "cached", plan) for root in (first, second)]
    if any(case["binding"].get("schema") != 2 for case in cases):
        raise ValueError("Cached repeat localization requires both expanded schema-2 captures")
    policies = [c["binding"].get("capture_policy", "synchronous") for c in cases]
    if any(policy not in {"synchronous", "deferred_downstream"} for policy in policies) or (policies[0] != policies[1] and not observation_ablation):
        raise ValueError("Cached observation policies differ outside the explicit capture-placement ablation")
    for key in ("sources", "qkv_quant_method", "qkv_gemm", "o_proj_quant_method", "o_proj_gemm", "norms", "rope",
                "loaded_q_sha256", "loaded_k_sha256", "loaded_v_sha256", "allow_bf16_reduced_precision_reduction", "batch_invariant", "torch_version", "cuda_version"):
        if cases[0]["binding"][key] != cases[1]["binding"][key]:
            raise ValueError("Cached repeat numerical/profile binding differs: " + key)
    if any(not np.array_equal(cases[0]["constants"][key], cases[1]["constants"][key]) for key in cases[0]["constants"]):
        raise ValueError("Cached repeat loaded constants changed")
    comparisons = {str(p): {stage: bit_difference(cases[0]["rows"][p][stage], cases[1]["rows"][p][stage]) for stage in ORDER + EXTRA_ORDER} for p in (31, 32)}
    first_different = {str(p): next((stage for stage in ORDER + EXTRA_ORDER if not comparisons[str(p)][stage]["raw_bits_equal"]), None) for p in (31, 32)}
    prefix_comparisons = [{field: bit_difference(cases[0]["prefixes"][frame]["stored_prefix_" + field], cases[1]["prefixes"][frame]["stored_prefix_" + field]) for field in ("k", "v")} for frame in (0, 1)]
    return {**evidence.reference.CONTROLLED_ORIGIN, "checkpoint_revision": evidence.REVISION,
            "scope": "controlled_routing_fixture", "comparison": "capture_placement_ablation" if observation_ablation else "two_fresh_cached_lifecycles",
            "observation_policies": policies,
            "token_sha256": plan["token_sha256"], "schedule_sha256": plan["schedule_sha256"],
            "binding_sha256": [c["binding_sha256"] for c in cases], "stage_comparisons": comparisons,
            "first_differing_stage": first_different, "complete_layer0_prefix_comparisons": prefix_comparisons,
            "prefix_write_storage_and_decode_retention_exact": True,
            "operator_and_bf16_residual_associations_exact": True,
            "incoming_corrected_expert_rows_bound_to_norm_output": True,
            "native_accumulation_qualified": False, "full_cached_handoff_accepted": False,
            "independent_scheduler_qualified": False, "quality_gate_passed": False, "timing_qualified": False,
            "limits": ["Two retained operator rows and only the frozen 33-row layer-0 prefix",
                       "Copies synchronize CUDA and cannot establish timing or observer-free repeatability",
                       "Actual writer slots establish copy association, not independent scheduler correctness",
                       "No acceptance tolerance is derived from observed repeat differences"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "cached", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    peer = parser.add_mutually_exclusive_group(required=True)
    peer.add_argument("--full", type=Path)
    peer.add_argument("--cached-repeat", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--observation-ablation", action="store_true")
    args = parser.parse_args()
    if args.full is not None and (args.checkpoint is None or args.observation_ablation):
        parser.error("Full/cached arithmetic requires its original checkpoint and no observation-ablation flag")
    if args.cached_repeat is not None and args.checkpoint is not None:
        parser.error("Cached repeat localization reads captured operands only")
    inputs = [args.plan, args.cached, args.full or args.cached_repeat]
    if args.checkpoint is not None:
        inputs.append(args.checkpoint)
    target = args.output.resolve()
    if target.exists() or any(target.is_relative_to(p.resolve()) for p in inputs):
        parser.error("Report must be new and outside all input directories")
    report = compare_pair(args.plan, args.full, args.cached, args.checkpoint) if args.full is not None else compare_repeats(args.plan, args.cached, args.cached_repeat, observation_ablation=args.observation_ablation)
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"first_differing_stage": report["first_differing_stage"], "writer_storage_exact": True,
                      "full_cached_handoff_accepted": False, "native_accumulation_qualified": False}))


if __name__ == "__main__":
    main()
