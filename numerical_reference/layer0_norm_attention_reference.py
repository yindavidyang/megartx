"""CPU-only, conditional checks for the frozen layer-0 norm/attention pair.

Arithmetic intervals come from declared source operations and format precision,
never from observed errors. Exact cache/copy provenance remains a separate gate.
MMA accumulation and compiled transcendental behavior are explicit assumptions;
this module cannot grant whole-model, quality, timing, or native-ISA acceptance.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import layer0_reference as layer0
from router_reference import bf16_decode, bf16_rne_bits, gamma

U32, U64, UBF16 = 2.0 ** -24, 2.0 ** -53, 2.0 ** -8
F32 = np.finfo(np.float32)
NORM_NAMES = ("input_norm", "q_norm", "k_norm", "v_norm", "post_attention_norm", "pre_ff2_norm")
ORIGINAL_NORM_SHA256 = {
    "input_norm": "218744f656a797cef91f0f985f9f034c3e9e06c40d3dad26502edd12a811f068",
    "q_norm": "a80f3176d8c4e9c8d113aee90863711847acaf9bc112dc49f85690fd355c7866",
    "k_norm": "82dbd230fa6158cdafdf769a7ed09db864b9b957b379d356408db8c75496b1b4",
    "post_attention_norm": "3b7c7fff8d5d5c07ddbd6588c6ce7433cf3852017b7ee9eebcd91d4a45d0f1ab",
    "pre_ff2_norm": "2e70831425408d1cb6c980c48d1132e10b72ff3ed2a81dccb2265efd1d036f1b",
}
REPLAY_SOURCE_SHA256 = {
    "vllm.kernels.vllm_c": "ef37e71736807026f0b8ad187155b63f147984c03d287a78790ef8f8ed642ec1",
    "flashinfer.decode": "d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a",
    "flashinfer.prefill": "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de",
    "flashinfer.xqa": "6a3f7330980b976201cff0fcdf51e2958e454c4c5951ad2b30c1f2aa13722d33",
    "_C_stable_libtorch.abi3.so": "7a283666e16f66348351ed0e877425016b09e1986a2a1086b0aaad08dbadd101",
}


def _decode(bits):
    if not isinstance(bits, np.ndarray) or bits.dtype != np.uint16:
        raise ValueError("Expected raw finite BF16 uint16 arrays")
    return bf16_decode(bits).astype(np.float64)


def _normal_or_zero(values):
    return bool(np.all((values == 0) | ((np.abs(values) >= F32.tiny) & (np.abs(values) <= F32.max))))


def _endpoints(low, high):
    """Outward F32 endpoints followed by monotone final BF16 RNE."""
    if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("Invalid finite arithmetic interval")
    low32, high32 = low.astype(np.float32), high.astype(np.float32)
    if not np.isfinite(low32).all() or not np.isfinite(high32).all():
        raise ValueError("F32 interval overflow")
    low32 = np.where(low32.astype(float) > low, np.nextafter(low32, np.float32(-np.inf)), low32)
    high32 = np.where(high32.astype(float) < high, np.nextafter(high32, np.float32(np.inf)), high32)
    return bf16_decode(bf16_rne_bits(low32)).astype(float), bf16_decode(bf16_rne_bits(high32)).astype(float)


def check_interval(candidate_bits, interval):
    observed = _decode(candidate_bits)
    low, high = _endpoints(interval["low"], interval["high"])
    if observed.shape != low.shape:
        raise ValueError("Candidate/interval geometry changed")
    outside = (observed < low) | (observed > high)
    ideal = bf16_rne_bits(interval["ideal"].astype(np.float32))
    return {"outputs": int(observed.size), "outside_conditional_interval": int(outside.sum()),
            "conditional_interval_pass": bool(not outside.any()),
            "ideal_then_f32_bf16_different": int(np.count_nonzero(candidate_bits != ideal)),
            "maximum_error_from_ideal": float(np.max(np.abs(observed - interval["ideal"]), initial=0)),
            "maximum_interval_width_before_final_cast": float(np.max(interval["high"] - interval["low"], initial=0)),
            "native_arithmetic_qualified": False}


def rms_interval(input_bits, weight_bits=None, epsilon=1e-6):
    """vLLM ordinary RMSNorm: FP32 reduction, scaling, then one BF16 cast.

    BF16 square products are exactly representable in normal F32. Any F32
    reduction tree is covered by gamma-D. The source's division and rsqrt use
    the CUDA documented two-ULP allowance plus the reference rounding half ULP:
    at most 5*u32 relative error for each, including imprecise division builds.
    """
    x = _decode(input_bits)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 2816 or epsilon != 1e-6:
        raise ValueError("Unsupported RMS fixture geometry/epsilon")
    w = np.ones(x.shape[1]) if weight_bits is None else _decode(weight_bits)
    if w.shape != (x.shape[1],):
        raise ValueError("RMS learned-weight geometry changed")
    squares = x * x
    if not _normal_or_zero(squares):
        raise ValueError("RMS square product is outside the normal F32 lane")
    d = x.shape[1]
    total = np.sum(squares, axis=1, keepdims=True)
    g64, g32 = gamma(d, U64), gamma(d, U32)
    upper_total = np.nextafter(total / (1 - g64), np.inf)
    error = np.nextafter((g64 + g32) * upper_total, np.inf)
    if np.any(upper_total * (1 + g32) > F32.max):
        raise ValueError("RMS reduction may overflow")
    eps = float(np.float32(epsilon))
    low_den = np.maximum(0, total - error) / d * (1 - 5 * U32)
    high_den = (total + error) / d * (1 + 5 * U32)
    low_den = (low_den + eps) * (1 - U32)
    high_den = (high_den + eps) * (1 + U32)
    if np.any(low_den <= 0) or not _normal_or_zero(low_den) or not _normal_or_zero(high_den):
        raise ValueError("RMS inverse square root leaves the declared lane")
    # FP64 endpoint evaluation is also widened; no libm bit identity is used.
    inv_low = (1 / np.sqrt(high_den)) * (1 - 5 * U32 - gamma(4, U64))
    inv_high = (1 / np.sqrt(low_den)) * (1 + 5 * U32 + gamma(4, U64))
    factors = x * w
    round_error = gamma(1 if weight_bits is None else 2, U32) + gamma(8, U64)
    endpoints = np.stack((factors * inv_low, factors * inv_high))
    low, high = endpoints.min(axis=0), endpoints.max(axis=0)
    radius = np.maximum(np.abs(low), np.abs(high)) * round_error
    ideal = factors / np.sqrt(total / d + eps)
    if not _normal_or_zero(x * inv_low) or not _normal_or_zero(ideal):
        raise ValueError("RMS scaling is FTZ-sensitive or overflows")
    return {"low": np.nextafter(low - radius, -np.inf), "high": np.nextafter(high + radius, np.inf),
            "ideal": ideal, "reduction_gamma": g32, "conditional": True}


def rope_reference(input_bits, cache_bits):
    """Exact source arithmetic for the retained BF16-cache NeoX RoPE lane.

    Both BF16 products fit F32 exactly in the guarded lane; fused and separate
    implementations consequently give the same F32 sum, then BF16 cast.
    """
    x, cache = _decode(input_bits), _decode(cache_bits)
    if x.ndim != 2 or x.shape[1] % 256 or cache.shape != (len(x), 256):
        raise ValueError("Unsupported NeoX RoPE fixture geometry")
    heads = x.reshape(len(x), -1, 256)
    a, b = heads[..., :128], heads[..., 128:]
    c, s = cache[:, None, :128], cache[:, None, 128:]
    products = (a * c, b * s, b * c, a * s)
    if any(not _normal_or_zero(p) for p in products):
        raise ValueError("RoPE product is outside the normal F32 lane")
    sums = (products[0] - products[1], products[2] + products[3])
    if any(not _normal_or_zero(p) for p in sums):
        raise ValueError("RoPE cancellation is FTZ-sensitive or overflows")
    result = np.concatenate(sums, axis=-1).reshape(x.shape).astype(np.float32)
    return bf16_rne_bits(result)


def _exp64(values):
    """Correctly rounded Decimal exp at 80 digits, then binary64 conversion."""
    with localcontext() as ctx:
        ctx.prec = 80
        return np.array([float(Decimal.from_float(float(v)).exp()) for v in values.flat]).reshape(values.shape)


def attention_interval(query_bits, key_bits, value_bits, *, provider):
    """Conditional BF16-exp-weight attention for one causal query, <=33 keys.

    QK/PV and row sums assume faithful F32 operations (u=2^-23), full BF16
    products and no reduced-precision partial sums. FA2 uses <=ceil(T/16)
    online updates; XQA's one nonempty 256-key CTA also stores a BF16 partial
    output before the multi-CTA merge. Those source casts are included, rather
    than treating mathematical FP64 softmax as the native BF16 oracle.
    """
    q, k, v = _decode(query_bits), _decode(key_bits), _decode(value_bits)
    if provider not in {"fa2", "xqa"} or q.ndim != 2 or k.ndim != 3 or k.shape != v.shape or not 1 <= len(k) <= 33:
        raise ValueError("Unsupported attention provider/fixture geometry")
    if q.shape[1] != k.shape[2] or q.shape[0] % k.shape[1]:
        raise ValueError("Grouped-query geometry changed")
    mapping = np.arange(len(q)) // (len(q) // k.shape[1])
    kg, vg = k[:, mapping], v[:, mapping]
    products = q[None, :, :] * kg
    if not _normal_or_zero(products):
        raise ValueError("Attention QK product is FTZ-sensitive or overflows")
    dim, count = q.shape[1], len(k)
    score = np.sum(products, axis=-1).T  # scale=1.0, no sink or soft cap
    sumabs = np.sum(np.abs(products), axis=-1).T
    g64 = gamma(dim, U64)
    score_error = (gamma(2 * dim + 2, 2 * U32) + g64) * np.nextafter(sumabs / (1 - g64), np.inf)
    maximum = score.max(axis=1, keepdims=True)
    maxabs = np.max(np.abs(score), axis=1, keepdims=True) + score_error.max(axis=1, keepdims=True)
    phases = math.ceil(count / 16) + 1 if provider == "fa2" else 3
    # Six ordinary F32 operations per bias/rescale phase cover log2(e), its
    # multiplication/subtraction, and subsequent weight rescaling. The exp
    # allowance covers documented exp2f and also fast-math __expf in rescaling.
    exp_relative = (5 + 2 * np.ceil(1.173 * (2 * maxabs + 1))) * U32
    exponent_error = score_error + score_error.max(axis=1, keepdims=True)
    exponent_error += gamma(6 * phases, U32) * (2 * maxabs + 1) + phases * exp_relative
    shifted = score - maximum
    if np.any(shifted - exponent_error < math.log(F32.tiny)):
        raise ValueError("Attention exponent/weight can enter the FTZ lane")
    exp_low = _exp64(shifted - exponent_error) * (1 - UBF16)
    exp_high = _exp64(shifted + exponent_error) * (1 + UBF16)
    reduction_error = gamma(2 * count + 2, 2 * U32) + gamma(2 * count + 8, U64)
    denom_low = exp_low.sum(axis=1, keepdims=True) * (1 - reduction_error)
    denom_high = exp_high.sum(axis=1, keepdims=True) * (1 + reduction_error)
    values = vg.transpose(1, 0, 2)
    nlow = np.sum(np.where(values >= 0, exp_low[..., None], exp_high[..., None]) * values, axis=1)
    nhigh = np.sum(np.where(values >= 0, exp_high[..., None], exp_low[..., None]) * values, axis=1)
    nradius = reduction_error * np.sum(exp_high[..., None] * np.abs(values), axis=1)
    candidates = np.stack(((nlow - nradius) / denom_low, (nlow - nradius) / denom_high,
                           (nhigh + nradius) / denom_low, (nhigh + nradius) / denom_high))
    low, high = candidates.min(axis=0), candidates.max(axis=0)
    quotient_error = gamma(4, U32) + gamma(8, U64)
    radius = np.maximum(np.abs(low), np.abs(high)) * quotient_error
    if provider == "xqa":
        # Exactly one nonempty CTA for T<=33<256; all other partials are zero.
        # Its normalized partial is stored in BF16 and converted back to F32.
        radius += np.maximum(np.abs(low), np.abs(high)) * UBF16
        radius += np.maximum(np.abs(low), np.abs(high)) * quotient_error
    probability = _exp64(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    ideal = np.sum(probability[..., None] * values, axis=1)
    if not _normal_or_zero(exp_low[..., None] * values) or not _normal_or_zero(ideal):
        raise ValueError("Attention PV/reference leaves the normal F32 lane")
    return {"low": np.nextafter(low - radius, -np.inf), "high": np.nextafter(high + radius, np.inf),
            "ideal": ideal, "score_error_max": float(score_error.max()),
            "weight_bf16_casts": 1, "partial_output_bf16_casts": int(provider == "xqa"),
            "provider": provider, "conditional": True}


def validate_binding(case):
    binding = case["binding"]
    if binding.get("schema") != 2:
        raise ValueError("The norm/attention contract requires schema-2 boundaries")
    for name in NORM_NAMES:
        spec = binding["norms"][name]
        if spec["epsilon"] != 1e-6 or spec["variance_size"] is not None or spec["pass_weight"] != (name != "v_norm"):
            raise ValueError("Ordinary RMSNorm parameters changed")
        if name != "v_norm":
            value = case["constants"][name + "_weight_bits"]
            if value.dtype != np.uint16 or hashlib.sha256(value.tobytes()).hexdigest() != ORIGINAL_NORM_SHA256[name]:
                raise ValueError("Loaded norm differs from the original frozen tensor: " + name)
        for frame in binding["frames"]:
            dispatch = frame["dispatch"][name]
            if dispatch["provider"] != "vllm_c" or dispatch["implementation"] != {
                "module": "vllm.kernels.vllm_c", "qualname": "rms_norm",
                "source_sha256": "ef37e71736807026f0b8ad187155b63f147984c03d287a78790ef8f8ed642ec1"}:
                raise ValueError("Native ordinary RMS provider changed")
    rope = binding["rope"]
    if rope["use_flashinfer"] is not False or rope["is_neox_style"] is not True or rope["head_size"] != 256 or rope["rotary_dim"] != 256:
        raise ValueError("Pinned BF16-cache NeoX RoPE lane changed")


def span_launches(events, span):
    """Bind asynchronous kernels via runtime launch correlation, including
    launches whose profiler parent is a user annotation rather than cpu_op.
    GPU timestamps need not overlap the CPU span.
    """
    if any(type(span.get(k)) not in {int, float} or not math.isfinite(span[k]) for k in ("ts", "dur")) or span["dur"] <= 0:
        raise ValueError("Invalid operator span coordinates")
    correlations = set()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        if event.get("pid") == span.get("pid") and event.get("tid") == span.get("tid") and span["ts"] <= event.get("ts", -math.inf) < span["ts"] + span["dur"]:
            correlation = event.get("args", {}).get("correlation")
            if type(correlation) is int:
                correlations.add(correlation)
    launches = [event for event in events if event.get("ph") == "X" and event.get("cat") == "kernel"
                and type(event.get("args", {}).get("correlation")) is int
                and event["args"]["correlation"] in correlations]
    return launches


def require_launch_profile(events, span, profile):
    main = [e for e in span_launches(events, span) if hashlib.sha256(e["name"].encode()).hexdigest() == profile["kernel_name_sha256"]]
    if len(main) != 1 or any(main[0]["args"][key] != profile[key] for key in ("grid", "block")):
        raise ValueError("Replayed attention kernel/launch differs from retained provider")


def bind_attention_trace(root, path):
    directory = Path(root) / "controlled" / path
    manifest = layer0.evidence.read_json(directory / "controlled-manifest.json")
    trace = layer0.evidence.local_file(directory, "controlled-trace.json.gz")
    if layer0.evidence.digest_file(trace, layer0.evidence.MAX_TRACE) != manifest["cuda_trace_sha256"]:
        raise ValueError("Retained attention trace digest changed")
    with gzip.open(trace, "rb") as source:
        raw = source.read(layer0.evidence.MAX_TRACE + 1)
    if len(raw) > layer0.evidence.MAX_TRACE:
        raise ValueError("Attention trace exceeds the bounded read")
    events = json.loads(raw)["traceEvents"]
    spans = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name") == "megartx.layer0.attention"]
    expected = [("fa2", [2, 1, 8], [32, 4, 1])] if path == "full" else [("fa2", [1, 1, 8], [32, 4, 1]), ("xqa", [21, 8, 1], [128, 1, 2])]
    if len(spans) != len(expected):
        raise ValueError("Frozen attention span count changed")
    records = []
    for span, (provider, grid, block) in zip(sorted(spans, key=lambda e: e["ts"]), expected):
        launches = span_launches(events, span)
        main = [e for e in launches if ("BatchPrefillWithPagedKVCacheKernel" in e["name"] if provider == "fa2" else e["name"].startswith("kernel_mha("))]
        if len(main) != 1 or main[0]["args"]["grid"] != grid or main[0]["args"]["block"] != block:
            raise ValueError("Retained attention provider/launch geometry differs from the declared lane")
        name = main[0]["name"]
        if provider == "fa2" and ("__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float" not in name or "DefaultAttention<false, true, false, false>" not in name):
            raise ValueError("Retained FA2 precision/variant changed")
        if provider == "xqa" and "Vec<__nv_bfloat16, 256u>" not in name:
            raise ValueError("Retained XQA precision/head dimension changed")
        records.append({"provider": provider, "grid": grid, "block": block,
                        "kernel_name_sha256": hashlib.sha256(name.encode()).hexdigest()})
    return {"trace_sha256": manifest["cuda_trace_sha256"], "launches": records}


def analyze_case(case, path):
    validate_binding(case)
    reports, intervals = {}, {}
    for position in (31, 32):
        row = case["rows"][position]
        norms = {}
        for name in NORM_NAMES:
            width = 256 if name in {"q_norm", "k_norm", "v_norm"} else 2816
            weight = None if name == "v_norm" else case["constants"][name + "_weight_bits"]
            norms[name] = check_interval(row[name + "_out"].reshape(-1, width),
                                         rms_interval(row[name + "_in"].reshape(-1, width), weight))
        ropes = {}
        for name in ("q", "k"):
            expected = rope_reference(row["rope_" + name + "_in"][None, :], row["rope_cache_bits"][None, :])
            ropes[name] = layer0.bit_difference(expected, row["rope_" + name + "_out"][None, :])
        prefix = case["prefixes"][-1]
        key = prefix["stored_prefix_k"][:position + 1].reshape(-1, 8, 256)
        value = prefix["stored_prefix_v"][:position + 1].reshape(-1, 8, 256)
        provider = "xqa" if path == "cached" and position == 32 else "fa2"
        interval = attention_interval(row["attention_q"].reshape(16, 256), key, value, provider=provider)
        attention = check_interval(row["attention_out"].reshape(16, 256), interval)
        attention.update({k: interval[k] for k in ("provider", "score_error_max", "weight_bf16_casts", "partial_output_bf16_casts")})
        reports[str(position)] = {"norms": norms, "rope": ropes, "attention": attention}
        intervals[position] = interval
    return reports, intervals


def analyze_pair(full, cached):
    for key in ("token_sha256", "schedule_sha256", "loaded_q_sha256", "loaded_k_sha256", "loaded_v_sha256"):
        if full["binding"][key] != cached["binding"][key]:
            raise ValueError("Paired frozen provenance changed: " + key)
    if full["constants"].keys() != cached["constants"].keys() or any(
        not np.array_equal(value, cached["constants"][key]) for key, value in full["constants"].items()
    ):
        raise ValueError("Paired loaded constants changed")
    left, li = analyze_case(full, "full")
    right, ri = analyze_case(cached, "cached")
    pair = {}
    for p in (31, 32):
        a = _decode(full["rows"][p]["attention_out"]).reshape(16, 256)
        b = _decode(cached["rows"][p]["attention_out"]).reshape(16, 256)
        alo, ahi = _endpoints(li[p]["low"], li[p]["high"])
        blo, bhi = _endpoints(ri[p]["low"], ri[p]["high"])
        difference_low, difference_high = alo - bhi, ahi - blo
        actual = a - b
        pair[str(p)] = {"maximum_ideal_input_sensitivity": float(np.max(np.abs(li[p]["ideal"] - ri[p]["ideal"]))),
                       "maximum_observed_attention_difference": float(np.max(np.abs(actual))),
                       "outside_combined_conditional_difference_interval": int(np.count_nonzero((actual < difference_low) | (actual > difference_high)))}
    passed = all(r["conditional_interval_pass"] for path in (left, right) for row in path.values()
                 for r in (*row["norms"].values(), row["attention"]))
    ropes_exact = all(r["raw_bits_equal"] for path in (left, right) for row in path.values() for r in row["rope"].values())
    controls = {}
    for path, case, intervals in (("full", full, li), ("cached", cached, ri)):
        p = 32
        row, prefix = case["rows"][p], case["prefixes"][-1]
        q = row["attention_q"].reshape(16, 256)
        k = prefix["stored_prefix_k"].reshape(33, 8, 256)
        v = prefix["stored_prefix_v"].reshape(33, 8, 256)
        provider = intervals[p]["provider"]
        wrong_scale = bf16_rne_bits((bf16_decode(q) / 16).astype(np.float32))
        for label, qi, ki, vi in (("wrong_scale_1_over_sqrt_d", wrong_scale, k, v),
                                 ("wrong_gqa_head", q, np.roll(k, 1, axis=1), np.roll(v, 1, axis=1)),
                                 ("v_only_row_permutation", q, k, np.roll(v, 1, axis=0))):
            wrong = attention_interval(qi, ki, vi, provider=provider)
            wrong_bits = bf16_rne_bits(wrong["ideal"].astype(np.float32))
            controls[path + "_" + label] = check_interval(wrong_bits, intervals[p])["outside_conditional_interval"]
    return {"schema": 1, "full": left, "cached": right, "pair": pair, "negative_controls_rejected_outputs": controls,
            "conditional_selected_norm_attention_compatible": passed and ropes_exact and all(controls.values()),
            "native_mma_or_transcendental_qualified": False,
            "whole_layer_accepted": False, "whole_model_accepted": False,
            "natural_routing_quality_accepted": False, "timing_qualified": False,
            "scope": "Frozen layer-0 selected boundaries with exact provenance; declared source arithmetic only",
            "assumptions": ["Faithful F32 QK/PV and row-sum arithmetic; full BF16 products; no reduced-precision partials",
                            "Source-declared BF16 exponent weights, ordinary RMS, NeoX RoPE, scale 1, contiguous GQA grouping",
                            "CUDA documented transcendental error allowances; normal guarded F32 lane",
                            "XQA <=33 keys in one nonempty 256-key CTA; FA2 causal 16-key online tiles"]}


def analyze_replay(directory, full, cached, retained_dispatch):
    """Re-read every repeat and its bound launch; do not trust report pass flags."""
    directory = Path(directory)
    if layer0.evidence.read_bytes(directory.parent / (directory.name + ".exit"), 32).strip() != b"0":
        raise ValueError("Operator replay did not exit successfully")
    report = layer0.evidence.read_json(directory / "report.json")
    if report["sources"] != REPLAY_SOURCE_SHA256 or report["repeats"] != 3 or report["torch"] != "2.13.0+cu130" or report["cuda"] != "13.0" or report.get("xqa_table_capacity_tokens") != 8448:
        raise ValueError("Operator replay source/runtime/repetition binding changed")
    if report["retained_dispatch"] != retained_dispatch:
        raise ValueError("Replay does not bind the independently re-read original trace")
    expected, profiles = {}, {}
    for path, case, count, positions in (("full", full, 33, [31, 32]), ("cached", cached, 32, [31]), ("cached", cached, 1, [32])):
        label = path + "_m" + str(count)
        for name in NORM_NAMES:
            expected[label + "_" + name] = np.stack([case["rows"][p][name + "_out"] for p in positions])
        expected[label + "_rope"] = np.stack([np.concatenate((case["rows"][p]["rope_q_out"], case["rows"][p]["rope_k_out"])) for p in positions])
    for label, case, positions, profile in (("full_fa2", full, [31, 32], retained_dispatch["full"]["launches"][0]),
                                          ("cached_fa2_m32", cached, [31], retained_dispatch["cached"]["launches"][0]),
                                          ("cached_xqa", cached, [32], retained_dispatch["cached"]["launches"][1])):
        expected[label] = np.stack([case["rows"][p]["attention_out"].reshape(16, 256) for p in positions])
        profiles[label] = profile
    for label, provider in (("cached_inputs_fa2", "fa2"), ("full_inputs_xqa", "xqa"), ("constant_fa2", "fa2"), ("constant_xqa", "xqa")):
        profiles[label] = retained_dispatch["full"]["launches"][0] if provider == "fa2" else retained_dispatch["cached"]["launches"][1]
    labels = set(expected) | set(profiles)
    fields = {label + "_r" + str(i) for label in labels for i in range(3)}
    arrays = layer0.evidence.read_npz(directory, "results.npz", report["results_sha256"], fields)
    for label in labels:
        values = [arrays[label + "_r" + str(i)] for i in range(3)]
        if any(v.dtype != np.uint16 for v in values) or any(not np.array_equal(values[0], v) for v in values[1:]):
            raise ValueError("Operator repeats differ or lost raw BF16 dtype")
        if label in expected and not np.array_equal(values[0], expected[label]):
            raise ValueError("Operator replay differs from retained boundary: " + label)
        if label.startswith("constant_") and (values[0].shape != (1, 16, 256) or not np.all(values[0] == 0x3f80)):
            raise ValueError("Exact constant attention control failed")
    trace = directory / "trace.json.gz"
    if layer0.evidence.digest_file(trace, layer0.evidence.MAX_TRACE) != report["trace_sha256"]:
        raise ValueError("Operator replay trace digest changed")
    with gzip.open(trace, "rb") as source:
        raw = source.read(layer0.evidence.MAX_TRACE + 1)
    if len(raw) > layer0.evidence.MAX_TRACE:
        raise ValueError("Operator replay trace exceeds bounded read")
    events = json.loads(raw)["traceEvents"]
    for label, profile in profiles.items():
        for i in range(3):
            name = "megartx.norm_attention." + label + ".r" + str(i)
            spans = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name") == name]
            if len(spans) != 1:
                raise ValueError("Operator replay annotation is absent or duplicated")
            require_launch_profile(events, spans[0], profile)
    cross_checks = {}
    for label, case, provider in (("cached_inputs_fa2", cached, "fa2"), ("full_inputs_xqa", full, "xqa")):
        row, prefix = case["rows"][32], case["prefixes"][-1]
        interval = attention_interval(row["attention_q"].reshape(16, 256), prefix["stored_prefix_k"].reshape(33, 8, 256),
                                      prefix["stored_prefix_v"].reshape(33, 8, 256), provider=provider)
        cross_checks[label] = check_interval(arrays[label + "_r0"][0], interval)
    if any(not r["conditional_interval_pass"] for r in cross_checks.values()):
        raise ValueError("Crossed provider output falls outside the predeclared conditional model")
    comparisons = {
        "same_full_inputs_provider_difference": (arrays["full_fa2_r0"][1:], arrays["full_inputs_xqa_r0"]),
        "same_cached_inputs_provider_difference": (arrays["cached_inputs_fa2_r0"], arrays["cached_xqa_r0"]),
        "fa2_input_sensitivity": (arrays["full_fa2_r0"][1:], arrays["cached_inputs_fa2_r0"]),
        "xqa_input_sensitivity": (arrays["full_inputs_xqa_r0"], arrays["cached_xqa_r0"]),
    }
    return {"repeats": 3, "raw_boundary_reproduction": True, "constant_controls_pass": True,
            "attention_launches_bound": len(profiles) * 3, "crossed_conditional_checks": cross_checks,
            "comparisons": {key: layer0.bit_difference(*values) for key, values in comparisons.items()},
            "results_sha256": report["results_sha256"], "trace_sha256": report["trace_sha256"],
            "peak_allocated_bytes": report["peak_allocated_bytes"], "timing_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--cached", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    plan = layer0.evidence.load_plan(args.plan)
    full, cached = layer0.load_case(args.full, "full", plan), layer0.load_case(args.cached, "cached", plan)
    # Reuse all paired provenance/constant/copy guards; no original-weight
    # qualification is implied by the CPU-only norm/attention diagnostic.
    for key in full["constants"]:
        if not np.array_equal(full["constants"][key], cached["constants"][key]):
            raise ValueError("Paired loaded constants changed")
    report = analyze_pair(full, cached)
    report["retained_dispatch"] = {"full": bind_attention_trace(args.full, "full"), "cached": bind_attention_trace(args.cached, "cached")}
    if args.replay:
        report["operator_replay"] = analyze_replay(args.replay, full, cached, report["retained_dispatch"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("conditional_selected_norm_attention_compatible", "whole_layer_accepted", "whole_model_accepted")}))


if __name__ == "__main__":
    main()
