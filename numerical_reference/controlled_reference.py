"""CPU-only controlled-route guards and captured-operand numerical replay.

Uses the existing independent NVFP4 format oracle, never production helpers.
Exact paired comparisons and conditional numerical diagnostics are kept
separate; this module cannot grant model, native-kernel or quality acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import nvfp4_reference as format_ref


TARGETS = ((0, 42), (0, 82), (1, 126), (2, 89), (3, 7), (5, 12))
INPUT_TOKENS = 33
BASE_IDS = (0, 1, 2, 3, 4, 5, 6, 8)
TARGET_POSITIONS = {(0, 42): 31, (0, 82): 32, (1, 126): 32,
                    (2, 89): 32, (3, 7): 32, (5, 12): 32}
TARGET_SLOTS = {(0, 42): 7, (0, 82): 7, (1, 126): 7,
                (2, 89): 7, (3, 7): 7, (5, 12): 7}
CONFIG_SHA256 = "4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98"
CONTROLLED_ORIGIN = {"route_origin": "controlled", "routing_intervention": True,
                     "scope": "controlled_routing_fixture"}


def _controlled_origin(record):
    if any(record.get(key) != value or type(record.get(key)) is not type(value)
           for key, value in CONTROLLED_ORIGIN.items()):
        raise ValueError("Controlled origin is missing or inconsistent; this is not natural coverage")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _array(value, dtype, shape, name):
    value = np.asarray(value)
    if value.dtype != np.dtype(dtype) or value.shape != tuple(shape):
        raise ValueError(f"{name} has the wrong captured dtype/shape")
    return value


def _f32_scalar(bits, name):
    value = _array(bits, np.uint32, (1,), name).view(np.float32)[0]
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite F32 scalar")
    return value


def _f32_bits(value):
    return np.asarray([value], dtype=np.float32).view(np.uint32)


def bf16_values(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise ValueError("BF16 storage must be uint16 raw bits")
    result = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite captured BF16 value")
    return result


def bf16_bits(value):
    rounded = format_ref.round_bf16(value)
    if not np.isfinite(rounded).all():
        raise ValueError("BF16 cast overflows or is nonfinite")
    return (rounded.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def token_sha256(tokens):
    tokens = np.asarray(tokens)
    if tokens.dtype != np.int64 or tokens.ndim != 1 or not 1 <= len(tokens) <= 1033:
        raise ValueError("Expected one bounded immutable I64 token sequence")
    if np.any(tokens < 0) or np.any(tokens >= 262144):
        raise ValueError("Token ID outside the frozen vocabulary")
    return hashlib.sha256(tokens.astype("<i8").tobytes()).hexdigest()


def schedule_sha256(schedule):
    digest = hashlib.sha256()
    _controlled_origin(schedule)
    digest.update(json.dumps(CONTROLLED_ORIGIN, sort_keys=True, separators=(",", ":")).encode() + b"\0")
    for name, dtype in (("tokens", "<i8"), ("ids", "<i4"), ("weight_bits", "<u4")):
        array = np.asarray(schedule[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.astype(dtype).tobytes())
    return digest.hexdigest()


def validate_schedule(schedule):
    _controlled_origin(schedule)
    tokens = np.asarray(schedule["tokens"])
    token_sha256(tokens)
    if len(tokens) != INPUT_TOKENS:
        raise ValueError("Minimal controlled experiment requires exactly 33 supplied input tokens")
    ids = _array(schedule["ids"], np.int32, (30, len(tokens), 8), "frozen route IDs")
    weight_bits = _array(schedule["weight_bits"], np.uint32, ids.shape, "frozen route weight bits")
    weights = weight_bits.view(np.float32)
    if np.any(ids < 0) or np.any(ids >= 128):
        raise ValueError("Frozen route expert IDs are not local original IDs")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate expert IDs in a frozen top-eight row")
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(np.sum(weights, axis=-1) <= 0):
        raise ValueError("Invalid frozen F32 route weights")
    expected = np.broadcast_to(np.asarray(BASE_IDS, dtype=np.int32), ids.shape).copy()
    for layer, expert in TARGETS:
        expected[layer, TARGET_POSITIONS[layer, expert], TARGET_SLOTS[layer, expert]] = expert
    if not np.array_equal(ids, expected):
        raise ValueError("Route table differs from the predeclared deterministic ordinary/intervention IDs")
    if not np.array_equal(weight_bits, np.full(ids.shape, _f32_bits(0.125)[0], dtype=np.uint32)):
        raise ValueError("Route weights differ from the predeclared F32 one-eighth weights")
    if schedule.get("token_sha256") != token_sha256(tokens):
        raise ValueError("Frozen token digest differs")
    if schedule.get("schedule_sha256") != schedule_sha256(schedule):
        raise ValueError("Frozen route table digest differs")
    return schedule


def make_schedule(tokens):
    """Declare a deterministic intervention; no natural master GPU pass is needed."""
    tokens = np.asarray(tokens)
    token_sha256(tokens)
    if len(tokens) != INPUT_TOKENS:
        raise ValueError("Minimal controlled experiment requires exactly 33 supplied input tokens")
    ids = np.broadcast_to(np.asarray(BASE_IDS, dtype=np.int32), (30, len(tokens), 8)).copy()
    weights = np.full(ids.shape, _f32_bits(0.125)[0], dtype=np.uint32)
    for layer, expert in TARGETS:
        ids[layer, TARGET_POSITIONS[layer, expert], TARGET_SLOTS[layer, expert]] = expert
    result = {**CONTROLLED_ORIGIN, "tokens": tokens.copy(), "ids": ids, "weight_bits": weights,
              "token_sha256": token_sha256(tokens)}
    result["schedule_sha256"] = schedule_sha256(result)
    return validate_schedule(result)


def validate_dispatch(schedule, dispatch):
    """Validate reported dispatch/execution records; no live CUDA claim follows."""
    validate_schedule(schedule)
    _controlled_origin(dispatch)
    if dispatch["token_sha256"] != schedule["token_sha256"] or dispatch["schedule_sha256"] != schedule["schedule_sha256"]:
        raise ValueError("Dispatched prefix/route-table identity differs")
    if dispatch["mode"] not in {"native", "paired_reference", "gate_only_negative_control"}:
        raise ValueError("Unknown controlled arithmetic mode")
    if not _sha256(dispatch["execution_proof_sha256"]) or not _sha256(dispatch["cuda_trace_sha256"]):
        raise ValueError("Reported dispatch is missing fixed execution proof/trace hashes")
    layers = _array(dispatch["layers"], np.int32, (30,), "dispatched layer ordinals")
    if not np.array_equal(layers, np.arange(30, dtype=np.int32)):
        raise ValueError("Dispatched layer identity is missing, duplicate or out of order")
    positions = _array(dispatch["positions"], np.int64, (INPUT_TOKENS,), "dispatched absolute positions")
    if not np.array_equal(np.sort(positions), np.arange(INPUT_TOKENS, dtype=np.int64)):
        raise ValueError("Missing, duplicate or unexpected dispatched position")
    tokens = _array(dispatch["tokens"], np.int64, positions.shape, "dispatched token IDs")
    if not np.array_equal(tokens, schedule["tokens"][positions]):
        raise ValueError("Dispatched token IDs differ at actual positions")
    ids = _array(dispatch["ids"], np.int32, (30, INPUT_TOKENS, 8), "actual dispatched IDs")
    bits = _array(dispatch["weight_bits"], np.uint32, ids.shape, "actual dispatched weight bits")
    if not np.array_equal(ids, schedule["ids"][:, positions]) or not np.array_equal(bits, schedule["weight_bits"][:, positions]):
        raise ValueError("Actual dispatched IDs/weight bits differ from the frozen table")
    hits = dispatch["executed_interventions"]
    expected = {(layer, expert, TARGET_POSITIONS[layer, expert]) for layer, expert in TARGETS}
    observed = []
    for hit in hits:
        _controlled_origin(hit)
        if hit["mode"] != dispatch["mode"] or hit["completed"] is not True or hit["finite_output"] is not True or hit["registered_binding_verified"] is not True:
            raise ValueError("Intervention record lacks consistent completed/finite/registered execution assertions")
        if type(hit["nonzero_output_elements"]) is not int or not 1 <= hit["nonzero_output_elements"] <= 2816:
            raise ValueError("Intervention record lacks bounded nonzero completed output")
        if hit["execution_proof_sha256"] != dispatch["execution_proof_sha256"] or hit["cuda_trace_sha256"] != dispatch["cuda_trace_sha256"] or not _sha256(hit["stage_capture_sha256"]):
            raise ValueError("Intervention record differs from reported proof/trace/stage hashes")
        if any(type(hit[key]) is not int for key in ("layer", "expert", "position", "slot", "weight_bits")):
            raise ValueError("Intervention identities/weight bits must be exact integer fields")
        key = (hit["layer"], hit["expert"], hit["position"])
        if key not in expected or hit["slot"] != TARGET_SLOTS[key[:2]]:
            raise ValueError("Executed correction differs from the predeclared expert/position/slot")
        layer, expert, position = key
        if hit["weight_bits"] != int(schedule["weight_bits"][layer, position, hit["slot"]]):
            raise ValueError("Executed correction weight differs from its dispatched slot")
        observed.append(key)
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Missing or duplicate positive executed correction coverage")
    return {**CONTROLLED_ORIGIN, "all_reported_dispatched_rows_exact": True,
            "reported_completed_intervention_records": len(expected),
            "independent_execution_proof_trace_stage_verification": False,
            "model_or_natural_quality_qualified": False}


def original_expert(reader, layer, expert):
    """Read eighteen original projection triplets at most, one expert per call."""
    if (layer, expert) not in TARGETS:
        raise ValueError("Independent checkpoint reads are confined to the six affected experts")
    projections, hashes = {}, {}
    for label in ("gate", "up", "down"):
        prefix = f"model.language_model.layers.{layer}.experts.{expert}.{label}_proj"
        projections[label], hashes[label] = reader.projection(prefix)
    return projections, hashes


def _compare_bits(candidate, expected):
    candidate_values, expected_values = bf16_values(candidate), bf16_values(expected)
    if candidate.shape != expected.shape:
        raise ValueError("Captured/reference BF16 shapes differ")
    delta = candidate_values.astype(np.float64) - expected_values.astype(np.float64)
    different = np.argwhere(candidate != expected)
    return {"raw_bf16_bits_equal": len(different) == 0,
            "bf16_differing_elements": len(different),
            "first_differing_coordinates": different[:16].tolist(),
            "max_absolute_difference": float(np.abs(delta).max(initial=0)),
            "numerical_values_equal": bool(np.array_equal(candidate_values, expected_values))}


def replay_projection(q, physical_sf, original, *, input_global_bits,
                      alpha_bits, candidate_bits):
    """Original-weight NumPy replay from actual native q/SF; no quantizer sharing hidden."""
    q = np.asarray(q)
    if q.dtype != np.uint8 or q.ndim != 2 or not 1 <= q.shape[0] <= 3 or not 16 <= q.shape[1] * 2 <= 2816 or q.shape[1] * 2 % 16:
        raise ValueError("Captured projection requires bounded packed uint8 rows")
    if physical_sf.dtype != np.uint8 or physical_sf.ndim not in (1, 2):
        raise ValueError("Captured physical activation SF must retain raw E4M3 uint8 bytes")
    sf = format_ref.unswizzle_128x4(physical_sf, q.shape[0], q.shape[1] * 2 // 16)
    packed, weight_sf, weight_global = original
    if packed.dtype != np.uint8 or packed.ndim != 2 or not 1 <= packed.shape[0] <= 2816 or packed.shape[1] != q.shape[1]:
        raise ValueError("Original projection exceeds bounded N/K or has the wrong captured format")
    activation_global = _f32_scalar(input_global_bits, "activation dequant global")
    alpha = _f32_scalar(alpha_bits, "actual projection alpha")
    expected_alpha = np.float32(np.float32(weight_global) * activation_global)
    if not np.array_equal(alpha_bits, _f32_bits(expected_alpha)):
        raise ValueError("Actual alpha does not preserve the original projection/global product")
    reference = format_ref.gemm_reference(q, sf, packed, weight_sf, alpha_fp32=float(alpha), output_chunk_rows=64)
    expected = bf16_bits(reference.expected_fp32)
    candidate = _array(candidate_bits, np.uint16, expected.shape, "captured projection output")
    comparison = _compare_bits(candidate, expected)
    # Inflate F64 sumabs before using it in a bound, independent of candidate errors.
    k = q.shape[1] * 2
    g32, g64 = format_ref.gamma(k, 2 ** -24), format_ref.gamma(k, 2 ** -53)
    sum_upper = np.nextafter(reference.sum_absolute_products / (1 - g64), np.inf)
    dot_error = np.nextafter((g32 + g64) * sum_upper, np.inf)
    alpha_error = np.nextafter(abs(float(alpha)) * dot_error, np.inf)
    rounding_error = np.nextafter((2 ** -24) * (np.abs(reference.ideal_scaled_fp64) + alpha_error) + 2 ** -150, np.inf)
    # Account for this F64 epilogue/interval calculation as well as dot reduction.
    arithmetic_error = np.nextafter(format_ref.gamma(8, 2 ** -53) *
                                    (np.abs(reference.ideal_scaled_fp64) + alpha_error + rounding_error), np.inf)
    error = np.nextafter(alpha_error + rounding_error + arithmetic_error, np.inf)
    lower = format_ref.round_bf16(np.nextafter(reference.ideal_scaled_fp64 - error, -np.inf))
    upper = format_ref.round_bf16(np.nextafter(reference.ideal_scaled_fp64 + error, np.inf))
    observed = bf16_values(candidate)
    outside = (observed < lower) | (observed > upper)
    comparison.update({**CONTROLLED_ORIGIN, "conditional_f32_envelope_pass": not bool(outside.any()),
                       "outside_conditional_envelope": int(outside.sum()),
                       "gamma_k_f32": g32, "gamma_k_f64": g64,
                       "native_mma_qualified": False,
                       "bound_scope": "Conditional F32 RNE accumulation/epilogue; PTX MMA rounding remains unqualified",
                       "effective_sf_layout": "independent coordinate inverse of 128x4"})
    return {"comparison": comparison, "expected_bits": expected,
            "linear_activation_sf": sf, "expected_alpha_bits": _f32_bits(expected_alpha)}


def quantizer_diagnostics(input_bits, *, input_global_bits, actual_quant_global_bits, packed, physical_sf):
    """Independent named IEEE profiles; discrepancies never fit a CUDA tolerance."""
    x = bf16_values(input_bits)
    if x.ndim != 2 or not 1 <= x.shape[0] <= 3 or not 16 <= x.shape[1] <= 2816 or x.shape[1] % 16:
        raise ValueError("Quantizer replay exceeds the fixed M/K fixture bounds")
    global_a = _f32_scalar(input_global_bits, "activation dequant global")
    quant_g = _f32_scalar(actual_quant_global_bits, "actual activation quant multiplier")
    if not np.array_equal(actual_quant_global_bits, _f32_bits(np.float32(np.float32(1) / global_a))):
        raise ValueError("Actual quant multiplier differs from the explicit F32 reciprocal")
    sf = format_ref.unswizzle_128x4(physical_sf, x.shape[0], x.shape[1] // 16)
    actual_effective = format_ref.decode_qsf(packed, sf)
    report = {}
    for profile in ("vllm_python_rn", "flashinfer_strict_rn"):
        predicted = format_ref.quantize_activation_rn(x, float(quant_g), profile=profile)
        effective = format_ref.decode_qsf(predicted.packed, predicted.linear_sf)
        report[profile] = {"packed_bytes_equal": bool(np.array_equal(packed, predicted.packed)),
                           "sf_bytes_equal": bool(np.array_equal(sf, predicted.linear_sf)),
                           "decoded_units_equal": bool(np.array_equal(actual_effective, effective)),
                           "max_decoded_difference": float(np.abs(actual_effective - effective).max(initial=0))}
    report.update({**CONTROLLED_ORIGIN, "cuda_quantizer_qualified": False})
    return report


def replay_expert(stages, originals, *, a1_bits, a2_bits, mode):
    """Localize gate/up/down with each mode's actual inputs, not assumed equal states."""
    if mode not in {"native", "paired_reference", "gate_only_negative_control"}:
        raise ValueError("Unknown controlled replay mode")
    result = {}
    for label in ("gate", "up", "down"):
        projection = originals[label]
        if label == "up" and mode == "gate_only_negative_control":
            projection = (projection[0], projection[1], originals["gate"][2])
        number = 2 if label == "down" else 1
        replay = replay_projection(stages[f"q{number}"], stages[f"sf{number}"], projection,
                                   input_global_bits=a2_bits if number == 2 else a1_bits,
                                   alpha_bits=stages[label + "_alpha_bits"],
                                   candidate_bits=stages[label + "_bits"])
        result[label] = replay["comparison"]
    result["a1_quantizer_profiles"] = quantizer_diagnostics(
        stages["input_bits"], input_global_bits=a1_bits, actual_quant_global_bits=stages["quant1_global_bits"],
        packed=stages["q1"], physical_sf=stages["sf1"])
    result["a2_quantizer_profiles"] = quantizer_diagnostics(
        stages["activation_bits"], input_global_bits=a2_bits, actual_quant_global_bits=stages["quant2_global_bits"],
        packed=stages["q2"], physical_sf=stages["sf2"])
    semantic = format_ref.gelu_tanh_semantic_fp64(bf16_values(stages["gate_bits"]))
    mathematical = bf16_bits(semantic * bf16_values(stages["up_bits"]).astype(np.float64))
    result["mathematical_activation_diagnostic"] = _compare_bits(stages["activation_bits"], mathematical)
    result["native_activation_qualified"] = False
    result["whole_expert_or_model_qualified"] = False
    result.update(CONTROLLED_ORIGIN)
    return result


def _aligned_logit_rows(snapshot, schedule, expected_vocab):
    _controlled_origin(snapshot)
    if snapshot["token_sha256"] != schedule["token_sha256"] or snapshot["schedule_sha256"] != schedule["schedule_sha256"]:
        raise ValueError("Logit prefix/route-table identity differs")
    positions = np.asarray(snapshot["input_positions"])
    if positions.dtype != np.int64 or positions.ndim != 1 or not 1 <= len(positions) <= 6:
        raise ValueError("Logit rows require bounded I64 absolute input positions")
    if len(set(positions.tolist())) != len(positions) or np.any(positions < 0) or np.any(positions >= INPUT_TOKENS):
        raise ValueError("Duplicate or out-of-scope logit input positions")
    if snapshot["row_identity_verified"] is not True:
        raise ValueError("Full hidden-row correspondence was not verified")
    tokens = _array(snapshot["input_token_ids"], np.int64, positions.shape, "logit input token IDs")
    if not np.array_equal(tokens, schedule["tokens"][positions]):
        raise ValueError("Captured logit row has the wrong actual input token")
    logits = _array(snapshot["logits"], np.float32, (len(positions), expected_vocab), "full-vocabulary logits")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite full-vocabulary logits")
    order = np.argsort(positions)
    if not np.array_equal(positions[order], [31, 32]):
        raise ValueError("Missing required handoff logit positions 31 and 32")
    return positions[order], logits[order]


def compare_logits(candidate, reference, schedule, *, expected_vocab=262144):
    if type(expected_vocab) is not int or not 2 <= expected_vocab <= 262144:
        raise ValueError("Compared vocabulary exceeds the fixed full-vocabulary bound")
    validate_schedule(schedule)
    cp, x = _aligned_logit_rows(candidate, schedule, expected_vocab)
    rp, y = _aligned_logit_rows(reference, schedule, expected_vocab)
    if not np.array_equal(cp, rp):
        raise ValueError("Compared absolute logit positions differ")
    delta = x.astype(np.float64) - y.astype(np.float64)
    equal_bits = x.view(np.uint32) == y.view(np.uint32)
    lx, ly = [], []
    for array, output in ((x, lx), (y, ly)):
        for row in array.astype(np.float64):
            maximum = float(row.max())
            output.append(row - maximum - np.log(np.exp(row - maximum).sum(dtype=np.float64)))
    lx, ly = np.asarray(lx), np.asarray(ly)
    kl = np.sum(np.exp(ly) * (ly - lx), axis=-1, dtype=np.float64)
    nll = []
    for index, position in enumerate(cp):
        prediction = int(position) + 1
        if prediction < len(schedule["tokens"]):
            token = int(schedule["tokens"][prediction])
            if token >= expected_vocab:
                raise ValueError("Supplied NLL target outside the compared vocabulary")
            nll.append((prediction, float(-lx[index, token]), float(-ly[index, token])))
    changed = cp[np.any(~equal_bits, axis=-1)]
    return {**CONTROLLED_ORIGIN, "input_positions": cp.tolist(), "prediction_positions": (cp + 1).tolist(),
            "full_f32_bits_equal": bool(equal_bits.all()), "f32_differing_elements": int((~equal_bits).sum()),
            "first_changed_input_position": int(changed[0]) if len(changed) else None,
            "max_absolute_logit_difference": float(np.abs(delta).max(initial=0)),
            "logit_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "top1_agreement": float(np.mean(x.argmax(axis=-1) == y.argmax(axis=-1))),
            "reference_top2_margins": (np.partition(y, -2, axis=-1)[:, -1] - np.partition(y, -2, axis=-1)[:, -2]).tolist(),
            "reference_to_candidate_kl": kl.tolist(),
            "unique_scored_prediction_positions": len(nll),
            "scored_nll": [{"prediction_position": p, "candidate": c, "reference": r} for p, c, r in nll],
            "numerical_acceptance_tolerance": None, "quality_gate_passed": False}


def cache_contract_sha256(contract):
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def nominal_cache_contract(config_path):
    """Read the pinned config; actual cache owner/layout remains unverified."""
    path = Path(config_path)
    if path.stat().st_size > 1 << 20:
        raise ValueError("Pinned nominal config exceeds its bounded read size")
    payload = path.read_bytes()
    config_hash = hashlib.sha256(payload).hexdigest()
    if config_hash != CONFIG_SHA256:
        raise ValueError("Nominal cache config differs from the immutable checkpoint hash")
    config = json.loads(payload)["text_config"]
    layers = []
    for ordinal, kind in enumerate(config["layer_types"]):
        global_layer = kind == "full_attention"
        layers.append({"layer": ordinal, "attention_type": kind,
                       "kv_heads": config["num_global_key_value_heads"] if global_layer else config["num_key_value_heads"],
                       "head_dim": config["global_head_dim"] if global_layer else config["head_dim"],
                       "window_size": None if global_layer else config["sliding_window"],
                       "logical_layout": "position,kv_head,head_dim", "dtype": "bf16"})
    contract = {"checkpoint_revision": "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
                "provenance": {"scope": "checkpoint_config_only", "config_sha256": config_hash,
                               "source_sha256": config_hash, "kv_owner_layout_confirmed": False},
                "layers": layers}
    contract["sha256"] = cache_contract_sha256(contract)
    return validate_cache_contract(contract)


def validate_cache_contract(contract):
    """Bind declared logical geometry, without certifying installed cache owners."""
    scope = contract["provenance"]["scope"]
    if scope not in {"synthetic_cpu_fixture", "checkpoint_config_only", "installed_runtime_confirmed"}:
        raise ValueError("Unknown cache descriptor provenance scope")
    for label in ("config_sha256", "source_sha256"):
        value = contract["provenance"][label]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Cache descriptor is missing fixed config/source hashes")
    if contract["provenance"]["kv_owner_layout_confirmed"] is not (scope == "installed_runtime_confirmed"):
        raise ValueError("Cache owner/layout assertion exceeds its descriptor provenance scope")
    descriptors = contract["layers"]
    if len(descriptors) != 30 or [item["layer"] for item in descriptors] != list(range(30)):
        raise ValueError("Cache descriptor must bind all thirty layer identities")
    for item in descriptors:
        if item["logical_layout"] != "position,kv_head,head_dim" or item["dtype"] != "bf16":
            raise ValueError("Cache descriptor does not match the canonical BF16 logical sample")
        if type(item["kv_heads"]) is not int or not 1 <= item["kv_heads"] <= 8 or type(item["head_dim"]) is not int or not 1 <= item["head_dim"] <= 512:
            raise ValueError("Cache descriptor exceeds the bounded head geometry")
        kind, window = item["attention_type"], item["window_size"]
        if kind == "full_attention":
            if window is not None:
                raise ValueError("Full-attention descriptor must not invent a sliding window")
        elif kind == "sliding_attention":
            if type(window) is not int or not 1 <= window <= 1024:
                raise ValueError("Sliding-attention descriptor has an invalid bounded window")
        else:
            raise ValueError("Unknown cache attention type")
    if contract["sha256"] != cache_contract_sha256(contract):
        raise ValueError("Frozen cache descriptor digest differs")
    return contract


def compare_cache(candidate, reference, schedule, *, cache_contract):
    """Compare selected logical K/V entries; physical allocation IDs are irrelevant."""
    validate_schedule(schedule)
    validate_cache_contract(cache_contract)
    aligned = []
    for snapshot in (candidate, reference):
        _controlled_origin(snapshot)
        if snapshot["token_sha256"] != schedule["token_sha256"] or snapshot["schedule_sha256"] != schedule["schedule_sha256"]:
            raise ValueError("Cache prefix/route-table identity differs")
        layer, length = snapshot["layer"], snapshot["committed_length"]
        if type(layer) is not int or not 0 <= layer < 30 or length != INPUT_TOKENS:
            raise ValueError("Cache layer or committed sequence length differs")
        if snapshot["cache_contract_sha256"] != cache_contract["sha256"]:
            raise ValueError("Cache snapshot is not bound to the frozen descriptor")
        descriptor = cache_contract["layers"][layer]
        window = descriptor["window_size"]
        start = max(0, length - window) if window is not None else 0
        expected = (descriptor["attention_type"], descriptor["kv_heads"], descriptor["head_dim"], start)
        if (snapshot["attention_type"], snapshot["kv_heads"], snapshot["head_dim"], snapshot["mask_window_start"]) != expected:
            raise ValueError("Cache mask/window/head geometry differs")
        if snapshot["logical_mapping_verified"] is not True:
            raise ValueError("Cache page/logical-position mapping was not verified")
        positions = np.asarray(snapshot["logical_positions"])
        if positions.dtype != np.int64 or positions.ndim != 1 or not 1 <= len(positions) <= 6:
            raise ValueError("Cache sample requires bounded I64 logical positions")
        if len(set(positions.tolist())) != len(positions) or np.any(positions < expected[3]) or np.any(positions >= length):
            raise ValueError("Cache positions are duplicate, stale, evicted or uncommitted")
        keys = _array(snapshot["key_bits"], np.uint16, (len(positions), expected[1], expected[2]), "logical BF16 K")
        values = _array(snapshot["value_bits"], np.uint16, keys.shape, "logical BF16 V")
        bf16_values(keys)
        bf16_values(values)
        order = np.argsort(positions)
        if not np.array_equal(positions[order], [31, 32]):
            raise ValueError("Missing required logical handoff cache positions 31 and 32")
        aligned.append((positions[order], keys[order], values[order]))
    if candidate["layer"] != reference["layer"] or not np.array_equal(aligned[0][0], aligned[1][0]):
        raise ValueError("Cache logical layer/position comparison differs")
    return {**CONTROLLED_ORIGIN, "layer": candidate["layer"], "logical_positions": aligned[0][0].tolist(),
            "key": _compare_bits(aligned[0][1], aligned[1][1]),
            "value": _compare_bits(aligned[0][2], aligned[1][2]),
            "cache_descriptor_sha256": cache_contract["sha256"],
            "cache_descriptor_scope": cache_contract["provenance"]["scope"],
            "installed_owner_layout_independently_verified": False,
            "independent_cache_correctness_qualified": False}


def compare_cache_collection(candidate, reference, schedule, *, cache_contract):
    """Require every nominal layer's selected K/V rows; still a bounded comparison."""
    if len(candidate) != 30 or len(reference) != 30:
        raise ValueError("Cache collection is missing nominal layer coverage")
    expected = list(range(30))
    if sorted(item["layer"] for item in candidate) != expected or sorted(item["layer"] for item in reference) != expected:
        raise ValueError("Cache collection has duplicate or missing nominal layer identities")
    left = {item["layer"]: item for item in candidate}
    right = {item["layer"]: item for item in reference}
    layers = [compare_cache(left[i], right[i], schedule, cache_contract=cache_contract) for i in expected]
    return {**CONTROLLED_ORIGIN, "layers": layers,
            "all_selected_logical_k_v_bits_equal": all(item["key"]["raw_bf16_bits_equal"] and item["value"]["raw_bf16_bits_equal"] for item in layers),
            "installed_owner_layout_independently_verified": False,
            "independent_cache_correctness_qualified": False}
