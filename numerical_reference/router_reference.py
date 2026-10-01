"""Independent NumPy/stdlib oracle for the pinned Gemma4 CUDA router.

No production helpers, Torch, Triton or GPU runtime are imported. Captured F32
score ordering is exact under the inspected finite-bit key contract. FP64
projection and softmax profiles are separate from native-kernel qualification.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


F32_U = 2.0 ** -24
F64_U = 2.0 ** -53
LOG2E_SOURCE_F32 = np.float32(1.4426950408889634)


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return value


def bf16_decode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.dtype.kind != "u" or bits.dtype.itemsize != 2:
        raise ValueError("BF16 storage must use uint16 bits")
    result = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return _finite(result, "BF16 operand")


def bf16_rne_bits(value: np.ndarray) -> np.ndarray:
    """Explicit F32 conversion, then software round-to-nearest-even BF16."""
    value = _finite(np.asarray(value, dtype=np.float32), "F32 BF16-cast input")
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    result = (rounded >> np.uint32(16)).astype(np.uint16)
    bf16_decode(result)  # Reject finite F32 values that overflow BF16 on casting.
    return result


def bf16_product_bits(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Separate F32 multiplication followed by a BF16 RNE store."""
    product = np.multiply(bf16_decode(left), bf16_decode(right), dtype=np.float32)
    return bf16_rne_bits(product)


def gamma(length: int, unit_roundoff: float) -> float:
    if length < 1 or length * unit_roundoff >= 1:
        raise ValueError("Invalid length for a finite gamma bound")
    return length * unit_roundoff / (1.0 - length * unit_roundoff)


def projection_diagnostic(
    input_bits: np.ndarray, weight_bits: np.ndarray, observed_logits: np.ndarray
) -> dict[str, Any]:
    """Bounded FP64 dot and conditional, unfitted F32 gamma-K diagnostic.

    BF16 products have at most 16 significant bits. For products in the normal
    F32 range they are exactly representable in F32. The bound still requires
    full-precision F32 accumulation/RNE, no partial BF16 reduction, no overflow
    or FTZ affecting the computation. The API's F32 output is not such proof.
    FP64 dot/sumabs rounding is included conservatively in the envelope.
    """
    x = bf16_decode(input_bits).astype(np.float64)
    w = bf16_decode(weight_bits).astype(np.float64)
    observed = _finite(np.asarray(observed_logits), "Captured router logits")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Projection operands must be compatible matrices")
    if observed.dtype != np.float32 or observed.shape != (x.shape[0], w.shape[0]):
        raise ValueError("Projection output must be the matching captured F32 matrix")
    dot = _finite(x @ w.T, "FP64 projection")
    sumabs = _finite(np.abs(x) @ np.abs(w).T, "FP64 absolute-product sum")
    g64, g32 = gamma(x.shape[1], F64_U), gamma(x.shape[1], F32_U)
    sumabs_upper = np.nextafter(sumabs / (1.0 - g64), np.inf)
    envelope = np.nextafter((g32 + g64) * sumabs_upper, np.inf)
    nonzero_min, product_max = math.inf, 0.0
    for row in x:
        products = np.abs(w * row[None, :])
        nonzero = products[products > 0]
        if nonzero.size:
            nonzero_min = min(nonzero_min, float(nonzero.min()))
        product_max = max(product_max, float(products.max(initial=0.0)))
    f32 = np.finfo(np.float32)
    normal_products = nonzero_min >= float(f32.tiny) and product_max <= float(f32.max)
    no_overflow = bool(np.all(sumabs_upper < float(f32.max)))
    usable = bool(normal_products and no_overflow)
    difference = observed.astype(np.float64) - dot
    ieee32 = _finite(dot.astype(np.float32), "FP64-dot to F32 reference")
    report = {
        "dot_f64": dot,
        "envelope": envelope,
        "sumabs_upper": sumabs_upper,
        "absolute_error": np.abs(difference),
        "conditional_envelope_usable": usable,
        "conditional_envelope_pass": bool(np.all(np.abs(difference) <= envelope)) if usable else None,
        "max_absolute_error": float(np.abs(difference).max(initial=0.0)),
        "outputs_compared": int(observed.size),
        "observed_ieee32_value_equal_count": int(np.count_nonzero(observed == ieee32)),
        "observed_ieee32_value_equal_fraction": float(np.mean(observed == ieee32)) if observed.size else 1.0,
        "largest_conditional_envelope": float(envelope.max(initial=0.0)),
        "max_error_over_conditional_envelope": float(np.max(np.divide(
            np.abs(difference), envelope, out=np.zeros_like(envelope), where=envelope > 0
        ), initial=0.0)) if usable else None,
        "gamma_k_f32": g32,
        "gamma_k_f64": g64,
        "all_nonzero_products_normal_f32": bool(normal_products),
        "sumabs_below_f32_max": no_overflow,
        "native_accumulation_qualified": False,
        "assumptions": [
            "Full BF16 products accumulated with F32 RNE; no partial BF16 reduction",
            "No overflow or FTZ/subnormal effect; no unmodeled accumulator cast",
            "FP64 reference/sumabs use RNE without overflow; gamma-K includes their uncertainty",
        ],
        "residual_blocker": "torch.mm out_dtype=F32 does not establish native accumulation order/precision",
    }
    return report


def pinned_bit_order(logits: np.ndarray) -> np.ndarray:
    """Inspected CUDA key: signed-int64 ascending, finite F32 bits then ID.

    Same bits tie by ascending expert ID. Positive zero precedes negative zero.
    This is deliberately independent of floating point argsort tie behavior.
    """
    logits = _finite(np.asarray(logits), "Router scores")
    if logits.dtype != np.float32 or logits.ndim != 2:
        raise ValueError("Router scores must be an F32 matrix")
    experts = logits.shape[1]
    if not 1 <= experts < 2 ** 31:
        raise ValueError("Invalid expert count")
    bits = np.ascontiguousarray(logits).view(np.uint32)
    high = np.where(
        (bits & np.uint32(0x80000000)) == 0,
        bits ^ np.uint32(0xFFFFFFFF), bits ^ np.uint32(0x80000000),
    ).astype(np.uint64)
    packed = (high << np.uint64(32)) | np.arange(experts, dtype=np.uint64)[None, :]
    signed = packed.view(np.int64)
    return np.argsort(signed, axis=1)


def _balanced_f32_sum(values: np.ndarray) -> np.float32:
    count = 1 << (len(values) - 1).bit_length()
    work = np.zeros(count, dtype=np.float32)
    work[:len(values)] = values
    while len(work) > 1:
        work = np.add(work[0::2], work[1::2], dtype=np.float32)
    return work[0]


def weight_profiles(
    logits: np.ndarray, selected_ids: np.ndarray, expert_scale_bits: np.ndarray
) -> dict[str, np.ndarray | str | bool]:
    """High-precision semantics and a named F32 exp2/div/reduction profile.

    The native exp2/div implementation and reduction order are not assumed.
    No selected-weight acceptance threshold is fitted or supplied here.
    Learned expert factors multiply AFTER top-eight selection/normalization.
    """
    logits = _finite(np.asarray(logits), "Router scores")
    ids = np.asarray(selected_ids)
    scales = bf16_decode(expert_scale_bits)
    if logits.dtype != np.float32 or logits.ndim != 2:
        raise ValueError("Router scores must be an F32 matrix")
    if ids.ndim != 2 or ids.shape[0] != logits.shape[0] or ids.dtype.kind not in "iu":
        raise ValueError("Selected IDs must be a matching integer matrix")
    if scales.shape != (logits.shape[1],) or np.any(scales <= 0):
        raise ValueError("Selected expert factors must match E and be positive/finite")
    if np.any(ids < 0) or np.any(ids >= logits.shape[1]):
        raise ValueError("Selected expert ID outside frozen local scope")
    if any(len(set(row.tolist())) != len(row) for row in ids):
        raise ValueError("Selected expert IDs must be distinct")
    semantic, ieee = [], []
    full_probabilities = []
    for row, chosen in zip(logits, ids):
        d64 = row.astype(np.float64) - float(row.max())
        exponent64 = np.exp(d64)
        selected64 = exponent64[chosen]
        semantic.append(selected64 / selected64.sum() * scales[chosen].astype(np.float64))
        full_probabilities.append(exponent64 / exponent64.sum())
        d32 = np.subtract(row, row.max(), dtype=np.float32)
        exponent_argument32 = np.multiply(d32, LOG2E_SOURCE_F32, dtype=np.float32)
        # libm/NumPy F64 exp2 followed by an F32 cast, not native approximate exp2.
        exponential32 = np.exp2(exponent_argument32.astype(np.float64)).astype(np.float32)
        denominator32 = _balanced_f32_sum(exponential32[chosen])
        if denominator32 <= 0:
            denominator32 = np.float32(1)
        inverse32 = np.float32(np.float32(1) / denominator32)
        normalized32 = np.multiply(exponential32[chosen], inverse32, dtype=np.float32)
        ieee.append(np.multiply(normalized32, scales[chosen], dtype=np.float32))
    return {
        "semantic_weights_f64": np.asarray(semantic, dtype=np.float64),
        "semantic_full_probabilities_f64": np.asarray(full_probabilities, dtype=np.float64),
        "named_f32_weights": np.asarray(ieee, dtype=np.float32),
        "named_profile": "F32 subtract/log2e multiply; F64 exp2->F32; balanced F32 selected sum; RN div; two F32 products",
        "native_weight_kernel_qualified": False,
        "residual_blocker": "No invocation-specific exp2/div PTX error contract or reduction-order proof",
    }


def interval_rank(
    scores: np.ndarray, absolute_bounds: np.ndarray, expert: int, topk: int = 8
) -> dict[str, Any]:
    scores = _finite(np.asarray(scores, dtype=np.float64), "Reference scores")
    bound = _finite(np.asarray(absolute_bounds, dtype=np.float64), "Score uncertainty")
    if scores.ndim != 1 or bound.shape != scores.shape or np.any(bound < 0):
        raise ValueError("Score bounds must be a nonnegative matching vector")
    if not 0 <= expert < len(scores) or not 1 <= topk <= len(scores):
        raise ValueError("Invalid target/topk")
    lower = np.nextafter(scores - bound, -np.inf)
    upper = np.nextafter(scores + bound, np.inf)
    others = np.arange(len(scores)) != expert
    minimum = 1 + int(np.count_nonzero(lower[others] > upper[expert]))
    maximum = 1 + int(np.count_nonzero(upper[others] >= lower[expert]))
    membership = "inside" if maximum <= topk else "outside" if minimum > topk else "uncertain"
    return {"conditional_minimum_rank": minimum, "conditional_maximum_rank": maximum,
            "conditional_topk_membership": membership,
            "qualification": "Conditional F32 dot envelope; exact ties are conservatively unresolved"}


def score_diagnostic(
    logits: np.ndarray, selected_ids: np.ndarray, selected_weights: np.ndarray,
    expert_scale_bits: np.ndarray, targets: list[int], topk: int = 8,
) -> dict[str, Any]:
    logits = _finite(np.asarray(logits), "Router scores")
    ids = np.asarray(selected_ids)
    weights = _finite(np.asarray(selected_weights), "Selected router weights")
    if logits.ndim != 2 or logits.dtype != np.float32:
        raise ValueError("Router scores must be an F32 matrix")
    if ids.shape != (len(logits), topk) or weights.shape != ids.shape or weights.dtype != np.float32:
        raise ValueError("Selected IDs/weights must have the matching top-k F32 shape")
    if np.any(weights < 0):
        raise ValueError("Selected router weights must be nonnegative")
    if len(set(targets)) != len(targets) or any(not 0 <= e < logits.shape[1] for e in targets):
        raise ValueError("Targets must be distinct local expert IDs")
    order = pinned_bit_order(logits)
    expected = order[:, :topk]
    profiles = weight_profiles(logits, ids, expert_scale_bits)
    semantic = profiles["semantic_weights_f64"]
    named = profiles["named_f32_weights"]
    ranking = []
    for row_index, (row, sorted_ids) in enumerate(zip(logits, order)):
        row64 = row.astype(np.float64)
        inverse = np.empty(len(row), dtype=np.int64)
        inverse[sorted_ids] = np.arange(1, len(row) + 1)
        eighth = float(row[sorted_ids[topk - 1]])
        ninth = float(row[sorted_ids[topk]]) if topk < len(row) else None
        target_metrics = []
        for target in targets:
            greater = int(np.count_nonzero(row > row[target]))
            equal = int(np.count_nonzero(row == row[target]))
            other_order = sorted_ids[sorted_ids != target]
            other_threshold = float(row[other_order[topk - 1]]) if topk <= len(other_order) else None
            target_metrics.append({
                "expert": target, "raw_score": float(row[target]),
                "pinned_bit_rank": int(inverse[target]),
                "numeric_tie_minimum_rank": greater + 1, "numeric_tie_maximum_rank": greater + equal,
                "same_bits_tie_count": int(np.count_nonzero(row.view(np.uint32) == row.view(np.uint32)[target])),
                "target_minus_kth_logit": float(row64[target] - eighth),
                "target_minus_kth_other_logit": float(row64[target] - other_threshold) if other_threshold is not None else None,
                "semantic_full_probability_f64": float(profiles["semantic_full_probabilities_f64"][row_index, target]),
                "selected": bool(np.any(ids[row_index] == target)),
            })
        ranking.append({
            "row_index": row_index, "targets": target_metrics,
            "kth_minus_next_logit": eighth - ninth if ninth is not None else None,
            "selected_ids_match_pinned_bit_order": bool(np.array_equal(ids[row_index], expected[row_index])),
        })
    return {
        "all_full_scores_finite": True,
        "selected_ids_match_pinned_bit_order": bool(np.array_equal(ids, expected)),
        "weight_semantic_max_absolute_error": float(np.abs(weights.astype(np.float64) - semantic).max(initial=0.0)),
        "weight_named_f32_max_absolute_error": float(np.abs(weights.astype(np.float64) - named.astype(np.float64)).max(initial=0.0)),
        "weight_named_f32_value_equal_fraction": float(np.mean(weights == named)) if weights.size else 1.0,
        "weight_values_compared": int(weights.size),
        "weight_named_f32_value_equal_count": int(np.count_nonzero(weights == named)),
        "weight_comparison_has_acceptance_bound": False,
        "native_weight_kernel_qualified": False,
        "named_weight_profile": profiles["named_profile"],
        "weight_residual_blocker": profiles["residual_blocker"],
        "row_metrics": ranking,
    }


def analyze_forward(
    arrays: dict[str, np.ndarray], reference: dict[str, np.ndarray], targets: list[int],
    hidden: int = 2816, experts: int = 128, topk: int = 8,
) -> dict[str, Any]:
    """Check one captured forward; independent projection covers checked rows only."""
    logits = np.asarray(arrays["logits_f32"])
    positions, token_ids = np.asarray(arrays["positions"]), np.asarray(arrays["token_ids"])
    checked = np.asarray(arrays["checked_indices"])
    if not 1 <= len(positions) <= 256 or logits.shape != (len(positions), experts) or positions.shape != token_ids.shape or positions.ndim != 1:
        raise ValueError("Forward score/token/position shape mismatch")
    if positions.dtype.kind not in "iu" or token_ids.dtype.kind not in "iu" or np.any(positions < 0) or np.any(token_ids < 0):
        raise ValueError("Forward positions/tokens must be nonnegative integers")
    if checked.ndim != 1 or checked.dtype.kind not in "iu" or len(checked) > 8:
        raise ValueError("Checked rows must be a bounded integer vector")
    if len(set(checked.tolist())) != len(checked) or np.any(checked < 0) or np.any(checked >= len(positions)):
        raise ValueError("Checked rows are duplicate or outside the forward")
    weight_bits = np.asarray(reference["weight_bits"])
    dimension_bits = np.asarray(reference["dimension_scale_bits"])
    root_bits = np.asarray(reference["root_cast_bf16_bits"])
    if weight_bits.shape != (experts, hidden) or dimension_bits.shape != (hidden,) or root_bits.size != 1:
        raise ValueError("Original router reference shape mismatch")
    bf16_decode(weight_bits)
    stage_bits = {key: np.asarray(arrays[key]) for key in ("residual_bits", "norm_bits", "projection_bits")}
    if any(value.shape != (len(checked), hidden) for value in stage_bits.values()):
        raise ValueError("Checked router stage shape mismatch")
    for bits in stage_bits.values():
        bf16_decode(bits)
    reconstructed_root = bf16_product_bits(stage_bits["norm_bits"], root_bits.reshape(()))
    reconstructed_input = bf16_product_bits(reconstructed_root, dimension_bits[None, :])
    projection_equal = bool(np.array_equal(reconstructed_input, stage_bits["projection_bits"]))
    scoring = score_diagnostic(logits, arrays["selected_ids"], arrays["selected_weights_f32"], reference["expert_scale_bits"], targets, topk)
    retained = []
    dot_report = None
    rms = None
    if len(checked):
        dot = projection_diagnostic(stage_bits["projection_bits"], weight_bits, logits[checked])
        dot_report = {key: value for key, value in dot.items() if not isinstance(value, np.ndarray)}
        for index, forward_index in enumerate(checked):
            retained.append({
                "forward_row": int(forward_index), "position": int(positions[forward_index]),
                "targets": [{"expert": target, **interval_rank(dot["dot_f64"][index], dot["envelope"][index], target, topk)} for target in targets]
                if dot["conditional_envelope_usable"] else [],
            })
        residual = bf16_decode(stage_bits["residual_bits"]).astype(np.float64)
        # Source epsilon is an F32 kernel scalar. Native IR/rsqrt/reduction
        # dispatch is unproved; this ideal profile is a separate diagnostic.
        epsilon = float(np.float32(1e-6))
        ideal = residual / np.sqrt(np.mean(residual * residual, axis=1, keepdims=True) + epsilon)
        rms_bits = bf16_rne_bits(ideal)
        captured_norm = bf16_decode(stage_bits["norm_bits"]).astype(np.float64)
        rms = {
            "values_compared": int(rms_bits.size),
            "ideal_bf16_bits_equal_count": int(np.count_nonzero(rms_bits == stage_bits["norm_bits"])),
            "ideal_f64_then_f32_bf16_bits_equal_fraction": float(np.mean(rms_bits == stage_bits["norm_bits"])),
            "ideal_bf16_value_max_absolute_difference": float(np.abs(captured_norm - bf16_decode(rms_bits).astype(np.float64)).max(initial=0.0)),
            "ideal_f64_max_absolute_difference_before_cast": float(np.abs(captured_norm - ideal).max(initial=0.0)),
            "native_rms_qualified": False,
            "residual_blocker": "RMSNorm IR reduction/rsqrt dispatch is not independently specified",
        }
    return {
        "rows": len(positions), "checked_projection_rows": len(checked),
        "projection_reconstructed_from_norm_root_dimension_bits_equal": projection_equal,
        "root_intermediate": "Reconstructed with separate F32/BF16 stores; not captured",
        "projection_profile": dot_report, "rms_semantic_diagnostic": rms,
        "checked_row_rank_envelopes": retained, "scoring": scoring,
        "whole_router_numerically_qualified": False,
        "scope": "Captured operand projection on retained rows; full-score ordering/ranks on all rows; earlier hidden-state semantics remain shared",
    }
