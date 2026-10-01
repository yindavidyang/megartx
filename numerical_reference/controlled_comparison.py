"""Independent CPU comparisons for saved controlled expert stages.

No model/runtime imports. Counterfactual negative checks always distinguish the
actual wrong alpha from a deliberately recomputed correct-scale output.
"""

from __future__ import annotations

import numpy as np

import controlled_reference as reference
import nvfp4_reference as formats


POSITIVE_STAGE_FIELDS = ("input_bits", "q1", "sf1", "gate_alpha_bits", "up_alpha_bits",
                         "gate_bits", "up_bits", "activation_bits", "q2", "sf2",
                         "down_alpha_bits", "down_bits", "quant1_global_bits", "quant2_global_bits")


def compare_activation_operands(left, right, number):
    """Compare logical coordinates; arbitrary physical SF padding is diagnostic."""
    qname, sfname = f"q{number}", f"sf{number}"
    lq, rq = np.asarray(left[qname]), np.asarray(right[qname])
    if lq.dtype != np.uint8 or rq.dtype != np.uint8 or lq.shape != rq.shape or lq.ndim != 2 or not 1 <= lq.shape[0] <= 3 or not 16 <= lq.shape[1] * 2 <= 2816 or lq.shape[1] * 2 % 16:
        raise ValueError("Paired activation packed operand dtype/shape/bound differs")
    linear = []
    for stages in (left, right):
        physical = np.asarray(stages[sfname])
        if physical.dtype != np.uint8:
            raise ValueError("Activation SF must retain raw E4M3 bytes")
        linear.append(formats.unswizzle_128x4(physical, lq.shape[0], lq.shape[1] * 2 // 16))
    values = [formats.decode_qsf(q, sf) for q, sf in zip((lq, rq), linear)]
    logical_sf_equal = bool(np.array_equal(*linear))
    effective_equal = bool(np.array_equal(*values))
    return {"packed_bytes_equal": bool(np.array_equal(lq, rq)),
            "physical_sf_bytes_equal": bool(np.array_equal(left[sfname], right[sfname])),
            "logical_sf_bytes_equal": logical_sf_equal,
            "decoded_units_equal": effective_equal,
            "decoded_f64_bits_equal": bool(np.array_equal(values[0].view(np.uint64), values[1].view(np.uint64))),
            "max_decoded_difference": float(np.abs(values[0] - values[1]).max(initial=0)),
            "logical_operand_stage_gate_pass": logical_sf_equal and effective_equal}


def compare_positive_stages(native, paired_reference):
    """Strict observed-stage equality gate, with no fitted tolerance."""
    report = {}
    first, first_raw = None, None
    for name in POSITIVE_STAGE_FIELDS:
        left, right = np.asarray(native[name]), np.asarray(paired_reference[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError("Paired stage dtype/shape differs: " + name)
        equal = bool(np.array_equal(left, right))
        item = {"raw_array_equal": equal, "differing_elements": int(np.count_nonzero(left != right))}
        if name.endswith("_bits") and left.dtype == np.uint16:
            item.update(reference._compare_bits(left, right))
        if not equal and first_raw is None:
            first_raw = name
        if not equal and first is None and name not in {"q1", "sf1", "q2", "sf2"}:
            first = name
        report[name] = item
    operands = {f"a{number}": compare_activation_operands(native, paired_reference, number) for number in (1, 2)}
    if first is None:
        first = next((name for name, item in operands.items() if not item["logical_operand_stage_gate_pass"]), None)
    return {**reference.CONTROLLED_ORIGIN, "strict_observed_stage_gate_pass": first is None,
            "first_differing_stage": first, "all_raw_payloads_equal": first_raw is None,
            "first_differing_raw_payload": first_raw, "stages": report,
            "activation_operands": operands,
            "native_mma_or_model_qualified": False}


def compare_negative_common_input(native, paired_reference, negative, originals, *, a1_bits):
    """Only call for first forced L0/E42/position31, before any prior correction.

    The caller binds metadata/position. Shared-input agreement is checked here
    rather than presumed for later negative-control activations.
    """
    common = ("input_bits", "gate_alpha_bits", "gate_bits", "quant1_global_bits")
    for name in common:
        arrays = [np.asarray(item[name]) for item in (native, paired_reference, negative)]
        if any(item.dtype != arrays[0].dtype or item.shape != arrays[0].shape or not np.array_equal(item, arrays[0]) for item in arrays[1:]):
            raise ValueError("First negative control lacks common actual input/gate stage: " + name)
    operand_pairs = [compare_activation_operands(native, other, 1) for other in (paired_reference, negative)]
    if not all(item["logical_operand_stage_gate_pass"] for item in operand_pairs):
        raise ValueError("First negative control lacks common logical q1/SF1 operands")
    a1 = reference._f32_scalar(a1_bits, "shared A1 dequant global")
    correct_alpha = np.float32(np.float32(originals["up"][2]) * a1)
    wrong_alpha = np.float32(np.float32(originals["gate"][2]) * a1)
    correct_bits, wrong_bits = reference._f32_bits(correct_alpha), reference._f32_bits(wrong_alpha)
    for label, stages in (("native", native), ("paired reference", paired_reference)):
        if not np.array_equal(stages["up_alpha_bits"], correct_bits):
            raise ValueError("Positive up alpha is not the original up product: " + label)
    if not np.array_equal(negative["up_alpha_bits"], wrong_bits):
        raise ValueError("Negative control did not change only to the gate-global up alpha")
    wrong_projection = (originals["up"][0], originals["up"][1], originals["gate"][2])
    wrong = reference.replay_projection(negative["q1"], negative["sf1"], wrong_projection,
                                        input_global_bits=a1_bits, alpha_bits=wrong_bits,
                                        candidate_bits=negative["up_bits"])
    # Explicitly counterfactual: this alpha was not used by the negative run.
    correct = reference.replay_projection(negative["q1"], negative["sf1"], originals["up"],
                                          input_global_bits=a1_bits, alpha_bits=correct_bits,
                                          candidate_bits=negative["up_bits"])
    positive_up = reference._compare_bits(native["up_bits"], paired_reference["up_bits"])
    negative_vs_native = reference._compare_bits(negative["up_bits"], native["up_bits"])
    wrong_equal = wrong["comparison"]["raw_bf16_bits_equal"]
    correct_equal = correct["comparison"]["raw_bf16_bits_equal"]
    alpha_distinct = not np.array_equal(wrong_bits, correct_bits)
    return {**reference.CONTROLLED_ORIGIN,
            "common_input_and_gate_bits_verified": True,
            "common_logical_a1_operands": operand_pairs,
            "actual_wrong_alpha_bits": int(wrong_bits[0]),
            "counterfactual_correct_alpha_bits": int(correct_bits[0]),
            "alphas_distinct": alpha_distinct,
            "positive_up": positive_up, "negative_vs_native_up": negative_vs_native,
            "negative_matches_wrong_alpha_oracle": wrong_equal,
            "negative_differs_from_correct_alpha_oracle": not correct_equal,
            "negative_control_effect_observed": alpha_distinct and wrong_equal and not correct_equal and positive_up["raw_bf16_bits_equal"],
            "wrong_alpha_oracle_comparison": wrong["comparison"],
            "correct_alpha_counterfactual_comparison": correct["comparison"],
            "native_mma_or_model_qualified": False}
