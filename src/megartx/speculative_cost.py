"""Pure CPU accounting from measured scalars; no invented device timings."""

import math


def staging_budget(*, candidates, absolute_start, page_tokens):
    """Pinned Gemma BF16-KV allocation lower bounds, never a peak-fit receipt.

    Assume one new/COW page for each page touched by the verifier per layer.
    Existing prefix/weights/allocator state, attention/MoE scratch, promotion,
    metadata, head intermediates and runtime padding must be measured separately.
    One-row FP32 logits describes a serialized diagnostic head, not throughput.
    """
    for label, value in (("candidates", candidates), ("absolute_start", absolute_start),
                         ("page_tokens", page_tokens)):
        if type(value) is not int or value < (1 if label == "page_tokens" else 0):
            raise ValueError(label + " must be a valid integer")
    if candidates > 7:
        raise ValueError("candidate length outside proposed bounded sweep")
    rows = candidates + 1
    bytes_per_row = 25 * (2 * 8 * 256 * 2) + 5 * (2 * 2 * 512 * 2)
    pages = (absolute_start % page_tokens + rows + page_tokens - 1) // page_tokens
    page_bytes = pages * page_tokens * bytes_per_row
    logits_row = 262144 * 4
    return {"input_rows": rows, "kv_bytes_per_row": bytes_per_row,
            "packed_kv_payload_bytes": rows * bytes_per_row,
            "new_or_COW_pages_per_layer": pages,
            "new_or_COW_storage_bytes": page_bytes,
            "FP32_batched_logits_bytes": rows * logits_row,
            "FP32_one_row_logits_bytes": logits_row,
            "page_storage_plus_one_row_logits_bytes": page_bytes + logits_row,
            "remaining_under_8MiB_before_other_scratch": (8 << 20) - page_bytes - logits_row}


def _nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(label + " must be finite and nonnegative")


def expected_yield(prefix_survival, *, budget=None):
    """1 + sum P(first i candidates accepted), before EOS/budget censoring.

    Supply prefix survival, not marginal token agreement. For terminal/censored
    requests use measured committed yield instead of this uncensored model.
    """
    prefix_survival = tuple(prefix_survival)
    previous = 1
    for probability in prefix_survival:
        _nonnegative(probability, "survival probability")
        if probability > previous:
            raise ValueError("prefix survival must be nonincreasing and <=1")
        previous = probability
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("budget must be a positive integer")
    included = prefix_survival if budget is None else prefix_survival[:budget - 1]
    return 1 + sum(included)


def break_even(*, target_only_ms, verify_ms, draft_ms, overhead_ms, committed_yield):
    """Steady-state mean cost; request p95 and uncertainty are separate gates."""
    for label, value in locals().copy().items():
        _nonnegative(value, label)
    if target_only_ms == 0 or committed_yield == 0:
        raise ValueError("baseline and committed yield must be positive")
    cycle_ms = verify_ms + draft_ms + overhead_ms
    if cycle_ms == 0:
        raise ValueError("cycle time must be positive")
    return {
        "cycle_ms": cycle_ms,
        "ms_per_committed_token": cycle_ms / committed_yield,
        "speedup": target_only_ms * committed_yield / cycle_ms,
        "max_profitable_draft_ms": target_only_ms * committed_yield - verify_ms - overhead_ms,
    }


def expert_reuse(selected_rows, *, experts=128, top_k=8):
    """Per-layer route union over ALL verifier inputs, including discarded ones.

    Slot reuse is a potential traffic indicator, never measured residency or
    speedup. Compare repeated IDs, actual weight bytes and hardware counters.
    """
    if type(experts) is not int or experts < 1 or type(top_k) is not int or not 1 <= top_k <= experts:
        raise ValueError("invalid expert geometry")
    if not selected_rows:
        raise ValueError("at least one verifier input row is required")
    union = set()
    for row in selected_rows:
        if len(row) != top_k or len(set(row)) != top_k or any(
                type(e) is not int or not 0 <= e < experts for e in row):
            raise ValueError("each row must contain distinct valid top-k expert IDs")
        union.update(row)
    assignments = len(selected_rows) * top_k
    return {"rows": len(selected_rows), "assignments": assignments, "union": len(union),
            "repeated_assignments": assignments - len(union),
            "potential_reuse_fraction": 1 - len(union) / assignments}
