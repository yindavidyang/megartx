"""Descriptive paired request-level statistics, not a performance gate evaluator."""

import math
import random
import statistics

from .contracts import ContractError, validate


def quantile(values, probability):
    """Linear interpolation between order statistics (Hyndman-Fan type 7)."""
    data = sorted(values)
    if not data or not 0 <= probability <= 1:
        raise ContractError("quantile needs data and a probability in [0, 1]")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in data):
        raise ContractError("quantile requires finite numeric values")
    position = (len(data) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def paired_median_reduction(baseline, candidate, samples=2000, seed=0, confidence=0.95):
    """Resample matched request pairs, never individual correlated tokens.

    Estimand: 1 - median(candidate request medians)/median(baseline request medians).
    Returned percentile CI covers this latency estimand only, not quality/tail gates.
    """
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ContractError("bootstrap needs at least two matched request pairs")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ContractError("bootstrap requires at least 100 resamples")
    if not 0 < confidence < 1:
        raise ContractError("confidence must be in (0, 1)")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in [*baseline, *candidate]):
        raise ContractError("timings must be positive finite numbers")
    rng = random.Random(seed)
    reductions = []
    count = len(baseline)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        b = statistics.median(baseline[i] for i in indices)
        c = statistics.median(candidate[i] for i in indices)
        reductions.append(1 - c / b)
    alpha = (1 - confidence) / 2
    return {"reduction_fraction": 1 - statistics.median(candidate) / statistics.median(baseline),
            "confidence_interval": [quantile(reductions, alpha), quantile(reductions, 1 - alpha)],
            "confidence": confidence, "bootstrap_resamples": samples, "bootstrap_seed": seed,
            "bootstrap_unit": "matched_request_pair", "request_pair_count": count}


def summarize(record, samples=2000, seed=0, min_pairs=30):
    validate(record, "results")
    if record["boundary"] != "delivered_token_end_to_end":
        raise ContractError("this summary only supports delivered-token end-to-end records; do not relabel GPU or microbenchmark timing")
    pairs = record["pairs"]
    if len(pairs) < 2:
        raise ContractError("summary requires at least two request pairs; templates are not measurements")
    for pair in pairs:
        if len(pair["baseline_itl_ms"]) != len(pair["candidate_itl_ms"]):
            raise ContractError("paired requests must use matched output lengths/EOS policy")
        for name in ("baseline", "candidate"):
            total = pair[f"{name}_ttft_ms"] + sum(pair[f"{name}_itl_ms"])
            if not math.isclose(total, pair[f"{name}_total_response_ms"], rel_tol=1e-6, abs_tol=1e-6):
                raise ContractError("total response must equal TTFT plus delivered-token intervals")
    baseline = [statistics.median(p["baseline_itl_ms"]) for p in pairs]
    candidate = [statistics.median(p["candidate_itl_ms"]) for p in pairs]
    result = paired_median_reduction(baseline, candidate, samples, seed)
    result.update({"run_id": record["run_id"], "provenance": record["provenance"],
                   "boundary": record["boundary"],
                   "omitted_critical_path_costs": record["omitted_critical_path_costs"],
                   "prerequisite_statuses": {key: record[key] for key in (
                       "correctness", "quality", "memory", "baseline_dispatch")},
                   "numerical_lane": record["numerical_lane"], "units": "milliseconds",
                   "estimand": "one minus candidate/baseline ratio of medians of per-request median delivered-token ITL",
                   "minimum_pair_count_met": len(pairs) >= min_pairs,
                   "minimum_pair_count": min_pairs,
                   "both_orders_present": len({p["order"] for p in pairs}) == 2,
                   "gate_decision": "not_evaluated",
                   "gate_note": "Statistics alone cannot pass GPU correctness, quality, compatibility, memory or performance gates."})
    for name, medians in (("baseline", baseline), ("candidate", candidate)):
        tokens = [v for pair in pairs for v in pair[f"{name}_itl_ms"]]
        result[name] = {
            "request_median_itl_p50_ms": statistics.median(medians),
            "request_median_itl_p95_ms": quantile(medians, 0.95),
            "pooled_token_itl_p95_ms": quantile(tokens, 0.95),
            "ttft_p50_ms": statistics.median(p[f"{name}_ttft_ms"] for p in pairs),
            "total_response_p50_ms": statistics.median(p[f"{name}_total_response_ms"] for p in pairs),
            "token_interval_count": len(tokens)}
    result["limitations"] = ["Tail, TTFT, total-response and quality uncertainty require separate analysis",
                              "Both order labels do not prove randomization; review the trial protocol separately",
                              "No gate approval or dispatch verification is inferred from result labels",
                              "Per-token intervals exclude the first token; cold load/compile and profiling are separate"]
    return result
