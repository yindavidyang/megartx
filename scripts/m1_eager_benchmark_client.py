"""Host-local SSE timings for one explicit source-bound, guarded eager plan."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from megartx.m1_eager_benchmark import OUTPUTS, drain_marker, load_plan, marker
from megartx.m1_normal_plan import digest
from megartx.stats import paired_median_reduction, quantile


def completion(session, url, payload, clock=time.perf_counter_ns):
    # Encoding, marker writes, tokenization and post-response validation excluded.
    # All guarded server work and local HTTP/SSE delivery are included.
    encoded = json.dumps(payload).encode()
    tokens, timestamps = [], []
    usage = finish = None
    multi = 0
    start = clock()
    with session.post(url + "/v1/completions", data=encoded,
                      headers={"Content-Type": "application/json"}, stream=True, timeout=(5, 900)) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=None):
            if not raw.startswith(b"data: "):
                continue
            now = clock()
            if raw == b"data: [DONE]":
                done = now
                break
            data = json.loads(raw[6:])
            if "error" in data:
                raise RuntimeError("SSE server error")
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                ids = choice.get("token_ids") or []
                if any(type(t) is not int or not 0 <= t < 262144 for t in ids):
                    raise RuntimeError("invalid SSE token IDs")
                multi += int(len(ids) > 1)
                tokens.extend(ids)
                timestamps.extend([now - start] * len(ids))
                finish = choice.get("finish_reason") or finish
        else:
            raise RuntimeError("SSE stream omitted DONE")
    end = clock()
    expected = payload["max_tokens"]
    if (usage is None or usage.get("prompt_tokens") != len(payload["prompt"])
            or usage.get("completion_tokens") != expected or len(tokens) != expected
            or finish != "length"):
        raise RuntimeError("SSE usage/output length/finish differs")
    gaps = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    return {"token_ids": tokens, "token_elapsed_ns": timestamps,
            "request_start_monotonic_ns": start, "request_end_monotonic_ns": end,
            "ttft_ms": timestamps[0] / 1e6, "last_token_ms": timestamps[-1] / 1e6,
            "done_ms": (done - start) / 1e6, "response_ms": (end - start) / 1e6,
            "amortized_itl_ms": (timestamps[-1] - timestamps[0]) / (expected - 1) / 1e6 if expected > 1 else None,
            "itl_p50_ms": statistics.median(gaps) if gaps and not multi else None,
            "itl_p95_ms": quantile(gaps, .95) if gaps and not multi else None,
            "multi_token_chunks": multi, "usage": usage, "finish_reason": finish}


def payload(tokens, row, outputs=OUTPUTS):
    return {"model": "gemma4-nvfp4", "prompt": tokens, "add_special_tokens": False,
            "max_tokens": outputs, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
            "seed": row["seed"], "n": 1, "stream": True, "stream_interval": 1,
            "stream_options": {"include_usage": True}, "return_token_ids": True,
            "ignore_eos": True, "skip_special_tokens": False, "request_id": row["id"]}


def validate_dispatch(plan, report, records):
    if (report.get("plan_sha256") != plan["plan_sha256"] or report.get("source_head") != plan["source_head"]
            or report.get("observer_off") is not True or len(report.get("records", [])) != len(records)
            or len(records) != len(plan["schedule"])):
        raise RuntimeError("dispatch report identity/count differs")
    cases = {c["id"]: c for c in plan["cases"]}
    for row, receipt, result in zip(plan["schedule"], report["records"], records):
        if (any(receipt.get(k) != v or result.get(k) != v for k, v in row.items())
                or receipt["input_transcript_sha256"] != digest(cases[row["case"]]["prompt_token_ids"] + result["token_ids"][:-1])):
            raise RuntimeError("dispatch/input/SSE transcript differs")
        n = int(row["case"])
        if receipt["counts"] != {"stock": 7650 if row["lane"] == "stock" else 0,
                                 "fused": 7650 if row["lane"] == "fused" else 0,
                                 "prefill_fallback": 30 * (n // 256)}:
            raise RuntimeError("dispatch/backend count differs")
    for a, b in zip(records[::2], records[1::2]):
        if a["token_ids"] != b["token_ids"] or a["usage"] != b["usage"]:
            raise RuntimeError("matched eager outputs/usage differ")
    for a, b in zip(report["records"][::2], report["records"][1::2]):
        if a["natural_correction_selected_rows"] != b["natural_correction_selected_rows"]:
            raise RuntimeError("matched natural correction counters differ")


def summarize_run(directory):
    directory = Path(directory)
    plan = load_plan(directory / "eager-benchmark-plan.json")
    records = [json.loads(line) for line in (directory / "eager-requests.jsonl").read_text().splitlines()]
    dispatch = json.loads((directory / "eager-benchmark/dispatch.json").read_text())
    validate_dispatch(plan, dispatch, records)
    cleanup = json.loads((directory / "eager-benchmark-cleanup.json").read_text())
    if (not cleanup.get("cleanup_complete") or (directory / "benchmark.exit").read_text().strip() != "0"
            or (directory / "run.exit").read_text().strip() != "0"):
        raise RuntimeError("eager benchmark lifecycle did not pass")
    telemetry = [json.loads(line) for line in (directory / "gpu-telemetry.jsonl").read_text().splitlines()]
    valid = [r for r in telemetry if r.get("exit") == 0 and "memory.used" in r.get("fields", [])]
    used = lambda row: float(row["values"][row["fields"].index("memory.used")])
    summary = {"schema": "megartx-m1-guarded-eager-summary-v1", "source_head": plan["source_head"],
        "plan_sha256": plan["plan_sha256"], "scope": "guarded stock eager versus guarded fused eager, exploratory",
        "timing_boundary": "Host-local client perf_counter_ns before POST through SSE delivery, DONE and response close. JSON encoding/tokenization/marker writes excluded. Native validation, correction, stream handoffs, prefill, head, sampler, KV and local streaming included.",
        "gpu_event_timing": "not measured; no per-token fences or kernel-only inference",
        "policy": "BF16 KV, corrected native quantizer/model, synchronous eager concurrency one, greedy 256 tokens with ignore_eos, prefix caching off",
        "retained_guard_costs": "Stock one route-ID readback/fence; fused two; both seven descriptor table readbacks and one fence per stage; all owners, lease checks and stream handoffs retained",
        "quality_qualified": False, "graphs_qualified": False, "performance_gate_passed": False,
        "minimum_30_pairs_met": plan["trials"] >= 30,
        "memory": {"sampled_lifecycle_device_peak_mib": max(map(used, valid)) if valid else None,
                   "scope": "nvidia-smi samples at approximately 200 ms; transient exact peak and Torch allocated/reserved peaks not measured"},
        "contexts": {}}
    for n in ("2048", "8192"):
        cells = [r for r in records if r["phase"] == "measurement" and r["case"] == n]
        pairs = [{r["lane"]: r for r in cells[i:i+2]} for i in range(0, len(cells), 2)]
        result = {"matched_pairs": len(pairs), "first_lane_order": [cells[i]["lane"] for i in range(0, len(cells), 2)],
                  "warmups_per_lane": plan["warmups"], "individual_itl_observable": not any(r["multi_token_chunks"] for r in cells),
                  "multi_token_chunks": sum(r["multi_token_chunks"] for r in cells)}
        for field in ("ttft_ms", "amortized_itl_ms", "last_token_ms", "done_ms", "response_ms"):
            b, c = ([p[lane][field] for p in pairs] for lane in ("stock", "fused"))
            result[field] = {"stock_p50": quantile(b, .5), "stock_p95": quantile(b, .95),
                             "fused_p50": quantile(c, .5), "fused_p95": quantile(c, .95),
                             "paired_95pct_reduction": paired_median_reduction(b, c, seed=plan["seed"]) if len(pairs) >= 2 and min(b+c) > 0 else None}
        for lane in ("stock", "fused"):
            lane_cells = [r for r in cells if r["lane"] == lane]
            samples = [r for r in valid if any(c["request_start_monotonic_ns"] <= r["monotonic_ns"] <= c["request_end_monotonic_ns"] for c in lane_cells)]
            hits = {}
            for receipt in dispatch["records"]:
                if receipt["case"] == n and receipt["phase"] == "measurement" and receipt["lane"] == lane:
                    for key, count in receipt["natural_correction_selected_rows"].items(): hits[key] = hits.get(key, 0) + count
            result[lane] = {"sampled_request_device_peak_mib": max(map(used, samples)) if samples else None,
                            "natural_correction_selected_rows": hits, "zero_natural_correction_hits": not bool(hits),
                            "individual_itl_p95_ms": quantile([(b-a)/1e6 for r in lane_cells for a,b in zip(r["token_elapsed_ns"],r["token_elapsed_ns"][1:])], .95) if result["individual_itl_observable"] else None}
        summary["contexts"][n] = result
    (directory / "eager-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    args = parser.parse_args()
    if args.url != "http://127.0.0.1:18000":
        parser.error("timing requires the owned host-local server")
    plan = load_plan(args.plan)
    launch = json.loads((args.output / "launch-manifest.json").read_text())
    env = launch["environment_overrides"]
    if (launch.get("eager_benchmark_plan_sha256") != plan["plan_sha256"]
            or launch.get("m1_external_observer_requested") is not False
            or env.get("MEGARTX_M1_EXECUTION") != "capture-free"
            or any(k in env for k in ("MEGARTX_M1_EXTERNAL_OBSERVER_DIR", "MEGARTX_LOGITS_DIR", "MEGARTX_CONTROLLED_DIR"))
            or not {"--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching"}.issubset(launch["command"])):
        raise RuntimeError("launch is not an admitted observer-off eager benchmark")
    import requests
    session = requests.Session()
    session.trust_env = False
    marker_path = args.output / "capture-request.json"
    records = []
    cases = {c["id"]: c for c in plan["cases"]}
    try:
        with (args.output / "eager-requests.jsonl").open("x", buffering=1) as raw:
            for row in plan["schedule"]:
                marker_path.write_text(json.dumps(marker(plan, row)))
                record = {**row, **completion(session, args.url, payload(cases[row["case"]]["prompt_token_ids"], row))}
                raw.write(json.dumps(record) + "\n")
                records.append(record)
                print(json.dumps({k: record[k] for k in ("id", "lane", "ttft_ms", "amortized_itl_ms", "response_ms")}), flush=True)
        # Only this separate one-token control request serializes the in-memory
        # server ledger. Its timing and allocation are absent from measurements.
        marker_path.write_text(json.dumps(drain_marker(plan)))
        completion(session, args.url, payload([7], {"id": "drain", "seed": plan["seed"]}, outputs=1))
        report = json.loads((args.output / "eager-benchmark/dispatch.json").read_text())
        validate_dispatch(plan, report, records)
        (args.output / "eager-client-validation.json").write_text(json.dumps({
            "plan_sha256": plan["plan_sha256"], "requests": len(records), "matched_tokens_usage": True,
            "actual_input_transcripts_verified": True, "native_backends_verified": True,
            "quality_qualified": False, "graphs_qualified": False, "performance_gate_passed": False}, indent=2))
    finally:
        marker_path.unlink(missing_ok=True)
        session.close()


if __name__ == "__main__":
    main()
