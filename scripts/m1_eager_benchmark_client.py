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
from megartx.m1_eager_benchmark import (OUTPUTS, PROFILE_SCHEMA, PROFILE_RULE, drain_marker,
                                      load_plan, marker, require_profile_intent, WARMED_SCHEMA, WARMED_RULE,
                                      WARMED_SUMMARY_SCHEMA, require_warmed_intent, warmed_boundaries,
                                      validate_warmed_boundaries, validate_warmed_telemetry)
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
    if (report.get("schema") != plan["schema"]
            or (plan["schema"] == PROFILE_SCHEMA and report.get("diagnostic_admission") != PROFILE_RULE)
            or (plan["schema"] == WARMED_SCHEMA and report.get("timing_admission") != WARMED_RULE)
            or report.get("plan_sha256") != plan["plan_sha256"] or report.get("source_head") != plan["source_head"]
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
    launch = directory / "launch-manifest.json"
    if ((directory / "decode-profile").exists() or
            (launch.is_file() and json.loads(launch.read_text()).get("m1_decode_profile_requested"))):
        raise RuntimeError("instrumented decode attribution cannot admit performance timings")
    plan = load_plan(directory / "eager-benchmark-plan.json")
    if plan["schema"] == PROFILE_SCHEMA:
        raise RuntimeError("diagnostic plan permanently excludes performance timings")
    if launch.is_file():
        launch_record = json.loads(launch.read_text())
        require_warmed_intent(plan, launch_record.get("m1_warmed_timing_requested", False))
        if plan["schema"] != WARMED_SCHEMA and launch_record.get("timing_admission") is not None:
            raise RuntimeError("warmed launch policy cannot replay as strict lifetime timing")
    records = [json.loads(line) for line in (directory / "eager-requests.jsonl").read_text().splitlines()]
    dispatch = json.loads((directory / "eager-benchmark/dispatch.json").read_text())
    validate_dispatch(plan, dispatch, records)
    cleanup = json.loads((directory / "eager-benchmark-cleanup.json").read_text())
    if (not cleanup.get("cleanup_complete") or (directory / "benchmark.exit").read_text().strip() != "0"
            or (directory / "run.exit").read_text().strip() != "0"):
        raise RuntimeError("eager benchmark lifecycle did not pass")
    warmed_receipt = None
    if plan["schema"] == WARMED_SCHEMA:
        warmed_receipt = validate_warmed_run(directory, plan, cleanup, replay=True)
    elif cleanup.get("warmed_runtime_files") is not None or any((directory / name).exists() for name in ("warmed-boundaries.json", "warmed-timing-admission.json",
                                                    "warmed-launch-boundaries.json")):
        raise RuntimeError("warmed evidence cannot replay as strict lifetime timing")
    telemetry = [json.loads(line) for line in (directory / "gpu-telemetry.jsonl").read_text().splitlines()]
    valid = [r for r in telemetry if r.get("exit") == 0 and "memory.used" in r.get("fields", [])]
    used = lambda row: float(row["values"][row["fields"].index("memory.used")])
    summary = {"schema": WARMED_SUMMARY_SCHEMA if warmed_receipt else "megartx-m1-guarded-eager-summary-v1", "source_head": plan["source_head"],
        "plan_sha256": plan["plan_sha256"], "scope": "guarded stock eager versus guarded fused eager, exploratory",
        "timing_boundary": "Host-local client " + ("monotonic_ns" if warmed_receipt else "perf_counter_ns") + " before POST through SSE delivery, DONE and response close. JSON encoding/tokenization/marker writes excluded. Native validation, correction, stream handoffs, prefill, head, sampler, KV and local streaming included.",
        "gpu_event_timing": "not measured; no per-token fences or kernel-only inference",
        "policy": "BF16 KV, corrected native quantizer/model, synchronous eager concurrency one, greedy 256 tokens with ignore_eos, prefix caching off",
        "retained_guard_costs": "Stock one route-ID readback/fence; fused two; both seven descriptor table readbacks and one fence per stage; all owners, lease checks and stream handoffs retained",
        "quality_qualified": False, "graphs_qualified": False, "performance_gate_passed": False,
        "minimum_30_pairs_met": plan["trials"] >= 30,
        "memory": {"sampled_lifecycle_device_peak_mib": max(map(used, valid)) if valid else None,
                   "scope": "nvidia-smi samples at approximately 200 ms; transient exact peak and Torch allocated/reserved peaks not measured"},
        "contexts": {}}
    if warmed_receipt:
        summary.update(timing_admission=WARMED_RULE, sampled_window_quiescence=True,
                       warmed_admission_sha256=digest(warmed_receipt),
                       compiler_quiescence=warmed_receipt["compiler_quiescence"])
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



def read_evidence(path, limit=32 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise RuntimeError("warmed timing evidence must be bounded regular file: " + path.name)
    return path.read_text()


class WarmedRuntimeFiles:
    """Lifetime file-version checks, with initial and final byte verification.

    The diagnostic module supplies only its existing exact build/file verifier;
    its relaxed compiler admission never participates in timing authority.
    """
    def __init__(self, plan, launch):
        from m1_decode_profile import verify_runtime_files
        from m1_owned_processes import hash_file_version
        require_warmed_intent(plan, True)
        self.verify_source_head(plan)
        verify_runtime_files(plan, launch)
        self.plan, self.launch, self.final = plan, launch, None
        env = launch["environment_overrides"]
        build_path = Path(env["MEGARTX_M1_BUILD_RECEIPT"])
        build = json.loads(read_evidence(build_path))
        aot = Path(env["MEGARTX_M1_PRIVATE_AOT"])
        manifest = json.loads(read_evidence(aot / "manifest.json"))
        paths = [build_path, Path(env["MEGARTX_M1_BRIDGE"]), aot / "manifest.json", aot / "cpu-dry-run.json",
                 *(ROOT / p for p in build["source_hashes"]), *(Path(p) for p in build["installed_pins"]),
                 *(Path(manifest["flashinfer_root"]) / p for p in manifest["loader_hashes"]),
                 *(aot / "python" / p for p in manifest["shim_hashes"]),
                 *(aot / "aot" / n / (n + ".so") for n in manifest["module_hashes"])]
        self.files = {str(p): {"resolved": str(p.resolve()), "sha256": (value := hash_file_version(p))[0],
                               "version": value[1]} for p in paths}
        self.error = None
        verify_runtime_files(plan, launch)
        if error := self.version_error():
            raise RuntimeError(error)

    @staticmethod
    def verify_source_head(plan):
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if head != plan["source_head"]:
            raise RuntimeError("warmed timing repository source HEAD differs")

    def version_error(self):
        from m1_owned_processes import file_version
        if self.error:
            return self.error
        try:
            for path, expected in self.files.items():
                if str(Path(path).resolve()) != expected["resolved"] or file_version(Path(path).stat()) != expected["version"]:
                    raise RuntimeError("source/AOT/build file version drift: " + path)
        except (OSError, RuntimeError) as error:
            self.error = "warmed timing " + str(error)
        return self.error

    def finalize(self):
        from m1_decode_profile import verify_runtime_files
        from m1_owned_processes import hash_file_version
        errors = [error] if (error := self.version_error()) else []
        for path, expected in self.files.items():
            try:
                actual, version = hash_file_version(path)
                if actual != expected["sha256"] or version != expected["version"]:
                    errors.append("warmed timing final source/AOT/build hash/version differs: " + path)
            except (OSError, RuntimeError) as error:
                errors.append("warmed timing final source/AOT/build verification failed: " + str(error))
        try:
            self.verify_source_head(self.plan)
            verify_runtime_files(self.plan, self.launch)
        except Exception as error:
            errors.append(str(error))
        self.final = {"passed": not errors, "errors": errors}
        if errors and self.error is None:
            self.error = errors[0]
        return self.final

    def report(self):
        return {"policy": "whole_run_file_versions_initial_final_hashes_v1", "files": self.files,
                "failure": self.error, "final": self.final}


def validate_warmed_run(directory, plan, owned, *, replay=False):
    from m1_owned_processes import evaluate_warmed_quiescence
    root = Path(directory)
    owned = json.loads(json.dumps(owned))  # Canonical on-disk tuples/lists for first admission and replay.
    read = lambda name: json.loads(read_evidence(root / name))
    require_warmed_intent(plan, True)
    if load_plan(root / "eager-benchmark-plan.json") != plan:
        raise RuntimeError("warmed timing saved plan differs")
    launch = read("launch-manifest.json")
    require_profile_intent(plan, launch.get("m1_decode_profile_requested", False))
    require_warmed_intent(plan, launch.get("m1_warmed_timing_requested", False))
    if (launch.get("timing_admission") != WARMED_RULE
            or launch.get("eager_benchmark_plan_sha256") != plan["plan_sha256"]
            or (root / "decode-profile").exists() or (root / "decode-diagnostic-admission.json").exists()):
        raise RuntimeError("warmed timing launch policy differs")
    records = [json.loads(line) for line in read_evidence(root / "eager-requests.jsonl").splitlines()]
    validate_dispatch(plan, read("eager-benchmark/dispatch.json"), records)
    lifecycle = read("warmed-launch-boundaries.json")
    boundary = validate_warmed_boundaries(plan, records, read("warmed-boundaries.json"), lifecycle)
    telemetry = [json.loads(line) for line in read_evidence(root / "gpu-telemetry.jsonl").splitlines()]
    resource_evidence = validate_warmed_telemetry(telemetry, boundary, lifecycle)
    saved_owned = read("owned-processes.json")
    saved_cleanup = read("eager-benchmark-cleanup.json")
    if (any(saved_cleanup.get(k) != v or owned.get(k) != v for k, v in saved_owned.items())
            or any(saved_cleanup.get(k) != v for k, v in owned.items())):
        raise RuntimeError("warmed timing ownership/cleanup evidence differs")
    files = owned.get("warmed_runtime_files") or {}
    if (owned.get("cleanup_complete") is not True or owned.get("cleanup_errors") != []
            or owned.get("owned_identities_remaining") != [] or owned.get("owned_gpu_pids_remaining") != []
            or files.get("failure") is not None or not files.get("files")
            or files.get("policy") != "whole_run_file_versions_initial_final_hashes_v1"
            or (files.get("final") or {}).get("passed") is not True
            or (files.get("final") or {}).get("errors") != []
            or any(read_evidence(root / name).strip() != "0" for name in ("run.exit", "benchmark.exit"))):
        raise RuntimeError("warmed timing whole-run file/resource/cleanup evidence did not pass")
    compiler = evaluate_warmed_quiescence(owned, boundary)
    names = ("launch-manifest.json", "eager-benchmark-plan.json", "eager-requests.jsonl",
             "eager-benchmark/dispatch.json", "warmed-boundaries.json", "warmed-launch-boundaries.json",
             "eager-benchmark-cleanup.json", "owned-processes.json", "gpu-telemetry.jsonl", "run.exit", "benchmark.exit")
    receipt = {"schema": "megartx-m1-warmed-timing-admission-v1", "timing_admission": WARMED_RULE,
               "plan_sha256": plan["plan_sha256"], "source_head": plan["source_head"],
               "boundary_sha256": boundary["boundary_sha256"], "compiler_quiescence": compiler,
               "evidence_sha256": {n: hashlib.sha256(read_evidence(root / n).encode()).hexdigest() for n in names},
               "whole_run_integrity_passed": True, "resource_evidence": resource_evidence, "performance_gate_passed": False,
               "quality_qualified": False, "graphs_qualified": False}
    # Normalize tuples to the on-disk JSON representation before comparison.
    receipt = json.loads(json.dumps(receipt))
    if replay and read("warmed-timing-admission.json") != receipt:
        raise RuntimeError("warmed timing admission receipt/evidence differs")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--m1-warmed-timing", action="store_true")
    args = parser.parse_args()
    if args.url != "http://127.0.0.1:18000":
        parser.error("timing requires the owned host-local server")
    plan = load_plan(args.plan)
    launch = json.loads((args.output / "launch-manifest.json").read_text())
    require_profile_intent(plan, launch.get("m1_decode_profile_requested", False))
    require_warmed_intent(plan, args.m1_warmed_timing)
    require_warmed_intent(plan, launch.get("m1_warmed_timing_requested", False))
    if plan["schema"] == WARMED_SCHEMA and launch.get("timing_admission") != WARMED_RULE:
        raise RuntimeError("launch warmed timing admission differs")
    if plan["schema"] == PROFILE_SCHEMA and launch.get("diagnostic_admission") != PROFILE_RULE:
        raise RuntimeError("launch diagnostic admission differs")
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
    warmups_completed_ns = None
    try:
        with (args.output / "eager-requests.jsonl").open("x", buffering=1) as raw:
            for row in plan["schedule"]:
                marker_path.write_text(json.dumps(marker(plan, row)))
                clock = time.monotonic_ns if args.m1_warmed_timing else time.perf_counter_ns
                record = {**row, **completion(session, args.url, payload(cases[row["case"]]["prompt_token_ids"], row), clock)}
                raw.write(json.dumps(record) + "\n")
                records.append(record)
                if row["phase"] == "warmup":
                    warmups_completed_ns = time.monotonic_ns()
                fields = ("id", "lane") if plan["schema"] == PROFILE_SCHEMA else ("id", "lane", "ttft_ms", "amortized_itl_ms", "response_ms")
                print(json.dumps({k: record[k] for k in fields}), flush=True)
        if args.m1_warmed_timing:
            boundary = warmed_boundaries(plan, records, warmups_completed_ns)
            with (args.output / "warmed-boundaries.json").open("x") as stream:
                json.dump(boundary, stream, indent=2)
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
