"""Compare capture-free M1 observer runs with exact captured controls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from compare_controlled_capture import (read_bytes, read_json, read_npz, MAX_TRACE,
    ROUTE_FIELDS, STAGE_FIELDS, LOGIT_FIELDS, KV_FIELDS)
from compare_m1_live import check_run, correlate_trace, load_build, compare_runs, require
from controlled_reference import TARGETS


SCHEMA = "megartx-m1-external-observer-v1"
STREAM_ID_API = "cuptiGetStreamIdEx"
CUPTI_STREAM_ID_PROVIDER = {
    "distribution": "nvidia-cuda-cupti", "version": "13.0.85",
    "library_name": "libcupti.so.13",
    "library_sha256": "e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e",
}
EVENT_COUNTS = {"lease_begin": 1, "lease_end": 1, "runner_identity": 1,
    "runner_workspace": 1, "candidate_status": 1, "installed_map_call": 1,
    "installed_expand_call": 1, "payload": 15, "json": 2}
PAYLOADS = {
    "sf-before.bin": 2_883_584, "sf-after.bin": 2_883_584,
    "input-aq.bin": 1_408, "input-sf.bin": 22_528,
    "ids.bin": 32, "route-weights.bin": 32,
    "fc1-act-global.bin": 512, "fc1-global.bin": 512,
    "fc2-act-global.bin": 512, "fc2-global.bin": 512,
    "expanded-aq.bin": 11_264, "slot-to-sorted.bin": 32,
    "sorted-to-slot.bin": 32, "offsets.bin": 1_032,
    "routed-output.bin": 5_632,
}
MODEL_FIELDS = {"routes": ROUTE_FIELDS, "stages": STAGE_FIELDS,
                "logits": LOGIT_FIELDS, "kv": KV_FIELDS}


def _sha(path, budget):
    return hashlib.sha256(read_bytes(path, budget)).hexdigest()


def _valid_cupti_provider(value):
    return value == CUPTI_STREAM_ID_PROVIDER


def _load_trace(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= MAX_TRACE,
            "Observer trace is missing or too large")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_TRACE + 1)
    require(len(raw) <= MAX_TRACE, "Expanded observer trace is too large")
    return json.loads(raw)["traceEvents"]


def _per_call_trace(events, lane, call_bindings):
    """Bind each native observer call to its profiler scope and CUDA launches."""
    require(len(call_bindings) == 30, "Trace needs thirty native stream bindings")
    thread_ids, stream_handles, profiler_stream_ids, provider_identities = [], [], [], []
    for binding in call_bindings:
        require(isinstance(binding, dict)
                and binding.get("stream_id_api") == STREAM_ID_API
                and binding.get("per_thread_stream") is False
                and type(binding.get("thread_id")) is int
                and type(binding.get("stream_handle")) is int and binding["stream_handle"] != 0
                and type(binding.get("profiler_stream_id")) is int
                and binding["profiler_stream_id"] != 0
                and _valid_cupti_provider(binding.get("cupti_stream_id_provider")),
                "Native callback stream lacks a valid CUPTI profiler-stream mapping")
        thread_ids.append(binding["thread_id"])
        stream_handles.append(binding["stream_handle"])
        profiler_stream_ids.append(binding["profiler_stream_id"])
        provider_identities.append(binding["cupti_stream_id_provider"])
    require(len(set(stream_handles)) == 1 and len(set(profiler_stream_ids)) == 1,
            "Native callback stream identity changed during the controlled request")
    require(all(identity == provider_identities[0] for identity in provider_identities),
            "CUPTI stream-ID provider changed during the controlled request")
    summary = correlate_trace(events, lane, ["controlled_live_request"] * 30,
                              request_count=30, artificial_count=0)
    scopes = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X" or event.get("cat") != "user_annotation":
            continue
        match = re.fullmatch(r"megartx::m1_external_call_(\d{4})", event.get("name", ""))
        if match:
            index = int(match[1])
            require(index not in scopes, "Duplicate external profiler call scope")
            scopes[index] = event
    require(set(scopes) == set(range(30)), "Trace lacks its numbered external call scopes")
    preparation = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"
                   and e.get("cat") == "user_annotation"
                   and e.get("name") == "megartx::m1_preparation_" + lane]
    require(len(preparation) == 30, "Trace lacks thirty M1 preparation spans")
    counters = [Counter() for _ in range(30)]
    owners = {}
    streams_by_call = [set() for _ in range(30)]
    for index, scope in scopes.items():
        require(scope.get("tid") == thread_ids[index],
                "Native callback and profiler scope use different threads")
        nested = [e for e in preparation if e.get("pid") == scope.get("pid")
                  and e.get("tid") == scope.get("tid")
                  and scope.get("ts", -1) <= e.get("ts", -1)
                  and e.get("ts", -1) + e.get("dur", -1)
                      <= scope.get("ts", -1) + scope.get("dur", -1)]
        require(len(nested) == 1, "External call scope lacks one M1 preparation span")
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X" or event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        correlation = event.get("args", {}).get("correlation")
        if type(correlation) is not int:
            continue
        selected = [i for i, scope in scopes.items()
                    if event.get("pid") == scope.get("pid") and event.get("tid") == scope.get("tid")
                    and scope.get("ts", -1) <= event.get("ts", -1)
                    < scope.get("ts", -1) + scope.get("dur", -1)]
        require(len(selected) <= 1 and not (selected and correlation in owners),
                "External CUDA launch correlation is ambiguous")
        if selected:
            owners[correlation] = selected[0]
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation in owners:
            counters[owners[correlation]][event["name"]] += 1
            streams_by_call[owners[correlation]].add(event.get("args", {}).get("stream"))
    require(all(streams_by_call[i] == {profiler_stream_ids[i]} for i in range(30)),
            "Profiler kernel stream ID differs from the CUPTI mapping of its native callback stream")
    incumbent = []
    for counts in counters:
        fused = sum(v for k, v in counts.items() if "m1_maps_expand" in k)
        maps = sum(v for k, v in counts.items() if "fusedBuildExpertMapsSortFirstTokenKernel" in k)
        expand = sum(v for k, v in counts.items() if "expandInputRowsKernel<" in k)
        gemms = sum(v for k, v in counts.items()
                    if "MainloopSm120ArrayTmaWarpSpecializedBlockScaled" in k)
        require((fused, maps, expand, gemms) == ((1, 0, 0, 2) if lane == "fused" else (0, 1, 1, 2)),
                "External call trace lacks candidate/stock proof or two incumbent GEMMs")
        incumbent.append({k: v for k, v in counts.items() if "m1_maps_expand" not in k
                           and "fusedBuildExpertMapsSortFirstTokenKernel" not in k
                           and "expandInputRowsKernel<" not in k})
    return {"launch_to_external_call_correlation_verified": True,
            "call_count": 30, "owned_nondefault_stream_verified": True,
            "stream_id_mapping_api": STREAM_ID_API,
            "cupti_stream_id_provider": provider_identities[0],
            "native_stream_handle_bound_to_profiler_trace_id": True,
            "raw_cuda_handle_assumed_equal_to_profiler_id": False,
            "profiler_stream_ids": sorted(set(profiler_stream_ids)),
            "positive_candidate_launches": 30 if lane == "fused" else 0,
            "per_call_incumbent_kernel_counts": incumbent,
            "generic_m1_trace": summary}


def _check_event_counts(events, receipt_counts, lane):
    expected = dict(EVENT_COUNTS)
    if lane == "fused":
        expected["installed_map_call"] = expected["installed_expand_call"] = 0
    expected_receipt = {key: expected[key] for key in sorted(EVENT_COUNTS)}
    actual = Counter(event.get("event") for event in events if isinstance(event, dict))
    require(receipt_counts == expected_receipt and actual == Counter(expected),
            "Native observer event log/count receipt differs")
    return expected


def _array_set(observer_root, records, category, case, baseline_records):
    fields = MODEL_FIELDS[category]
    expected = {name: sha for name, sha in baseline_records}
    require(len(records) == len(expected), "Observer array count differs: " + category)
    seen = set()
    for record in records:
        name = record.get("file")
        require(name in expected and name not in seen,
                "Observer array file identity differs: " + category)
        seen.add(name)
        observed = read_npz(observer_root / "model" / "cached", name,
                            record.get("sha256"), fields)
        captured = read_npz(case, name, expected[name], fields)
        require(set(observed) == set(captured), "Observer array fields differ: " + name)
        for key in observed:
            left, right = observed[key], captured[key]
            require(left.dtype == right.dtype and left.shape == right.shape
                    and left.tobytes() == right.tobytes(),
                    "Capture-free model output differs: " + name + ":" + key)
    return len(seen)


def _compare_model_arrays(observer_root, observer, captured):
    manifest, case = captured["manifest"], captured["case"]
    records = {
        "routes": [(r["file"], r["sha256"]) for r in manifest["routes"]],
        "stages": [(r["file"], r["stage_capture_sha256"])
                   for r in manifest["executed_interventions"]],
        "logits": [(r["file"], r["sha256"]) for r in map(json.loads,
                    read_bytes(case / "logits-records.jsonl", 1 << 20).splitlines())],
        "kv": [(r["file"], r["sha256"]) for r in read_json(case / "kv-records.json")],
    }
    counts = {key: _array_set(observer_root, observer["model_arrays"][key], key, case, value)
              for key, value in records.items()}
    return {"counts": counts, "arrays": sum(counts.values()),
            "bit_exact_to_captured_lane": True}


def check_external_run(run, lane, captured_run, build):
    run, captured_run = Path(run), Path(captured_run)
    build_report = load_build(build)
    compiled = build_report["live_contract"]
    provider = compiled.get("cupti_stream_id_provider")
    installed_pins = build_report.get("installed_pins", {})
    require(_valid_cupti_provider(provider)
            and build_report.get("cupti_stream_id_provider") == provider
            and build_report.get("installed_package_versions", {}).get("nvidia-cuda-cupti")
                == provider["version"]
            and sum(1 for path, sha256 in installed_pins.items()
                    if Path(path).name == provider["library_name"]
                    and sha256 == provider["library_sha256"]) == 1,
            "Build receipt does not pin the expected CUPTI stream-ID provider")
    launch = read_json(run / "launch-manifest.json")
    command = launch.get("command")
    env = launch.get("environment_overrides", {})
    observer_root = run / "m1-external-observer"
    require(launch.get("m1_execution_requested") == "capture-free"
            and isinstance(command, list) and "--no-async-scheduling" in command
            and launch.get("m1_external_observer_requested") is True
            and launch.get("m1_preparation_requested") == lane
            and launch.get("controlled_request_count") == 1
            and launch.get("controlled_path") == "cached"
            and env.get("MEGARTX_M1_EXECUTION") == "capture-free"
            and env.get("MEGARTX_M1_PREPARATION") == lane
            and env.get("MEGARTX_M1_EXTERNAL_OBSERVER_DIR") == str(observer_root.resolve())
            and "MEGARTX_M1_CAPTURE_DIR" not in env,
            "Launch manifest does not bind explicit observer-only execution")
    require(read_json(run / "status.json").get("phase") == "cleanup_complete"
            and not (run / "QUALIFICATION-INVALIDATED.json").exists()
            and not (run / "preparation").exists(),
            "Observer run lifecycle failed or created internal preparation captures")
    contract = read_json(observer_root / "observer-contract.json")
    require(contract.get("schema") == SCHEMA and contract.get("execution_mode") == "capture-free"
            and contract.get("internal_capture_enabled") is False
            and contract.get("perturbs_execution") is True
            and contract.get("stream_id_mapping_api") == STREAM_ID_API
            and contract.get("profiler_trace_stream_field") == "kernel.args.stream"
            and contract.get("cupti_stream_id_provider") == provider
            and contract.get("timing_qualified") is False
            and contract.get("native_contract") == compiled,
            "External observer contract differs from built sources")
    observer = read_json(observer_root / "observer-manifest.json")
    require(observer.get("schema") == SCHEMA and observer.get("case") == "cached"
            and observer.get("execution_mode") == "capture-free"
            and observer.get("observer_enabled") is True
            and observer.get("internal_capture_enabled") is False
            and observer.get("observer_perturbs_execution") is True
            and observer.get("native_call_count") == 30
            and observer.get("stream_id_mapping_api") == STREAM_ID_API
            and observer.get("cupti_stream_id_provider") == provider
            and observer.get("quality_gate_passed") is False
            and observer.get("graph_qualified") is False
            and observer.get("timing_qualified") is False,
            "Observer final manifest lacks its bounded nonqualification fields")
    case = run / "controlled" / "cached"
    require(case.is_dir() and not case.is_symlink()
            and {p.name for p in case.iterdir()} == {"capture-free-request.json"},
            "Controlled request directory must contain only its scalar completion record")
    scalar = read_json(case / "capture-free-request.json")
    require(scalar.get("id") == "controlled-cached" and scalar.get("request_count") == 1
            and scalar.get("m1_execution") == "capture-free"
            and scalar.get("capture_comparison_available") is False,
            "Capture-free client scalar record differs")
    captured = check_run(captured_run, lane)
    captured_manifest = captured["manifest"]
    require(scalar.get("token_sha256") == captured["request"].get("token_sha256")
            and scalar.get("schedule_sha256") == captured["request"].get("schedule_sha256")
            and observer.get("model_context", {}).get("token_sha256") == captured_manifest.get("token_sha256")
            and observer.get("model_context", {}).get("schedule_sha256") == captured_manifest.get("schedule_sha256"),
            "Observer and captured control use different controlled plans")

    calls_root = observer_root / "calls"
    dirs = sorted(calls_root.glob("call-*"))
    require([p.name for p in dirs] == [f"call-{i:04d}" for i in range(30)]
            and all(p.is_dir() and not p.is_symlink() for p in dirs),
            "Observer native call count differs from the 30-call controlled bound")
    baseline_calls = [c for c in captured["calls"] if c[0].get("scope") == "controlled_live_request"]
    require(len(baseline_calls) == 30, "Captured lane lacks thirty request-bound calls")
    route_index = {}
    for record in captured_manifest["routes"]:
        route_index[(record["forward"], record["layer"])] = read_npz(
            captured["case"], record["file"], record["sha256"], ROUTE_FIELDS)

    call_bindings = []
    for index, (directory, baseline) in enumerate(zip(dirs, baseline_calls)):
        receipt = read_json(directory / "observer-receipt.json")
        frame = receipt.get("frame", {})
        events = [json.loads(line) for line in read_bytes(directory / "events.jsonl", 4 << 20).splitlines()]
        require(receipt.get("schema") == SCHEMA and receipt.get("call_index") == index
                and receipt.get("lane_requested") == lane
                and receipt.get("native_status") == (1 if lane == "fused" else 0)
                and receipt.get("lease_released") is True
                and receipt.get("internal_capture_enabled") is False
                and receipt.get("callback_error") is None
                and receipt.get("stream_id_api") == STREAM_ID_API
                and receipt.get("cupti_stream_id_provider") == provider
                and receipt.get("per_thread_stream") is False
                and type(receipt.get("stream")) is int and receipt["stream"] != 0
                and type(receipt.get("profiler_stream_id")) is int
                and receipt["profiler_stream_id"] != 0
                and frame.get("request_id") == scalar["id"]
                and frame.get("binary_sha256") == build_report["binary_sha256"]
                and frame.get("owner_extents") == baseline[0].get("owner_extents")
                and frame.get("layer_name") == baseline[0].get("layer_name")
                and frame.get("forward_index") == baseline[0].get("forward_index")
                and frame.get("positions") == baseline[0].get("positions")
                and frame.get("token_ids") == baseline[0].get("tokens"),
                "Observer lease identity differs from its captured control")
        counts = _check_event_counts(events, receipt.get("event_counts"), lane)
        require(len(events) == sum(counts.values())
                and all(e.get("sequence") == i and e.get("thread_id") == receipt.get("thread_id")
                        and e.get("stream") == receipt.get("stream")
                        and e.get("stream_id_api") == receipt.get("stream_id_api")
                        and e.get("cupti_stream_id_provider") == provider
                        and e.get("profiler_stream_id") == receipt.get("profiler_stream_id")
                        for i, e in enumerate(events))
                and receipt.get("stream") != 0,
                "Native observer event counts, order, thread or stream differ")
        require(events[0]["event"] == "lease_begin" and events[-1]["event"] == "lease_end"
                and events[0]["value"] == 0 and events[-1]["value"] == receipt["native_status"],
                "Observer did not prove native begin/end with capture disabled")
        status = json.loads(read_bytes(directory / "preparation.json", 1 << 20))
        require(status == {"backend": lane, "incumbent_map_result": lane == "stock"},
                "Native preparation backend differs from the requested lane")
        runner = read_bytes(directory / "runner.json", 1 << 20)
        workspace = read_bytes(directory / "workspace.json", 1 << 20)
        require(b"tile shape ID: 128x128x128" in runner
                and b"epilogue fusion type: 0" in runner
                and b"workspace_bytes=3185408" in workspace,
                "Actual runner tactic/workspace identity is missing")
        baseline_path = next(p for p in (captured_run / "preparation").glob("call-*")
            if read_json(p / "receipt.json").get("scope") == "controlled_live_request"
            and read_json(p / "receipt.json").get("layer_name") == baseline[0].get("layer_name")
            and read_json(p / "receipt.json").get("forward_index") == baseline[0].get("forward_index"))
        native_payloads = {}
        for name, size in PAYLOADS.items():
            value = read_bytes(directory / "payloads" / name, size)
            require(len(value) == size and value == baseline[1][name],
                    "Capture-free native payload differs from captured control: " + name)
            native_payloads[name] = value
        for name in ("consumer-envelopes.json", "consumer-masks.json"):
            require(read_bytes(directory / name, 2 << 20)
                    == read_bytes(baseline_path / name, 2 << 20),
                    "Capture-free consumer descriptor evidence differs: " + name)
        require((runner + workspace).decode() == baseline[0].get("native_metadata"),
                "Capture-free native runner metadata differs from captured control")
        layer = int(re.search(r"layers\.(\d+)\.moe\.experts$", frame["layer_name"])[1])
        route = route_index[(frame["forward_index"], layer)]
        ids = np.frombuffer(native_payloads["ids.bin"], dtype="<i4").reshape(1, 8)
        weights = np.frombuffer(native_payloads["route-weights.bin"], dtype="<u4").reshape(1, 8)
        expected_weights = route["weight_bits"].copy()
        for target_layer, expert in TARGETS:
            if target_layer == layer:
                expected_weights[route["ids"] == expert] = 0
        require(ids.tobytes() == route["ids"].tobytes()
                and weights.tobytes() == expected_weights.tobytes()
                and frame["positions"] == route["positions"].tolist()
                and frame["token_ids"] == route["tokens"].tolist(),
                "Native candidate inputs differ from actual controlled route tensors")
        call_bindings.append({"thread_id": receipt["thread_id"],
                              "stream_handle": receipt["stream"],
                              "profiler_stream_id": receipt["profiler_stream_id"],
                              "stream_id_api": receipt["stream_id_api"],
                              "cupti_stream_id_provider": receipt["cupti_stream_id_provider"],
                              "per_thread_stream": receipt["per_thread_stream"]})

    trace_path = observer_root / observer["trace"]
    require(observer.get("trace_sha256") == _sha(trace_path, MAX_TRACE),
            "External observer trace hash differs")
    trace = _per_call_trace(_load_trace(trace_path), lane, call_bindings)
    arrays = _compare_model_arrays(observer_root, observer, captured)
    return {"lane": lane, "native_calls": 30,
            "candidate_launches_correlated": 30 if lane == "fused" else 0,
            "native_payloads_bit_exact_to_captured_lane": True,
            "model_arrays": arrays, "trace": trace,
            "internal_capture_hooks_inactive": True,
            "observer_perturbs_execution": True, "timing_qualified": False}


def compare_observers(build, stock_captured, fused_captured, stock_observed, fused_observed):
    captured_pair = compare_runs(stock_captured, fused_captured, build)
    stock = check_external_run(stock_observed, "stock", stock_captured, build)
    fused = check_external_run(fused_observed, "fused", fused_captured, build)
    require(stock["trace"]["per_call_incumbent_kernel_counts"]
            == fused["trace"]["per_call_incumbent_kernel_counts"],
            "Capture-free stock/fused incumbent kernel choices differ")
    require(stock["model_arrays"]["counts"] == fused["model_arrays"]["counts"],
            "Capture-free stock/fused model-array counts differ")
    return {"schema": SCHEMA, "scope": "capture_free_external_observer_validation",
            "passed": True, "stock": stock, "fused": fused,
            "captured_baseline": captured_pair,
            "native_outputs_bit_exact_to_captured": True,
            "model_outputs_bit_exact_to_captured": True,
            "stock_fused_incumbent_kernels_match": True,
            "quality_qualified": False, "graph_qualified": False,
            "timing_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--stock-captured", type=Path, required=True)
    parser.add_argument("--fused-captured", type=Path, required=True)
    parser.add_argument("--stock-observed", type=Path, required=True)
    parser.add_argument("--fused-observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    inputs = (args.build, args.stock_captured, args.fused_captured,
              args.stock_observed, args.fused_observed)
    require(all(destination != path.resolve() and path.resolve() not in destination.parents
                for path in inputs), "Report must be outside each input evidence path")
    report = compare_observers(args.build, args.stock_captured, args.fused_captured,
                               args.stock_observed, args.fused_observed)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
