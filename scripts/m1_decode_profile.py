"""Count-only admission for the explicit four-step lean decode diagnostic.

CPU timestamps establish containment/correlation only. No elapsed, utilization,
gap, performance or quality result is produced. Raw traces and tokens stay private.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from megartx.m1_eager_benchmark import PROFILE_DRIVER_SOURCES, PROFILE_RULE, require_profile_intent

# Full GPU operation names, launch geometry/shared memory and transfer bytes.
# Derived prospectively from accepted 2c65 stock/fused traces: stock unchanged;
# fused removes exactly 120 32-byte route D2H copies. No failed-run admission.
OPERATION_DIGESTS = {
    "stock": "c0189366d990def557ae2a9bbcb6b0027c9bf3e90e3ba12244d92e8cbfe4f033",
    "fused": "e9ed3d4c73bfa0240adc6c0f8af176baaab33984cd690d6622ae0a09fd0f79cc",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("decode diagnostic " + message)


def read_json(path, limit=1 << 20):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit,
            "evidence must be bounded regular JSON: " + path.name)
    return json.loads(path.read_text())


def operation_digest(events):
    inventory = Counter(json.dumps({"category": e["cat"], "name": e["name"],
        "geometry": {k: e["args"][k] for k in ("grid", "block", "shared memory", "bytes")
                     if k in e["args"]}}, sort_keys=True) for e in events)
    return hashlib.sha256(json.dumps(dict(inventory), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contains(parent, event):
    return (parent["pid"] == event["pid"] and parent["tid"] == event["tid"]
            and parent["ts"] <= event["ts"]
            and event["ts"] + event["dur"] <= parent["ts"] + parent["dur"] + .001)


def verify_runtime_files(plan, launch):
    """Recheck the already bound build/source/package/AOT/binary files untimed."""
    from m1_private_aot import sha, validate_cache
    project = Path(__file__).resolve().parents[1]
    env = launch["environment_overrides"]
    require(env.get("MEGARTX_M1_EXECUTION") == "capture-free"
            and launch.get("adapter_mode") == "native" and launch.get("m1_external_observer_requested") is False
            and {"--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching"}.issubset(launch["command"])
            and not any(env.get(k) for k in ("MEGARTX_M1_EXTERNAL_OBSERVER_DIR", "MEGARTX_LOGITS_DIR",
                                           "MEGARTX_CONTROLLED_DIR", "MEGARTX_M1_CAPTURE_DIR")), "execution/observer mode differs")
    build = read_json(env["MEGARTX_M1_BUILD_RECEIPT"])
    require(build.get("base_head") == plan["source_head"] and build.get("returncode") == 0
            and not build.get("reason") and build.get("compiled_lease_controls_returncode") == 0
            and build.get("compiled_binding_controls_returncode") == 0 and build.get("required_exports_present")
            and sha(env["MEGARTX_M1_BRIDGE"]) == build.get("binary_sha256")
            and build.get("live_contract", {}).get("controller_source_hashes") == plan["controller_source_hashes"],
            "final build/binary/source identity differs")
    sources = build["source_hashes"]
    required_sources = set(PROFILE_DRIVER_SOURCES) | {"src/megartx/" + p for p in plan["controller_source_hashes"]} | {
        "probes/m1_live_bridge.cu", "probes/m1_installed_bridge.cuh", "kernels/m1_installed_preparation.cuh",
        "kernels/m1_maps_expand.cuh", "scripts/build_m1_live_bridge.py", "scripts/check_m1_live_bridge.py",
        "scripts/check_m1_live_bindings.py"}
    require(set(sources) == required_sources and all(sources[p] == h for p, h in plan["driver_source_hashes"].items()),
            "build driver inventory differs")
    require(all(sources.get("src/megartx/" + p) == h for p, h in plan["controller_source_hashes"].items()),
            "build controller inventory differs")
    for path, expected in sources.items():
        relative = Path(path)
        require(not relative.is_absolute() and ".." not in relative.parts
                and not any((project / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1))
                and sha(project / relative) == expected, "final repository source drift")
    require(bool(build["installed_pins"]) and all(sha(p) == h for p, h in build["installed_pins"].items()), "final installed source/binary drift")
    packages = {"vllm": "0.30.0", "flashinfer-python": "0.6.18.post1", "torch": "2.13.0", "nvidia-cuda-cupti": "13.0.85"}
    require(build["installed_package_versions"] == packages
            and all(importlib.metadata.version(p) == h for p, h in packages.items()), "final installed package drift")
    aot = Path(env["MEGARTX_M1_PRIVATE_AOT"])
    validate_cache(aot, plan["source_head"])
    require(sha(aot / "manifest.json") == launch["private_aot_manifest_sha256"]
            and sha(aot / "cpu-dry-run.json") == launch["private_aot_cpu_dry_run_sha256"], "final AOT receipt drift")


def validate_trace(trace, lane):
    devices = trace.get("deviceProperties", [])
    require(len(devices) == 1 and devices[0].get("id") == 0
            and devices[0].get("name") == "NVIDIA GeForce RTX 5090"
            and devices[0].get("computeMajor") == 12 and devices[0].get("computeMinor") == 0,
            "trace device identity differs")
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    for e in events:
        require(all(type(e.get(k)) in (int, float) and math.isfinite(e[k]) and e[k] >= 0
                    for k in ("ts", "dur")), "trace interval is invalid")
    api = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    # gpu_user_annotation is Kineto range metadata, not a device operation.
    gpu = [e for e in events if e.get("cat") == "kernel"
           or (e.get("cat", "").startswith("gpu_") and e["cat"] != "gpu_user_annotation")]
    require(len(gpu) == (6440 if lane == "stock" else 6320), "GPU operation count differs")
    require(all(e["cat"] in ("kernel", "gpu_memcpy") for e in gpu), "unexpected GPU operation category")
    require(operation_digest(gpu) == OPERATION_DIGESTS[lane], "GPU operation identity/geometry/bytes differ")
    require(all(type(e["args"].get("stream")) is int and e["args"]["stream"] >= 0
                and e["args"].get("device") == 0 and e["args"].get("graph id") == 0 for e in gpu)
            and len({e["args"].get("context") for e in gpu}) == 1, "GPU stream/context/graph identity differs")
    correlations = defaultdict(list)
    for e in api:
        if "correlation" in e.get("args", {}):
            correlations[e["args"]["correlation"]].append(e)
    launches = {}
    for e in gpu:
        matches = correlations.get(e["args"].get("correlation"), [])
        require(len(matches) == 1, "GPU/API correlation is missing or ambiguous")
        launches[id(e)] = matches[0]
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
    named = lambda name: sorted((e for e in cpu if e["name"] == name), key=lambda e: e["ts"])
    models, heads, samples = (named(n) for n in ("megartx::model_forward", "megartx::logits_head", "aten::argmax"))
    require(len(models) == len(heads) == len(samples) == 4, "full model/head/sampler coverage differs")
    execution = {(e["pid"], e["tid"]) for e in (*models, *heads, *samples)}
    require(len(execution) == 1, "model/head/sampler execution identity differs")
    process, _ = next(iter(execution))
    require(all(call["pid"] == process for call in api), "CUDA API owner process differs")
    prep = sorted((e for e in cpu if e["name"].startswith("megartx::m1_preparation_")), key=lambda e: e["ts"])
    routed = [e for e in cpu if e["name"].startswith("megartx::m1_routed_")]
    corrections = named("megartx::correction_selection")
    require(len(prep) == len(routed) == 120 and len(corrections) == 24, "preparation/routed/correction coverage differs")
    for i, model in enumerate(models):
        require(model["ts"] + model["dur"] <= heads[i]["ts"]
                and heads[i]["ts"] + heads[i]["dur"] <= samples[i]["ts"], "model/head/sampler order differs")
        if i < 3:
            require(samples[i]["ts"] + samples[i]["dur"] <= models[i+1]["ts"], "decode frame order differs")
        layers = [e for e in prep if contains(model, e)]
        expected = [f"megartx::m1_preparation_{lane}::language_model.model.layers.{layer}.moe.experts" for layer in range(30)]
        require([e["name"] for e in layers] == expected, "frame/layer preparation identities differ")
        routes = sorted((e for e in routed if contains(model, e)), key=lambda e: e["ts"])
        require(len(routes) == 30 and all(e["name"] == "megartx::m1_routed_" + lane for e in routes)
                and sum(contains(model, e) for e in corrections) == 6, "frame routed/correction identity differs")
        require(all(contains(route, layer) and sum(contains(route, e) for e in layers) == 1
                    for route, layer in zip(routes, layers)), "preparation/routed containment differs")
    stream = None
    for scope in prep:
        calls = [e for e in api if contains(scope, e)]
        device = [e for e in gpu if contains(scope, launches[id(e)])]
        counts = Counter(e["name"] for e in calls)
        require(counts["cudaMemcpyAsync"] == 15 and counts["cudaStreamSynchronize"] == 3
                and counts["cudaPointerGetAttributes"] == 10, "preparation copy/fence/pointer counts differ")
        require(Counter(e["args"]["bytes"] for e in device if e["cat"] == "gpu_memcpy")
                == {32: 1, 3072: 2, 2560: 2, 1024: 10}, "descriptor/route transfer counts differ")
        require(sum(e["cat"] == "kernel" for e in device) == (7 if lane == "stock" else 6), "preparation kernel count differs")
        streams = {e["args"]["stream"] for e in device}
        require(len(streams) == 1 and (stream is None or streams == {stream}), "preparation stream dependency differs")
        stream = next(iter(streams))
    return {"gpu_operations": len(gpu), "unique_gpu_api_correlations": len(launches),
            "gpu_operation_inventory_sha256": operation_digest(gpu), "model_frames": 4, "head_frames": 4,
            "sampler_frames": 4, "preparation_calls": 120, "routed_calls": 120, "correction_selections": 24,
            "preparation_d2h_copies": 1800, "preparation_stream_fences": 360, "preparation_pointer_queries": 1200,
            "correction_layer_attribution": "withheld; selection is outside the routed annotation",
            "stream_wait_dependency_duration": "not measured"}


def validate_profile_run(directory, plan, owned):
    from m1_eager_benchmark_client import validate_dispatch
    from megartx.nvfp4_qualification import activation as validate_activation
    require_profile_intent(plan, True)
    root = Path(directory)
    require((root / "run.exit").read_text().strip() == "0"
            and (root / "benchmark.exit").read_text().strip() == "0", "lifecycle did not pass")
    require(owned["cleanup_complete"] is True and not owned["cleanup_errors"]
            and not owned["owned_identities_remaining"] and not owned["owned_gpu_pids_remaining"]
            and owned["failure"] is None and not owned["timing_metadata_policy_invalid"]
            and owned["compiler_rss_limit_bytes"] == 2 << 30 and owned["shared_compiler_seconds_limit"] == 300
            and owned["sampled_peak_compiler_rss_bytes"] <= 2 << 30
            and owned["shared_compiler_elapsed_seconds"] <= 300, "owned resource/cleanup admission differs")
    require(owned["timing_metadata_policy"]["final"]["passed"] is True, "final metadata files did not pass")
    launch = read_json(root / "launch-manifest.json")
    require(launch.get("m1_decode_profile_requested") is True and launch.get("diagnostic_admission") == PROFILE_RULE
            and launch.get("eager_benchmark_plan_sha256") == plan["plan_sha256"], "launch intent differs")
    require(load_same_plan(root, plan), "saved plan identity differs")
    verify_runtime_files(plan, launch)
    requests = [json.loads(line) for line in (root / "eager-requests.jsonl").read_text().splitlines()]
    dispatch = read_json(root / "eager-benchmark/dispatch.json")
    validate_dispatch(plan, dispatch, requests)
    for r in requests:
        require(len(r["token_ids"]) == 256 and all(type(t) is int and 0 <= t < 262144 for t in r["token_ids"])
                and r["usage"]["prompt_tokens"] == int(r["case"])
                and r["usage"]["completion_tokens"] == 256 and r["finish_reason"] == "length", "output/usage differs")
    for case in ("2048", "8192"):
        rows = [r for r in requests if r["case"] == case]
        require(len(rows) == 4 and all(r["token_ids"] == rows[0]["token_ids"] for r in rows), "warmup/measurement output equality differs")
    activation = read_json(root / "activation-proof.json")
    validate_activation(root)
    require(activation["forced_all_six_executed"] is True and activation["natural_model_forward_verified"] is True
            and {r["layer_name"] for r in activation["registered_layers"]}
                == {f"language_model.model.layers.{i}.moe.experts" for i in range(30)}
            and {(r["layer_name"], r["expert"]) for r in activation["forced_fixtures"]}
                == {(f"language_model.model.layers.{i}.moe.experts", e) for i, e in ((0,42),(0,82),(1,126),(2,89),(3,7),(5,12))}
            and all(r["observed_bf16_value_equal"] is True and r["max_absolute_difference"] == 0
                    for r in activation["forced_fixtures"]), "startup numerical fixtures differ")
    client = read_json(root / "eager-client-validation.json")
    require(client["plan_sha256"] == plan["plan_sha256"] and client["requests"] == 8
            and all(client[k] is True for k in ("matched_tokens_usage", "actual_input_transcripts_verified", "native_backends_verified")),
            "client transcript/dispatch validation differs")
    lanes = {}
    for lane in ("stock", "fused"):
        path = root / "decode-profile" / (lane + ".json")
        record = read_json(path.with_name(lane + "-scalars.json"))
        row = next(r for r in plan["schedule"] if r["phase"] == "measurement" and r["case"] == "2048" and r["lane"] == lane)
        require(record["lane"] == lane and record["decode_steps"] == 4 and record["request_id"] == row["id"]
                and record["positions_relative_to_context"] == [0, 1, 2, 3]
                and record["source_head"] == plan["source_head"] and record["plan_sha256"] == plan["plan_sha256"]
                and record["timing_qualified"] is False, "window/source/request evidence differs")
        phases = record["host_phases"]
        require({k: v["count"] for k, v in phases.items()} == {"native_runner": 120, "map_eligibility": 0,
            "map_dispatch": 120, "descriptor_readback_fence": 240, "descriptor_validation_enumeration": 240}, "native phase counts differ")
        lanes[lane] = validate_trace(read_json(path, 64 << 20), lane)
        lanes[lane]["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema": "megartx-m1-decode-diagnostic-admission-v1", "source_head": plan["source_head"],
            "plan_sha256": plan["plan_sha256"], "diagnostic_admission": PROFILE_RULE, "diagnostic_admission_passed": True,
            "timing_qualified": False, "performance_gate_passed": False, "quality_qualified": False, "graphs_qualified": False,
            "observed_stock_fused_warmup_measurement_tokens_equal": True, "six_forced_bf16_fixtures_passed": True,
            "natural_correction_selected_rows": sum(sum(r["natural_correction_selected_rows"].values()) for r in dispatch["records"]),
            "compiler_unknown_or_work_identities": len(owned["timing_unknown_or_work_identities"]),
            "compiler_sampled_peak_rss_bytes": owned["sampled_peak_compiler_rss_bytes"],
            "compiler_shared_budget_seconds_observed": owned["shared_compiler_elapsed_seconds"],
            "cleanup_complete": True, "lanes": lanes}


def load_same_plan(root, plan):
    from megartx.m1_eager_benchmark import load_plan
    return load_plan(root / "eager-benchmark-plan.json") == plan
