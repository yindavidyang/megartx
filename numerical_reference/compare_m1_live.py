"""Strict, read-only proof of request-bound M1 stock/fused live execution.

Controlled cached routing and eager execution only. Artificial routes are kept
separate. The byte oracle does not qualify model quality, graphs or timing.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
import m1_preparation_reference as oracle
from controlled_reference import TARGETS
from compare_controlled_capture import read_bytes, read_json, read_npz, MAX_TRACE, trace_summary, check_proof
from compare_controlled_capture import ROUTE_FIELDS, STAGE_FIELDS, LOGIT_FIELDS, KV_FIELDS


EXTENTS = [1408, 22528, 32, 32, 5632, 3185408, 32, 253755392, 126877696,
           512, 31719424, 512, 512, 15859712, 512]
BINARY_FILES = {"input-aq.bin": 1408, "input-sf.bin": 22528, "ids.bin": 32,
                "route-weights.bin": 32, "sf-before.bin": 2883584, "sf-after.bin": 2883584,
                "expanded-aq.bin": 11264, "slot-to-sorted.bin": 32,
                "sorted-to-slot.bin": 32, "offsets.bin": 1032, "routed-output.bin": 5632,
                **{n+".bin": 512 for n in ("fc1-act-global", "fc1-global", "fc2-act-global", "fc2-global")}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payload(directory, name, size):
    path = directory / name
    require(not path.is_symlink(), "Captured payload cannot be a symlink")
    result = read_bytes(path, size)
    require(len(result) == size, "Incorrect captured extent: " + name)
    return result


def check_call(directory, receipt, lane):
    require(receipt.get("schema") == "megartx-m1-live-v2" and receipt.get("lane") == lane
            and receipt.get("actual_backend") == lane, "Missing positive actual backend")
    require(receipt.get("execution_mode") == "eager" and receipt.get("pdl") is False
            and receipt.get("finalize_fusion") is False, "Unmatched execution mode")
    require(all(receipt.get(k) is True for k in ("producer_wait_inserted", "consumer_wait_inserted", "lease_released"))
            and receipt.get("routed_call_failed") is False and "error" not in receipt,
            "Missing successful lease/stream cleanup")
    require(receipt.get("owner_extents") == EXTENTS and receipt.get("workspace_bytes") == EXTENTS[5],
            "Actual owner extents differ")
    require(receipt.get("forward_index") == 1 and receipt.get("positions") == [32]
            and len(receipt.get("tokens", [])) == 1, "Missing actual M1 forward identity")
    require(receipt.get("live_contract", {}).get("abi_version") == 2
            and receipt["live_contract"].get("view_count") == 15
            and receipt["live_contract"].get("view_bytes") == 32, "Unqualified owner ABI")
    metadata = receipt.get("native_metadata", "")
    for text in ("activation=4 expected_activation=4", "per_expert_scale=1 fc2_per_expert_scale=1",
                 "tile shape ID: 128x128x128", "epilogue fusion type: 0", "swap_ab: false"):
        require(text in metadata, "Actual typed runner/tactic identity differs")
    data = {name: payload(directory, name, size) for name, size in BINARY_FILES.items()}
    row = oracle.InputRow(struct.unpack("<8i", data["ids.bin"]), data["route-weights.bin"],
                          data["input-aq.bin"], data["input-sf.bin"])
    # StageContext is scaffolding for the preparation byte oracle only. Actual
    # per-expert QuantParams are captured/retained and compared separately.
    one = struct.pack("<f", 1.)
    contexts = [oracle.StageContext(stage, False, "none", one*128, one,
                (one*128,) * (2 if stage == "fc1" else 1), "byte_oracle_only") for stage in ("fc1", "fc2")]
    expected = oracle.prepare(row, *contexts, reference_abi=oracle.REFERENCE_ABI)
    for name, value in (("slot-to-sorted.bin", expected.slot_to_sorted),
                        ("sorted-to-slot.bin", expected.sorted_to_slot), ("offsets.bin", expected.expert_offsets),
                        ("expanded-aq.bin", expected.expanded_aq)):
        require(data[name] == value, "Independent preparation oracle mismatch: " + name)
    oracle.require_sf_storage(expected, data["sf-before.bin"], data["sf-after.bin"])
    for name in ("fc1-act-global.bin", "fc1-global.bin", "fc2-act-global.bin", "fc2-global.bin"):
        values = struct.unpack("<128f", data[name])
        require(all(math.isfinite(v) and v > 0 for v in values), "Invalid original global tensor")
    shapes = read_json(directory / "consumer-envelopes.json")
    masks = read_json(directory / "consumer-masks.json")
    require(masks.get("scope") == "source_bound_actual_sm120_grouped_tma_payload_masks"
            and masks.get("tile_mn") == [128, 128] and masks.get("physical_tile_k") == 256
            and masks.get("sf_vector_size") == 16 and masks.get("inactive_group_metadata_may_be_read") is True,
            "Unqualified physical consumer mask")
    require(len(shapes.get("stages", [])) == 2 and len(masks.get("stages", [])) == 2, "Missing physical consumer stage")
    offsets = struct.unpack("<129q", data["offsets.bin"])
    active = sorted(row.selected_ids)
    for i, (stage_shapes, stage_mask) in enumerate(zip(shapes["stages"], masks["stages"])):
        n, k = (1408, 2816) if i == 0 else (2816, 704)
        require(stage_shapes["stage"] == i+1 and stage_shapes["swap_ab"] is False
                and len(stage_shapes["experts"]) == 128 and stage_mask["stage"] == i+1,
                "Actual descriptor stage differs")
        expected_masks = []
        for e, shape in enumerate(stage_shapes["experts"]):
            m = offsets[e+1] - offsets[e]
            base = oracle.grouped_sf_base(e, offsets[e], k)
            require(shape == [m, n, k, base if m else -1, 128*(k//16) if m else 0],
                    "Actual consumer descriptor differs from routed shape")
            if m:
                expected_masks.append({"expert": e, "shape": [1, n, k],
                    "aq_read_range": [2944+offsets[e]*(k//2), k//2],
                    "sf_read_range": [base, 128*(k//16)],
                    "gemm_output_write_range": [48000+offsets[e]*n*2, n*2]})
        require(stage_mask["active_payloads"] == expected_masks, "Physical read/write ranges differ")
        require([p["expert"] for p in expected_masks] == active, "Missing active consumer group")
    return data, masks, shapes


def load_build(directory):
    directory = Path(directory)
    report = read_json(directory / "build.json")
    require(report.get("returncode") == 0 and not report.get("reason")
            and report.get("compiled_lease_controls_returncode") == 0
            and report.get("compiled_binding_controls_returncode") == 0
            and report.get("required_exports_present") is True
            and report.get("installed_pins_unchanged") is True, "Build did not pass compiled gates")
    digest = hashlib.sha256(read_bytes(directory/"m1_live_bridge.so", 64<<20)).hexdigest()
    require(digest == report.get("binary_sha256"), "Compiled binary hash differs")
    for name, digest in report["source_hashes"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and hashlib.sha256(read_bytes(directory/name, 8<<20)).hexdigest() == digest,
                "Compiled source snapshot differs")
    contract = report["live_contract"]
    require(contract.get("abi_version") == 2 and contract.get("view_count") == 15 and contract.get("view_bytes") == 32
            and contract.get("native_source_sha256") == report["source_hashes"].get("probes/m1_live_bridge.cu")
            and contract.get("controller_source_hashes") == {name: report["source_hashes"].get("src/megartx/"+name)
               for name in ("m1_live.py", "vllm_scale_plugin.py")}, "Compiled source/ABI contract differs")
    header = read_bytes(directory/"m1_live_symbols.h", 1<<20).decode()
    declarations = [line.removeprefix("#define M1_LIVE_CONTRACT_JSON ") for line in header.splitlines()
                    if line.startswith("#define M1_LIVE_CONTRACT_JSON ")]
    require(len(declarations) == 1 and json.loads(json.loads(declarations[0])) == contract, "Generated compiled contract differs")
    lease, binding = read_json(directory/"lease-controls.json"), read_json(directory/"binding-controls.json")
    require(lease.get("live_contract") == contract and lease.get("invalid_framing_before_dereference") is True
            and lease.get("historical_begin_symbol_absent") is True and lease.get("active_after") == 0
            and binding.get("torch_runtime") == "2.13.0+cu130" and binding.get("relocations_bound_to_bridge") == 4
            and binding.get("active_after") == 0, "Compiled framing/actual binding controls differ")
    return report


def check_run(run, lane):
    run = Path(run)
    require(read_json(run / "status.json").get("phase") == "cleanup_complete", "Owned lifecycle incomplete")
    require(not (run / "QUALIFICATION-INVALIDATED.json").exists(), "Run explicitly invalidated")
    launch = read_json(run / "launch-manifest.json")
    require(launch.get("m1_preparation_requested") == lane and launch.get("controlled_request_count") == 1
            and launch.get("controlled_path") == "cached" and launch.get("m1_route_controls") is True,
            "Run lacks bounded matched request declaration")
    case = run / "controlled/cached"
    request = read_json(case / "request.json")
    manifest = read_json(case / "controlled-manifest.json")
    require(request.get("id") == "controlled-cached" and request.get("request_count") == 1
            and manifest.get("mode") == "native" and manifest.get("path") == "cached"
            and manifest.get("input_tokens") == 33 and manifest.get("forward_calls") == 2
            and manifest.get("route_origin") == "controlled" and manifest.get("routing_unchanged") is False,
            "Request identity or controlled model scope differs")
    paths = sorted((run / "preparation").glob("call-*"))
    require(all(p.is_dir() and not p.is_symlink() for p in paths), "Captured call directory cannot be a symlink")
    require([p.name for p in paths] == [f"call-{i:04d}" for i in range(32)], "Missing/extra live calls")
    calls, layers, artificial = [], set(), []
    for path in paths:
        record = read_json(path / "receipt.json")
        require(record.get("request", {}).get("id") == request["id"]
                and record["request"].get("token_sha256") == manifest["token_sha256"]
                and record["request"].get("schedule_sha256") == manifest["schedule_sha256"],
                "Lease is not bound to the captured request")
        values = check_call(path, record, lane)
        if record.get("scope") == "controlled_live_request":
            match = re.fullmatch(r"language_model.model.layers.(\d+).moe.experts", record.get("layer_name", ""))
            require(match is not None and int(match[1]) not in layers, "Repeated/unbound request layer")
            layers.add(int(match[1]))
            route_records = [r for r in manifest["routes"] if r["layer"] == int(match[1]) and r["forward"] == 1]
            require(len(route_records) == 1, "Missing unique actual model route")
            route = route_records[0]
            actual = read_npz(case, route["file"], route["sha256"], ROUTE_FIELDS)
            # Existing corrected runner masks its six corrected expert weights
            # in the ordinary grouped call, then adds their separate outputs.
            ordinary_weights = actual["weight_bits"].copy()
            for layer, expert in TARGETS:
                if layer == int(match[1]): ordinary_weights[actual["ids"] == expert] = 0
            require(actual["tokens"].tolist() == record["tokens"] and actual["positions"].tolist() == record["positions"]
                    and actual["ids"].dtype == np.int32 and actual["ids"].shape == (1,8)
                    and actual["ids"].tobytes() == values[0]["ids.bin"]
                    and actual["weight_bits"].dtype == np.uint32 and actual["weight_bits"].shape == (1,8)
                    and ordinary_weights.tobytes() == values[0]["route-weights.bin"],
                    "Lease inputs differ from actual captured model routes")
        elif record.get("scope") == "artificial_route_control":
            artificial.append((record, struct.unpack("<8i", values[0]["ids.bin"])))
        else:
            raise ValueError("Unknown live call scope")
        calls.append((record, *values))
    require(layers == set(range(30)), "Missing positive request layer dispatch")
    require(len(artificial) == 2 and artificial[0][1] == tuple(range(11, 19))
            and artificial[1][1] == tuple(range(65, 73))
            and artificial[0][0].get("route_control_case") == "first_use_omits_zero"
            and artificial[1][0].get("route_control_case") == "reuse_disjoint_omits_zero"
            and artificial[0][0].get("route_control_workspace_reused") is False
            and artificial[1][0].get("route_control_workspace_reused") is True
            and all(0 < r.get("route_control_scratch_bytes", 0) <= 8 << 20 for r, _ in artificial),
            "Artificial first-use/reuse coverage missing")
    fallback = [json.loads(line) for line in read_bytes(run / "preparation/stock-fallbacks.jsonl", 1 << 20).splitlines()]
    require(len(fallback) == 30 and {f.get("layer_name") for f in fallback} ==
            {f"language_model.model.layers.{i}.moe.experts" for i in range(30)}
            and all(f.get("rows") == 32 and f.get("forward_index") == 0
                    and f.get("request_id") == request["id"] for f in fallback), "Full prefill fallback missing")
    check_proof(run, manifest["execution_proof_sha256"])
    corrected = trace_summary(case, manifest)
    trace = check_trace(case, manifest, lane, [c[0]["scope"] for c in calls])
    require(trace["routed_spans"] == 60, "Trace lacks both thirty-layer model forwards")
    return {"run": run, "case": case, "request": request, "manifest": manifest, "calls": calls,
            "trace": trace, "corrected_trace": corrected, "fallback_calls": len(fallback)}


def check_trace(case, manifest, lane, scopes):
    path = case / "controlled-trace.json.gz"
    require(hashlib.sha256(read_bytes(path, MAX_TRACE)).hexdigest() == manifest["cuda_trace_sha256"], "Trace hash differs")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_TRACE+1)
    require(len(raw) <= MAX_TRACE, "Expanded trace exceeds bound")
    return correlate_trace(json.loads(raw)["traceEvents"], lane, scopes)


def correlate_trace(events, lane, scopes):
    require(isinstance(events, list) and len(events) <= 150000, "Trace event count exceeds bound")
    complete = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"]
    spans = sorted((e for e in complete if e.get("cat") == "user_annotation"
                    and e.get("name") == "megartx::m1_preparation_"+lane), key=lambda e: e["ts"])
    outer = [e for e in complete if e.get("cat") == "user_annotation" and e.get("name") == "megartx::m1_routed_"+lane]
    require(len(spans) == len(scopes) and scopes.count("controlled_live_request") == 30
            and scopes.count("artificial_route_control") == 2, "Trace lacks positive request spans")
    for span in spans:
        require(all(type(span.get(f)) in (int, float) and math.isfinite(span[f]) for f in ("ts", "dur"))
                and span["dur"] > 0, "Invalid CPU span")
        require(sum(p.get("pid") == span.get("pid") and p.get("tid") == span.get("tid")
                and p["ts"] <= span["ts"] and span["ts"]+span["dur"] <= p["ts"]+p["dur"] for p in outer) == 1,
                "Preparation lacks unique request routed parent")
    owners = {}
    for event in complete:
        if event.get("cat") not in {"cuda_runtime", "cuda_driver"}: continue
        correlation = event.get("args", {}).get("correlation")
        if type(correlation) is not int: continue
        selected = [i for i,s in enumerate(spans) if event.get("pid") == s.get("pid") and event.get("tid") == s.get("tid")
                    and s["ts"] <= event.get("ts", -1) < s["ts"]+s["dur"]]
        require(len(selected) <= 1 and not (selected and correlation in owners), "Ambiguous launch correlation")
        if selected: owners[correlation] = selected[0]
    counters = [Counter() for _ in spans]
    streams = set()
    for event in complete:
        if event.get("cat") != "kernel": continue
        c = event.get("args", {}).get("correlation")
        if c in owners:
            counters[owners[c]][event["name"]] += 1
            streams.add(event.get("args", {}).get("stream"))
    global_candidates = sum(e.get("cat") == "kernel" and "m1_maps_expand" in e.get("name", "") for e in complete)
    require(global_candidates == (32 if lane == "fused" else 0), "Unexpected candidate launch outside the bounded call set")
    require(len(streams) == 1 and None not in streams and 0 not in streams, "Consumer kernels lack one owned nondefault stream")
    incumbent = []
    for counts in counters:
        fused = sum(v for k,v in counts.items() if "m1_maps_expand" in k)
        maps = sum(v for k,v in counts.items() if "fusedBuildExpertMapsSortFirstTokenKernel" in k)
        expand = sum(v for k,v in counts.items() if "expandInputRowsKernel<" in k)
        gemms = sum(v for k,v in counts.items() if "MainloopSm120ArrayTmaWarpSpecializedBlockScaled" in k)
        require((fused, maps, expand, gemms) == ((1,0,0,2) if lane == "fused" else (0,1,1,2)),
                "Missing positive preparation replacement or unchanged GEMMs")
        incumbent.append({k:v for k,v in counts.items() if "m1_maps_expand" not in k
                          and "fusedBuildExpertMapsSortFirstTokenKernel" not in k and "expandInputRowsKernel<" not in k})
    return {"routed_spans": len(outer), "request_spans": 30, "artificial_spans": 2, "positive_request_fused_launches": 30 if lane == "fused" else 0,
            "positive_artificial_fused_launches": 2 if lane == "fused" else 0, "installed_gemm_launches": 64,
            "cpu_launch_kernel_correlation_verified": True, "owned_nondefault_stream_verified": True,
            "incumbent_kernel_counts": incumbent}


def compare_arrays(left, right):
    require(left["request"] == right["request"], "Controlled requests differ")
    for field in ("token_sha256", "schedule_sha256", "mode", "path", "forward_calls", "input_tokens"):
        require(left["manifest"][field] == right["manifest"][field], "Matched model identity differs")
    files = {}
    for side in (left, right):
        manifest, case = side["manifest"], side["case"]
        entries = [(r["file"], r["sha256"], ROUTE_FIELDS) for r in manifest["routes"]]
        entries += [(r["file"], r["stage_capture_sha256"], STAGE_FIELDS) for r in manifest["executed_interventions"]]
        entries += [(r["file"], r["sha256"], LOGIT_FIELDS) for r in
                    map(json.loads, read_bytes(case/"logits-records.jsonl", 1<<20).splitlines())]
        kv = read_json(case/"kv-records.json")
        entries += [(r["file"], r["sha256"], KV_FIELDS) for r in kv]
        require(len(entries) == 99 and len({e[0] for e in entries}) == len(entries), "Missing/repeated model array captures")
        files[id(side)] = {name: read_npz(case, name, digest, fields) for name,digest,fields in entries}
    require(set(files[id(left)]) == set(files[id(right)]), "Matched capture files differ")
    arrays = elements = 0
    for name,a in files[id(left)].items():
        b = files[id(right)][name]
        require(set(a) == set(b), "Matched capture fields differ")
        for field,x in a.items():
            y = b[field]
            require(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(),
                    "Matched model intermediate/output differs: "+name+":"+field)
            arrays += 1; elements += x.size
    return {"npz_files": len(files[id(left)]), "arrays": arrays, "elements": elements, "bit_exact": True}


def compare_runs(stock, fused, build):
    compiled = load_build(build)
    left, right = check_run(stock, "stock"), check_run(fused, "fused")
    for side in (left, right):
        require(all(c[0]["live_contract"] == compiled["live_contract"] and
                    c[0]["binary_sha256"] == compiled["binary_sha256"] for c in side["calls"]),
                "Actual request lacks matching compiled source/binary admission")
    for a,b in zip(left["calls"], right["calls"]):
        for key in ("scope", "layer_name", "forward_index", "tokens", "positions", "binary_sha256", "live_contract",
                    "owner_extents", "quantization_owners", "execution_mode", "pdl", "finalize_fusion", "native_metadata"):
            require(a[0].get(key) == b[0].get(key), "Unmatched live call identity: "+key)
        require(a[1:] == b[1:], "Matched preparation, whole SF, physical masks or routed output differ")
    require(left["trace"]["incumbent_kernel_counts"] == right["trace"]["incumbent_kernel_counts"],
            "Installed TMA/GEMM/activation/finalization kernels differ")
    return {"scope": "controlled_cached_live_preparation", "passed": True, "request_bound_layers": 30,
            "artificial_route_controls": 2, "whole_sf_bytes_per_call": 2883584,
            "preparation_oracle_bit_exact": True, "matched_payloads_and_routed_outputs_bit_exact": True,
            "model_arrays": compare_arrays(left,right), "stock_trace": {k:v for k,v in left["trace"].items() if k != "incumbent_kernel_counts"},
            "fused_trace": {k:v for k,v in right["trace"].items() if k != "incumbent_kernel_counts"},
            "stock_fallback_calls": left["fallback_calls"], "fused_fallback_calls": right["fallback_calls"],
            "binary_sha256": left["calls"][0][0]["binary_sha256"], "live_contract": left["calls"][0][0]["live_contract"],
            "quality_qualified": False, "natural_routing_qualified": False, "graph_qualified": False, "timing_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    require(all(destination != d.resolve() and d.resolve() not in destination.parents for d in (args.stock,args.fused,args.build)),
            "Report must be outside input evidence")
    report = compare_runs(args.stock,args.fused,args.build)
    with destination.open("x") as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
