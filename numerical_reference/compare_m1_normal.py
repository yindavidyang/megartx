"""Exact, source-bound comparison of the fixed normal-routing correctness plan."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from megartx.m1_normal_plan import frames, load_plan, request, LIVE_CALLS, FALLBACK_CALLS
from compare_m1_live import check_call, load_build, correlate_trace, BINARY_FILES, require
from compare_controlled_capture import read_bytes, read_json, read_npz, check_proof, KV_SOURCE_HASHES
from controlled_reference import TARGETS


def expected_records(plan):
    live, fallback, routes, kv = [], [], [], []
    forward = 0
    for case in plan["cases"]:
        for positions in frames(case):
            classification = "single_row_prefill_tail" if positions == [256] and case["id"] == "split257" else "cached_decode" if len(positions) == 1 else "prefill"
            for layer in range(30):
                value = (case["id"], forward, layer, positions, classification)
                routes.append(value)
                kv.append((case["id"],forward,layer,positions[-1]))
                (live if len(positions) == 1 else fallback).append(value)
            forward += 1
    require(len(live) == LIVE_CALLS and len(fallback) == FALLBACK_CALLS and forward == 12,"Expected normal scope differs")
    return live,fallback,routes,kv


def arrays_for(directory, records):
    for record in records:
        name = record["file"]
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe normal capture path")
        if name.startswith("routes-"):
            fields = {"positions","tokens","ids","weight_bits"} | ({"input_bits"} if record["input_rows"]==1 else set())
        elif name.startswith("output-"):
            fields = {"routed_bits"}
        elif name.startswith("stage-"):
            fields = {"input","q1","sf1","gate","up","activation","q2","sf2","down","quant1_global","quant2_global",
                      "gate_alpha","up_alpha","down_alpha","a1","a2","route_weight_bits","ordinary_bits","routed_bits","rows","slots"}
        elif name.startswith("kv-"):
            fields = {"logical_positions","token_ids","key_bits","value_bits"}
        elif name.startswith("logits-"):
            fields = {"input_positions","input_token_ids","logits"}
        else:
            raise ValueError("Unknown normal numeric capture")
        yield record,read_npz(directory,name,record["sha256"],fields)



def check_run(run, lane, plan, compiled):
    run = Path(run)
    require(read_json(run/"status.json").get("phase") == "cleanup_complete"
            and read_bytes(run/"run.exit",8).strip() == b"0"
            and read_bytes(run/"benchmark.exit",8).strip() == b"0"
            and not (run/"QUALIFICATION-INVALIDATED.json").exists(),"Normal owned lifecycle failed/incomplete")
    launch = read_json(run/"launch-manifest.json")
    require(launch.get("m1_preparation_requested") == lane and launch.get("normal_request_count") == 2
            and launch.get("normal_plan_sha256") == plan["plan_sha256"]
            and launch.get("continuation_constrained") is True and launch.get("m1_route_controls") is False
            and launch.get("adapter_mode") == "native" and "--enforce-eager" in launch["command"] and "--no-async-scheduling" in launch["command"]
            and not (run/"CONTROLLED-ROUTING.json").exists(),"Normal launch differs from the exact plan")
    require(load_plan(run/"normal-plan.json") == plan,"Run plan differs")
    live,fallback,expected_routes,expected_kv = expected_records(plan)
    routes, artifacts, traces = {},{},[]
    for case in plan["cases"]:
        directory = run/"normal"/case["id"]
        require(not (directory/"INVALIDATED.json").exists(),"Normal case invalidated")
        manifest = read_json(directory/"normal-manifest.json")
        marker = request(plan,case)
        observed = read_json(directory/"request.json")
        require(manifest.get("request") == marker and all(observed.get(k) == v for k,v in marker.items())
                and observed.get("request_count") == 1 and observed.get("completion_token_ids") == [plan["continuation_token_id"]]*4
                and observed["usage"]["prompt_tokens"] == len(case["prompt_token_ids"])
                and observed["usage"]["completion_tokens"] == 4
                and manifest.get("forward_calls") == len(frames(case)),"Normal case input/continuation differs")
        check_proof(run,manifest["execution_proof_sha256"])
        for record,arrays in arrays_for(directory,manifest["routes"]):
            key = (case["id"],record["forward_index"],record["layer"])
            require(key not in routes and record.get("routing_unchanged") is True,"Duplicate/intervened normal route")
            routes[key] = (record,arrays)
        rows = [value for value in expected_routes if value[0] == case["id"]]
        require(len(manifest["routes"]) == len(rows),"Missing/extra normal routes")
        for cid,forward,layer,positions,classification in rows:
            record,arrays = routes[(cid,forward,layer)]
            expected_tokens = [marker["prompt_token_ids"][p] for p in positions]
            require(record.get("classification") == classification and record.get("input_rows") == len(positions)
                    and arrays["positions"].dtype == np.int64 and arrays["positions"].tolist() == positions
                    and arrays["tokens"].dtype == np.int64 and arrays["tokens"].tolist() == expected_tokens
                    and arrays["ids"].dtype == np.int32 and arrays["ids"].shape == (len(positions),8)
                    and arrays["weight_bits"].dtype == np.uint32 and arrays["weight_bits"].shape == (len(positions),8),"Normal route identity differs")
            require(set(arrays) == {"positions","tokens","ids","weight_bits"} | ({"input_bits"} if len(positions)==1 else set()),"Normal route capture fields differ")
        outputs = [(case["id"],r["forward_index"],r["layer"]) for r in manifest["outputs"]]
        require(outputs == [(cid,f,l) for cid,f,l,_,_ in live if cid == case["id"]],"Normal routed outputs missing/extra")
        kv_rows = [(case["id"],r["forward_index"],r["layer"],r["position"]) for r in manifest["kv"]]
        require(kv_rows == [v for v in expected_kv if v[0] == case["id"]]
                and all(r.get("logical_mapping_verified") is True and 0 <= r["writer_slot"] < r["capacity"] for r in manifest["kv"]),"Normal immediate writer snapshots differ")
        for record,values in arrays_for(directory,manifest["kv"]):
            layer=record["layer"];heads,dim=(2,512)if layer%6==5 else(8,256)
            pos=record["position"];binding=record["binding"]
            require(record.get("source_sha256")==KV_SOURCE_HASHES
                    and binding.get("layer")==layer and binding.get("dtype")=="bf16"
                    and binding.get("registry_owner_verified")is True
                    and binding.get("metadata_writer_slots_equal")is True
                    and values["logical_positions"].dtype==np.int64 and values["logical_positions"].tolist()==[pos]
                    and values["token_ids"].dtype==np.int64 and values["token_ids"].tolist()==[marker["prompt_token_ids"][pos]],
                    "Normal K/V snapshot lacks source-bound actual writer row identity")
            for field in ("key_bits","value_bits"):
                require(values[field].dtype==np.uint16 and values[field].shape==(1,heads,dim)
                        and not np.any((values[field]&0x7F80)==0x7F80),"Normal K/V payload geometry/nonfinite bits differ")
        logits = [json.loads(line) for line in read_bytes(directory/"logits-records.jsonl",65536).splitlines()]
        require(len(logits)==4 and all(r.get("row_identity_verified") is True for r in logits),"Missing normal raw logits")
        for offset,(_,values) in enumerate(arrays_for(directory,logits)):
            pos = len(case["prompt_token_ids"])-1+offset
            require(values["input_positions"].tolist()==[pos] and values["input_token_ids"].tolist()==[marker["prompt_token_ids"][pos]]
                    and values["logits"].dtype==np.float32 and values["logits"].shape==(1,262144)
                    and np.isfinite(values["logits"]).all(),"Normal raw-logit row identity differs")
        entries = manifest["routes"]+manifest["outputs"]+manifest["stages"]+manifest["kv"]+logits
        require(len({r["file"] for r in entries})==len(entries),"Repeated normal array file")
        artifacts[case["id"]] = (directory,entries)
        trace = directory/"normal-trace.json.gz"
        require(hashlib.sha256(read_bytes(trace,128<<20)).hexdigest()==manifest["trace_sha256"],"Normal trace hash differs")
        with gzip.open(trace,"rb") as stream:
            raw = stream.read((512<<20)+1)
        require(len(raw)<=512<<20,"Normal trace exceeds its bound")
        events = json.loads(raw)["traceEvents"]
        count = sum(v[0]==case["id"] for v in live)
        correlated = correlate_trace(events,lane,["normal_live_request"]*count,
            request_scope="normal_live_request",request_count=count,artificial_count=0)
        require(correlated["routed_spans"]==30*len(frames(case)),"Normal trace lacks complete model forwards")
        traces.append(correlated)
    paths = sorted((run/"preparation").glob("call-*"))
    require([p.name for p in paths]==[f"call-{i:04d}" for i in range(LIVE_CALLS)],"Normal preparation count exceeds/drops expected calls")
    calls,workspace = [],None
    for index,(path,(cid,forward,layer,positions,classification)) in enumerate(zip(paths,live)):
        receipt = read_json(path/"receipt.json")
        marker = request(plan,next(c for c in plan["cases"] if c["id"]==cid))
        require(receipt.get("scope")=="normal_live_request" and receipt.get("request")==marker
                and receipt.get("layer_name")==f"language_model.model.layers.{layer}.moe.experts"
                and receipt.get("live_contract")==compiled["live_contract"] and receipt.get("binary_sha256")==compiled["binary_sha256"],"Normal lease is unbound/mixed-version")
        data,_,_ = check_call(path,receipt,lane,forward_identity=(forward,positions[0]))
        arrays = routes[(cid,forward,layer)][1]
        weights = arrays["weight_bits"].copy()
        for target_layer,expert in TARGETS:
            if target_layer==layer:
                weights[arrays["ids"]==expert]=0
        require(arrays["tokens"].tolist()==receipt["tokens"] and arrays["ids"].tobytes()==data["ids.bin"]
                and weights.tobytes()==data["route-weights.bin"],"Normal preparation inputs differ from actual unchanged routes")
        require(receipt.get("workspace_reused") is (index>0) and 0<receipt.get("additional_scratch_bytes",0)<=8<<20,
                "Normal workspace reuse/scratch bound missing")
        if workspace is None:
            workspace = receipt["workspace_storage_pointer"]
        require(workspace==receipt["workspace_storage_pointer"],"Normal workspace owner changed between calls/requests")
        calls.append((path,receipt,classification))
    actual = [json.loads(line) for line in read_bytes(run/"preparation/stock-fallbacks.jsonl",1<<20).splitlines()]
    require(len(actual)==FALLBACK_CALLS,"Normal full-stock fallback count differs")
    for record,(cid,forward,layer,positions,_) in zip(actual,fallback):
        require(record.get("request_id")==cid and record.get("forward_index")==forward
                and record.get("layer_name")==f"language_model.model.layers.{layer}.moe.experts"
                and record.get("rows")==len(positions),"Normal prefill fallback identity differs")
    return calls,artifacts,traces


def compare_runs(stock,fused,build,plan):
    compiled = load_build(build)
    left = check_run(stock,"stock",plan,compiled)
    right = check_run(fused,"fused",plan,compiled)
    for (a,ra,ca),(b,rb,cb) in zip(left[0],right[0]):
        require(ca==cb,"M1 prefill/decode classifications differ")
        for field in ("request","scope","layer_name","forward_index","tokens","positions","owner_extents","quantization_owners","native_metadata","additional_scratch_bytes","workspace_reused"):
            require(ra.get(field)==rb.get(field),"Matched normal lease identity differs: "+field)
        for name,size in BINARY_FILES.items():
            require(read_bytes(a/name,size)==read_bytes(b/name,size),"Matched normal preparation/output bytes differ: "+name)
        for name in ("consumer-masks.json","consumer-envelopes.json"):
            require(read_json(a/name)==read_json(b/name),"Matched actual consumer masks/descriptors differ")
    arrays = elements = files = stages = 0
    for case in plan["cases"]:
        ld,le = left[1][case["id"]];rd,re = right[1][case["id"]]
        require([r["file"] for r in le]==[r["file"] for r in re],"Matched normal array set differs")
        for (_,a),(_,b) in zip(arrays_for(ld,le),arrays_for(rd,re)):
            require(set(a)==set(b),"Matched normal fields differ")
            for field,value in a.items():
                other = b[field]
                require(value.dtype==other.dtype and value.shape==other.shape and value.tobytes()==other.tobytes(),"Matched normal model array differs: "+field)
                arrays += 1;elements += value.size
            files += 1
        stages += sum(r["file"].startswith("stage-") for r in le)
    for a,b in zip(left[2],right[2]):
        require(a["incumbent_kernel_counts"]==b["incumbent_kernel_counts"],"Matched installed kernels differ")
    return {"scope": "bounded_normal_routing_constrained_continuation", "passed": True,
        "plan_sha256": plan["plan_sha256"], "prompt_lengths": [257,1023], "requests_per_lane": 2,
        "outputs_per_request": 4, "continuation_constrained": True,"routing_intervention": False,
        "scheduling": "explicit_synchronous",
        "request_bound_m1_calls_per_lane": LIVE_CALLS,"single_row_prefill_tail_calls_per_lane": 30,
        "cached_decode_calls_per_lane": 180,"cached_decode_positions": [[257,258,259],[1023,1024,1025]],
        "stock_fallback_calls_per_lane": FALLBACK_CALLS,"preparation_oracle_bit_exact": True,
        "matched_preparation_masks_routed_outputs_bit_exact": True,"workspace_reuse_across_requests_verified": True,
        "additional_scratch_bytes": left[0][0][1]["additional_scratch_bytes"],"observed_corrected_stage_captures": stages,
        "model_arrays": {"npz_files": files,"arrays": arrays,"elements": elements,"bit_exact": True},
        "stock_traces": [{k:v for k,v in t.items() if k!="incumbent_kernel_counts"} for t in left[2]],
        "fused_traces": [{k:v for k,v in t.items() if k!="incumbent_kernel_counts"} for t in right[2]],
        "binary_sha256": compiled["binary_sha256"],"live_contract": compiled["live_contract"],
        "quality_qualified": False,"unconstrained_generation_qualified": False,"broad_natural_routing_qualified": False,
        "graph_qualified": False,"timing_qualified": False,"async_scheduling_qualified": False,
        "independent_attention_scheduler_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stock","fused","build","plan","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    require(all(d.resolve()!=destination and d.resolve() not in destination.parents for d in (args.stock,args.fused,args.build)),"Report must be outside input evidence")
    result = compare_runs(args.stock,args.fused,args.build,load_plan(args.plan))
    with destination.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
