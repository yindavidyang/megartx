"""Synthetic CPU diagnostics; no private prompts, native imports or GPU runs."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m1_decode_profile as diagnostic
from m1_eager_benchmark_client import summarize_run
from megartx.m1_eager_benchmark import (EagerBenchmark, PROFILE_DRIVER_SOURCES, PROFILE_RULE, PROFILE_SCHEMA,
                                       drain_marker, require_profile_intent, validate_plan)
from megartx.m1_normal_plan import digest
from test_m1_eager_benchmark import finish, plan as ordinary_plan
from megartx.m1_execution import CONTROLLER_SOURCES


def profile_plan():
    p = ordinary_plan()
    p.update(schema=PROFILE_SCHEMA, diagnostic_admission=PROFILE_RULE, metadata_help_timing=True)
    p["driver_source_hashes"]["scripts/m1_decode_profile.py"] = "c" * 64
    p["plan_sha256"] = digest({k: v for k, v in p.items() if k != "plan_sha256"})
    return validate_plan(p)


def synthetic_trace(lane):
    events, correlation = [], 0
    def cpu(name, start, duration, category="user_annotation", args=None):
        e = dict(ph="X", cat=category, name=name, ts=start, dur=duration, pid=100, tid=100, args=args or {})
        events.append(e)
        return e
    def device(name, category, start, size=None):
        nonlocal correlation
        correlation += 1
        cpu("cudaMemcpyAsync" if category == "gpu_memcpy" else "cudaLaunchKernel", start, 1,
            "cuda_runtime", {"correlation": correlation})
        args = dict(correlation=correlation, stream=17, context=1, device=0)
        args["graph id"] = 0
        if size is None: args.update(grid=[1,1,1], block=[32,1,1], **{"shared memory":0})
        else: args["bytes"] = size
        events.append(dict(ph="X", cat=category, name=name, ts=start+1, dur=1, pid=0, tid=17, args=args))
    for frame in range(4):
        base = frame * 1_000_000
        cpu("megartx::model_forward", base, 700_000)
        cpu("megartx::logits_head", base+700_100, 100)
        cpu("aten::argmax", base+700_300, 100, "cpu_op")
        for layer in range(30):
            begin = base + layer * 10_000 + 100
            cpu(f"megartx::m1_preparation_{lane}::language_model.model.layers.{layer}.moe.experts", begin, 5_000)
            cpu(f"megartx::m1_routed_{lane}::language_model.model.layers.{layer}.moe.experts", begin+5_100, 3_000)
            if layer < 6: cpu("megartx::correction_selection", begin+8_200, 10)
            for i, size in enumerate([32,3072,2560,*[1024]*5,3072,2560,*[1024]*5]):
                device("synthetic DtoH", "gpu_memcpy", begin+10+i*10, size)
            for i in range(7 if lane == "stock" else 6):
                device("synthetic preparation kernel " + str(i), "kernel", begin+200+i*10)
            for i in range(3): cpu("cudaStreamSynchronize", begin+300+i*10, 1, "cuda_runtime")
            for i in range(10): cpu("cudaPointerGetAttributes", begin+400+i*10, 1, "cuda_runtime")
        for i in range(950): device("synthetic dense/head/sampler", "kernel", base+400_000+i*10)
    return {"deviceProperties":[dict(id=0,name="NVIDIA GeForce RTX 5090",computeMajor=12,computeMinor=0)], "traceEvents": events}


def write_profile_packet(root):
    root = Path(root); profile = root / "decode-profile"; profile.mkdir(exist_ok=True)
    p = profile_plan()
    with patch.dict(os.environ, {"MEGARTX_M1_DECODE_PROFILE_DIR":str(profile)}):
        ledger = EagerBenchmark(p, root / "eager-benchmark", "stock")
    requests = []
    for row in p["schedule"]:
        finish(ledger, row)
        requests.append({**row, "token_ids":[7]*256,
            "usage":dict(prompt_tokens=int(row["case"]),completion_tokens=256), "finish_reason":"length"})
    ledger.begin(drain_marker(p), [7], [0])
    write = lambda path, value: path.write_text(json.dumps(value))
    write(root / "eager-benchmark-plan.json", p)
    write(root / "launch-manifest.json", dict(m1_decode_profile_requested=True, diagnostic_admission=PROFILE_RULE,
                                               eager_benchmark_plan_sha256=p["plan_sha256"]))
    (root / "eager-requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests))
    for name in ("run.exit", "benchmark.exit"): (root / name).write_text("0\n")
    write(root / "eager-client-validation.json", dict(plan_sha256=p["plan_sha256"],requests=8,
         matched_tokens_usage=True,actual_input_transcripts_verified=True,native_backends_verified=True))
    forced = [{"ph":"X", "cat":"user_annotation", "name":name} for name in
              ("megartx::corrected_expert_native", "megartx::corrected_expert_reference") for _ in range(6)]
    forced += [{"ph":"X", "cat":"kernel", "name":"MainloopSm120TmaWarpSpecializedBlockScaled"} for _ in range(18)]
    forced += [{"ph":"X", "cat":"kernel", "name":"_gelu_product"} for _ in range(6)]
    with gzip.open(root / "activation-forced.json.gz", "wt") as stream: json.dump({"traceEvents":forced},stream)
    write(root / "activation-proof.json", dict(forced_all_six_executed=True,natural_model_forward_verified=True,
          registered_layers=[dict(layer_name=f"language_model.model.layers.{i}.moe.experts") for i in range(30)],
          original_tensor_captures=[dict(tensor=str(i),original_bytes_equal=True) for i in range(54)],
          forced_fixtures=[dict(layer_name=f"language_model.model.layers.{i}.moe.experts",expert=e,
              forced_correction_rows=2,observed_bf16_value_equal=True,max_absolute_difference=0,nonzero_output_elements=1)
              for i,e in ((0,42),(0,82),(1,126),(2,89),(3,7),(5,12))],
          forced_trace_sha256=hashlib.sha256((root / "activation-forced.json.gz").read_bytes()).hexdigest()))
    traces = {}
    for lane in ("stock","fused"):
        traces[lane] = synthetic_trace(lane)
        write(profile / (lane + ".json"), traces[lane])
        row = next(r for r in p["schedule"] if r["phase"]=="measurement" and r["case"]=="2048" and r["lane"]==lane)
        write(profile / (lane + "-scalars.json"), dict(lane=lane,decode_steps=4,request_id=row["id"],
              positions_relative_to_context=[0,1,2,3],source_head=p["source_head"],plan_sha256=p["plan_sha256"],timing_qualified=False,
              host_phases={k:dict(count=v,nanoseconds=123) for k,v in dict(native_runner=120,map_eligibility=0,map_dispatch=120,
                    descriptor_readback_fence=240,descriptor_validation_enumeration=240).items()}))
    return p, {lane:diagnostic.operation_digest([e for e in t["traceEvents"] if e["cat"] in ("kernel","gpu_memcpy")])
               for lane,t in traces.items()}


class ProfilePlanTests(unittest.TestCase):
    def test_intent_is_digest_bound_and_requires_exact_rule_and_pilot(self):
        p = profile_plan(); require_profile_intent(p,True)
        for base, enabled in ((p,False),(ordinary_plan(),True),(p,1)):
            with self.assertRaisesRegex(RuntimeError,"intent"): require_profile_intent(base,enabled)
        for key,value in (("schema","unknown"),("diagnostic_admission","unknown"),("trials",2),
                          ("warmups",2),("metadata_help_timing",False)):
            bad=copy.deepcopy(p);bad[key]=value
            bad["plan_sha256"]=digest({k:v for k,v in bad.items() if k!="plan_sha256"})
            with self.subTest(key=key),self.assertRaises(RuntimeError):validate_plan(bad)
        bad=copy.deepcopy(p);bad.pop("diagnostic_admission")
        with self.assertRaises(RuntimeError):validate_plan(bad)
        bad=copy.deepcopy(p);bad["schema"]=ordinary_plan()["schema"];bad.pop("diagnostic_admission")
        bad["driver_source_hashes"].pop("scripts/m1_decode_profile.py")
        with self.assertRaisesRegex(RuntimeError,"digest"):validate_plan(bad)

    def test_replay_cannot_remove_or_falsify_optional_profile_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=profile_plan()
            (root/"eager-benchmark-plan.json").write_text(json.dumps(p))
            for launch in (None,{}, {"m1_decode_profile_requested":False}):
                path=root/"launch-manifest.json"
                if launch is None:path.unlink(missing_ok=True)
                else:path.write_text(json.dumps(launch))
                with self.subTest(launch=launch),self.assertRaisesRegex(RuntimeError,"diagnostic plan"):
                    summarize_run(root)
            self.assertFalse((root/"eager-summary.json").exists())

    def test_real_cli_intent_mismatch_rejects_before_aot_or_gpu(self):
        script=Path(__file__).resolve().parents[1]/"scripts/run_scale_validation.py"
        code="import runpy,sys,types;sys.modules['requests']=types.SimpleNamespace();sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"plan.json"
            for p,flag in ((profile_plan(),False),(ordinary_plan(),True)):
                path.write_text(json.dumps(p))
                cmd=[sys.executable,"-c",code,str(script),"--mode","native","--label","cpu-only", "--client","m1-eager-benchmark",
                     "--m1-eager-benchmark-plan",str(path),"--m1-private-aot","not-read","--m1-preparation","stock",
                     "--m1-execution","capture-free","--m1-bridge","not-read","--m1-build-receipt","not-read"]
                if flag:cmd.append("--m1-decode-profile")
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
                self.assertEqual(r.returncode,2,r.stderr);self.assertIn("intent differs",r.stderr)

    def test_actual_controller_requires_profile_destination_for_diagnostic_plan(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,"intent"):
            EagerBenchmark(profile_plan(),"unused","stock")
        with patch.dict(os.environ,{"MEGARTX_M1_DECODE_PROFILE_DIR":"unused"},clear=True),self.assertRaisesRegex(RuntimeError,"intent"):
            EagerBenchmark(ordinary_plan(),"unused","stock")


class ProfileEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.plan,self.inventories=write_profile_packet(self.root)
        self.owned=dict(cleanup_complete=True,cleanup_errors=[],owned_identities_remaining=[],owned_gpu_pids_remaining=[],
            failure=None,timing_metadata_policy_invalid=False,compiler_rss_limit_bytes=2<<30,shared_compiler_seconds_limit=300,
            sampled_peak_compiler_rss_bytes=1024,shared_compiler_elapsed_seconds=.1,
            timing_metadata_policy=dict(final=dict(passed=True)),timing_unknown_or_work_identities=[(30,30)])
        self.patch_inventory=patch.object(diagnostic,"OPERATION_DIGESTS",self.inventories);self.patch_inventory.start();self.addCleanup(self.patch_inventory.stop)
        self.patch_files=patch.object(diagnostic,"verify_runtime_files");self.patch_files.start();self.addCleanup(self.patch_files.stop)
    def admit(self): return diagnostic.validate_profile_run(self.root,self.plan,self.owned)
    def test_complete_counts_and_equality_accept_without_durations_or_timing_claims(self):
        result=self.admit();self.assertTrue(result["diagnostic_admission_passed"])
        for field in ("timing_qualified","performance_gate_passed","quality_qualified","graphs_qualified"):self.assertIs(result[field],False)
        self.assertEqual(result["compiler_unknown_or_work_identities"],1)
        self.assertEqual(result["lanes"]["fused"]["preparation_d2h_copies"],1800)
        self.assertNotIn("nanoseconds",json.dumps(result));self.assertNotIn("elapsed",json.dumps(result))
    def test_resource_metadata_cleanup_or_old_failed_run_never_admits(self):
        for key,value in (("failure","latched resource breach"),("sampled_peak_compiler_rss_bytes",(2<<30)+1),
                ("shared_compiler_elapsed_seconds",301),("compiler_rss_limit_bytes",4<<30),
                ("cleanup_errors",["signal failed"]),("owned_identities_remaining",[(30,30)]),
                ("owned_gpu_pids_remaining",[30]),("timing_metadata_policy_invalid",True),
                ("timing_metadata_policy",dict(final=dict(passed=False)))):
            old=self.owned[key];self.owned[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.admit()
            self.owned[key]=old
        (self.root/"run.exit").write_text("1\n")
        with self.assertRaisesRegex(RuntimeError,"lifecycle"):self.admit()
        with self.assertRaisesRegex(RuntimeError,"intent"):diagnostic.validate_profile_run(self.root,ordinary_plan(),self.owned)
    def test_wrong_window_request_source_and_native_counts_reject(self):
        path=self.root/"decode-profile/fused-scalars.json";original=json.loads(path.read_text())
        for key,value in (("lane","stock"),("source_head","f"*40),("plan_sha256","f"*64),("request_id","other"),
                          ("decode_steps",3),("positions_relative_to_context",[1,2,3,4]),("timing_qualified",True)):
            changed=copy.deepcopy(original);changed[key]=value;path.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,"window/source"):self.admit()
        changed=copy.deepcopy(original);changed["host_phases"]["map_eligibility"]["count"]=120;path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError,"native phase"):self.admit()
    def test_frame_layer_stream_correlation_kernel_descriptor_and_unknown_work_reject(self):
        path=self.root/"decode-profile/fused.json";original=json.loads(path.read_text())
        for kind in ("missing_gpu","symbol","geometry","bytes","stream","missing_correlation","ambiguous_correlation","layer","frame","graph","unknown_gpu"):
            changed=copy.deepcopy(original);events=changed["traceEvents"]
            gpu=next(e for e in events if e["cat"]=="gpu_memcpy")
            api=next(e for e in events if e["cat"]=="cuda_runtime" and "correlation" in e["args"])
            if kind=="missing_gpu":events.remove(gpu)
            elif kind=="symbol":gpu["name"]="unknown operation"
            elif kind=="geometry":next(e for e in events if e["cat"]=="kernel")["args"]["grid"]=[2,1,1]
            elif kind=="bytes":gpu["args"]["bytes"]=64
            elif kind=="stream":gpu["args"]["stream"]=18
            elif kind=="missing_correlation":gpu["args"]["correlation"]=-1
            elif kind=="ambiguous_correlation":events.append(copy.deepcopy(api))
            elif kind=="layer":next(e for e in events if e["name"].startswith("megartx::m1_preparation_"))["name"]="megartx::m1_preparation_fused::wrong.layer"
            elif kind=="frame":events.remove(next(e for e in events if e["name"]=="megartx::model_forward"))
            elif kind=="graph":gpu["args"]["graph id"]=1
            else:gpu["cat"]="gpu_memset"
            path.write_text(json.dumps(changed))
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.admit()
    def test_token_usage_dispatch_fixture_and_launch_identity_fail_closed(self):
        for filename,field,value in (("launch-manifest.json","diagnostic_admission","unknown"),
              ("eager-client-validation.json","native_backends_verified",False),("activation-proof.json","forced_all_six_executed",False)):
            path=self.root/filename;original=path.read_text();changed=json.loads(original);changed[field]=value;path.write_text(json.dumps(changed))
            with self.subTest(file=filename),self.assertRaises(RuntimeError):self.admit()
            path.write_text(original)
        path=self.root/"eager-requests.jsonl";original=path.read_text();rows=[json.loads(line) for line in original.splitlines()]
        rows[0]["token_ids"][0]=8;path.write_text("\n".join(json.dumps(r) for r in rows))
        with self.assertRaisesRegex(RuntimeError,"transcript"):self.admit()


class RuntimeFileTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        project=Path(diagnostic.__file__).resolve().parents[1]
        self.plan=profile_plan()
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.plan["controller_source_hashes"]={p:sha(project/"src/megartx"/p) for p in CONTROLLER_SOURCES}
        self.plan["driver_source_hashes"]={p:sha(project/p) for p in PROFILE_DRIVER_SOURCES}
        self.plan["plan_sha256"]=digest({k:v for k,v in self.plan.items() if k!="plan_sha256"})
        self.binary=self.root/"bridge.so";self.binary.write_bytes(b"synthetic bridge bytes")
        self.installed=self.root/"incumbent.so";self.installed.write_bytes(b"synthetic installed bytes")
        self.aot=self.root/"aot";self.aot.mkdir()
        for name in ("manifest.json","cpu-dry-run.json"):(self.aot/name).write_text('{}')
        source_names=set(PROFILE_DRIVER_SOURCES)|{"src/megartx/"+p for p in CONTROLLER_SOURCES}|{
             "probes/m1_live_bridge.cu","probes/m1_installed_bridge.cuh","kernels/m1_installed_preparation.cuh",
             "kernels/m1_maps_expand.cuh","scripts/build_m1_live_bridge.py","scripts/check_m1_live_bridge.py","scripts/check_m1_live_bindings.py"}
        self.packages={"vllm":"0.30.0","flashinfer-python":"0.6.18.post1","torch":"2.13.0","nvidia-cuda-cupti":"13.0.85"}
        self.build=dict(base_head=self.plan["source_head"],returncode=0,reason=None,compiled_lease_controls_returncode=0,
             compiled_binding_controls_returncode=0,required_exports_present=True,binary_sha256=sha(self.binary),
             live_contract=dict(controller_source_hashes=self.plan["controller_source_hashes"]),
             source_hashes={p:sha(project/p) for p in source_names},installed_pins={str(self.installed):sha(self.installed)},
             installed_package_versions=self.packages)
        self.build_path=self.root/"build.json";self.build_path.write_text(json.dumps(self.build))
        self.launch=dict(adapter_mode="native",m1_external_observer_requested=False,
             command=["--enforce-eager","--no-async-scheduling","--no-enable-prefix-caching"],
             environment_overrides=dict(MEGARTX_M1_EXECUTION="capture-free",MEGARTX_M1_BUILD_RECEIPT=str(self.build_path),
                   MEGARTX_M1_BRIDGE=str(self.binary),MEGARTX_M1_PRIVATE_AOT=str(self.aot)),
             private_aot_manifest_sha256=sha(self.aot/"manifest.json"),private_aot_cpu_dry_run_sha256=sha(self.aot/"cpu-dry-run.json"))
        for patcher in (patch("m1_private_aot.validate_cache"),patch("m1_decode_profile.importlib.metadata.version",side_effect=self.packages.__getitem__)):
            patcher.start();self.addCleanup(patcher.stop)
    def check(self):diagnostic.verify_runtime_files(self.plan,self.launch)
    def test_final_file_recheck_covers_build_binary_controller_driver_native_and_installed_drift(self):
        self.check()
        mutations=[("head",lambda b:b.update(base_head="f"*40)),("build",lambda b:b.update(returncode=1)),
             ("abi",lambda b:b.update(compiled_lease_controls_returncode=1)),("binary",lambda b:b.update(binary_sha256="f"*64)),
             ("controller",lambda b:b["live_contract"]["controller_source_hashes"].update({"m1_eager_benchmark.py":"f"*64})),
             ("driver",lambda b:b["source_hashes"].update({"scripts/m1_decode_profile.py":"f"*64})),
             ("native",lambda b:b["source_hashes"].update({"kernels/m1_installed_preparation.cuh":"f"*64})),
             ("missing_source",lambda b:b["source_hashes"].pop("kernels/m1_maps_expand.cuh")),
             ("installed",lambda b:b["installed_pins"].update({str(self.installed):"f"*64})),
             ("packages",lambda b:b["installed_package_versions"].update(torch="unknown"))]
        for kind,mutate in mutations:
            changed=copy.deepcopy(self.build);mutate(changed);self.build_path.write_text(json.dumps(changed))
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.check()
        self.build_path.write_text(json.dumps(self.build))
    def test_final_aot_package_observer_and_execution_drift_remain_failures(self):
        self.check()
        with patch("m1_private_aot.validate_cache",side_effect=RuntimeError("AOT module drift")),self.assertRaisesRegex(RuntimeError,"AOT"):
            self.check()
        with patch("m1_decode_profile.importlib.metadata.version",return_value="unknown"),self.assertRaisesRegex(RuntimeError,"package drift"):
            self.check()
        for key,value in (("adapter_mode","control"),("m1_external_observer_requested",True),("command",["--enforce-eager"])):
            original=self.launch[key];self.launch[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,"execution/observer"):self.check()
            self.launch[key]=original
        (self.aot/"cpu-dry-run.json").write_text('{"drift":true}')
        with self.assertRaisesRegex(RuntimeError,"AOT receipt"):self.check()


if __name__ == "__main__": unittest.main()
