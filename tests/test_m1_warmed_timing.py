"""Prospective warmed timing admission through real CPU client/launcher paths.

Only HTTP, /proc, device queries and pinned target files are substituted. No
model imports, CUDA, private prompts, package installation or experiment.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m1_eager_benchmark_client as client
from m1_owned_processes import OwnedProcesses, Process, evaluate_warmed_quiescence
from prepare_m1_eager_benchmark import make_plan
from megartx.m1_eager_benchmark import (EagerBenchmark, PROFILE_DRIVER_SOURCES, WARMED_RULE, WARMED_SCHEMA,
    WARMED_SUMMARY_SCHEMA, RESOURCE_FIELDS, drain_marker, require_warmed_intent, validate_plan, validate_warmed_boundaries)
from megartx.m1_execution import CONTROLLER_SOURCES
from megartx.m1_normal_plan import digest
from test_m1_eager_benchmark import finish
import test_m1_launcher_resources as launcher_tests
from test_m1_launcher_resources import LAUNCHER, RUN, code, index


def plan():
    return make_plan({2048:[3]*2048,8192:[3]*8192}, "a"*40,
        {n:"b"*64 for n in CONTROLLER_SOURCES}, {n:"c"*64 for n in PROFILE_DRIVER_SOURCES},
        trials=1, warmups=1, warmed_timing=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def client_packet(root, *, completed=True, explicit=True, launch_enabled=True):
    """Run actual main/completion through fake local HTTP/SSE delivery."""
    p = plan()
    ledger = EagerBenchmark(p, root / "eager-benchmark", "stock")
    for row in p["schedule"]:
        finish(ledger, row)
    ledger.begin(drain_marker(p), [7], [0])
    write_json(root / "eager-benchmark-plan.json", p)
    launch = {"eager_benchmark_plan_sha256":p["plan_sha256"], "m1_decode_profile_requested":False,
              "m1_warmed_timing_requested":launch_enabled, "timing_admission":WARMED_RULE,
              "m1_external_observer_requested":False,
              "environment_overrides":{"MEGARTX_M1_EXECUTION":"capture-free"},
              "command":["--enforce-eager","--no-async-scheduling","--no-enable-prefix-caching"]}
    write_json(root / "launch-manifest.json", launch)
    clock_value = [1_000_000_000]
    def clock():
        clock_value[0] += 1_000_000
        return clock_value[0]
    calls = []
    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): calls.append(("close", self.payload["request_id"]))
        def raise_for_status(self): pass
        def iter_lines(self, **kwargs):
            for _ in range(self.payload["max_tokens"]):
                yield b"data: " + json.dumps({"choices":[{"token_ids":[7]}]}).encode()
            yield b"data: " + json.dumps({"choices":[{"finish_reason":"length"}], "usage":{
                "prompt_tokens":len(self.payload["prompt"]),"completion_tokens":self.payload["max_tokens"]}}).encode()
            if completed: yield b"data: [DONE]"
    class Session:
        def post(self, url, data, **kwargs):
            payload = json.loads(data); calls.append(("POST", payload["request_id"]))
            return Response(payload)
        def close(self): calls.append(("session_close", None))
    argv = ["client", "--plan", str(root / "eager-benchmark-plan.json"), "--output", str(root)]
    if explicit: argv.append("--m1-warmed-timing")
    with patch.object(sys,"argv",argv), patch.dict(sys.modules,{"requests":SimpleNamespace(Session=Session)}), \
         patch.object(client.time,"monotonic_ns",clock), patch("builtins.print"):
        client.main()
    records = [json.loads(r) for r in (root / "eager-requests.jsonl").read_text().splitlines()]
    boundary = json.loads((root / "warmed-boundaries.json").read_text())
    lifecycle = {"schema":"megartx-m1-warmed-launch-boundaries-v1", "timing_admission":WARMED_RULE,
                 "plan_sha256":p["plan_sha256"], "server_ready_ns":100_000_000,
                 "client_launch_ns":200_000_000, "client_returned_ns":clock_value[0]+1_000_000}
    write_json(root / "warmed-launch-boundaries.json", lifecycle)
    for name in ("run.exit", "benchmark.exit"): (root / name).write_text("0\n")
    telemetry=[{"sample_started_ns":ns, "monotonic_ns":ns+1_000_000, "unix_ns":ns,
                "host_available_kib":16*1024*1024, "fields":list(RESOURCE_FIELDS),
                "values":["23000","9000","40","100","40","1000","1000","P0"], "exit":0}
               for ns in range(50_000_000,clock_value[0]+201_000_000,100_000_000)]
    (root / "gpu-telemetry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in telemetry))
    return p, records, boundary, lifecycle, calls


class FileGuard:
    def __init__(self): self.error = None
    def version_error(self): return self.error
    def finalize(self): return {"passed":self.error is None,"errors":[self.error] if self.error else []}
    def report(self):
        return {"policy":"whole_run_file_versions_initial_final_hashes_v1", "files":{"fixture":{"version":(1,2,3,4,5),"sha256":"a"*64,"resolved":"fixture"}},
                "failure":self.error, "final":{"passed":self.error is None,"errors":[self.error] if self.error else []}}


def observe_exact(owner, processes, now):
    ns=round(now*1e9)
    return owner.observe(processes,now,sampled_start_ns=ns,observed_ns=ns)


def sampled_owner(end, samples=()):
    root = Process(20,20,10)
    owner = OwnedProcesses(Process(10,10,1)); owner.register(root)
    for when, processes in samples:
        observe_exact(owner,{20:root, **{p.pid:p for p in processes}},when)
    for ns in range(1_000_000_000, end+200_000_000, 50_000_000):
        observe_exact(owner,{20:root}, ns/1e9)
    owner.runtime_files = FileGuard()
    return owner


def write_owned(root, owner):
    owned = {**owner.report(), "cleanup_complete":True, "cleanup_errors":[],
             "owned_identities_remaining":[], "owned_gpu_pids_remaining":[]}
    write_json(root / "owned-processes.json", owned)
    write_json(root / "eager-benchmark-cleanup.json", owned)
    return json.loads(json.dumps(owned))


class WarmedContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)

    def test_default_strict_and_profile_never_inherit_warmed_policy(self):
        from test_m1_eager_benchmark import plan as strict
        from test_m1_profile_admission import profile_plan
        for p in (strict(),profile_plan()):
            with self.assertRaisesRegex(RuntimeError,"opt-in differs"): require_warmed_intent(p,True)
        p = plan(); require_warmed_intent(p,True)
        for value in (False, 1, None):
            with self.assertRaises(RuntimeError): require_warmed_intent(p,value)
        for change in ({"schema":strict()["schema"]},{"timing_admission":"compiler_accounted_operation_diagnostic_v1"}):
            bad = {**p,**change};bad["plan_sha256"] = digest({k:v for k,v in bad.items() if k != "plan_sha256"})
            with self.assertRaises(RuntimeError): validate_plan(bad)

    def test_real_client_actual_post_close_and_completed_warmup_boundaries(self):
        p, records, boundary, lifecycle, calls = client_packet(self.root)
        validate_warmed_boundaries(p,records,boundary,lifecycle)
        measured = [r for r in records if r["phase"] == "measurement"]
        self.assertEqual(boundary["measurement_start_ns"],measured[0]["request_start_monotonic_ns"])
        self.assertEqual(boundary["measurement_end_ns"],measured[-1]["request_end_monotonic_ns"])
        self.assertEqual(len(boundary["completed_warmup_ids"]),4)
        self.assertEqual(calls[-3:],[("POST","drain"),("close","drain"),("session_close",None)])
        self.assertFalse((self.root / "capture-request.json").exists())

    def test_real_client_requires_explicit_flag_and_matching_launch(self):
        for explicit,launch in ((False,True),(True,False)):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError,"opt-in differs"):
                    client_packet(Path(directory),explicit=explicit,launch_enabled=launch)
                self.assertFalse((Path(directory)/"eager-requests.jsonl").exists())

    def test_real_client_incomplete_warmup_writes_no_boundary(self):
        with self.assertRaisesRegex(RuntimeError,"omitted DONE"):client_packet(self.root,completed=False)
        self.assertFalse((self.root / "warmed-boundaries.json").exists())
        self.assertFalse((self.root / "capture-request.json").exists())

    def test_actual_launcher_cli_plan_mismatch_stops_before_aot_or_gpu(self):
        write_json(self.root / "plan.json",plan())
        stub = "import runpy,sys,types; sys.modules['requests']=types.SimpleNamespace(); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        common = [sys.executable,"-c",stub,str(LAUNCHER),"--label","cpu-only","--mode","native",
            "--client","m1-eager-benchmark","--m1-eager-benchmark-plan",str(self.root / "plan.json"),
            "--m1-private-aot","not-read","--m1-preparation","stock","--m1-execution","capture-free",
            "--m1-bridge","not-read","--m1-build-receipt","not-read"]
        result = subprocess.run(common,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2);self.assertIn("warmed timing opt-in differs",result.stderr)
        from test_m1_eager_benchmark import plan as strict
        write_json(self.root / "plan.json",strict())
        result = subprocess.run([*common,"--m1-warmed-timing"],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2);self.assertIn("warmed timing opt-in differs",result.stderr)

    def test_forged_missing_reordered_and_incomplete_boundaries_reject(self):
        p, records, boundary, lifecycle, _ = client_packet(self.root)
        for change in ({"measurement_start_ns":boundary["measurement_start_ns"]+1},
                       {"measurement_end_ns":boundary["measurement_end_ns"]-1},
                       {"completed_warmup_ids":boundary["completed_warmup_ids"][:-1]},
                       {"warmups_completed_ns":boundary["measurement_start_ns"]},
                       {"clock":"perf_counter_ns"},{"plan_sha256":"d"*64}):
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                validate_warmed_boundaries(p,records,{**boundary,**change},lifecycle)
        for changed in (records[:-1],list(reversed(records)),[records[1],records[0],*records[2:]]):
            with self.assertRaises(RuntimeError):validate_warmed_boundaries(p,changed,boundary,lifecycle)
        for key in ("server_ready_ns","client_launch_ns","client_returned_ns"):
            with self.assertRaises(RuntimeError):validate_warmed_boundaries(p,records,boundary,{**lifecycle,key:0})
        missing = dict(boundary);missing.pop("warmups_completed_ns")
        with self.assertRaises(RuntimeError):validate_warmed_boundaries(p,records,missing,lifecycle)

    def test_summary_receipt_replay_and_strict_confusion_fail_closed(self):
        p, records, boundary, lifecycle, _ = client_packet(self.root)
        owner = sampled_owner(boundary["measurement_end_ns"])
        owned = write_owned(self.root,owner)
        with self.assertRaises(RuntimeError):client.summarize_run(self.root)
        receipt = client.validate_warmed_run(self.root,p,owned)
        write_json(self.root / "warmed-timing-admission.json",receipt)
        summary = client.summarize_run(self.root)
        self.assertEqual(summary["schema"],WARMED_SUMMARY_SCHEMA)
        self.assertEqual(summary["timing_admission"],WARMED_RULE)
        self.assertFalse(summary["performance_gate_passed"])
        owned["sampled_peak_compiler_rss_bytes"] = 99
        write_json(self.root / "eager-benchmark-cleanup.json",owned)
        with self.assertRaisesRegex(RuntimeError,"ownership/cleanup evidence differs"):client.summarize_run(self.root)
        write_owned(self.root,owner)
        (self.root / "run.exit").write_text("1\n")
        with self.assertRaisesRegex(RuntimeError,"lifecycle"):client.summarize_run(self.root)
        (self.root / "run.exit").write_text("0\n")
        strict = copy.deepcopy(p);strict["schema"]="megartx-m1-eager-benchmark-v1";strict.pop("timing_admission")
        strict["driver_source_hashes"].pop("scripts/m1_decode_profile.py")
        strict["plan_sha256"] = digest({k:v for k,v in strict.items() if k != "plan_sha256"})
        write_json(self.root / "eager-benchmark-plan.json",strict)
        with self.assertRaises(RuntimeError):client.summarize_run(self.root)


class WarmedOverlapTests(unittest.TestCase):
    def setUp(self):
        self.window = {"measurement_start_ns":2_000_000_000,"measurement_end_ns":4_000_000_000}
        self.compiler = Process(30,30,20,10,"tileiras", "Z", compiler_identity_verified=True)

    def test_completed_unknown_startup_stays_unknown_and_strict_rejected(self):
        owner = sampled_owner(4_000_000_000,[(.4,[self.compiler]),(.5,[])])
        result = owner.require_warmed_quiescence(self.window)
        self.assertTrue(result["sampled_quiescence_passed"])
        self.assertIn((30,30),owner.non_metadata_identities)
        self.assertIsNotNone(owner.report()["timing_classification_history"][0]["first_unknown_or_work_sample"])
        with self.assertRaisesRegex(RuntimeError,"compiler activity"):owner.require_compiler_quiescence()

    def test_live_across_boundary_and_gap_activity_reject(self):
        for start,end in ((1.8,2.1),(2.5,2.6),(3.9,4.1)):
            owner = OwnedProcesses(Process(10,10,1));root = Process(20,20,10);owner.register(root)
            for tick in range(10,43):
                now = tick/10
                processes = {20:root}
                if start <= now <= end:processes[30] = replace(self.compiler,state="S")
                observe_exact(owner,processes,now)
            with self.subTest(start=start),self.assertRaisesRegex(RuntimeError,"overlap or late"):
                owner.require_warmed_quiescence(self.window)

    def test_late_zombie_terminal_ambiguity_and_missing_observation_reject(self):
        owner = sampled_owner(4_000_000_000)
        observe_exact(owner,{20:Process(20,20,10),30:self.compiler},4.5)
        with self.assertRaisesRegex(RuntimeError,"late/uncertain"):owner.require_warmed_quiescence(self.window)
        owner = OwnedProcesses(Process(10,10,1));owner.register(Process(20,20,10))
        for tick in range(10,43):
            observe_exact(owner,{20:Process(20,20,10),30:replace(self.compiler,compiler_identity_verified=False)},tick/10)
        with self.assertRaisesRegex(RuntimeError,"uncertain"):owner.require_warmed_quiescence(self.window)
        report = sampled_owner(4_000_000_000).report()
        report["compiler_observation_intervals_ns"] = [s for s in report["compiler_observation_intervals_ns"] if not 2.5e9 < s[0] < 3e9]
        with self.assertRaisesRegex(RuntimeError,"gap uncertain"):evaluate_warmed_quiescence(report,self.window)

    def test_pre_boundary_unknown_to_help_and_reappearance_do_not_erase_history(self):
        owner = OwnedProcesses(Process(10,10,1));root=Process(20,20,10);owner.register(root)
        for tick in range(10,43):
            now=tick/10
            observed={20:root}
            if tick in (10,12):observed[30]=replace(self.compiler,state="S")
            observe_exact(owner,observed,now)
        with self.assertRaisesRegex(RuntimeError,"uncertain"):owner.require_warmed_quiescence(self.window)
        self.assertTrue(owner.compiler_lifetimes[(30,30)]["completion_conflict"])

    def test_whole_run_resource_and_file_failures_survive_window(self):
        for message in ("source/AOT version drift","Owned compiler aggregate RSS exceeded 2 GiB",
                        "Owned shared compiler budget exceeded 300 seconds"):
            owner = sampled_owner(4_000_000_000);owner.fail(message)
            with self.subTest(message=message),self.assertRaisesRegex(RuntimeError,message):owner.require_warmed_quiescence(self.window)


class WarmedLauncherTests(unittest.TestCase):
    def setUp(self):
        launcher_tests.LauncherResourceTests.setUp(self)
        self.p, self.records, self.boundary, _, _ = client_packet(self.output)
        self.owner = sampled_owner(self.boundary["measurement_end_ns"],[(.4,[Process(30,30,20,0,"tileiras","Z",compiler_identity_verified=True)]),(.5,[])])
        self.env.update(ownership=self.owner,benchmark_plan=self.p)
        self.env["args"].m1_warmed_timing = True
    final = launcher_tests.LauncherResourceTests.final

    def test_actual_final_admission_accepts_completed_startup_and_preserves_ledger(self):
        self.final([{}, {}, {}])
        self.summary.assert_called_once_with(self.output)
        receipt = json.loads((self.output / "warmed-timing-admission.json").read_text())
        self.assertEqual(receipt["timing_admission"],WARMED_RULE)
        self.assertTrue(receipt["whole_run_integrity_passed"])
        self.assertFalse(receipt["performance_gate_passed"])

    def test_actual_final_missing_boundary_cleans_up_and_rejects(self):
        (self.output / "warmed-boundaries.json").unlink()
        with self.assertRaises(RuntimeError):self.final([{}, {}, {}])
        self.assertTrue(json.loads((self.output / "eager-benchmark-cleanup.json").read_text())["cleanup_complete"])
        self.assertEqual((self.output / "run.exit").read_text(),"1\n")
        self.summary.assert_not_called()

    def test_actual_final_late_cleanup_compiler_rejects_without_hiding_cleanup(self):
        zombie = Process(40,40,10,0,"tileiras","Z",compiler_identity_verified=True)
        with self.assertRaisesRegex(RuntimeError,"late/uncertain"):
            self.final([{10:Process(10,10,1),40:zombie},{},{}])
        cleanup = json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertTrue(cleanup["cleanup_complete"])
        self.assertTrue(cleanup["timing_unknown_or_work_identities"])
        self.summary.assert_not_called()

    def test_actual_final_source_failure_and_primary_error_keep_cleanup(self):
        self.owner.runtime_files.error = "warmed timing final source/AOT hash differs"
        primary = ValueError("primary client failure")
        with self.assertRaises(ValueError) as error:self.final([{}, {}, {}],primary=primary)
        self.assertIs(error.exception,primary)
        cleanup=json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertTrue(cleanup["cleanup_complete"])
        self.assertIn("source/AOT",cleanup["failure"])
        self.summary.assert_not_called()

    def test_actual_dispatch_records_independent_launcher_boundaries_and_cli(self):
        self.env.update(server_ready_ns=100,WARMED_RULE=WARMED_RULE)
        self.env["phase"] = lambda *a,**k: 200
        self.env["time"] = SimpleNamespace(monotonic_ns=lambda:300)
        (self.output / "warmed-launch-boundaries.json").unlink()
        nodes=RUN.body[index("bench_command ="):index("benchmark.exit")+1]
        exec(code(nodes),self.env)
        command=self.client_run.call_args.args[0]
        self.assertIn("--m1-warmed-timing",command)
        lifecycle=json.loads((self.output / "warmed-launch-boundaries.json").read_text())
        self.assertEqual([lifecycle[k] for k in ("server_ready_ns","client_launch_ns","client_returned_ns")],[100,200,300])
        self.assertEqual(lifecycle["plan_sha256"],self.p["plan_sha256"])

    def test_actual_final_unexpected_hash_error_preserves_primary_and_cleanup(self):
        primary=ValueError("primary client failure")
        with patch.object(self.owner.runtime_files,"finalize",side_effect=OSError("hash unavailable")), \
                self.assertRaises(ValueError) as error:
            self.final([{}, {}, {}],primary=primary)
        self.assertIs(error.exception,primary)
        cleanup=json.loads((self.output / "eager-benchmark-cleanup.json").read_text())
        self.assertTrue(cleanup["cleanup_complete"])
        self.assertIn("final source/AOT",cleanup["failure"])

    def test_actual_final_cleanup_fault_rejects_even_after_valid_boundaries(self):
        with self.assertRaisesRegex(RuntimeError,"cleanup did not complete"):
            self.final([{20:self.root},{},{},{}],signal_error=OSError(22,"unknown cleanup fault"))
        self.assertFalse(json.loads((self.output / "eager-benchmark-cleanup.json").read_text())["cleanup_complete"])
        self.summary.assert_not_called()

    def test_actual_final_resource_failure_and_cleanup_fault_still_reject(self):
        self.owner.fail("Owned compiler aggregate RSS exceeded 2 GiB")
        with self.assertRaisesRegex(RuntimeError,"aggregate RSS"):self.final([{}, {}, {}])
        self.assertFalse((self.output / "warmed-timing-admission.json").exists())
        self.summary.assert_not_called()


class WarmedRuntimeFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        for name in ("source.py","installed.py","bridge.so","loader.py","python/shim.py","aot/mod/mod.so"):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"pinned CPU fixture")
        write_json(self.root/"build.json",{"source_hashes":{"source.py":"bound"},"installed_pins":{str(self.root/"installed.py"):"bound"}})
        write_json(self.root/"manifest.json",{"flashinfer_root":str(self.root),"loader_hashes":{"loader.py":"bound"},
            "shim_hashes":{"shim.py":"bound"},"module_hashes":{"mod":"bound"}})
        write_json(self.root/"cpu-dry-run.json",{"passed":True})
        self.launch={"environment_overrides":{"MEGARTX_M1_BUILD_RECEIPT":str(self.root/"build.json"),
            "MEGARTX_M1_BRIDGE":str(self.root/"bridge.so"),"MEGARTX_M1_PRIVATE_AOT":str(self.root)}}
        # Existing target build/package/AOT verifier has its independent suite.
        # Here only that target-dependent verifier and git target HEAD are replaced;
        # all initial hashes, sampled versions and final byte checks are real.
        for patcher in (patch("m1_decode_profile.verify_runtime_files"),patch.object(client,"ROOT",self.root),
                        patch.object(client.WarmedRuntimeFiles,"verify_source_head")):
            patcher.start();self.addCleanup(patcher.stop)
        self.guard=client.WarmedRuntimeFiles(plan(),self.launch)

    def test_real_guard_versions_and_final_hashes_accept_unchanged_files(self):
        self.assertIsNone(self.guard.version_error())
        self.assertTrue(self.guard.finalize()["passed"])
        self.assertEqual(len(self.guard.files),9)

    def test_real_guard_sticky_changed_restored_or_replaced_source_aot_bytes_reject(self):
        for name in ("source.py","aot/mod/mod.so","bridge.so","build.json","installed.py"):
            with self.subTest(name=name):
                guard=client.WarmedRuntimeFiles(plan(),self.launch)
                p=self.root/name;old=p.read_bytes();p.write_bytes(b"changed")
                self.assertIn("version drift",guard.version_error())
                p.write_bytes(old)
                self.assertIn("version drift",guard.version_error())
                self.assertFalse(guard.finalize()["passed"])

    def test_real_guard_final_hash_drift_even_when_version_read_is_stale(self):
        from m1_owned_processes import hash_file_version
        p=self.root/"source.py";p.write_bytes(b"altered fixture")
        def stale(path):
            hashed,_=hash_file_version(path)
            return hashed,self.guard.files[str(path)]["version"]
        with patch.object(self.guard,"version_error",return_value=None),patch("m1_owned_processes.hash_file_version",side_effect=stale):
            result=self.guard.finalize()
        self.assertFalse(result["passed"])
        self.assertTrue(any("final source/AOT/build hash/version" in e for e in result["errors"]))

    def test_actual_guard_latches_ownership_failure_during_measurement(self):
        owner=sampled_owner(4_000_000_000);owner.runtime_files=self.guard
        (self.root/"source.py").write_bytes(b"drift")
        observe_exact(owner,{},4.3)
        self.assertIn("version drift",owner.failure)
        with self.assertRaisesRegex(RuntimeError,"version drift"):
            owner.require_warmed_quiescence({"measurement_start_ns":2_000_000_000,"measurement_end_ns":4_000_000_000})


class WarmedReplayHardeningTests(unittest.TestCase):
    def test_strict_summary_rejects_warmed_launch_intent_without_boundary_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p,_,_,_,_=client_packet(root)
            p["schema"]="megartx-m1-eager-benchmark-v1";p.pop("timing_admission")
            p["driver_source_hashes"].pop("scripts/m1_decode_profile.py")
            p["plan_sha256"]=digest({k:v for k,v in p.items() if k != "plan_sha256"})
            write_json(root / "eager-benchmark-plan.json",p)
            dispatch=json.loads((root / "eager-benchmark/dispatch.json").read_text())
            dispatch.update(schema=p["schema"],plan_sha256=p["plan_sha256"]);dispatch.pop("timing_admission")
            write_json(root / "eager-benchmark/dispatch.json",dispatch)
            for name in ("warmed-boundaries.json","warmed-launch-boundaries.json"):(root / name).unlink()
            write_json(root / "eager-benchmark-cleanup.json",{"cleanup_complete":True})
            with self.assertRaisesRegex(RuntimeError,"opt-in differs"):client.summarize_run(root)
            launch=json.loads((root / "launch-manifest.json").read_text());launch["m1_warmed_timing_requested"]=False
            write_json(root / "launch-manifest.json",launch)
            with self.assertRaisesRegex(RuntimeError,"warmed launch policy"):client.summarize_run(root)

    def test_identity_reappearing_as_zombie_after_disappearance_is_uncertain(self):
        owner=sampled_owner(4_000_000_000,[(.4,[Process(30,30,20,0,"tileiras","S")]),(.5,[])])
        observe_exact(owner,{20:Process(20,20,10),30:Process(30,30,20,0,"tileiras","Z",compiler_identity_verified=True)},4.3)
        with self.assertRaisesRegex(RuntimeError,"uncertain"):
            owner.require_warmed_quiescence({"measurement_start_ns":2_000_000_000,"measurement_end_ns":4_000_000_000})


    def test_resource_nan_inf_boolean_negative_or_missing_is_never_evidence(self):
        report=sampled_owner(4_000_000_000).report()
        window={"measurement_start_ns":2_000_000_000,"measurement_end_ns":4_000_000_000}
        for field in ("sampled_peak_compiler_rss_bytes","shared_compiler_elapsed_seconds"):
            for value in (float("nan"),float("inf"),True,-1,None,"0"):
                with self.subTest(field=field,value=value),self.assertRaises(RuntimeError):
                    evaluate_warmed_quiescence({**report,field:value},window)

    def test_sticky_classification_cannot_be_removed_from_completed_startup(self):
        sample=Process(30,30,20,0,"tileiras","Z",compiler_identity_verified=True)
        report=sampled_owner(4_000_000_000,[(.4,[sample]),(.5,[])]).report()
        for field in ("timing_classification_history","timing_unknown_or_work_identities","compiler_lifetimes"):
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                evaluate_warmed_quiescence({**report,field:[]},{"measurement_start_ns":2_000_000_000,"measurement_end_ns":4_000_000_000})


class WarmedResourceEvidenceTests(unittest.TestCase):
    def test_initial_admission_and_replay_reject_empty_malformed_and_nonfinite_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p,_,boundary,_,_=client_packet(root)
            owned=write_owned(root,sampled_owner(boundary["measurement_end_ns"]))
            path=root/"gpu-telemetry.jsonl"
            good=[json.loads(line) for line in path.read_text().splitlines()]
            receipt=client.validate_warmed_run(root,p,owned)
            write_json(root/"warmed-timing-admission.json",receipt)
            mutations=[[],good[1:-1],good[:1],list(reversed(good))]
            for field in range(7):
                for value in ("NaN","Infinity","-1",True,None):
                    rows=copy.deepcopy(good);rows[0]["values"][field]=value;mutations.append(rows)
            for field,value in (("fields",["memory.used","memory.free"]),("values",["23000","9000"]),
                                ("exit",1),("exit",False),("host_available_kib",None),
                                ("host_available_kib",float("nan")),("host_available_kib",-1),
                                ("sample_started_ns",True)):
                rows=copy.deepcopy(good);rows[0][field]=value;mutations.append(rows)
            rows=copy.deepcopy(good);rows[0]["values"][1]="2047";mutations.append(rows)
            rows=copy.deepcopy(good);rows[0]["values"][2]="101";mutations.append(rows)
            rows=copy.deepcopy(good);rows[0]["values"][-1]="unknown";mutations.append(rows)
            for i,rows in enumerate(mutations):
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.subTest(mutation=i),self.assertRaises(RuntimeError):client.validate_warmed_run(root,p,owned)
                with self.subTest(replay=i),self.assertRaises(RuntimeError):client.summarize_run(root)

    def test_actual_sampler_rejects_nan_inf_negative_and_missing_scalars_and_writes_final_sample(self):
        import ast
        from test_m1_launcher_resources import TREE
        sampler_code=code([next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="sampler")])
        for used,free,host in (("NaN","NaN",16*1024*1024),("23000","Infinity",16*1024*1024),
                               ("-1","9000",16*1024*1024),("23000","9000",-1),("23000","9000",16*1024*1024)):
            with self.subTest(used=used,free=free,host=host),tempfile.TemporaryDirectory() as directory:
                stop=SimpleNamespace(is_set=Mock(side_effect=[False,True]),wait=Mock())
                fail=Mock();values=f"{used}, {free}, 0, 100, 40, 1000, 1000, P0"
                env={"output":Path(directory),"stop_sample":stop,"subprocess":SimpleNamespace(run=Mock(
                    return_value=SimpleNamespace(returncode=0,stdout=values))),"json":json,
                    "time":SimpleNamespace(monotonic_ns=Mock(side_effect=[1,2,3,4]),time_ns=lambda:1),
                    "eager_benchmark":True,"server":SimpleNamespace(poll=lambda:None),"fail_guard":fail,
                    "pathlib":__import__("pathlib")}
                with patch.object(Path,"read_text",return_value=f"MemAvailable: {host} kB\n"):
                    exec(sampler_code,env);env["sampler"]()
                self.assertEqual(fail.call_count,0 if used=="23000" and free=="9000" and host>0 else 2)
                rows=[json.loads(r) for r in (Path(directory)/"gpu-telemetry.jsonl").read_text().splitlines()]
                samples=[r for r in rows if "values" in r]
                self.assertEqual(len(samples),2)
                self.assertEqual(samples[-1]["sample_started_ns"],3)
                self.assertEqual(samples[-1]["monotonic_ns"],4)

    def test_actual_preflight_rejects_nonfinite_negative_or_unavailable_gpu_headroom(self):
        import ast, math
        block=RUN.body[0].body
        i=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="idle_free")
        preflight=code(block[i:i+2])
        for value,status in (("NaN",0),("Infinity",0),("-1",0),("2047",0),("9000",1)):
            with self.subTest(value=value,status=status),self.assertRaisesRegex(RuntimeError,"preflight unavailable"):
                exec(preflight,{"math":math,"idle":SimpleNamespace(returncode=status,stdout=f"name,12.0,24000,{value},driver,450")})


class WarmedIntegerClockTests(unittest.TestCase):
    def test_actual_snapshot_exact_ns_terminal_at_opening_boundary_rejects_without_rounding(self):
        ns=2_000_000_003
        for delta in (-1,0,1):
            owner=OwnedProcesses(Process(10,10,1));root=Process(20,20,10);owner.register(root)
            zombie=Process(30,30,20,0,"tileiras","Z",compiler_identity_verified=True)
            def sample(at,processes):
                with patch("m1_owned_processes.snapshot",return_value=processes), \
                     patch("m1_owned_processes.time.monotonic_ns",side_effect=[at,at]):owner.sample()
            for at in range(1_800_000_000,2_000_000_000,50_000_000):sample(at,{20:root})
            sample(ns+delta,{20:root,30:zombie})
            for at in range(2_050_000_000,4_200_000_000,50_000_000):sample(at,{20:root})
            self.assertEqual(owner.compiler_lifetimes[(30,30)]["completed_by_ns"],ns+delta)
            self.assertEqual(owner.timing_history[(30,30)]["first_sample_ns"],ns+delta)
            with self.subTest(delta=delta):
                if delta<0:self.assertTrue(owner.require_warmed_quiescence({"measurement_start_ns":ns,"measurement_end_ns":4_000_000_000})["sampled_quiescence_passed"])
                else:
                    with self.assertRaisesRegex(RuntimeError,"late/uncertain"):
                        owner.require_warmed_quiescence({"measurement_start_ns":ns,"measurement_end_ns":4_000_000_000})

    def test_legacy_float_observation_never_supplies_warmed_boundary_authority(self):
        ns=2_000_000_003
        owner=sampled_owner(4_000_000_000)
        owner.observe({},ns/1e9)
        with self.assertRaisesRegex(RuntimeError,"exact integer observation clock"):
            owner.require_warmed_quiescence({"measurement_start_ns":ns,"measurement_end_ns":4_000_000_000})
