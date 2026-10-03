"""CPU trace-correlation controls for the perturbing capture-free observer."""
from pathlib import Path
import gzip
import hashlib
import json
import copy
import sys
import tempfile
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_m1_external_observer import (CUPTI_STREAM_ID_PROVIDER, EVENT_COUNTS,
    _check_event_counts, _per_call_trace)
from megartx.m1_external_observer import ExternalObserver, SETTER
import megartx.m1_external_observer as observer_module
import megartx.m1_process_lifecycle as lifecycle_module
import compare_m1_external_observer as observer_compare


def stream_bindings():
    # Handles and CUPTI/profiler IDs intentionally differ.
    return [{"thread_id": 7, "stream_handle": 77, "profiler_stream_id": 19,
             "stream_id_api": "cuptiGetStreamIdEx",
             "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER,
             "per_thread_stream": False}
            for _ in range(30)]


def trace(lane):
    events = []
    for i in range(30):
        start = i * 100
        common = {"ph": "X", "pid": 5, "tid": 7}
        events.extend([
            dict(common, cat="user_annotation", name="megartx::m1_routed_" + lane,
                 ts=start, dur=35),
            dict(common, cat="user_annotation", name=f"megartx::m1_external_call_{i:04d}",
                 ts=start + 1, dur=24),
            dict(common, cat="user_annotation", name="megartx::m1_preparation_" + lane,
                 ts=start + 2, dur=20),
        ])
        kernels = ["MainloopSm120ArrayTmaWarpSpecializedBlockScaled_GEMM"] * 2
        kernels += (["m1_maps_expand"] if lane == "fused" else
                    ["fusedBuildExpertMapsSortFirstTokenKernel", "expandInputRowsKernel<fp4>"])
        for j, name in enumerate(kernels):
            correlation = i * 10 + j
            events.append(dict(common, cat="cuda_runtime", name="cudaLaunchKernel",
                               ts=start + 3 + j, dur=0.2,
                               args={"correlation": correlation}))
            events.append({"ph": "X", "cat": "kernel", "name": name,
                           "pid": 0, "tid": 19, "ts": 100000 + start + j,
                           "dur": 1, "args": {"correlation": correlation, "stream": 19}})
    return events


def process_owner_fixture(root):
    Path(root).mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence = Path(root) / "process-evidence"
    evidence.mkdir(mode=0o700)
    identity = {"pid": 51, "ppid": 50, "process_name": "EngineCore"}
    api_pid_path = Path(root) / "owned-server.pid"
    api_pid_path.write_text("50\n")
    probe_sha = "d" * 64
    context_id = "50-1"
    records = [
        {"event": "process_probe_ready", "pid": 50, "ppid": 40,
         "process_name": "MainProcess", "probe_sha256": probe_sha},
        {"event": "multiprocessing_context_created", "pid": 50, "ppid": 40,
         "context_id": context_id, "actual_start_method": "spawn",
         "requested_start_method": "spawn", "probe_sha256": probe_sha},
        {"event": "engine_core_process_started", "pid": 50, "ppid": 40,
         "context_id": context_id, "context_pid": 50, "parent_pid": 50,
         "parent_ppid": 40, "child_pid": 51, "child_name": "EngineCore",
         "actual_start_method": "spawn", "probe_sha256": probe_sha},
        {"event": "process_probe_ready", "pid": 51, "ppid": 50,
         "process_name": "MainProcess", "probe_sha256": probe_sha},
    ]
    (evidence / "process-events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    bridge = {"path": "/fixture/m1_live_bridge.so", "sha256": "b" * 64}
    return identity, evidence, api_pid_path, probe_sha, bridge


def patched_owner(identity):
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(observer_module, "process_identity", return_value=identity))
    stack.enter_context(patch.object(lifecycle_module, "process_identity", return_value=identity))
    stack.enter_context(patch.dict(os.environ, {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "MEGARTX_M1_PROCESS_PROBE_ACTIVE": "1"}))
    return stack


def construct_observer(native, destination, root, contract=None):
    identity, evidence, api_pid_path, probe_sha, bridge = process_owner_fixture(root)
    with patched_owner(identity):
        observer = ExternalObserver(
            native, destination, contract or {"cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER},
            stream_id_query=lambda handle: 19,
            registration_identity=identity, bridge_identity=bridge,
            process_evidence_dir=evidence, api_pid_path=api_pid_path,
            process_probe_sha256=probe_sha)
    return observer, identity, evidence, api_pid_path, probe_sha, bridge


class ExternalTraceTests(unittest.TestCase):
    def test_each_native_call_has_its_own_correlated_candidate_or_stock_kernels(self):
        stock = _per_call_trace(trace("stock"), "stock", stream_bindings())
        fused = _per_call_trace(trace("fused"), "fused", stream_bindings())
        self.assertEqual(stock["positive_candidate_launches"], 0)
        self.assertEqual(fused["positive_candidate_launches"], 30)
        self.assertTrue(stock["native_stream_handle_bound_to_profiler_trace_id"])
        self.assertFalse(stock["raw_cuda_handle_assumed_equal_to_profiler_id"])
        self.assertEqual(stock["per_call_incumbent_kernel_counts"],
                         fused["per_call_incumbent_kernel_counts"])

    def test_global_kernel_without_call_correlation_is_rejected(self):
        events = trace("fused")
        for event in events:
            if event.get("cat") == "cuda_runtime":
                event["ts"] += 1_000_000
        with self.assertRaisesRegex(ValueError, "Consumer kernels|positive preparation|bounded call set"):
            _per_call_trace(events, "fused", stream_bindings())

    def test_wrong_profiler_thread_or_missing_numbered_scope_is_rejected(self):
        events = trace("stock")
        changed = copy.deepcopy(events)
        next(e for e in changed if e.get("name") == "megartx::m1_external_call_0000")["tid"] = 99
        with self.assertRaisesRegex(ValueError, "different threads"):
            _per_call_trace(changed, "stock", stream_bindings())
        missing = [e for e in events if e.get("name") != "megartx::m1_external_call_0003"]
        with self.assertRaisesRegex(ValueError, "numbered external call"):
            _per_call_trace(missing, "stock", stream_bindings())

    def test_profiler_stream_id_must_match_verified_native_handle_mapping(self):
        bindings = stream_bindings()
        for binding in bindings:
            binding["profiler_stream_id"] = 77
        with self.assertRaisesRegex(ValueError, "differs from the CUPTI mapping"):
            _per_call_trace(trace("fused"), "fused", bindings)


class NativeEventCountTests(unittest.TestCase):
    def _evidence(self, lane):
        expected = dict(EVENT_COUNTS)
        if lane == "fused":
            expected["installed_map_call"] = expected["installed_expand_call"] = 0
        receipt = {key: expected[key] for key in sorted(EVENT_COUNTS)}
        events = [{"event": name} for name, count in expected.items() for _ in range(count)]
        return events, receipt

    def test_event_log_and_complete_count_receipt_match_both_lanes(self):
        for lane in ("stock", "fused"):
            with self.subTest(lane=lane):
                events, receipt = self._evidence(lane)
                self.assertEqual(_check_event_counts(events, receipt, lane),
                                 {**dict(EVENT_COUNTS), **({
                                     "installed_map_call": 0,
                                     "installed_expand_call": 0} if lane == "fused" else {})})

    def test_missing_or_unexpected_event_kind_is_rejected(self):
        events, receipt = self._evidence("stock")
        receipt.pop("payload")
        with self.assertRaisesRegex(ValueError, "event log/count receipt"):
            _check_event_counts(events, receipt, "stock")
        events, receipt = self._evidence("fused")
        events[0]["event"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "event log/count receipt"):
            _check_event_counts(events, receipt, "fused")


class ExternalObserverLifecycleTests(unittest.TestCase):
    def test_sidecar_is_not_claimed_until_the_first_native_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "m1-external-observer"
            native = SimpleNamespace()
            setattr(native, SETTER, lambda callback: 0)
            observer, identity, *_ = construct_observer(
                native, destination, Path(temporary) / "owner")
            with patched_owner(identity):
                self.assertFalse(destination.exists())
                observer.begin_case("cached", {"request_id": "controlled-cached"})
                self.assertFalse(destination.exists())

                observer.begin_call(0, "stock", 77, {"request_id": "controlled-cached"})
                self.assertTrue((destination / "observer-contract.json").is_file())
                self.assertTrue((destination / "calls/call-0000/events.jsonl").is_file())
                observer.abort_case(RuntimeError("test cleanup"))

    def test_same_native_bridge_rejects_callback_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            registrations = []
            native = SimpleNamespace()
            setattr(native, SETTER,
                    lambda callback: registrations.append(callback) or 0)
            contract = {"cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER}
            first, identity, *_ = construct_observer(
                native, Path(temporary) / "first", Path(temporary) / "owner-one", contract)
            registered_callback = registrations[0]

            with self.assertRaisesRegex(RuntimeError, "already registered"):
                construct_observer(native, Path(temporary) / "second",
                                   Path(temporary) / "owner-two", contract)

            self.assertEqual(registrations, [registered_callback])
            self.assertIs(first._callback_ref, registered_callback)
            self.assertFalse((Path(temporary) / "first").exists())
            self.assertFalse((Path(temporary) / "second").exists())
            with patched_owner(identity):
                first.abort_case(RuntimeError("test cleanup"))

    def test_two_observers_do_not_claim_or_overwrite_each_others_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "m1-external-observer"

            def create_observer():
                native = SimpleNamespace()
                setattr(native, SETTER, lambda callback: 0)
                return construct_observer(native, destination,
                    Path(temporary) / ("owner-" + str(id(native))))

            owner, owner_identity, *_ = create_observer()
            nonowner, nonowner_identity, *_ = create_observer()
            with patched_owner(owner_identity):
                owner.begin_case("cached", {"request_id": "controlled-cached"})
            with patched_owner(nonowner_identity):
                nonowner.begin_case("cached", {"request_id": "controlled-cached"})
            self.assertFalse(destination.exists())

            with patched_owner(owner_identity):
                owner.begin_call(0, "stock", 77, {"request_id": "controlled-cached"})
                owner.finish_call(0, True)
                owner.abort_case(RuntimeError("owner aborted"))
            with patched_owner(nonowner_identity):
                with self.assertRaises(FileExistsError):
                    nonowner.begin_call(0, "stock", 77, {"request_id": "controlled-cached"})

            invalidation = destination / "INVALIDATED.json"
            original = invalidation.read_bytes()
            with patched_owner(nonowner_identity):
                with self.assertRaises(FileExistsError):
                    nonowner.abort_case(RuntimeError("nonowner must not overwrite"))
            self.assertEqual(invalidation.read_bytes(), original)
            self.assertEqual(json.loads(original)["error"], "owner aborted")


class ExternalObserverEntryPointTests(unittest.TestCase):
    def _fixture(self, root, lane):
        run = root / (lane + "-observed")
        captured_run = root / (lane + "-captured")
        run.mkdir()
        captured_run.mkdir()
        observer_root = run / "m1-external-observer"
        calls_root = observer_root / "calls"
        model_root = observer_root / "model" / "cached"
        calls_root.mkdir(parents=True)
        model_root.mkdir(parents=True)
        observer_case = run / "controlled" / "cached"
        observer_case.mkdir(parents=True)
        captured_case = captured_run / "controlled" / "cached"
        captured_case.mkdir(parents=True)
        captured_preparation = captured_run / "preparation"
        captured_preparation.mkdir()
        (captured_case / "logits-records.jsonl").write_bytes(b"")
        (captured_case / "kv-records.json").write_text("[]")

        token_sha, schedule_sha = "a" * 64, "c" * 64
        binary_sha = "b" * 64
        api_pid, engine_pid = 50, 51
        process_root = run / "m1-process-evidence"
        process_root.mkdir()
        api_pid_path = run / "owned-server.pid"
        api_pid_path.write_text(f"{api_pid}\n")
        probe_path = Path(__file__).resolve().parents[1] / "scripts/m1_process_probe/sitecustomize.py"
        process_probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        bridge_identity = {"path": "/fixture/m1_live_bridge.so", "sha256": binary_sha}
        registration = {"pid": engine_pid, "ppid": api_pid, "process_name": "EngineCore"}
        lifecycle = {"actual_start_method": "spawn", "api_server_pid": api_pid,
            "api_server_ppid": 40, "engine_core_pid": engine_pid,
            "engine_core_ppid": api_pid, "engine_core_process_name": "EngineCore",
            "observer_registration_pid": engine_pid,
            "observer_registration_owner_pid": engine_pid,
            "observer_registered_by_api_server": False,
            "observer_callback_unregistered": True,
            "process_probe_sha256": process_probe_sha,
            "bridge_identity": bridge_identity}
        pre_dispatch = {"schema": "megartx-m1-pre-dispatch-owner-v1",
            "passed": True, "before_request_dispatch": True, **lifecycle}
        (process_root / "pre-dispatch.json").write_text(json.dumps(pre_dispatch))
        (process_root / "cleanup.json").write_text(json.dumps({
            "schema": "megartx-m1-process-cleanup-v1", "server_pid": api_pid,
            "cleanup_complete": True, "owned_process_group_alive": False,
            "owned_gpu_pids_remaining": []}))
        contract = {"abi_version": 2, "view_count": 15, "view_bytes": 32,
                    "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER}
        build = {"live_contract": contract, "binary_sha256": binary_sha,
                 "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER,
                 "installed_package_versions": {
                     "nvidia-cuda-cupti": CUPTI_STREAM_ID_PROVIDER["version"]},
                 "installed_pins": {"/fixture/libcupti.so.13":
                                     CUPTI_STREAM_ID_PROVIDER["library_sha256"]}}
        ids = np.arange(8, dtype="<i4").reshape(1, 8)
        weight_bits = np.zeros((1, 8), dtype="<u4")
        route_arrays = {"positions": np.asarray([32], dtype=np.int64),
                        "tokens": np.asarray([42], dtype=np.int64),
                        "ids": ids, "weight_bits": weight_bits}
        payloads = {"ids.bin": ids.tobytes(), "route-weights.bin": weight_bits.tobytes()}
        routes, baseline_calls, model_routes = [], [], []
        extents = [32]
        runner = b"tile shape ID: 128x128x128\nepilogue fusion type: 0"
        workspace = b"workspace_bytes=3185408"
        metadata = (runner + workspace).decode()
        native_status = 1 if lane == "fused" else 0

        for layer in range(30):
            layer_name = f"language_model.model.layers.{layer}.moe.experts"
            route_file = f"route-{layer:02d}.npz"
            route_path = captured_case / route_file
            route_path.write_bytes(b"bounded route fixture")
            route_sha = hashlib.sha256(route_path.read_bytes()).hexdigest()
            routes.append({"forward": 1, "layer": layer, "file": route_file,
                           "sha256": route_sha})
            model_routes.append({"file": route_file, "sha256": route_sha})
            (model_root / route_file).write_bytes(b"bounded route fixture")

            record = {"scope": "controlled_live_request", "forward_index": 1,
                      "layer_name": layer_name, "positions": [32], "tokens": [42],
                      "owner_extents": extents, "native_metadata": metadata}
            baseline_calls.append((record, payloads))
            base_call = captured_preparation / f"call-{layer:04d}"
            base_call.mkdir()
            (base_call / "receipt.json").write_text(json.dumps({
                "scope": record["scope"], "layer_name": layer_name,
                "forward_index": 1}))
            (base_call / "consumer-envelopes.json").write_bytes(b"{\"stages\":[]}")
            (base_call / "consumer-masks.json").write_bytes(b"{\"stages\":[]}")

        manifest = {"routes": routes, "executed_interventions": [],
                    "token_sha256": token_sha, "schedule_sha256": schedule_sha}
        captured = {"manifest": manifest,
                    "request": {"token_sha256": token_sha, "schedule_sha256": schedule_sha},
                    "calls": baseline_calls, "case": captured_case}

        (run / "status.json").write_text(json.dumps({"phase": "cleanup_complete"}))
        launch = {"m1_execution_requested": "capture-free",
                  "command": ["vllm", "serve", "--no-async-scheduling"],
                  "m1_external_observer_requested": True,
                  "m1_preparation_requested": lane,
                  "controlled_request_count": 1, "controlled_path": "cached",
                  "m1_process_lifecycle_probe_sha256": process_probe_sha,
                  "m1_process_lifecycle_expected_method": "spawn",
                  "m1_process_lifecycle_policy": "actual_spawn_owner_pre_dispatch",
                  "environment_overrides": {
                      "MEGARTX_M1_EXECUTION": "capture-free",
                      "MEGARTX_M1_PREPARATION": lane,
                      "MEGARTX_M1_EXTERNAL_OBSERVER_DIR": str(observer_root.resolve()),
                      "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                      "MEGARTX_M1_PROCESS_EXPECTED_METHOD": "spawn",
                      "MEGARTX_M1_PROCESS_EVIDENCE_DIR": str(process_root.resolve()),
                      "MEGARTX_M1_PROCESS_PROBE_SHA256": process_probe_sha,
                      "MEGARTX_M1_API_PID_FILE": str(api_pid_path.resolve()),
                      "PYTHONPATH": str(probe_path.parent.resolve()) + ":/fixture/site-packages"}}
        (run / "launch-manifest.json").write_text(json.dumps(launch))
        scalar = {"id": "controlled-cached", "request_count": 1,
                  "m1_execution": "capture-free", "capture_comparison_available": False,
                  "token_sha256": token_sha, "schedule_sha256": schedule_sha}
        (observer_case / "capture-free-request.json").write_text(json.dumps(scalar))
        (observer_root / "observer-contract.json").write_text(json.dumps({
            "schema": observer_compare.SCHEMA, "execution_mode": "capture-free",
            "internal_capture_enabled": False, "perturbs_execution": True,
            "timing_qualified": False, "native_contract": contract,
            "registration_identity": registration, "bridge_identity": bridge_identity,
            "process_probe_sha256": process_probe_sha,
            "stream_id_mapping_api": "cuptiGetStreamIdEx",
            "profiler_trace_stream_field": "kernel.args.stream",
            "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER}))

        event_names = ["lease_begin", "runner_identity", "runner_workspace",
                       "candidate_status"]
        if lane == "stock":
            event_names += ["installed_map_call", "installed_expand_call"]
        event_names += ["payload"] * 15 + ["json"] * 2 + ["lease_end"]
        event_counts = {key: event_names.count(key) for key in sorted(EVENT_COUNTS)}
        call_records = []
        for layer in range(30):
            directory = calls_root / f"call-{layer:04d}"
            (directory / "payloads").mkdir(parents=True)
            for name, value in payloads.items():
                (directory / "payloads" / name).write_bytes(value)
            (directory / "runner.json").write_bytes(runner)
            (directory / "workspace.json").write_bytes(workspace)
            (directory / "preparation.json").write_text(json.dumps({
                "backend": lane, "incumbent_map_result": lane == "stock"}))
            (directory / "consumer-envelopes.json").write_bytes(b"{\"stages\":[]}")
            (directory / "consumer-masks.json").write_bytes(b"{\"stages\":[]}")
            frame = {"request_id": "controlled-cached", "binary_sha256": binary_sha,
                     "owner_extents": extents,
                     "layer_name": f"language_model.model.layers.{layer}.moe.experts",
                     "forward_index": 1, "positions": [32], "token_ids": [42]}
            receipt = {"schema": observer_compare.SCHEMA, "call_index": layer,
                       "lane_requested": lane, "thread_id": 7, "stream": 77,
                       "pid": engine_pid, "ppid": api_pid,
                       "registration_owner_pid": engine_pid,
                       "bridge_identity": bridge_identity,
                       "profiler_stream_id": 19, "stream_id_api": "cuptiGetStreamIdEx",
                       "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER,
                       "per_thread_stream": False, "native_status": native_status,
                       "lease_released": True, "internal_capture_enabled": False,
                       "callback_error": None, "event_counts": event_counts,
                       "frame": frame}
            (directory / "observer-receipt.json").write_text(json.dumps(receipt))
            events = []
            for sequence, name in enumerate(event_names):
                payload_name = "ids.bin" if sequence % 2 == 0 else "route-weights.bin"
                event = {"sequence": sequence, "event": name, "name": payload_name,
                         "size": len(payloads.get(payload_name, b"")), "stream": 77,
                         "pid": engine_pid, "ppid": api_pid,
                         "registration_owner_pid": engine_pid,
                         "bridge_identity": bridge_identity,
                         "profiler_stream_id": 19, "stream_id_api": "cuptiGetStreamIdEx",
                         "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER,
                         "thread_id": 7, "value": 0}
                if name == "lease_end":
                    event["value"] = native_status
                events.append(event)
            (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            call_records.append(receipt)

        trace_path = observer_root / "controlled-trace.json.gz"
        with gzip.open(trace_path, "wb") as compressed:
            compressed.write(json.dumps({"traceEvents": trace(lane)}).encode())
        trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        observer_manifest = {"schema": observer_compare.SCHEMA, "case": "cached",
            "execution_mode": "capture-free", "observer_enabled": True,
            "internal_capture_enabled": False, "observer_perturbs_execution": True,
            "native_call_count": 30, "stream_id_mapping_api": "cuptiGetStreamIdEx",
            "cupti_stream_id_provider": CUPTI_STREAM_ID_PROVIDER,
            "quality_gate_passed": False, "graph_qualified": False,
            "timing_qualified": False, "trace": trace_path.name,
            "trace_sha256": trace_sha,
            "bridge_identity": bridge_identity,
            "process_lifecycle": lifecycle,
            "model_context": {"token_sha256": token_sha, "schedule_sha256": schedule_sha},
            "model_arrays": {"routes": model_routes, "stages": [], "logits": [], "kv": []}}
        (observer_root / "observer-manifest.json").write_text(json.dumps(observer_manifest))
        return {"run": run, "captured_run": captured_run, "captured": captured,
                "build": build, "route_arrays": route_arrays, "observer_receipts": call_records}

    def _with_patches(self, fixtures):
        build = fixtures[0]["build"]
        captured = {str(f["captured_run"]): f["captured"] for f in fixtures}
        route_arrays = {f"route-{layer:02d}.npz": f["route_arrays"]
                        for f in fixtures for layer in range(30)}

        def load_captured(path, lane):
            return captured[str(Path(path))]

        def fake_read_npz(directory, name, sha256, fields):
            if fields == observer_compare.ROUTE_FIELDS:
                return route_arrays[name]
            return {"stub": np.asarray([1], dtype=np.int32)}

        return (patch.object(observer_compare, "load_build", return_value=build),
                patch.object(observer_compare, "check_run", side_effect=load_captured),
                patch.object(observer_compare, "read_npz", side_effect=fake_read_npz),
                patch.object(observer_compare, "PAYLOADS",
                             {"ids.bin": 32, "route-weights.bin": 32}))

    def test_compare_observers_runs_both_external_entrypoints_on_a_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stock = self._fixture(root, "stock")
            fused = self._fixture(root, "fused")
            patches = self._with_patches([stock, fused])
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(observer_compare, "compare_runs", return_value={"passed": True}):
                report = observer_compare.compare_observers(
                    root / "build", stock["captured_run"], fused["captured_run"],
                    stock["run"], fused["run"])
            self.assertTrue(report["passed"])
            self.assertTrue(report["stock"]["trace"]["native_stream_handle_bound_to_profiler_trace_id"])
            self.assertEqual(report["fused"]["trace"]["positive_candidate_launches"], 30)
            self.assertFalse(report["timing_qualified"])

    def test_missing_output_payload_and_consumer_json_are_rejected(self):
        mutations = (
            ("payload", lambda f: (f["run"] / "m1-external-observer/calls/call-0000/payloads/ids.bin").unlink()),
            ("output", lambda f: (f["run"] / "controlled/cached/capture-free-request.json").unlink()),
            ("consumer", lambda f: (f["run"] / "m1-external-observer/calls/call-0000/consumer-masks.json").unlink()),
        )
        for label, mutate in mutations:
            with self.subTest(missing=label), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary), "stock")
                mutate(fixture)
                patches = self._with_patches([fixture])
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaises(ValueError):
                        observer_compare.check_external_run(
                            fixture["run"], "stock", fixture["captured_run"], Path(temporary) / "build")

    def test_missing_or_altered_synchronous_scheduler_flag_is_rejected(self):
        for flag in (None, "--async-scheduling"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary), "stock")
                path = fixture["run"] / "launch-manifest.json"
                launch = json.loads(path.read_text())
                launch["command"] = ["vllm", "serve"]
                if flag is not None:
                    launch["command"].append(flag)
                path.write_text(json.dumps(launch))
                patches = self._with_patches([fixture])
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaisesRegex(ValueError, "Launch manifest"):
                        observer_compare.check_external_run(
                            fixture["run"], "stock", fixture["captured_run"], Path(temporary) / "build")

    def test_late_or_unowned_pre_dispatch_process_gate_is_rejected(self):
        for field, value in (("before_request_dispatch", False),
                             ("actual_start_method", "fork"),
                             ("observer_registration_owner_pid", 99)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary), "stock")
                path = fixture["run"] / "m1-process-evidence/pre-dispatch.json"
                pre_dispatch = json.loads(path.read_text())
                pre_dispatch[field] = value
                path.write_text(json.dumps(pre_dispatch))
                patches = self._with_patches([fixture])
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaisesRegex(ValueError, "process ownership or owned-server cleanup"):
                        observer_compare.check_external_run(
                            fixture["run"], "stock", fixture["captured_run"], Path(temporary) / "build")

    def test_stale_receipt_frame_identity_fields_are_rejected(self):
        mutations = {
            "layer_name": "language_model.model.layers.29.moe.experts",
            "forward_index": 2,
            "positions": [31],
            "token_ids": [43],
            "binary_sha256": "d" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(frame_field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary), "fused")
                path = fixture["run"] / "m1-external-observer/calls/call-0000/observer-receipt.json"
                receipt = json.loads(path.read_text())
                receipt["frame"][field] = value
                path.write_text(json.dumps(receipt))
                patches = self._with_patches([fixture])
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaisesRegex(ValueError, "Observer lease identity"):
                        observer_compare.check_external_run(
                            fixture["run"], "fused", fixture["captured_run"], Path(temporary) / "build")

    def test_cupti_library_identity_is_bound_from_build_to_each_native_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary), "stock")
            path = fixture["run"] / "m1-external-observer/calls/call-0000/observer-receipt.json"
            receipt = json.loads(path.read_text())
            receipt["cupti_stream_id_provider"]["library_sha256"] = "0" * 64
            path.write_text(json.dumps(receipt))
            patches = self._with_patches([fixture])
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(ValueError, "Observer lease identity"):
                    observer_compare.check_external_run(
                        fixture["run"], "stock", fixture["captured_run"], Path(temporary) / "build")


if __name__ == "__main__":
    unittest.main()
