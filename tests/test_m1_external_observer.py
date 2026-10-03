"""CPU protocol tests for the opt-in capture-free M1 sidecar."""
import ctypes
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import megartx.m1_external_observer as external
import megartx.m1_process_lifecycle as lifecycle
from megartx.m1_external_observer import ExternalObserver, SCHEMA
from megartx.m1_process_lifecycle import PROCESS_EVIDENCE, process_identity


CUPTI_PROVIDER = {
    "distribution": "nvidia-cuda-cupti", "version": "13.0.85",
    "library_name": "libcupti.so.13",
    "library_sha256": "e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e",
}


class Function:
    def __init__(self):
        self.callback = None

    def __call__(self, callback):
        self.callback = callback
        return 0


class Native:
    def __init__(self):
        self.megartx_m1_set_external_observer_v1 = Function()
        self.megartx_m1_active = lambda: 0


def lifecycle_fixture(directory, identity=None):
    evidence = Path(directory) / "process-evidence"
    evidence.mkdir(mode=0o700)
    identity = dict(identity or process_identity())
    identity["process_name"] = "EngineCore"
    api_pid = identity["ppid"]
    api_pid_path = Path(directory) / "owned-server.pid"
    api_pid_path.write_text(str(api_pid) + "\n")
    probe_sha = "a" * 64
    context_id = f"{api_pid}-1"
    rows = [
        {"event": "process_probe_ready", "pid": api_pid, "ppid": identity["ppid"],
         "process_name": "MainProcess", "probe_sha256": probe_sha},
        {"event": "multiprocessing_context_created", "pid": api_pid,
         "ppid": identity["ppid"], "context_id": context_id,
         "actual_start_method": "spawn", "requested_start_method": "spawn",
         "probe_sha256": probe_sha},
        {"event": "engine_core_process_started", "pid": api_pid,
         "ppid": identity["ppid"], "context_id": context_id,
         "context_pid": api_pid, "parent_pid": api_pid,
         "parent_ppid": identity["ppid"], "child_pid": identity["pid"],
         "child_name": "EngineCore", "actual_start_method": "spawn",
         "probe_sha256": probe_sha},
        # sitecustomize runs before multiprocessing applies the child's name.
        {"event": "process_probe_ready", "pid": identity["pid"],
         "ppid": api_pid, "process_name": "MainProcess", "probe_sha256": probe_sha},
    ]
    (evidence / PROCESS_EVIDENCE).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return identity, evidence, api_pid_path, probe_sha


def patched_identity(identity):
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(external, "process_identity", return_value=identity))
    stack.enter_context(patch.object(lifecycle, "process_identity", return_value=identity))
    return stack


def make_observer(directory, native, identity=None):
    identity, evidence, api_pid_path, probe_sha = lifecycle_fixture(directory, identity)
    bridge = {"path": "/task-local/bridge.so", "sha256": "b" * 64}
    env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "MEGARTX_M1_PROCESS_PROBE_ACTIVE": "1"}
    with patch.dict(os.environ, env), patched_identity(identity):
        observer = ExternalObserver(
            native, Path(directory) / "observer",
            {"abi_version": 2, "cupti_stream_id_provider": CUPTI_PROVIDER},
            stream_id_query=lambda handle: 901,
            registration_identity=identity, bridge_identity=bridge,
            process_evidence_dir=evidence, api_pid_path=api_pid_path,
            process_probe_sha256=probe_sha)
    return observer, identity, evidence, api_pid_path, probe_sha, bridge


class ExternalObserverTests(unittest.TestCase):
    def test_pre_dispatch_gate_waits_for_registration_then_validates_exact_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            identity, evidence, api_pid_path, probe_sha = lifecycle_fixture(directory)
            bridge = {"path": "/task-local/bridge.so", "sha256": "b" * 64}
            observer_root = Path(directory) / "observer"
            with patched_identity(identity):
                self.assertIsNone(lifecycle.validate_engine_core_registration_before_dispatch(
                    evidence, api_pid_path, probe_sha, bridge))
            with patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                                         "MEGARTX_M1_PROCESS_PROBE_ACTIVE": "1"}), \
                 patched_identity(identity):
                observer = ExternalObserver(
                    native, observer_root,
                    {"abi_version": 2, "cupti_stream_id_provider": CUPTI_PROVIDER},
                    stream_id_query=lambda handle: 901,
                    registration_identity=identity, bridge_identity=bridge,
                    process_evidence_dir=evidence, api_pid_path=api_pid_path,
                    process_probe_sha256=probe_sha)
                lifecycle_record = lifecycle.validate_engine_core_registration_before_dispatch(
                    evidence, api_pid_path, probe_sha, bridge)
            self.assertTrue(lifecycle_record["actual_start_method"] == "spawn")
            self.assertEqual(lifecycle_record["engine_core_pid"], identity["pid"])
            self.assertEqual(lifecycle_record["bridge_identity"], bridge)
            self.assertFalse(observer_root.exists(), "registration must leave sidecar ownership lazy")
            with patched_identity(identity):
                observer.abort_case(RuntimeError("test cleanup"))

    def test_capture_free_begin_end_and_bounded_payload_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            observer, identity, evidence, api_pid_path, probe_sha, bridge = make_observer(directory, native)
            self.assertFalse(observer.root.exists(), "sidecar ownership must remain lazy")
            with patched_identity(identity):
                pre_dispatch = lifecycle.validate_engine_core_registration_before_dispatch(
                    evidence, api_pid_path, probe_sha, bridge)
            self.assertTrue(pre_dispatch["actual_start_method"] == "spawn")
            self.assertEqual(pre_dispatch["engine_core_pid"], identity["pid"])
            self.assertFalse(observer.root.exists(), "pre-dispatch proof must not claim the lazy sidecar")
            frame = {"request_id": "controlled-cached", "positions": [32]}
            with patched_identity(identity):
                observer.begin_call(0, "fused", 77, frame)
                callback = observer._callback_ref
                self.assertEqual(callback(b"lease_begin", b"fused", None, 0, 77, 0), 0)
                payload = ctypes.create_string_buffer(b"i" * 32)
                self.assertEqual(callback(b"payload", b"ids.bin", ctypes.addressof(payload), 32, 77, 0), 0)
                status = ctypes.create_string_buffer(b'{"backend":"fused"}')
                self.assertEqual(callback(b"candidate_status", b"preparation.json",
                    ctypes.addressof(status), len(b'{"backend":"fused"}'), 77, 1), 0)
                self.assertEqual(callback(b"lease_end", b"fused", None, 0, 77, 1), 0)
                receipt = observer.finish_call(1, True)
            self.assertFalse(receipt["internal_capture_enabled"])
            self.assertEqual(receipt["event_counts"]["payload"], 1)
            self.assertEqual(receipt["stream"], 77)
            self.assertEqual(receipt["profiler_stream_id"], 901)
            self.assertEqual(receipt["stream_id_api"], "cuptiGetStreamIdEx")
            self.assertEqual(receipt["cupti_stream_id_provider"], CUPTI_PROVIDER)
            self.assertEqual(receipt["pid"], identity["pid"])
            self.assertEqual(receipt["registration_owner_pid"], identity["pid"])
            self.assertEqual(receipt["bridge_identity"], bridge)
            observer_contract = json.loads((observer.root / "observer-contract.json").read_text())
            self.assertEqual(observer_contract["cupti_stream_id_provider"], CUPTI_PROVIDER)
            self.assertEqual(observer_contract["registration_identity"], identity)
            self.assertEqual((observer.root / "calls/call-0000/payloads/ids.bin").read_bytes(), b"i" * 32)
            with patched_identity(identity):
                observer.abort_case(RuntimeError("test cleanup"))
            self.assertFalse(bool(native.megartx_m1_set_external_observer_v1.callback))
            rows = [json.loads(line) for line in (evidence / PROCESS_EVIDENCE).read_text().splitlines()]
            self.assertEqual([r["event"] for r in rows[-2:]],
                             ["observer_registered", "observer_unregistered"])

    def test_thread_stream_or_event_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            observer, identity, *_ = make_observer(directory, native)
            with patched_identity(identity):
                observer.begin_call(0, "stock", 91, {})
                callback = observer._callback_ref
                self.assertEqual(callback(b"lease_begin", b"stock", None, 0, 92, 0), -1)
                self.assertIsNotNone(observer.callback_error)
                observer.finish_call(-1, False, RuntimeError("native begin failed"))
                observer.abort_case(RuntimeError("test cleanup"))
            self.assertFalse(bool(native.megartx_m1_set_external_observer_v1.callback))

    def test_duplicate_bridge_registration_fails_without_replacing_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            observer, identity, evidence, api_pid_path, probe_sha, bridge = make_observer(
                directory, native)
            callback = native.megartx_m1_set_external_observer_v1.callback
            with patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                                         "MEGARTX_M1_PROCESS_PROBE_ACTIVE": "1"}), \
                 patched_identity(identity), self.assertRaisesRegex(RuntimeError, "already registered"):
                ExternalObserver(native, Path(directory) / "other", observer.contract,
                    stream_id_query=lambda handle: 901,
                    registration_identity=identity, bridge_identity=bridge,
                    process_evidence_dir=evidence, api_pid_path=api_pid_path,
                    process_probe_sha256=probe_sha)
            self.assertIs(native.megartx_m1_set_external_observer_v1.callback, callback)
            with patched_identity(identity):
                observer.abort_case(RuntimeError("test cleanup"))

    def test_inherited_observer_owner_is_rejected_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            observer, identity, *_ = make_observer(directory, native)
            inherited = dict(identity, pid=identity["pid"] + 1000)
            with patched_identity(inherited), self.assertRaisesRegex(RuntimeError, "outside its EngineCore"):
                observer.begin_call(0, "fused", 77, {})
            with patched_identity(identity):
                observer.abort_case(RuntimeError("test cleanup"))
            self.assertFalse(bool(native.megartx_m1_set_external_observer_v1.callback))

    def test_nonspawn_context_is_rejected_during_startup_before_registration_or_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            identity, evidence, api_pid_path, probe_sha = lifecycle_fixture(directory)
            bridge = {"path": "/task-local/bridge.so", "sha256": "b" * 64}
            destination = Path(directory) / "observer"
            event_path = evidence / PROCESS_EVIDENCE
            raw = event_path.read_text().replace('"actual_start_method": "spawn"',
                                                  '"actual_start_method": "fork"')
            event_path.write_text(raw)
            with self.assertRaisesRegex(RuntimeError, "actual multiprocessing context is not spawn"):
                lifecycle.validate_engine_core_registration_before_dispatch(
                    evidence, api_pid_path, probe_sha, bridge)
            with patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                                         "MEGARTX_M1_PROCESS_PROBE_ACTIVE": "1"}), \
                 patched_identity(identity), self.assertRaisesRegex(RuntimeError, "actual multiprocessing context is not spawn"):
                ExternalObserver(
                    native, destination,
                    {"abi_version": 2, "cupti_stream_id_provider": CUPTI_PROVIDER},
                    stream_id_query=lambda handle: 901,
                    registration_identity=identity, bridge_identity=bridge,
                    process_evidence_dir=evidence, api_pid_path=api_pid_path,
                    process_probe_sha256=probe_sha)
            self.assertFalse(destination.exists())
            self.assertFalse(bool(native.megartx_m1_set_external_observer_v1.callback))
            rows = [json.loads(line) for line in event_path.read_text().splitlines()]
            self.assertFalse(any(row.get("event") == "observer_registered" for row in rows))

    def test_model_serialization_rejects_object_arrays_and_marks_perturbation(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native()
            observer, identity, *_ = make_observer(directory, native)
            with patched_identity(identity):
                observer.begin_case("cached", {"token_sha256": "a" * 64})
                with self.assertRaisesRegex(RuntimeError, "object arrays"):
                    observer.write_model_npz("routes", "bad.npz", {"unsafe": np.asarray([object()])}, {})
                observer.write_model_npz("routes", "routes-00-00.npz",
                    {"ids": np.asarray([[1]], dtype=np.int32)}, {"layer": 0})
            with gzip.open(observer.root / "controlled-trace.json.gz", "wb") as stream:
                stream.write(json.dumps({"traceEvents": []}).encode())
            observer.next_call = 30  # Bounded native-call protocol is tested separately.
            with patched_identity(identity):
                manifest = observer.finish_case(observer.root / "controlled-trace.json.gz")
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertTrue(manifest["observer_perturbs_execution"])
            self.assertFalse(manifest["timing_qualified"])
            self.assertEqual(manifest["process_lifecycle"]["actual_start_method"], "spawn")
            self.assertEqual(manifest["process_lifecycle"]["engine_core_pid"], identity["pid"])
            self.assertTrue(manifest["process_lifecycle"]["observer_callback_unregistered"])
            self.assertFalse(bool(native.megartx_m1_set_external_observer_v1.callback))
            self.assertTrue((observer.root / "observer-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
