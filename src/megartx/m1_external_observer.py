"""Explicit, perturbing sidecar for capture-free M1 validation only.

This observer is constructed only when the launcher passes a fresh destination.
It records native lease events/payloads through the versioned callback ABI and
keeps model arrays and the profiler trace outside the ordinary controlled
request directory. It is validation instrumentation and must never be used for
timing.
"""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading

import numpy as np


SCHEMA = "megartx-m1-external-observer-v1"
SETTER = "megartx_m1_set_external_observer_v1"
MAX_EVENT_BYTES = 8 << 20
MAX_JSON_BYTES = 2 << 20
MAX_MODEL_ARRAY_BYTES = 32 << 20
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
JSON_EVENTS = {"consumer-envelopes.json", "consumer-masks.json",
               "runner.json", "workspace.json", "preparation.json"}
EVENTS = {"lease_begin", "lease_end", "runner_identity", "runner_workspace",
          "candidate_status", "installed_map_call", "installed_expand_call",
          "payload", "json"}
STREAM_ID_API = "cuptiGetStreamIdEx"
_REGISTRATION_LOCK = threading.Lock()
_REGISTERED_OBSERVERS = {}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ExternalObserver:
    """Record bounded native and model evidence for one explicit validation run."""

    CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_void_p, ctypes.c_uint64,
                                ctypes.c_uint64, ctypes.c_int)

    def __init__(self, native, destination, contract, stream_id_query=None):
        self.native = native
        self._registration_key = _native_registration_key(native)
        self.root = Path(destination).resolve()
        self.calls_root = self.root / "calls"
        self.model_root = self.root / "model"
        self._root_initialized = False
        self.contract = contract
        self.callback_error = None
        self.call = None
        self.next_call = 0
        self.model_entries = {"routes": [], "stages": [], "logits": [], "kv": []}
        self.trace_sha256 = None
        self.case = None
        self._cupti = None
        self._cupti_path = None
        self._cupti_identity = contract.get("cupti_stream_id_provider")
        self._stream_id_query = stream_id_query or self._query_profiler_stream_id
        if stream_id_query is None:
            self._load_cupti()
        self._callback_ref = self.CALLBACK(self._callback)
        setter = getattr(native, SETTER)
        setter.argtypes = [self.CALLBACK]
        setter.restype = ctypes.c_int
        with _REGISTRATION_LOCK:
            if self._registration_key in _REGISTERED_OBSERVERS:
                raise RuntimeError("external observer callback is already registered for this bridge in this process")
            if setter(self._callback_ref) != 0:
                raise RuntimeError("external observer callback registration failed")
            # The native library retains this process-local function pointer.
            # Keep its owner alive and prevent a second instance replacing it.
            _REGISTERED_OBSERVERS[self._registration_key] = self

    def _ensure_root(self):
        """Claim this process's shared sidecar only when it records native work."""
        if self._root_initialized:
            return
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.calls_root.mkdir(mode=0o700)
        self.model_root.mkdir(mode=0o700)
        with (self.root / "observer-contract.json").open("x") as stream:
            json.dump({"schema": SCHEMA, "execution_mode": "capture-free",
                       "internal_capture_enabled": False,
                       "perturbs_execution": True,
                       "stream_id_mapping_api": STREAM_ID_API,
                       "profiler_trace_stream_field": "kernel.args.stream",
                       "cupti_stream_id_provider": self._cupti_identity,
                       "timing_qualified": False,
                       "native_contract": self.contract}, stream, indent=2)
        self._root_initialized = True

    def _load_cupti(self):
        """Load the source-bound CUPTI library used for trace stream IDs."""
        if self._cupti is None:
            distribution = importlib.metadata.distribution("nvidia-cuda-cupti")
            matches = [distribution.locate_file(path) for path in (distribution.files or ())
                       if Path(str(path)).name == "libcupti.so.13"]
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError("installed CUDA 13 CUPTI library identity is missing or ambiguous")
            path = matches[0].resolve()
            identity = {"distribution": distribution.metadata.get("Name"),
                        "version": distribution.version,
                        "library_name": path.name,
                        "library_sha256": _sha(path)}
            if identity != self._cupti_identity:
                raise RuntimeError("installed CUPTI stream-ID provider differs from the compiled build contract")
            self._cupti_path = path
            self._cupti = ctypes.CDLL(str(path))
            query = self._cupti.cuptiGetStreamIdEx
            query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint8,
                              ctypes.POINTER(ctypes.c_uint32)]
            query.restype = ctypes.c_int
        return self._cupti_identity

    def _query_profiler_stream_id(self, stream_handle):
        """Map a CUDA stream handle to the context-local ID used by CUPTI traces."""
        self._load_cupti()
        stream_id = ctypes.c_uint32()
        result = self._cupti.cuptiGetStreamIdEx(
            None, ctypes.c_void_p(stream_handle), ctypes.c_uint8(0), ctypes.byref(stream_id))
        if result != 0:
            raise RuntimeError("CUPTI cuptiGetStreamIdEx failed with status " + str(result))
        return int(stream_id.value)

    def begin_call(self, index, lane, stream, frame):
        if self.call is not None or index != self.next_call or lane not in {"stock", "fused"}:
            raise RuntimeError("external observer call sequence is nested or unbounded")
        self._ensure_root()
        if type(stream) is not int or stream == 0:
            raise RuntimeError("external observer requires the owned nondefault stream")
        profiler_stream_id = self._stream_id_query(stream)
        if type(profiler_stream_id) is not int or profiler_stream_id == 0:
            raise RuntimeError("external observer could not map its CUDA stream to a nondefault profiler stream ID")
        directory = self.calls_root / f"call-{index:04d}"
        directory.mkdir(mode=0o700, exist_ok=False)
        (directory / "payloads").mkdir(mode=0o700)
        (directory / "events.jsonl").touch(exist_ok=False)
        self.call = {"index": index, "lane": lane, "stream": stream,
                     "profiler_stream_id": profiler_stream_id,
                     "thread_id": threading.get_native_id(),
                     "directory": directory, "frame": frame,
                     "events": [], "files": []}
        self.callback_error = None

    def _decode(self, value, label):
        if value is None:
            raise RuntimeError("native observer omitted " + label)
        try:
            return value.decode("ascii")
        except UnicodeError as error:
            raise RuntimeError("native observer supplied invalid " + label) from error

    def _callback(self, event, name, data, size, stream, value):
        try:
            self._write_event(event, name, data, size, stream, value)
            return 0
        except BaseException as error:
            self.callback_error = error
            return -1

    def _write_event(self, raw_event, raw_name, pointer, size, stream, value):
        call = self.call
        if call is None:
            raise RuntimeError("native observer event arrived outside an active call")
        event, name = self._decode(raw_event, "event"), self._decode(raw_name, "name")
        if event not in EVENTS or type(size) is not int or size < 0 or size > MAX_EVENT_BYTES:
            raise RuntimeError("native observer event exceeds its bounded protocol")
        thread_id = threading.get_native_id()
        if thread_id != call["thread_id"] or stream != call["stream"]:
            raise RuntimeError("native observer event changed owner thread or CUDA stream")
        if size and not pointer:
            raise RuntimeError("native observer payload pointer is null")
        payload = ctypes.string_at(pointer, size) if size else b""
        if event == "lease_begin":
            if name != call["lane"] or size or value != 0:
                raise RuntimeError("capture-free native lease admission differs")
        elif event == "lease_end":
            if name != call["lane"] or size:
                raise RuntimeError("capture-free native lease end differs")
        elif event == "payload":
            if name not in PAYLOADS or size != PAYLOADS[name]:
                raise RuntimeError("native observer payload name or extent differs")
            path = call["directory"] / "payloads" / name
            with path.open("xb") as stream_out:
                stream_out.write(payload)
            call["files"].append({"path": str(path.relative_to(self.root)),
                                  "sha256": _sha(path), "bytes": size})
        elif event == "json":
            if name not in {"consumer-envelopes.json", "consumer-masks.json"} or size > MAX_JSON_BYTES:
                raise RuntimeError("native observer JSON destination differs")
            json.loads(payload)
            path = call["directory"] / name
            with path.open("xb") as stream_out:
                stream_out.write(payload)
            call["files"].append({"path": str(path.relative_to(self.root)),
                                  "sha256": _sha(path), "bytes": size})
        elif event in {"runner_identity", "runner_workspace", "candidate_status"}:
            expected = {"runner_identity": "runner.json", "runner_workspace": "workspace.json",
                        "candidate_status": "preparation.json"}[event]
            if name != expected or not size or size > MAX_JSON_BYTES:
                raise RuntimeError("native observer identity destination differs")
            if event == "candidate_status":
                json.loads(payload)
            path = call["directory"] / name
            with path.open("xb") as stream_out:
                stream_out.write(payload)
            call["files"].append({"path": str(path.relative_to(self.root)),
                                  "sha256": _sha(path), "bytes": size})
        elif event in {"installed_map_call", "installed_expand_call"}:
            if size or (event == "installed_map_call" and name != "map") or (
                    event == "installed_expand_call" and name != "expand"):
                raise RuntimeError("native observer incumbent event differs")
        with (call["directory"] / "events.jsonl").open("a") as stream_out:
            stream_out.write(json.dumps({"sequence": len(call["events"]), "event": event,
                "name": name, "size": size, "stream": stream,
                "profiler_stream_id": call["profiler_stream_id"],
                "stream_id_api": STREAM_ID_API,
                "cupti_stream_id_provider": self._cupti_identity,
                "thread_id": thread_id, "value": value}, separators=(",", ":")) + "\n")
        call["events"].append(event)

    def finish_call(self, status, released, error=None):
        call = self.call
        if call is None:
            raise RuntimeError("external observer has no active native call")
        receipt = {"schema": SCHEMA, "call_index": call["index"],
            "lane_requested": call["lane"], "thread_id": call["thread_id"],
            "stream": call["stream"], "profiler_stream_id": call["profiler_stream_id"],
            "stream_id_api": STREAM_ID_API, "per_thread_stream": False,
            "cupti_stream_id_provider": self._cupti_identity,
            "native_status": status,
            "lease_released": bool(released), "internal_capture_enabled": False,
            "primary_error": None if error is None else type(error).__name__,
            "callback_error": None if self.callback_error is None else type(self.callback_error).__name__,
            "event_counts": {key: call["events"].count(key) for key in sorted(EVENTS)},
            "files": call["files"], "frame": call["frame"]}
        path = call["directory"] / "observer-receipt.json"
        with path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        self.next_call += 1
        self.call = None
        if self.callback_error is not None and error is None:
            raise RuntimeError("external observer callback failed") from self.callback_error
        return receipt

    def begin_case(self, case, context):
        if self.case is not None or case != "cached":
            raise RuntimeError("external observer accepts only one cached controlled request")
        self.case = case
        self.model_entries = {"routes": [], "stages": [], "logits": [], "kv": []}
        self.model_directory = self.model_root / case
        self.model_context = context

    def write_model_npz(self, category, filename, arrays, metadata):
        if self.case is None or category not in self.model_entries:
            raise RuntimeError("external model serialization has no active case")
        self._ensure_root()
        if not self.model_directory.exists():
            self.model_directory.mkdir(mode=0o700, exist_ok=False)
        if Path(filename).name != filename or not filename.endswith(".npz"):
            raise RuntimeError("external model serialization path is not a simple NPZ name")
        total = 0
        checked = {}
        for name, array in arrays.items():
            value = np.asarray(array)
            if value.dtype.hasobject:
                raise RuntimeError("external model evidence cannot contain object arrays")
            total += value.nbytes
            checked[name] = value
        if total > MAX_MODEL_ARRAY_BYTES:
            raise RuntimeError("external model evidence exceeds its per-file bound")
        path = self.model_directory / filename
        with path.open("xb") as stream:
            np.savez_compressed(stream, **checked)
        record = {**metadata, "file": filename, "sha256": _sha(path),
                  "category": category}
        self.model_entries[category].append(record)
        return record

    def finish_case(self, trace_path):
        if self.case is None or self.call is not None:
            raise RuntimeError("external observer case is incomplete or has a live native lease")
        self._ensure_root()
        if self.next_call != 30:
            raise RuntimeError("external observer requires exactly thirty controlled request M1 calls")
        trace_path = Path(trace_path)
        if not trace_path.is_file() or trace_path.stat().st_size > 256 << 20:
            raise RuntimeError("external observer trace is absent or exceeds its bound")
        self.trace_sha256 = _sha(trace_path)
        payload = {"schema": SCHEMA, "case": self.case,
            "execution_mode": "capture-free", "observer_enabled": True,
            "internal_capture_enabled": False, "observer_perturbs_execution": True,
            "stream_id_mapping_api": STREAM_ID_API,
            "cupti_stream_id_provider": self._cupti_identity,
            "timing_qualified": False, "quality_gate_passed": False,
            "graph_qualified": False, "native_call_count": self.next_call,
            "trace": str(trace_path.relative_to(self.root)),
            "trace_sha256": self.trace_sha256,
            "model_context": self.model_context, "model_arrays": self.model_entries,
            "native_calls": [f"calls/call-{i:04d}/observer-receipt.json" for i in range(self.next_call)]}
        with (self.root / "observer-manifest.json").open("x") as stream:
            json.dump(payload, stream, indent=2)
        self.case = None
        return payload

    def abort_case(self, error):
        if self.case is None:
            return
        self._ensure_root()
        path = self.root / "INVALIDATED.json"
        with path.open("x") as stream:
            json.dump({"schema": SCHEMA, "case": self.case,
                       "reason": "external observer validation was incomplete",
                       "error": str(error), "quality_gate_passed": False,
                       "timing_qualified": False}, stream, indent=2)
        self.case = None


def _native_registration_key(native):
    handle = getattr(native, "_handle", None)
    name = getattr(native, "_name", None)
    bridge = str(Path(os.fsdecode(name)).resolve()) if name is not None else None
    identity = ("handle", int(handle)) if handle is not None else ("object", id(native))
    return os.getpid(), bridge, *identity
