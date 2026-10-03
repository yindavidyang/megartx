"""Task-local vLLM multiprocessing probe, enabled only for observer runs."""
import functools
import hashlib
import json
import multiprocessing
import multiprocessing.context
import os
from pathlib import Path
import threading


_lock = threading.Lock()
_next_context = 0
_probe_sha256 = None
_evidence_path = None


def _append_event(event, **fields):
    if _evidence_path is None:
        raise RuntimeError("M1 process probe has no evidence destination")
    current = multiprocessing.current_process()
    row = {"event": event, "pid": os.getpid(), "ppid": os.getppid(),
           "process_name": current.name, "probe_sha256": _probe_sha256, **fields}
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(_evidence_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


class _RecordingSpawnProcess(multiprocessing.context.SpawnProcess):
    def __init__(self, *args, _m1_context_id, _m1_context_method, **kwargs):
        super().__init__(*args, **kwargs)
        self._m1_context_id = _m1_context_id
        self._m1_context_method = _m1_context_method

    def start(self):
        if self._m1_context_method != "spawn" or self._start_method != "spawn":
            raise RuntimeError("M1 observer process must use the actual spawn context")
        parent_pid, parent_ppid = os.getpid(), os.getppid()
        parent_name = multiprocessing.current_process().name
        child_name = self.name
        super().start()
        event = "engine_core_process_started" if child_name == "EngineCore" else "child_process_started"
        _append_event(event, context_id=self._m1_context_id,
                      actual_start_method=self._m1_context_method,
                      context_pid=parent_pid, parent_pid=parent_pid,
                      parent_ppid=parent_ppid, parent_process_name=parent_name,
                      child_pid=self.pid, child_name=child_name)


def _install_probe():
    global _probe_sha256, _evidence_path
    evidence_dir = os.environ.get("MEGARTX_M1_PROCESS_EVIDENCE_DIR")
    if not evidence_dir:
        return
    expected_method = os.environ.get("MEGARTX_M1_PROCESS_EXPECTED_METHOD")
    requested_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    expected_sha256 = os.environ.get("MEGARTX_M1_PROCESS_PROBE_SHA256")
    source = Path(__file__).resolve()
    _probe_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if (expected_method != "spawn" or requested_method != "spawn"
            or expected_sha256 != _probe_sha256):
        raise RuntimeError("M1 process probe source or explicit spawn configuration differs")
    directory = Path(evidence_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("M1 process evidence destination must be a prepared directory")
    _evidence_path = directory / "process-events.jsonl"

    original_get_context = multiprocessing.get_context

    def recording_get_context(method=None):
        global _next_context
        context = original_get_context(method)
        actual_method = context.get_start_method()
        if actual_method != "spawn":
            _append_event("multiprocessing_context_created",
                          requested_start_method=method,
                          actual_start_method=actual_method)
            return context
        with _lock:
            _next_context += 1
            context_id = f"{os.getpid()}-{_next_context}"
        _append_event("multiprocessing_context_created", context_id=context_id,
                      requested_start_method=method,
                      actual_start_method=actual_method)
        context.Process = functools.partial(_RecordingSpawnProcess,
                                            _m1_context_id=context_id,
                                            _m1_context_method=actual_method)
        return context

    multiprocessing.get_context = recording_get_context
    _append_event("process_probe_ready", expected_start_method=expected_method,
                  requested_start_method=requested_method,
                  actual_start_method="deferred-to-vllm-context")
    os.environ["MEGARTX_M1_PROCESS_PROBE_ACTIVE"] = "1"


try:
    _install_probe()
except BaseException as _error:
    try:
        _append_event("process_probe_failure", error_type=type(_error).__name__)
    except BaseException:
        pass
    os.environ.pop("MEGARTX_M1_PROCESS_PROBE_ACTIVE", None)
    raise
