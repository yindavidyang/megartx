"""Fail-closed process ownership evidence for the external M1 observer."""
import json
import multiprocessing
import os
from pathlib import Path
import stat
import time


PROCESS_EVIDENCE = "process-events.jsonl"
MAX_PROCESS_EVIDENCE_BYTES = 1 << 20
MAX_PROCESS_EVIDENCE_ROWS = 1024
SPAWN_START_RECORD_WAIT_SECONDS = 10.0
SPAWN_START_RECORD_POLL_SECONDS = 0.05


def process_identity():
    current = multiprocessing.current_process()
    return {"pid": os.getpid(), "ppid": os.getppid(),
            "process_name": current.name}


def append_process_event(evidence_dir, event, *, probe_sha256=None, **fields):
    """Append one bounded process identity row to the task-local evidence file."""
    directory = Path(evidence_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("M1 process evidence destination must be a regular directory")
    identity = process_identity()
    row = {"event": event, **identity,
           "probe_sha256": (probe_sha256 if probe_sha256 is not None else
                            os.environ.get("MEGARTX_M1_PROCESS_PROBE_SHA256")),
           **fields}
    encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = directory / PROCESS_EVIDENCE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def _read_records(path, *, allow_partial_final=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("M1 process evidence must be a regular file")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROCESS_EVIDENCE_BYTES:
        raise RuntimeError("M1 process evidence exceeds its file bound")
    records = []
    raw = path.read_bytes()
    if len(raw) > MAX_PROCESS_EVIDENCE_BYTES:
        raise RuntimeError("M1 process evidence exceeds its file bound")
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if allow_partial_final and index == len(lines) - 1:
                break
            raise RuntimeError("M1 process evidence contains an incomplete record")
        if len(records) >= MAX_PROCESS_EVIDENCE_ROWS:
            raise RuntimeError("M1 process evidence exceeds its record bound")
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("M1 process evidence contains a non-object record")
        records.append(value)
    return records


def validate_engine_core_spawn_owner(evidence_dir, registration, api_pid_path,
                                     expected_probe_sha256):
    """Reject non-spawn or unowned EngineCore setup before a request can run."""
    if registration.get("process_name") != "EngineCore":
        raise RuntimeError("external M1 observer must register inside EngineCore")
    try:
        api_pid = int(Path(api_pid_path).read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("owned API server PID record is missing or invalid") from error
    if api_pid <= 0 or registration.get("ppid") != api_pid:
        raise RuntimeError("observer owner is not a child of the owned API server")

    evidence_path = Path(evidence_dir) / PROCESS_EVIDENCE
    deadline = time.monotonic() + SPAWN_START_RECORD_WAIT_SECONDS
    while True:
        records = _read_records(evidence_path, allow_partial_final=True)
        if not records:
            raise RuntimeError("M1 process probe emitted no runtime evidence")
        if any(record.get("probe_sha256") != expected_probe_sha256 for record in records):
            raise RuntimeError("M1 process evidence is not bound to the requested probe source")
        if any(record.get("event") == "process_probe_failure" for record in records):
            raise RuntimeError("task-local M1 process probe reported a startup failure")
        if any(record.get("event") == "multiprocessing_context_created"
               and record.get("actual_start_method") != "spawn" for record in records):
            raise RuntimeError("EngineCore actual multiprocessing context is not spawn")

        ready = [record for record in records if record.get("event") == "process_probe_ready"]
        api_ready = [record for record in ready if record.get("pid") == api_pid]
        if len(api_ready) != 1:
            raise RuntimeError("task-local process probe did not initialize in the owned API process")
        spawned = [record for record in records
                   if record.get("event") == "engine_core_process_started"]
        if len(spawned) > 1:
            raise RuntimeError("expected exactly one observed EngineCore process creation before dispatch")
        if spawned:
            child = spawned[0]
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("EngineCore spawn evidence did not arrive within the bounded startup window")
        time.sleep(min(SPAWN_START_RECORD_POLL_SECONDS, remaining))
    contexts = [record for record in records
                if record.get("event") == "multiprocessing_context_created"
                and record.get("context_id") == child.get("context_id")]
    if len(contexts) != 1:
        raise RuntimeError("EngineCore process has no unique actual multiprocessing context")
    context = contexts[0]
    engine_ready = [record for record in ready
                    if record.get("pid") == child.get("child_pid")
                    and record.get("ppid") == api_pid]
    if len(engine_ready) != 1:
        raise RuntimeError("task-local process probe did not initialize in the observed EngineCore child")
    if (context.get("actual_start_method") != "spawn"
            or context.get("requested_start_method") != "spawn"
            or child.get("actual_start_method") != "spawn"
            or context.get("pid") != api_pid
            or context.get("ppid") != api_ready[0].get("ppid")
            or child.get("parent_pid") != api_pid
            or child.get("parent_ppid") != context.get("ppid")
            or child.get("context_pid") != api_pid
            or child.get("child_pid") != registration.get("pid")
            or child.get("child_name") != "EngineCore"):
        raise RuntimeError("observer owner is not the EngineCore child of the actual spawn context")
    return {
        "actual_start_method": context["actual_start_method"],
        "api_server_pid": api_pid,
        "api_server_ppid": context.get("ppid"),
        "engine_core_pid": child["child_pid"],
        "engine_core_ppid": registration["ppid"],
        "engine_core_process_name": child["child_name"],
        "process_probe_sha256": expected_probe_sha256,
    }


def validate_engine_core_registration_before_dispatch(evidence_dir, api_pid_path,
                                                       expected_probe_sha256,
                                                       expected_bridge_identity):
    """Validate the registered EngineCore once its startup evidence is present."""
    records = _read_records(Path(evidence_dir) / PROCESS_EVIDENCE)
    if any(record.get("event") == "process_probe_failure" for record in records):
        raise RuntimeError("task-local M1 process probe reported a startup failure")
    if any(record.get("event") == "multiprocessing_context_created"
           and record.get("actual_start_method") != "spawn" for record in records):
        raise RuntimeError("EngineCore actual multiprocessing context is not spawn")
    registered = [record for record in records
                  if record.get("event") == "observer_registered"]
    if not registered:
        return None
    if len(registered) != 1:
        raise RuntimeError("observer callback has duplicate registration owners before dispatch")
    registration = registered[0]
    if registration.get("bridge_identity") != expected_bridge_identity:
        raise RuntimeError("observer registration is not bound to the requested bridge")
    lifecycle = validate_engine_core_spawn_owner(
        evidence_dir, registration, api_pid_path, expected_probe_sha256)
    if (registration.get("process_name") != "EngineCore"
            or registration.get("probe_sha256") != expected_probe_sha256):
        raise RuntimeError("observer registration evidence is not owned by the verified EngineCore")
    lifecycle.update({
        "observer_registration_pid": registration["pid"],
        "observer_registration_owner_pid": registration["pid"],
        "observer_registered_by_api_server": False,
        "bridge_identity": expected_bridge_identity,
    })
    return lifecycle


def validate_engine_core_spawn(evidence_dir, registration, api_pid_path,
                               expected_probe_sha256, expected_bridge_identity):
    """Bind the observer registration to the exact EngineCore spawned by vLLM."""
    validate_engine_core_spawn_owner(evidence_dir, registration, api_pid_path,
                                     expected_probe_sha256)
    if registration.get("process_name") != "EngineCore":
        raise RuntimeError("external M1 observer must register inside EngineCore")
    try:
        api_pid = int(Path(api_pid_path).read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("owned API server PID record is missing or invalid") from error
    if api_pid <= 0:
        raise RuntimeError("owned API server PID must be positive")

    records = _read_records(Path(evidence_dir) / PROCESS_EVIDENCE)
    if not records:
        raise RuntimeError("M1 process probe emitted no runtime evidence")
    for record in records:
        if record.get("probe_sha256") != expected_probe_sha256:
            raise RuntimeError("M1 process evidence is not bound to the requested probe source")
    ready = [record for record in records if record.get("event") == "process_probe_ready"]
    api_ready = [record for record in ready if record.get("pid") == api_pid]
    if len(api_ready) != 1:
        raise RuntimeError("task-local process probe did not initialize in the owned API process")

    spawned = [record for record in records
               if record.get("event") == "engine_core_process_started"]
    if len(spawned) != 1:
        raise RuntimeError("expected exactly one observed EngineCore process creation")
    child = spawned[0]
    engine_ready = [record for record in ready
                    if record.get("pid") == child.get("child_pid")
                    and record.get("ppid") == api_pid]
    if len(engine_ready) != 1:
        raise RuntimeError("task-local process probe did not initialize in the observed EngineCore child")
    contexts = [record for record in records
                if record.get("event") == "multiprocessing_context_created"
                and record.get("context_id") == child.get("context_id")]
    if len(contexts) != 1:
        raise RuntimeError("EngineCore process has no unique actual multiprocessing context")
    context = contexts[0]
    if (context.get("actual_start_method") != "spawn"
            or child.get("actual_start_method") != context.get("actual_start_method")
            or context.get("requested_start_method") != "spawn"):
        raise RuntimeError("EngineCore was not created by the requested actual spawn context")
    if (context.get("pid") != api_pid
            or context.get("ppid") != api_ready[0].get("ppid")
            or child.get("parent_pid") != api_pid
            or child.get("parent_ppid") != context.get("ppid")
            or child.get("context_pid") != api_pid
            or child.get("child_pid") != registration.get("pid")
            or child.get("child_name") != "EngineCore"
            or registration.get("ppid") != api_pid):
        raise RuntimeError("observer owner is not the EngineCore child of the owned API server")

    registered = [record for record in records
                  if record.get("event") == "observer_registered"]
    unregistered = [record for record in records
                    if record.get("event") == "observer_unregistered"]
    if (len(registered) != 1 or len(unregistered) != 1
            or registered[0].get("pid") != registration.get("pid")
            or unregistered[0].get("pid") != registration.get("pid")
            or registered[0].get("ppid") != registration.get("ppid")
            or unregistered[0].get("ppid") != registration.get("ppid")
            or registered[0].get("process_name") != "EngineCore"
            or unregistered[0].get("process_name") != "EngineCore"
            or registered[0].get("bridge_identity") != expected_bridge_identity
            or unregistered[0].get("bridge_identity") != expected_bridge_identity):
        raise RuntimeError("observer registration or callback cleanup is incomplete or duplicated")
    if any(record.get("event") == "observer_registered"
           and record.get("pid") != registration.get("pid") for record in records):
        raise RuntimeError("observer state was registered outside its EngineCore owner")
    return {
        "actual_start_method": context["actual_start_method"],
        "api_server_pid": api_pid,
        "api_server_ppid": context.get("ppid"),
        "engine_core_pid": child["child_pid"],
        "engine_core_ppid": registration["ppid"],
        "engine_core_process_name": child["child_name"],
        "observer_registration_pid": registered[0]["pid"],
        "observer_registration_owner_pid": registration["pid"],
        "observer_registered_by_api_server": False,
        "observer_callback_unregistered": True,
        "process_probe_sha256": expected_probe_sha256,
        "bridge_identity": expected_bridge_identity,
    }
