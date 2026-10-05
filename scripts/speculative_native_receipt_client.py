"""Future owned zero-forward execution; CPU/source blockers precede all queries.

Uses the actual pinned EngineCoreClient API, never an HTTP serving frontend.
--owned-child is an internal entry and revalidates authorization independently.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from megartx.speculative_native_plan import (BASE, SITE, LIMITS, PURPOSE, environment,
    read_json, validate_plan, validate_authorization, installed_preflight,
    checkpoint_preflight, receipt_admission)
from megartx.speculative_native_evidence import PrivateEvidence, error_detail, retain_error
from megartx.speculative_native_probe import ProbeError, _process_start
from megartx.speculative_native_preparation import runtime_preflight, create_runtime, validate_entrypoint_discovery


def gpu_processes():
    row = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
    lines = [s.strip() for s in row.stdout.splitlines() if s.strip()]
    if row.returncode or any(not s.isdigit() for s in lines):
        raise ProbeError("GPU owner telemetry unavailable")
    return [int(s) for s in lines]


def headroom():
    row = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
    lines = row.stdout.strip().splitlines()
    if row.returncode or len(lines) != 1 or not lines[0].strip().isdigit():
        raise ProbeError("Single GPU free-memory telemetry unavailable")
    gpu = int(lines[0]) << 20
    available = next((s for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")), None)
    if available is None:
        raise ProbeError("Host reserve telemetry unavailable")
    host = int(available.split()[1]) * 1024
    if gpu < LIMITS["minimum_gpu_free_bytes"] or host < LIMITS["minimum_host_free_bytes"]:
        raise ProbeError("2GiB GPU or 8GiB host reserve failed")
    return {"gpu_free_bytes": gpu, "host_free_bytes": host}


def bounded_log(source, target, fail):
    """Reject excess before write; drain/discard after a sticky guard failure."""
    total = 0
    try:
        with target.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            while True:
                data = source.read(65536)
                if not data:
                    break
                if total + len(data) > LIMITS["private_log_bytes"]:
                    fail("private_log_limit")
                    for _ in iter(lambda: source.read(65536), b""):
                        pass
                    break
                stream.write(data)
                total += len(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
    except BaseException:
        fail("private_log_capture_failed")


def retain_primary(primary, cleanup, phase):
    return retain_error(primary, cleanup, phase)


def write_cleanup_evidence(directory, cleanup, primary):
    """Never promote an unwritten identity graph to verified cleanup evidence."""
    if primary is not None:
        cleanup["primary_failure"] = error_detail(primary, "owned_execution")
    try:
        with PrivateEvidence(directory) as evidence:
            evidence.write("native-owned-cleanup.private.json", cleanup, cap=LIMITS["plan_bytes"])
    except BaseException as error:
        process_cleanup = cleanup.get("cleanup_complete") is True
        cleanup["cleanup_complete"] = False
        cleanup["cleanup_evidence_complete"] = False
        detail = error_detail(error, "owned_cleanup_evidence", "write_cleanup_receipt")
        primary = retain_primary(primary, error, "Owned cleanup evidence failed")
        # This smaller record is diagnostic only. It omits the identity graph and
        # cannot stand in for the failed complete receipt, even after no-jobs audit.
        fallback = {"schema": "megartx-native-cleanup-failure-v1", "cleanup_complete": False,
            "full_cleanup_receipt_written": False, "internal_identity_graph_verified": False,
            "process_cleanup_reported_complete": process_cleanup, "failure": detail,
            "primary_failure": error_detail(primary, "owned_execution"), "retry_allowed": False}
        try:
            with PrivateEvidence(directory) as evidence:
                evidence.write("native-owned-cleanup-failure.private.json", fallback, cap=64 << 10)
        except BaseException as secondary:
            primary = retain_primary(primary, secondary, "Owned cleanup diagnostic evidence failed")
        primary = retain_primary(primary, ProbeError("Internal cleanup receipt unavailable; cleanup uncertainty retained"),
                                 "Owned cleanup uncertainty")
    return primary


def child_environment(plan, private, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = {k: inherited[k] for k in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE") if k in inherited}
    env.update(environment(PROJECT, private, runner_lane=plan.get("runner_lane", "v1-legacy"),
                           runtime=plan.get("runtime_binding")))
    env["PATH"] = "/usr/local/cuda/bin:" + str(BASE / ".venv/bin") + ":/usr/bin:/bin"
    return env


def validate_child_environment(plan, private, observed):
    expected = child_environment(plan, private, observed)
    if any(observed.get(k) != v for k, v in expected.items()):
        raise ProbeError("Owned child environment differs")
    # Independent cache/config overrides must not supplement the fixed map.
    if plan.get("runner_lane") == "v2" and set(observed) != set(expected):
        raise ProbeError("Unbound inherited override in V2 owned child")


def child_entry(args):
    if args.control_fd is None:
        raise ProbeError("Owned supervisor control descriptor required")
    plan = validate_plan(read_json(args.plan, LIMITS["plan_bytes"]), PROJECT)
    packet = read_json(args.private_directory / "native-client-admission.private.json", LIMITS["plan_bytes"])
    validate_authorization(packet.get("authorization"), plan)  # child cannot bypass source blockers
    if packet.get("purpose") != plan["purpose"] or packet.get("plan_sha256") != plan["plan_sha256"]:
        raise ProbeError("Child admission source/purpose differs")
    from megartx.native_diagnostic_composition import diagnostic_mode
    if diagnostic_mode(os.environ) != plan.get("runner_lane", "v1-legacy"):
        raise ProbeError("Owned child shared diagnostic purpose differs")
    installed_preflight(SITE, runner_lane=plan.get("runner_lane", "v1-legacy"))  # repeated stable source read before runtime imports
    runtime = plan.get("runtime_binding")
    if plan.get("runner_lane") == "v2":
        runtime_preflight(runtime, args.private_directory, created=True)
        validate_entrypoint_discovery(runtime["entrypoint"])
    validate_child_environment(plan, args.private_directory, os.environ)
    if plan.get("runner_lane") == "v2":
        from megartx.speculative_native_v2_plan import FORBIDDEN_ENV
        if any(key in os.environ for key in FORBIDDEN_ENV):
            raise ProbeError("Conflicting V1/override environment in V2 owned child")
    with socket.socket(fileno=args.control_fd) as control:
        control.settimeout(10)
        stream = control.makefile("rwb", buffering=0)
        def gate():
            stream.write(b"gate\n")
            data = stream.readline(4097)
            if not data.endswith(b"\n") or len(data) > 4096:
                raise ProbeError("Owned resource gate malformed")
            row = json.loads(data)
            if (row.get("ok") is not True or row.get("plan_sha256") != plan["plan_sha256"]
                    or row.get("supervisor_pid") != os.getppid()
                    or row.get("supervisor_start") != _process_start(os.getppid())):
                raise ProbeError("Owned resource controls unavailable")
        admission, deadline = packet["admission"], packet["deadline_monotonic"]
        gate()
        # Evidence limits are enforced by their writers, not RLIMIT_FSIZE. A
        # process-wide file limit is inherited by JIT compiler descendants and
        # incorrectly applies the receipt's 8 MiB cap to runtime/cache artifacts.
        # V2's source-bound cache/tmp paths remain separate from evidence.
        from megartx.speculative_native_client import ReceiptSession, make_actual_client
        session = None
        async def run_async():
            nonlocal session
            client = make_actual_client(plan)  # inside live loop for AsyncMPClient
            session = ReceiptSession(client, deadline=deadline, resources=gate, runner_lane=plan.get("runner_lane", "v1-legacy"))
            return await session.run_async(admission)
        try:
            if plan["client_mode"] == "async":
                result = asyncio.run(run_async())
            else:
                client = make_actual_client(plan)
                session = ReceiptSession(client, deadline=deadline, resources=gate, runner_lane=plan.get("runner_lane", "v1-legacy"))
                result = session.run_sync(admission)
            with PrivateEvidence(args.private_directory) as evidence:
                evidence.write("native-receipt.scalars.json", result, cap=64 << 10)
                evidence.write("native-client-lifecycle.private.json", {"events": session.events,
                    "release_attempted": session.release_attempted, "shutdown_attempted": session.shutdown_attempted}, cap=64 << 10)
            stream.write(b"complete\n")
            return 0
        except BaseException as error:
            try:
                with PrivateEvidence(args.private_directory) as evidence:
                    evidence.write("native-client-failure.private.json", {"error_type": type(error).__name__,
                        "failure": error_detail(error, "owned_client_startup" if session is None else "owned_client_session"),
                        "events": session.events if session else [], "retry_allowed": False}, cap=64 << 10)
            except BaseException as secondary:
                retain_primary(error, secondary, "Owned client failure evidence failed")
            raise


def supervise(args):
    plan = validate_plan(read_json(args.plan, LIMITS["plan_bytes"]), PROJECT)
    auth = read_json(args.authorization, LIMITS["authorization_bytes"])
    validate_authorization(auth, plan)  # BEFORE queries, runtime imports or acquisition
    if sys.version_info[:3] != (3, 12, 3) or Path(sys.executable).resolve() != Path(plan["python"]).resolve():
        raise ProbeError("Pinned Python/environment required")
    installed = installed_preflight(SITE, runner_lane=plan.get("runner_lane", "v1-legacy"))
    checkpoint = checkpoint_preflight(args.checkpoint_manifest)
    runtime = plan.get("runtime_binding")
    if plan.get("runner_lane") == "v2":
        runtime_preflight(runtime, args.private_directory, installed_root=SITE)
    if gpu_processes():
        raise ProbeError("Existing GPU work blocks sole-owner startup")
    reserves = headroom()
    if args.private_directory.exists():
        raise ProbeError("Owned execution requires a fresh private directory")
    args.private_directory.mkdir(mode=0o700)
    if runtime is not None:
        create_runtime(runtime, args.private_directory)
    from m1_owned_processes import OwnedProcesses, enable_subreaper, read_process, snapshot
    ownership = OwnedProcesses(enable_subreaper(), rss_limit=2 << 30, compiler_seconds=300)
    started = time.monotonic()
    deadline = started + LIMITS["wall_seconds_including_cleanup"]
    admission = receipt_admission(plan, auth, checkpoint, deadline=deadline)
    admission.update(private_receipt_directory=str(args.private_directory.resolve()),
        client_evidence_source_sha256=plan["source_sha256"]["src/megartx/speculative_native_evidence.py"])
    with PrivateEvidence(args.private_directory) as evidence:
        evidence.write("native-client-admission.private.json", {"purpose": plan["purpose"], "plan_sha256": plan["plan_sha256"],
            "authorization": auth, "deadline_monotonic": deadline, "admission": admission}, cap=LIMITS["plan_bytes"])
        evidence.write("native-source-preflight.private.json", {"installed": installed, "checkpoint": checkpoint,
            "headroom": reserves}, cap=LIMITS["plan_bytes"])
    stop, failure, threads = threading.Event(), [], []
    failure_details = []
    lock = threading.RLock()
    child, primary = None, None
    def fail(kind, error=None, stage="owned_guard"):
        with lock:
            if error is not None and len(failure_details) < 8:
                failure_details.append(error_detail(error, stage))
            if not failure:
                failure.append(kind)
                ownership.fail(kind)
    def gate():
        if failure or ownership.failure:
            raise ProbeError("Owned resource failure retained")
        if time.monotonic() >= deadline - LIMITS["cleanup_seconds"]:
            raise TimeoutError("Wall bound reserves cleanup")
        processes = snapshot()
        ownership.observe(processes, time.monotonic())
        headroom()
        for pid in gpu_processes():
            if pid not in processes or processes[pid].identity not in ownership.remembered:
                raise ProbeError("Unrelated GPU owner appeared")
        if ownership.failure:
            raise ProbeError("Shared compiler budget failed")
        if (time.monotonic() >= started + LIMITS["startup_seconds"]
                and not (args.private_directory / "native-receipt.private.json").exists()):
            raise TimeoutError("Owned startup/receipt deadline")
    def watchdog():
        while not stop.is_set():
            try:
                gate()
            except BaseException as error:
                if stop.is_set():
                    return
                fail(type(error).__name__, error, "owned_watchdog")
                try:
                    ownership.stop()
                except BaseException as secondary:
                    fail("owned_stop_failed", secondary, "owned_watchdog_stop")
                return
            stop.wait(.2)
    parent_socket, child_socket = socket.socketpair()
    parent_socket.settimeout(.5)
    def controls():
        buffer = b""
        try:
            while not stop.is_set():
                try:
                    data = parent_socket.recv(128)
                except socket.timeout:
                    continue
                if not data:
                    return
                buffer += data
                if len(buffer) > 128:
                    raise ProbeError("Control message bound")
                while b"\n" in buffer:
                    row, buffer = buffer.split(b"\n", 1)
                    if row == b"complete":
                        return
                    if row != b"gate":
                        raise ProbeError("Unknown owned control message")
                    gate()
                    response = {"ok": True, "plan_sha256": plan["plan_sha256"],
                        "supervisor_pid": os.getpid(), "supervisor_start": _process_start(os.getpid())}
                    parent_socket.sendall(json.dumps(response).encode() + b"\n")
        except BaseException as error:
            if not stop.is_set():
                fail(type(error).__name__, error, "owned_control_channel")
    def interrupted(signum, frame):
        raise InterruptedError("Owned supervisor interrupted")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        argv = [plan["python"], str(Path(__file__).resolve()), "--owned-child", "--plan", str(args.plan.resolve()),
                "--private-directory", str(args.private_directory.resolve()), "--control-fd", str(child_socket.fileno())]
        env = child_environment(plan, args.private_directory)
        child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, pass_fds=(child_socket.fileno(),), start_new_session=True)
        ownership.register(read_process(child.pid))
        child_socket.close()
        threads = [threading.Thread(target=bounded_log, args=(child.stdout, args.private_directory / "runtime.private.log", fail), daemon=True),
                   threading.Thread(target=watchdog, daemon=True), threading.Thread(target=controls, daemon=True)]
        for thread in threads:
            thread.start()
        while child.poll() is None:
            if failure:
                raise ProbeError("Owned guard failure")
            if time.monotonic() >= deadline - LIMITS["cleanup_seconds"]:
                raise TimeoutError("Owned wall deadline")
            stop.wait(.1)
        if child.returncode:
            raise ProbeError("Owned child failed; private logs retained")
        gate()
    except BaseException as error:
        primary = error
    finally:
        stop.set()
        finalization_errors = []
        def finalization_failure(error, stage):
            nonlocal primary
            finalization_errors.append(error_detail(error, stage))
            primary = retain_primary(primary, error, stage)
        for sock in (parent_socket, child_socket):
            try:
                sock.close()
            except BaseException as error:
                finalization_failure(error, "Owned control socket close failed")
        try:
            cleanup = ownership.cleanup(gpu_processes, (child.poll if child else lambda: None), term_seconds=10, kill_seconds=10)
        except BaseException as error:
            cleanup = {"cleanup_complete": False, "cleanup_error_type": type(error).__name__,
                       "failure_detail": error_detail(error, "owned_process_cleanup")}
            primary = retain_primary(primary, error, "Owned cleanup failed")
        for thread in threads:
            try:
                thread.join(timeout=2)
                if thread.is_alive():
                    fail("owned_monitor_or_log_thread_uncertain")
            except BaseException as error:
                finalization_failure(error, "Owned monitor join failed")
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except BaseException as error:
                finalization_failure(error, "Owned signal handler restore failed")
        if finalization_errors:
            cleanup["cleanup_complete"] = False
            cleanup["finalization_errors"] = finalization_errors
        if failure_details:
            cleanup["guard_failure_details"] = failure_details
        if time.monotonic() > deadline:
            cleanup["cleanup_complete"] = False
            cleanup["wall_bound_exceeded"] = True
        primary = write_cleanup_evidence(args.private_directory, cleanup, primary)
        if not cleanup["cleanup_complete"] or ownership.failure or failure:
            if primary is None:
                primary = ProbeError("Owned cleanup/resource uncertainty retained")
            else:
                primary = retain_primary(primary, ProbeError("Owned cleanup/resource uncertainty retained"),
                                         "Owned cleanup/resource uncertainty retained")
    if primary:
        raise primary
    from megartx.speculative_native_compare import compare_files
    result = compare_files(args.private_directory, plan)
    with PrivateEvidence(args.private_directory) as evidence:
        evidence.write("native-receipt-comparison.scalars.json", result, cap=64 << 10)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--owned-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--private-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, default=BASE / "results/checkpoint-manifest.json")
    parser.add_argument("--control-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute and (args.authorization is None or args.control_fd is not None):
        parser.error("--execute requires exact-source authorization and owns its control descriptor")
    return child_entry(args) if args.owned_child else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
