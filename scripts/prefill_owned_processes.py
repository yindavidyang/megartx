"""Linux task-local subreaper ownership and sampled shared compiler bounds.

Ownership is PID/start-time lineage, never executable names or process groups.
pidfds close the identity-check/signal PID-reuse race. No GPU imports.
"""
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import os
from pathlib import Path
import signal
import threading
import time

COMPILERS = {"ninja", "nvcc", "cicc", "ptxas", "tileiras", "gcc", "g++", "cc", "c++",
             "cc1", "cc1plus", "clang", "clang++", "ld", "ld.gold", "ld.lld", "collect2"}


def is_compiler(name):
    return name in COMPILERS or name.endswith(("-gcc", "-g++", "-clang", "-clang++"))


def file_version(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def hash_file_version(path):
    """Untimed hash of a stable opened file, including path replacement checks."""
    with Path(path).open("rb") as stream:
        before = file_version(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1 << 20), b""):
            digest.update(data)
        if file_version(os.fstat(stream.fileno())) != before or file_version(Path(path).stat()) != before:
            raise RuntimeError("Metadata timing file changed during hashing")
    return digest.hexdigest(), before


class MetadataHelpTiming:
    """One reviewed command; all tool resource accounting stays independent."""
    EXECUTABLE = "/usr/local/cuda/bin/tileiras"
    RESOLVED = "/usr/local/cuda-13.3/bin/tileiras"
    BINARY_SHA = "88737a8be5c56bf73fb885a567a950f247a8cfa1d146dbc7a65eff77e7d62bf0"
    CALLER_SHA = "8b80053b84ad68fee19cc66f2b9a8f3e55df2c10be77a71e892da6aa22e35bae"
    PROPOSAL_SHA = "fecc1f67f9bbac374235ffbf82e0ce83d362ce3269237bfa2a8761e5ef027d1a"

    def __init__(self, flashinfer_root, source_head):
        self.caller = Path(flashinfer_root) / "cutile/cutile_common.py"
        self.source_head = source_head
        self.argv = (self.EXECUTABLE, "--help")
        if str(Path(self.EXECUTABLE).resolve()) != self.RESOLVED:
            raise RuntimeError("Metadata timing executable path differs")
        binary_sha, self.binary_version = hash_file_version(self.EXECUTABLE)
        caller_sha, self.caller_version = hash_file_version(self.caller)
        if (binary_sha, caller_sha) != (self.BINARY_SHA, self.CALLER_SHA):
            raise RuntimeError("Metadata timing executable/caller hash differs")
        if error := self.version_error():
            raise RuntimeError(error)
        self.final = None

    def version_error(self):
        try:
            if (str(Path(self.EXECUTABLE).resolve()) != self.RESOLVED
                    or file_version(Path(self.EXECUTABLE).stat()) != self.binary_version
                    or file_version(self.caller.stat()) != self.caller_version):
                return "Metadata timing executable/caller file-version drift"
        except OSError as error:
            return "Metadata timing file evidence unavailable: " + str(error)
        return None

    def qualifies(self, process):
        return (process.pid > 0 and process.compiler_identity_verified
                and process.executable == "tileiras" and process.state != "Z"
                and process.compiler_executable == self.RESOLVED
                and process.compiler_file_version == self.binary_version
                and process.compiler_argv == self.argv)

    def terminal_evidence_matches(self, process):
        # Exit can remove /proc evidence, but cannot erase newly captured work
        # or conflicting evidence. read_process also rejects unstable rechecks.
        return (process.executable == "tileiras"
                and all(observed is None or observed == expected for observed, expected in (
                    (process.compiler_argv, self.argv),
                    (process.compiler_executable, self.RESOLVED),
                    (process.compiler_file_version, self.binary_version))))

    def report(self):
        return {"policy": "pinned_tileiras_help_v1", "proposal_sha256": self.PROPOSAL_SHA,
                "source_head": self.source_head, "argv": self.argv, "resolved_executable": self.RESOLVED,
                "binary_sha256": self.BINARY_SHA, "caller_sha256": self.CALLER_SHA,
                "binary_file_version": self.binary_version, "caller_file_version": self.caller_version,
                "file_version_fields": ["dev", "ino", "size", "mtime_ns", "ctime_ns"], "final": self.final}

    def finalize(self):
        errors = [error] if (error := self.version_error()) else []
        hashes = {}
        for name, path, expected, version in (("binary", self.EXECUTABLE, self.BINARY_SHA, self.binary_version),
                                              ("caller", self.caller, self.CALLER_SHA, self.caller_version)):
            try:
                digest, current = hash_file_version(path)
                hashes[name] = digest
                if digest != expected or current != version:
                    errors.append("Metadata timing final " + name + " hash/version differs")
            except (OSError, RuntimeError) as error:
                errors.append("Metadata timing final " + name + " verification failed: " + str(error))
        self.final = {"passed": not errors, "hashes": hashes, "errors": errors}
        return self.final


@dataclass(frozen=True)
class Process:
    pid: int
    start_ticks: int
    ppid: int
    rss_bytes: int = 0
    executable: str = ""
    state: str = "S"
    age_seconds: float = 0
    compiler_argv: tuple | None = None
    compiler_executable: str | None = None
    compiler_file_version: tuple | None = None
    compiler_identity_verified: bool = False

    @property
    def identity(self):
        return self.pid, self.start_ticks

    @property
    def compiler_invocation(self):
        # Evidence classification only: metadata probes are NOT exempt from
        # resource accounting or the stricter timing-quiescence admission.
        if self.executable == "tileiras" and self.compiler_argv is not None:
            if self.compiler_argv[1:] == ("--help",):
                return "tileiras_help_probe"
            if self.compiler_argv[1:] == ("--version",):
                return "tileiras_version_probe"
        return "compiler_work_or_unknown"


def decode_cmdline(data):
    # /proc uses one terminating NUL per argument, including empty arguments.
    # Remove exactly the final terminator; malformed nonempty data is unknown.
    if not data or not data.endswith(b"\0"):
        return None
    return tuple(os.fsdecode(arg) for arg in data[:-1].split(b"\0"))


def read_process(pid, uptime=None):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text()
    fields = stat[stat.rindex(")") + 2:].split()
    executable = stat[stat.index("(") + 1:stat.rindex(")")]
    executable_path = None
    conflicting_evidence = False
    try:
        executable_path = os.readlink(directory / "exe")
        executable = Path(executable_path).name
    except PermissionError:
        conflicting_evidence = True
    except (FileNotFoundError, ProcessLookupError):
        pass
    argv = None
    version = None
    verified = False
    if is_compiler(executable):
        try:
            version = file_version((directory / "exe").stat())
        except PermissionError:
            conflicting_evidence = True
        except (FileNotFoundError, ProcessLookupError):
            pass
        try:
            data = (directory / "cmdline").read_bytes()
            argv = decode_cmdline(data)
            conflicting_evidence |= bool(data) and argv is None
        except PermissionError:
            conflicting_evidence = True
        except (FileNotFoundError, ProcessLookupError):
            pass  # Unknown argv stays conservative; no metadata-only exemption.
        stable_executable = False
        if version is not None and argv is not None:
            later_argv = later_path = later_version = None
            try:
                later_data = (directory / "cmdline").read_bytes()
                later_argv = decode_cmdline(later_data)
                conflicting_evidence |= bool(later_data) and later_argv != argv
            except PermissionError:
                conflicting_evidence = True
            except (FileNotFoundError, ProcessLookupError):
                pass
            try:
                later_path = os.readlink(directory / "exe")
                conflicting_evidence |= later_path != executable_path
            except PermissionError:
                conflicting_evidence = True
            except (FileNotFoundError, ProcessLookupError):
                pass
            try:
                later_version = file_version((directory / "exe").stat())
                conflicting_evidence |= later_version != version
            except PermissionError:
                conflicting_evidence = True
            except (FileNotFoundError, ProcessLookupError):
                pass
            stable_executable = (not conflicting_evidence and executable_path is not None
                and later_path == executable_path and later_version == version and later_argv == argv)
        try:
            later = (directory / "stat").read_text()
            later_fields = later[later.rindex(")") + 2:].split()
            same_identity = (int(stat.split("(", 1)[0]) == int(pid)
                             and int(later.split("(", 1)[0]) == int(pid)
                             and int(later_fields[19]) == int(fields[19]))
            if same_identity and later_fields[0] == "Z":
                fields[0] = "Z"
                verified = not conflicting_evidence  # Terminal identity/evidence only; never qualifies itself.
            elif same_identity:
                verified = stable_executable
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    start = int(fields[19])
    if uptime is None:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    return Process(int(pid), start, int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                   executable, fields[0], max(0, uptime - start / os.sysconf("SC_CLK_TCK")), argv,
                   executable_path, version, verified)


def snapshot():
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    result = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            try:
                process = read_process(int(directory.name), uptime)
                result[process.pid] = process
            except (FileNotFoundError, ProcessLookupError):
                pass
    return result


def enable_subreaper():
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("eager containment requires Linux pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "task-local subreaper admission failed")
    value = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
        raise RuntimeError("task-local subreaper verification failed")
    return read_process(os.getpid())


def _signal_operation(operation, function, *args):
    try:
        return function(*args)
    except OSError as error:
        error.megartx_operation = operation
        raise


def signal_identity(process, signum):
    """Signal exactly the retained identity; reused PID is never signalled."""
    try:
        fd = _signal_operation("pidfd_open", os.pidfd_open, process.pid)
    except ProcessLookupError:
        return False
    except OSError as error:
        if error.errno != errno.EINVAL or process.pid <= 0:
            raise
        # Linux 6.8 can lose the TGID task between find_get_pid() and
        # pidfd_prepare(), returning EINVAL instead of ESRCH during reaping.
        # Only fresh evidence that this identity is gone permits a no-op.
        try:
            current = _signal_operation("pidfd_error_identity_read", read_process, process.pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        if current.identity != process.identity:
            return False
        raise
    try:
        if _signal_operation("pidfd_identity_read", read_process, process.pid).identity != process.identity:
            return False
        _signal_operation("pidfd_send_signal", signal.pidfd_send_signal, fd, signum)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        _signal_operation("pidfd_close", os.close, fd)


class OwnedProcesses:
    def __init__(self, supervisor, *, rss_limit=2 << 30, compiler_seconds=300, metadata_timing=None):
        self.supervisor = supervisor
        self.remembered = {}
        self.compiler_identities = set()
        self.compiler_samples = {}
        self.metadata_timing = metadata_timing
        self.metadata_timing_invalid = False
        self.metadata_identities = set()
        self.non_metadata_identities = set()
        self.timing_history = {}
        self.root_identity = None
        self.rss_limit = rss_limit
        self.compiler_seconds = compiler_seconds
        self.compiler_started = None
        self.peak_compiler_rss = 0
        self.compiler_elapsed = 0
        self.failure = None
        self.failure_sample = None
        self.lock = threading.RLock()

    def register(self, root):
        with self.lock:
            self.root_identity = root.identity
            self.remembered[root.identity] = root

    def observe(self, processes, now):
        with self.lock:
            if self.metadata_timing is not None:
                if error := self.metadata_timing.version_error():
                    self.fail(error)
                    self.metadata_timing_invalid = True
            parents = {p.pid: p for p in processes.values()
                       if p.identity in self.remembered or p.identity == self.supervisor.identity}
            changed = True
            while changed:
                changed = False
                for process in processes.values():
                    parent = parents.get(process.ppid)
                    if (process.identity != self.supervisor.identity and parent is not None
                            and process.start_ticks >= parent.start_ticks
                            and process.identity not in self.remembered):
                        self.remembered[process.identity] = process
                        parents[process.pid] = process
                        changed = True
            alive = [p for p in processes.values() if p.identity in self.remembered]
            for process in alive:
                self.remembered[process.identity] = process
                name = process.executable
                if is_compiler(name) or process.identity in self.compiler_identities:
                    self.compiler_identities.add(process.identity)
                    if process.identity not in self.compiler_samples or process.compiler_argv is not None:
                        self.compiler_samples[process.identity] = process
                    self.classify_timing(process, now)
            changed = True
            while changed:
                changed = False
                parents = {p.pid for p in alive if p.identity in self.compiler_identities}
                for process in alive:
                    if process.ppid in parents and process.identity not in self.compiler_identities:
                        self.compiler_identities.add(process.identity)
                        self.classify_timing(process, now, descendant=True)
                        changed = True
                    elif process.ppid in parents:
                        self.classify_timing(process, now, descendant=True)
            compiling = [p for p in alive if p.identity in self.compiler_identities and p.state != "Z"]
            rss = sum(p.rss_bytes for p in compiling)
            self.peak_compiler_rss = max(self.peak_compiler_rss, rss)
            if compiling:
                started = min(now - p.age_seconds for p in compiling)
                self.compiler_started = started if self.compiler_started is None else min(self.compiler_started, started)
                self.compiler_elapsed = max(self.compiler_elapsed, now - self.compiler_started)
                reason = ("Owned compiler aggregate RSS exceeded 2 GiB" if rss > self.rss_limit else
                          "Owned shared compiler budget exceeded 300 seconds" if self.compiler_elapsed > self.compiler_seconds else None)
                if reason:
                    self.fail(reason, {"compiler_rss_bytes": rss, "compiler_elapsed_seconds": self.compiler_elapsed,
                                       "compiler_identities": [asdict(p) for p in compiling]})
            return alive

    def classify_timing(self, process, now, descendant=False):
        history = self.timing_history.setdefault(process.identity, {"first_sample": now, "last_sample": now,
                    "metadata_samples": 0, "first_unknown_or_work_sample": None})
        history["last_sample"] = now
        qualifies = (not descendant and not self.metadata_timing_invalid and self.metadata_timing is not None
                     and self.metadata_timing.qualifies(process))
        terminal = (not descendant and process.state == "Z" and process.compiler_identity_verified
                    and not self.metadata_timing_invalid and self.metadata_timing is not None
                    and process.identity in self.metadata_identities
                    and self.metadata_timing.terminal_evidence_matches(process))
        if qualifies:
            self.metadata_identities.add(process.identity)
            history["metadata_samples"] += 1
        elif not terminal:
            self.non_metadata_identities.add(process.identity)
            if history["first_unknown_or_work_sample"] is None:
                history["first_unknown_or_work_sample"] = {"time": now, "descendant": descendant, **asdict(process)}

    def fail(self, reason, sample=None):
        with self.lock:
            if self.failure is None:
                self.failure, self.failure_sample = reason, sample

    def current_owned(self, processes):
        return [p for p in processes.values() if p.identity in self.remembered]

    def stop(self, signum=signal.SIGTERM):
        alive = self.observe(snapshot(), time.monotonic())
        for process in alive:
            signal_identity(process, signum)

    def report(self):
        with self.lock:
            return {"policy": "subreaper_pid_start_time_pidfd", "compiler_rss_limit_bytes": self.rss_limit,
                    "shared_compiler_seconds_limit": self.compiler_seconds,
                    "sampled_peak_compiler_rss_bytes": self.peak_compiler_rss,
                    "shared_compiler_elapsed_seconds": self.compiler_elapsed,
                    "failure": self.failure, "failure_sample": self.failure_sample,
                    "sampled_compiler_identities": [asdict(self.remembered[i]) for i in sorted(self.compiler_identities)],
                    "sampled_compiler_invocations": [{**asdict(p), "classification": p.compiler_invocation}
                                                     for p in self.compiler_samples.values()],
                    "timing_metadata_policy": self.metadata_timing.report() if self.metadata_timing is not None else None,
                    "timing_metadata_policy_invalid": self.metadata_timing_invalid,
                    "timing_metadata_only_identities": sorted(self.metadata_identities - self.non_metadata_identities),
                    "timing_unknown_or_work_identities": sorted(self.non_metadata_identities),
                    "timing_classification_history": [{"identity": i, **h} for i,h in self.timing_history.items()],
                    "remembered_identities": [asdict(p) for p in self.remembered.values()]}

    def require_compiler_quiescence(self):
        """Enforce the reviewed, stricter all-server-lifetime timing gate."""
        with self.lock:
            if self.metadata_timing is not None:
                if self.metadata_timing_invalid:
                    raise RuntimeError(self.failure or "Metadata timing retained file evidence failure")
                if error := self.metadata_timing.version_error():
                    self.fail(error)
                    raise RuntimeError(error)
                if self.metadata_timing.final is None or not self.metadata_timing.final["passed"]:
                    raise RuntimeError("Metadata timing requires successful final file verification")
                if self.compiler_identities - (self.metadata_identities - self.non_metadata_identities):
                    raise RuntimeError("Owned compiler activity invalidates eager timing")
            elif self.compiler_identities or self.peak_compiler_rss or self.compiler_elapsed:
                raise RuntimeError("Owned compiler activity invalidates eager timing")

    def finalize_metadata(self):
        with self.lock:
            if self.metadata_timing is not None:
                report = self.metadata_timing.finalize()
                if not report["passed"]:
                    self.fail(report["errors"][0])
                return self.metadata_timing.report()

    def require_diagnostic_integrity(self):
        """Untimed diagnostics retain resource and pinned-file failures.

        Unknown/work identities and their sticky histories remain accounted;
        this method neither changes classification nor admits clean timing.
        """
        with self.lock:
            if self.failure or self.metadata_timing_invalid:
                raise RuntimeError(self.failure or "Metadata timing retained file evidence failure")
            if self.metadata_timing is None:
                raise RuntimeError("Decode diagnostic requires pinned metadata file verification")
            if error := self.metadata_timing.version_error():
                self.fail(error)
                raise RuntimeError(error)
            if self.metadata_timing.final is None or not self.metadata_timing.final["passed"]:
                raise RuntimeError("Metadata timing requires successful final file verification")

    def cleanup(self, gpu_query, root_poll, *, term_seconds=10, kill_seconds=10):
        errors = []
        details = []
        operation, identity = None, None
        def record_error(error):
            errors.append(str(error))
            details.append({"operation": getattr(error, "megartx_operation", operation),
                            "cleanup_stage": operation, "identity": identity,
                            "error_type": type(error).__name__, "errno": getattr(error, "errno", None),
                            "error": str(error)})
        alive = list(self.remembered.values())
        for signum, seconds in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
            deadline = time.monotonic() + seconds
            while True:
                try:
                    operation, identity = "observe_snapshot", None
                    alive = self.observe(snapshot(), time.monotonic())
                    operation = "root_poll"
                    root_poll()
                    for process in alive:
                        identity = process.identity
                        if process.state == "Z" and process.identity != self.root_identity:
                            try:
                                operation = "waitpid"
                                os.waitpid(process.pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                        else:
                            operation = "signal_identity"
                            signal_identity(process, signum)
                    operation, identity = "remaining_snapshot", None
                    alive = self.current_owned(snapshot())
                except Exception as error:
                    record_error(error)
                    break
                if not alive or time.monotonic() >= deadline:
                    break
                time.sleep(.05)
            if not alive:
                break
        try:
            operation, identity = "gpu_query", None
            queried_gpu_pids = gpu_query()
            operation = "final_snapshot"
            processes = snapshot()
            operation = "final_observe"
            remaining = self.observe(processes, time.monotonic())
            # Query PID then check current start time: a reused unrelated GPU PID
            # is not considered owned and is never selected for termination.
            gpu_pids = [pid for pid in queried_gpu_pids if pid in processes and processes[pid].identity in self.remembered]
        except Exception as error:
            record_error(error); remaining = list(self.remembered.values()); gpu_pids = []
        return {**self.report(), "owned_identities_remaining": [asdict(p) for p in remaining],
                "owned_gpu_pids_remaining": gpu_pids, "cleanup_errors": errors,
                "cleanup_error_details": details,
                "cleanup_complete": not remaining and not gpu_pids and not errors}


def preserve_primary(primary, cleanup_error):
    if primary is not None:
        if hasattr(primary, "add_note"):
            primary.add_note(cleanup_error)
        else:  # CPU scaffold also supports Python 3.10.
            primary.__notes__ = getattr(primary, "__notes__", []) + [cleanup_error]
    else:
        raise RuntimeError(cleanup_error)
