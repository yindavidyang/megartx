"""Linux task-local subreaper ownership and sampled shared compiler bounds.

Ownership is PID/start-time lineage, never executable names or process groups.
pidfds close the identity-check/signal PID-reuse race. No GPU imports.
"""
import ctypes
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import signal
import threading
import time

COMPILERS = {"ninja", "nvcc", "cicc", "ptxas", "gcc", "g++", "cc", "c++",
             "cc1", "cc1plus", "clang", "clang++", "ld", "ld.gold", "ld.lld", "collect2"}


@dataclass(frozen=True)
class Process:
    pid: int
    start_ticks: int
    ppid: int
    rss_bytes: int = 0
    executable: str = ""
    state: str = "S"
    age_seconds: float = 0

    @property
    def identity(self):
        return self.pid, self.start_ticks


def read_process(pid, uptime=None):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text()
    fields = stat[stat.rindex(")") + 2:].split()
    executable = stat[stat.index("(") + 1:stat.rindex(")")]
    try:
        executable = Path(os.readlink(directory / "exe")).name
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    start = int(fields[19])
    if uptime is None:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    return Process(int(pid), start, int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                   executable, fields[0], max(0, uptime - start / os.sysconf("SC_CLK_TCK")))


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


def signal_identity(process, signum):
    """Signal exactly the retained identity; reused PID is never signalled."""
    try:
        fd = os.pidfd_open(process.pid)
    except ProcessLookupError:
        return False
    try:
        if read_process(process.pid).identity != process.identity:
            return False
        signal.pidfd_send_signal(fd, signum)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        os.close(fd)


class OwnedProcesses:
    def __init__(self, supervisor, *, rss_limit=2 << 30, compiler_seconds=300):
        self.supervisor = supervisor
        self.remembered = {}
        self.compiler_identities = set()
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
                if name in COMPILERS or name.endswith(("-gcc", "-g++", "-clang", "-clang++")):
                    self.compiler_identities.add(process.identity)
            changed = True
            while changed:
                changed = False
                parents = {p.pid for p in alive if p.identity in self.compiler_identities}
                for process in alive:
                    if process.ppid in parents and process.identity not in self.compiler_identities:
                        self.compiler_identities.add(process.identity)
                        changed = True
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
                    "remembered_identities": [asdict(p) for p in self.remembered.values()]}

    def cleanup(self, gpu_query, root_poll, *, term_seconds=10, kill_seconds=10):
        errors = []
        alive = list(self.remembered.values())
        for signum, seconds in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
            deadline = time.monotonic() + seconds
            while True:
                try:
                    alive = self.observe(snapshot(), time.monotonic())
                    root_poll()
                    for process in alive:
                        if process.state == "Z" and process.identity != self.root_identity:
                            try:
                                os.waitpid(process.pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                        else:
                            signal_identity(process, signum)
                    alive = self.current_owned(snapshot())
                except Exception as error:
                    errors.append(str(error))
                    break
                if not alive or time.monotonic() >= deadline:
                    break
                time.sleep(.05)
            if not alive:
                break
        try:
            queried_gpu_pids = gpu_query()
            processes = snapshot()
            remaining = self.observe(processes, time.monotonic())
            # Query PID then check current start time: a reused unrelated GPU PID
            # is not considered owned and is never selected for termination.
            gpu_pids = [pid for pid in queried_gpu_pids if pid in processes and processes[pid].identity in self.remembered]
        except Exception as error:
            errors.append(str(error)); remaining = list(self.remembered.values()); gpu_pids = []
        return {**self.report(), "owned_identities_remaining": [asdict(p) for p in remaining],
                "owned_gpu_pids_remaining": gpu_pids, "cleanup_errors": errors,
                "cleanup_complete": not remaining and not gpu_pids and not errors}


def preserve_primary(primary, cleanup_error):
    if primary is not None:
        if hasattr(primary, "add_note"):
            primary.add_note(cleanup_error)
        else:  # CPU scaffold also supports Python 3.10.
            primary.__notes__ = getattr(primary, "__notes__", []) + [cleanup_error]
    else:
        raise RuntimeError(cleanup_error)
