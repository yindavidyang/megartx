"""Bounded immutable private evidence for the existing receipt utility only.

This writer grants no worker lease and no probe permission. Its byte cap does
not bound the pinned collector's earlier memory_snapshot materialization.
"""
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import uuid

from .speculative_native_probe import ProbeError, REVISION, _process_start
from .speculative_native_plan import PURPOSE, LIMITS, SHA, COMMIT, hash_file

CAP = LIMITS["receipt_host_metadata_bytes"]


def bounded_json_chunks(value, cap=CAP):
    """Check conservative encode size before encoding even the first chunk.

    The traversal creates no flattened copy. Strings are bounded individually;
    node count/depth and a conservative escaped upper bound limit encoding work.
    """
    nodes, upper = 0, 0
    def inspect(node, depth=0):
        nonlocal nodes, upper
        nodes += 1
        if depth > 32 or nodes > 100000:
            raise ProbeError("Private evidence topology exceeds its bound")
        if node is None or type(node) is bool:
            upper += 5
        elif type(node) is int:
            if abs(node) >= 1 << 64:
                raise ProbeError("Evidence integer exceeds uint64 magnitude")
            upper += 21
        elif type(node) is float:
            if not math.isfinite(node):
                raise ProbeError("Nonfinite private evidence")
            upper += 32
        elif type(node) is str:
            if len(node) > 4096:
                raise ProbeError("Private evidence string exceeds its bound")
            upper += 12 * len(node) + 2  # non-BMP surrogate pair worst case
        elif type(node) is dict:
            upper += 2 + 2 * len(node)
            for key, item in node.items():
                if type(key) is not str:
                    raise ProbeError("Private evidence requires string keys")
                inspect(key, depth + 1)
                inspect(item, depth + 1)
        elif type(node) in (list, tuple):
            upper += 2 + len(node)
            for item in node:
                inspect(item, depth + 1)
        else:
            raise ProbeError("Non-JSON private evidence")
        if upper > cap:
            raise ProbeError("Private evidence conservative byte bound exceeded before encoding")
    inspect(value)
    total = 0
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        data = chunk.encode()
        if total + len(data) > cap:
            raise ProbeError("Private evidence byte limit exceeded before write")
        total += len(data)
        yield data


class PrivateEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self.fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            self.close()
            raise ProbeError("Private evidence directory must be owned and mode 0700")
        filesystem = os.fstatvfs(self.fd)
        if filesystem.f_bavail * filesystem.f_frsize < LIMITS["private_run_evidence_bytes"]:
            self.close()
            raise ProbeError("Insufficient private evidence disk reserve")

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, name, value, *, cap=CAP):
        if Path(name).name != name or name.startswith("."):
            raise ProbeError("Invalid evidence filename")
        # Complete conservative validation before opening/acquiring output.
        chunks = bounded_json_chunks(value, cap)
        first = next(chunks, b"")
        # Serial owned writers share this directory. Reserve the complete future
        # baseline startup files and log even if they have not been written yet.
        # Counting is streaming and stops before accepting unbounded topology.
        count, present = 0, 0
        with os.scandir(self.fd) as entries:
            for entry in entries:
                count += 1
                if count > 64 or not entry.is_file(follow_symlinks=False):
                    raise ProbeError("Private evidence inventory exceeds its bound")
                present += entry.stat(follow_symlinks=False).st_size
                if present > LIMITS["private_run_evidence_bytes"]:
                    raise ProbeError("Private run evidence exceeds its frozen bound")
        future_startup_and_log = 4 * (8 << 20)
        if present + cap + future_startup_and_log > LIMITS["private_run_evidence_bytes"]:
            raise ProbeError("Private evidence disk budget exceeded before acquisition")
        tmp = ".pending-" + str(uuid.uuid4())
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        total, digest = 0, hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as stream:
                def put(data):
                    nonlocal total
                    if total + len(data) > cap:
                        raise ProbeError("Evidence limit before write")
                    stream.write(data)
                    total += len(data)
                    digest.update(data)
                put(first)
                for data in chunks:
                    put(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o400)
            # link is no-replace, unlike rename. Existing receipts never mutate.
            os.link(tmp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            os.unlink(tmp, dir_fd=self.fd)
            os.fsync(self.fd)
            return {"sha256": digest.hexdigest(), "bytes": total}
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=self.fd)
            except FileNotFoundError:
                pass
            raise


def persist_owned_receipt(core, admission, summary):
    """Observe the already bound real lease; no owner injection or new RPC."""
    from .speculative_native_lifecycle import PURPOSE as LEASE_PURPOSE, _require_core, UTILITIES
    _require_core(core, UTILITIES[0])
    lease = getattr(core, "_megartx_native_lease", None)
    if (lease is None or lease.receipt is None or lease.released or lease.poisoned
            or lease.ticket.get("purpose") != LEASE_PURPOSE
            or lease.ticket.get("engine_pid") != os.getpid()
            or lease.ticket.get("engine_start") != _process_start(os.getpid())
            or admission.get("client_purpose") != PURPOSE
            or not SHA.fullmatch(str(admission.get("client_plan_sha256", "")))
            or not COMMIT.fullmatch(str(admission.get("source_head", "")))):
        raise ProbeError("Private writer requires the existing live receipt owner/purpose")
    expected = admission.get("client_evidence_source_sha256")
    if expected != hash_file(__file__, 1 << 20):
        raise ProbeError("Private evidence implementation source differs")
    raw = lease.receipt
    if raw.get("purpose") != LEASE_PURPOSE or raw.get("checkpoint_revision") != REVISION:
        raise ProbeError("Wrong-purpose or wrong-checkpoint private receipt")
    if raw.get("diagnostic_target_forwards") != 0 or type(raw.get("diagnostic_target_forwards")) is not int:
        raise ProbeError("Additional target forwards are forbidden")
    with PrivateEvidence(admission["private_receipt_directory"]) as evidence:
        result = evidence.write("native-receipt.private.json", raw)
        if result["sha256"] != summary.get("receipt_sha256"):
            raise ProbeError("Private receipt digest differs from actual EngineCore result")
        identity = {"schema": "megartx-native-receipt-evidence-identity-v1", "purpose": PURPOSE,
                    "plan_sha256": admission["client_plan_sha256"], "source_head": admission["source_head"],
                    "checkpoint_revision": raw["checkpoint_revision"], "receipt_sha256": result["sha256"],
                    "receipt_bytes": result["bytes"], "engine_pid": lease.ticket["engine_pid"],
                    "engine_start": lease.ticket["engine_start"], "worker_pid": raw["worker_pid"],
                    "worker_start": raw["worker_start"], "lease_nonce": lease.ticket["nonce"],
                    "reserved_group_block_sizes": lease.ticket["block_sizes"],
                    "reserved_groups": lease.ticket["groups"],
                    "client_evidence_source_sha256": expected}
        evidence.write("native-receipt-identity.private.json", identity, cap=64 << 10)
    # Identity remains private; public output remains the original scalar map.
    return summary


def install_native_receipt_evidence():
    """Call immediately after install_native_diagnostic, before idempotence return.

    Wraps the same existing utility name. Does not register another capability,
    patch worker/Torch methods, free references or invoke target probes.
    """
    flag = os.environ.get("MEGARTX_NATIVE_RECEIPT_EVIDENCE")
    if flag is None:
        return False
    if flag != "1" or os.environ.get("MEGARTX_NATIVE_DIAGNOSTIC") != "1":
        raise ProbeError("Default-off private evidence mode differs")
    from . import speculative_native_lifecycle as lifecycle
    from vllm.v1.engine.core import EngineCoreProc
    original = getattr(EngineCoreProc, "megartx_owned_native_receipt", None)
    if original is not lifecycle.owned_receipt:
        raise ProbeError("Private writer requires exact owned receipt registration; duplicate rejected")
    @wraps(original)
    def receipt(self, admission):
        # Validate writer bounds/path/source before original reserves KV objects.
        if (admission.get("client_purpose") != PURPOSE
                or admission.get("client_evidence_source_sha256") != hash_file(__file__, 1 << 20)):
            raise ProbeError("Missing source-bound private writer admission")
        with PrivateEvidence(admission["private_receipt_directory"]):
            pass
        summary = original(self, admission)
        try:
            return persist_owned_receipt(self, admission, summary)
        except BaseException as primary:
            # The original RPC completed. A local writer failure can drain via
            # the existing release operation; uncertainty retains owned refs.
            try:
                lifecycle._release(self)
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note("Private writer cleanup uncertain: " + type(cleanup).__name__)
            raise
    EngineCoreProc.megartx_owned_native_receipt = receipt
    return True
