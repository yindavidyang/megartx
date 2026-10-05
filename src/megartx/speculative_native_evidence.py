"""Bounded immutable private evidence for the existing receipt utility only.

This writer grants no worker lease and no probe permission. Its byte cap does
not bound native allocator-counter acquisition; that peak remains unknown.
"""
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from .speculative_native_probe import ProbeError, REVISION, _process_start
from .speculative_native_plan import PURPOSE, LIMITS, SHA, COMMIT, hash_file

CAP = LIMITS["receipt_host_metadata_bytes"]


def error_detail(error, stage, operation=None):
    """Small diagnostic only, never a traceback, raw path, or cleanup proof."""
    def safe(value, size=384):
        try:
            value = str(value)[:size]
        except BaseException:
            return "<unprintable>"
        value = re.sub(r"(?<![\w])/(?:[^\s'\"<>]*)", "<path>", value)
        return "".join(char if char.isprintable() else " " for char in value)
    def row(item):
        return {"error_type": safe(type(item).__name__, 80),
                "message": safe(item),
                "errno": item.errno if isinstance(item, OSError) and type(item.errno) is int else None,
                "operation": safe(getattr(item, "megartx_operation", operation or stage), 80)}
    result = {"stage": safe(stage, 80), **row(error), "causes": [],
              "notes": [safe(note, 256) for note in getattr(error, "__notes__", ())[:2]]}
    seen, current = {id(error)}, error
    for _ in range(3):
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        result["causes"].append(row(current))
    return result


def annotate_operation(error, operation):
    if not hasattr(error, "megartx_operation"):
        error.megartx_operation = operation


def retain_error(primary, secondary, stage, operation=None):
    """Keep the original exception, including on Python 3.10, with safe cause."""
    if primary is None:
        primary = secondary
    note = stage + ": " + json.dumps(error_detail(secondary, stage, operation), sort_keys=True)
    if hasattr(primary, "add_note"):
        primary.add_note(note)
    else:
        primary.__notes__ = getattr(primary, "__notes__", []) + [note]
    return primary


def bounded_json_chunks(value, cap=CAP):
    """Bound structure, then exact encoded bytes before acquiring an output.

    The traversal creates no flattened copy. Strings are bounded individually;
    node count/depth and scalar limits bound each encoder chunk. A conservative
    upper bound is a fast acceptance check, not grounds to reject a valid file.
    If needed, a discard-only streaming pass counts exact bytes up to the cap.
    """
    if type(cap) is not int or not 0 < cap <= CAP:
        raise ProbeError("Private evidence cap exceeds the fixed serialized limit")
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
    inspect(value)
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    if upper > cap:
        exact = 0
        for chunk in encoder.iterencode(value):
            exact += len(chunk.encode())
            if exact > cap:
                raise ProbeError("Private evidence exact byte bound exceeded before acquisition")
    total = 0
    for chunk in encoder.iterencode(value):
        data = chunk.encode()
        if total + len(data) > cap:
            raise ProbeError("Private evidence byte limit exceeded before write")
        total += len(data)
        yield data


class PrivateEvidence:
    def __init__(self, directory):
        self.directory, self.fd = Path(directory), None
        operation = "evidence_directory_open"
        try:
            self.fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            operation = "evidence_directory_validate"
            info = os.fstat(self.fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ProbeError("Private evidence directory must be owned and mode 0700")
            operation = "evidence_disk_reserve"
            filesystem = os.fstatvfs(self.fd)
            if filesystem.f_bavail * filesystem.f_frsize < LIMITS["private_run_evidence_bytes"]:
                raise ProbeError("Insufficient private evidence disk reserve")
        except BaseException as primary:
            annotate_operation(primary, operation)
            try:
                self.close()
            except BaseException as secondary:
                retain_error(primary, secondary, "Evidence initialization cleanup failed")
            raise

    def close(self):
        if getattr(self, "fd", None) is not None:
            fd, self.fd = self.fd, None
            try:
                os.close(fd)
            except BaseException as error:
                annotate_operation(error, "evidence_directory_close")
                raise

    def __enter__(self):
        return self

    def __exit__(self, error_type, primary, traceback):
        try:
            self.close()
        except BaseException as secondary:
            if primary is None:
                raise
            retain_error(primary, secondary, "Evidence directory cleanup failed")

    def write(self, name, value, *, cap=CAP):
        operation, tmp, fd = "evidence_filename", None, None
        try:
            if Path(name).name != name or name.startswith("."):
                raise ProbeError("Invalid evidence filename")
            # Complete conservative validation before opening/acquiring output.
            operation = "evidence_serialization_preflight"
            chunks = bounded_json_chunks(value, cap)
            first = next(chunks, b"")
            # Serial owned writers share this directory. Reserve the complete future
            # baseline startup files and log even if they have not been written yet.
            # Counting is streaming and stops before accepting unbounded topology.
            operation = "evidence_inventory"
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
            operation = "evidence_temporary_open"
            candidate = ".pending-" + str(uuid.uuid4())
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
            tmp = candidate  # Only unlink a temporary file we actually acquired.
            total, digest = 0, hashlib.sha256()
            operation = "evidence_stream_open"
            stream = os.fdopen(fd, "wb")
            fd = None  # stream owns the descriptor after successful fdopen.
            try:
                operation = "evidence_write"
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
                operation = "evidence_flush"
                stream.flush()
                operation = "evidence_file_fsync"
                os.fsync(stream.fileno())
                operation = "evidence_file_seal"
                os.fchmod(stream.fileno(), 0o400)
            except BaseException as primary:
                annotate_operation(primary, operation)
                try:
                    stream.close()
                except BaseException as secondary:
                    retain_error(primary, secondary, "Evidence stream cleanup failed", "evidence_stream_close")
                raise
            operation = "evidence_stream_close"
            stream.close()
            # link is no-replace, unlike rename. Existing receipts never mutate.
            operation = "evidence_publish_no_replace"
            os.link(tmp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            operation = "evidence_temporary_unlink"
            os.unlink(tmp, dir_fd=self.fd)
            tmp = None
            operation = "evidence_directory_fsync"
            os.fsync(self.fd)
            return {"sha256": digest.hexdigest(), "bytes": total}
        except BaseException as primary:
            annotate_operation(primary, operation)
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as secondary:
                    retain_error(primary, secondary, "Evidence file cleanup failed", "evidence_file_close")
            if tmp is not None:
                try:
                    os.unlink(tmp, dir_fd=self.fd)
                except FileNotFoundError:
                    pass
                except BaseException as secondary:
                    retain_error(primary, secondary, "Evidence temporary cleanup failed", "evidence_temporary_unlink")
            raise


def persist_owned_receipt(core, admission, summary, *, runner_lane="v1-legacy"):
    """Observe the already bound real lease; no owner injection or new RPC."""
    if runner_lane == "v1-legacy":
        from .speculative_native_lifecycle import PURPOSE as LEASE_PURPOSE, _require_core, UTILITIES
        client_purpose, identity_schema = PURPOSE, "megartx-native-receipt-evidence-identity-v1"
    elif runner_lane == "v2":
        from .speculative_native_v2_lifecycle import PURPOSE as LEASE_PURPOSE, _require_core, UTILITIES, _check_result
        from .speculative_native_v2_plan import PURPOSE as client_purpose
        identity_schema = "megartx-native-v2-receipt-evidence-identity-v1"
    else:
        raise ProbeError("Unknown private evidence runner lane")
    _require_core(core, UTILITIES[0])
    lease = getattr(core, "_megartx_native_lease", None)
    if (lease is None or lease.receipt is None or lease.released or lease.poisoned
            or lease.ticket.get("purpose") != LEASE_PURPOSE
            or lease.ticket.get("engine_pid") != os.getpid()
            or lease.ticket.get("engine_start") != _process_start(os.getpid())
            or admission.get("client_purpose") != client_purpose
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
    if runner_lane == "v2":
        _check_result(raw)
        from .speculative_native_compare import validate_scalar_receipt
        validate_scalar_receipt(summary, runner_lane="v2")
    with PrivateEvidence(admission["private_receipt_directory"]) as evidence:
        result = evidence.write("native-receipt.private.json", raw)
        if result["sha256"] != summary.get("receipt_sha256"):
            raise ProbeError("Private receipt digest differs from actual EngineCore result")
        identity = {"schema": identity_schema, "purpose": client_purpose,
                    "plan_sha256": admission["client_plan_sha256"], "source_head": admission["source_head"],
                    "checkpoint_revision": raw["checkpoint_revision"], "receipt_sha256": result["sha256"],
                    "receipt_bytes": result["bytes"], "engine_pid": lease.ticket["engine_pid"],
                    "engine_start": lease.ticket["engine_start"], "worker_pid": raw["worker_pid"],
                    "worker_start": raw["worker_start"], "lease_nonce": lease.ticket["nonce"],
                    "reserved_group_block_sizes": lease.ticket["block_sizes"],
                    "reserved_groups": lease.ticket["groups"],
                    "client_evidence_source_sha256": expected}
        if runner_lane == "v2":
            from .speculative_native_v2 import POLICY
            identity.update(runner_lane="v2", runner_policy=dict(POLICY))
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
    if (flag != "1" or os.environ.get("MEGARTX_NATIVE_DIAGNOSTIC") != "1"
            or "MEGARTX_NATIVE_V2_DIAGNOSTIC" in os.environ
            or "MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE" in os.environ):
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


def persist_owned_v2_receipt(core, admission, summary):
    return persist_owned_receipt(core, admission, summary, runner_lane="v2")


def install_native_v2_receipt_evidence():
    """Wrap only the V2 owned receipt after its explicit plugin registration."""
    flag = os.environ.get("MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE")
    if flag is None:
        return False
    from . import speculative_native_v2_lifecycle as lifecycle
    from .speculative_native_v2_plan import FORBIDDEN_ENV, check_client_admission
    if (flag != "1" or os.environ.get("MEGARTX_NATIVE_V2_DIAGNOSTIC") != "1"
            or any(key in os.environ for key in FORBIDDEN_ENV)):
        raise ProbeError("Default-off exclusive V2 private evidence mode differs")
    from vllm.v1.engine.core import EngineCoreProc
    original = getattr(EngineCoreProc, lifecycle.UTILITIES[0], None)
    if original is not lifecycle.owned_receipt:
        raise ProbeError("V2 private writer requires exact owned receipt registration; duplicate rejected")
    @wraps(original)
    def receipt(self, admission):
        # Repeat the immutable source plan before any native reservation.
        check_client_admission(admission, Path(__file__).resolve().parents[2])
        with PrivateEvidence(admission["private_receipt_directory"]):
            pass
        summary = original(self, admission)
        try:
            return persist_owned_v2_receipt(self, admission, summary)
        except BaseException as primary:
            try:
                # We remain inside receipt utility context. The shared release
                # uses the retained V2 ticket/worker drain and keeps uncertain refs.
                lifecycle._require_core(self, lifecycle.UTILITIES[0])
                lifecycle.common._release(self)
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note("V2 private writer cleanup uncertain: " + type(cleanup).__name__)
            raise
    setattr(EngineCoreProc, lifecycle.UTILITIES[0], receipt)
    return True
