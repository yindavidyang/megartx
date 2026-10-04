"""CPU receipt checks and scalar-only comparison; missing coverage stays unknown."""
import hashlib
import uuid

from .speculative_native_probe import ProbeError, REVISION
from .speculative_native_plan import (LIMITS, PURPOSE, SHA, TORCH_BUILD_IDENTITY, integer, read_json)
from .speculative_native_evidence import CAP, bounded_json_chunks

SCALAR_KEYS = {"schema", "receipt_sha256", "diagnostic_target_forwards", "existing_startup_target_forwards",
               "existing_forced_expert_fixture_pairs", "existing_forced_correction_rows", "original_native_head_dtype",
               "allocation_lower_bound", "gpu_free_bytes", "host_free_bytes", "decision"}
ALLOCATION_KEYS = {"private_page_bytes", "head_peak_bytes", "hidden_bytes", "metadata_bytes", "known_bytes",
                   "remaining_for_native_scratch_and_allocator"}
DECISION_KEYS = {"admitted", "blockers", "cached_slack_bytes", "known_plus_slack_bytes", "bounded_incremental_bytes",
                 "cap_bytes", "required_bytes_above_cap"}
BLOCKERS = {"external_CUDA_allocation_bound_unresolved", "M1_M2_M256_original_lane_temporary_bound_unresolved",
            "FFI_argument_exchange_allocator_coverage_unverified", "known_buffers_plus_cached_slack_exceed_8MiB",
            "GPU_free_below_2GiB", "host_free_below_8GiB", "bounded_incremental_peak_exceeds_8MiB"}
UNRESOLVED = {"external_CUDA_allocation_bound_unresolved", "M1_M2_M256_original_lane_temporary_bound_unresolved",
              "FFI_argument_exchange_allocator_coverage_unverified"}


def validate_scalar_receipt(receipt):
    if type(receipt) is not dict or set(receipt) != SCALAR_KEYS:
        raise ProbeError("Malformed or non-allowlisted scalar receipt")
    if (receipt["schema"] != "megartx-native-zero-forward-receipt-v1"
            or not SHA.fullmatch(str(receipt["receipt_sha256"]))
            or type(receipt["diagnostic_target_forwards"]) is not int or receipt["diagnostic_target_forwards"] != 0):
        raise ProbeError("Receipt source digest or zero-forward schema differs")
    if integer(receipt["existing_startup_target_forwards"]) > LIMITS["startup_target_forwards"]:
        raise ProbeError("Existing startup frame allowance exceeded")
    if integer(receipt["existing_forced_expert_fixture_pairs"]) != 6:
        raise ProbeError("Existing six fixture pairs required")
    integer(receipt["existing_forced_correction_rows"])
    if receipt["original_native_head_dtype"] not in ("torch.bfloat16", "torch.float32"):
        raise ProbeError("Unresolved original head dtype")
    for key in ("gpu_free_bytes", "host_free_bytes"):
        integer(receipt[key])
    allocation, decision = receipt["allocation_lower_bound"], receipt["decision"]
    if type(allocation) is not dict or set(allocation) != ALLOCATION_KEYS:
        raise ProbeError("Malformed actual allocation lower bound")
    parts = ("private_page_bytes", "head_peak_bytes", "hidden_bytes", "metadata_bytes")
    for key in (*parts, "known_bytes"):
        integer(allocation[key])
    if (allocation["known_bytes"] != sum(allocation[k] for k in parts)
            or type(allocation["remaining_for_native_scratch_and_allocator"]) is not int
            or allocation["remaining_for_native_scratch_and_allocator"] != (8 << 20) - allocation["known_bytes"]):
        raise ProbeError("Allocation lower-bound arithmetic differs")
    if type(decision) is not dict or set(decision) != DECISION_KEYS:
        raise ProbeError("Malformed scalar measurement decision")
    if (decision["admitted"] is not False or decision["bounded_incremental_bytes"] is not None
            or decision["cap_bytes"] != 8 << 20 or type(decision["cap_bytes"]) is not int):
        raise ProbeError("First receipt cannot grant later target-probe/unknown-fit admission")
    blockers = decision["blockers"]
    if (type(blockers) is not list or any(type(b) is not str for b in blockers)
            or len(set(blockers)) != len(blockers) or not UNRESOLVED <= set(blockers) <= BLOCKERS):
        raise ProbeError("Unknown allocation bounds were omitted or fabricated")
    for key in ("cached_slack_bytes", "known_plus_slack_bytes", "required_bytes_above_cap"):
        integer(decision[key])
    known = allocation["known_bytes"] + decision["cached_slack_bytes"]
    if (decision["known_plus_slack_bytes"] != known
            or decision["required_bytes_above_cap"] != max(0, known - (8 << 20))):
        raise ProbeError("Cached-slack/lower-bound arithmetic differs")
    required = set(UNRESOLVED)
    if known > 8 << 20:
        required.add("known_buffers_plus_cached_slack_exceed_8MiB")
    if receipt["gpu_free_bytes"] < 2 << 30:
        required.add("GPU_free_below_2GiB")
    if receipt["host_free_bytes"] < 8 << 30:
        required.add("host_free_below_8GiB")
    if set(blockers) != required:
        raise ProbeError("Scalar measurement blockers differ from observed bounds")
    return receipt


def stream_digest(value):
    digest = hashlib.sha256()
    for chunk in bounded_json_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def verify_private_receipt(raw, identity, scalar, plan):
    validate_scalar_receipt(scalar)
    from .speculative_native_lifecycle import PURPOSE as LEASE_PURPOSE
    from .speculative_native_probe import source_manifest
    if (type(raw) is not dict or raw.get("purpose") != LEASE_PURPOSE
            or raw.get("checkpoint_revision") != REVISION
            or raw.get("drained") is not True
            or identity.get("schema") != "megartx-native-receipt-evidence-identity-v1"
            or identity.get("purpose") != PURPOSE
            or identity.get("plan_sha256") != plan["plan_sha256"]
            or identity.get("source_head") != plan["source_head"]
            or identity.get("checkpoint_revision") != REVISION
            or identity.get("receipt_sha256") != scalar["receipt_sha256"]
            or stream_digest(raw) != scalar["receipt_sha256"]):
        raise ProbeError("Unbound private receipt source/owner/purpose/digest")
    if str(uuid.UUID(raw["lease_nonce"])) != raw["lease_nonce"] or identity.get("lease_nonce") != raw["lease_nonce"]:
        raise ProbeError("Receipt lease nonce differs")
    for key in ("engine_pid", "worker_pid"):
        integer(identity[key], 1)
    for key in ("engine_start", "worker_start"):
        if type(identity[key]) is not str or not identity[key].isdigit():
            raise ProbeError("Missing actual PID start-time identity")
    for key in ("worker_pid", "worker_start"):
        if raw.get(key) != identity[key]:
            raise ProbeError("Actual worker owner differs")
    if identity.get("client_evidence_source_sha256") != plan["source_sha256"]["src/megartx/speculative_native_evidence.py"]:
        raise ProbeError("Private writer source differs")
    expected = {k: v["sha256"] for k, v in source_manifest()["files"].items()}
    if raw.get("source_sha256") != expected:
        raise ProbeError("Actual loaded receipt source differs")
    if raw.get("torch_build_identity") != TORCH_BUILD_IDENTITY:
        raise ProbeError("Actual loaded Torch distribution/CUDA build identity differs")
    from .speculative_native_probe import ADAPTER_FILES
    if raw.get("adapter_source_sha256") != {k: plan["source_sha256"]["src/megartx/" + k] for k in ADAPTER_FILES}:
        raise ProbeError("Actual adapter source differs")
    for key in SCALAR_KEYS - {"receipt_sha256"}:
        if raw.get(key) != scalar[key]:
            raise ProbeError("Private/public scalar field differs: " + key)
    allocator = raw.get("allocator", {})
    allocated, reserved = integer(allocator.get("allocated_bytes")), integer(allocator.get("reserved_bytes"))
    if reserved < allocated or allocator.get("cached_slack_bytes") != reserved - allocated or reserved - allocated != scalar["decision"]["cached_slack_bytes"]:
        raise ProbeError("Actual allocator counters differ")
    # A missing/global snapshot is unknown coverage, never zero allocation.
    # Never publish arbitrary private coverage text/objects. Presence of an
    # explicit source coverage field still cannot establish transitive bounds.
    coverage = "explicit_partial_or_unavailable" if ("snapshot_coverage" in allocator
        or "snapshot_coverage" in raw or "collection_coverage" in raw
        or "coverage" in allocator or "snapshot_status" in allocator) else "legacy_observation_only"
    if "acquisition" in allocator:
        acquisition = allocator["acquisition"]
        if (type(acquisition) is not dict or acquisition.get("native_query_preallocation_bound_bytes") is not None
                or acquisition.get("queries") != 1):
            raise ProbeError("Native allocator-query acquisition coverage was fabricated")
    if raw.get("external_gpu_workspace_bound_bytes") is not None or raw.get("ffi_allocator", {}).get("argument_exchange_coverage_verified") is not False:
        raise ProbeError("First receipt unexpectedly claims external allocation coverage")
    from .speculative_native_receipt import FFI_SOURCES
    ffi = raw["ffi_allocator"]
    if ffi.get("source_sha256") != FFI_SOURCES:
        raise ProbeError("Loaded FFI helper/ABI source differs")
    for key in ("managed_allocator_callback", "callback_binary_sha256"):
        if key not in ffi:
            raise ProbeError("Missing loaded callback observation/explicit unknown")
    if ffi["callback_binary_sha256"] is not None and not SHA.fullmatch(str(ffi["callback_binary_sha256"])):
        raise ProbeError("Malformed loaded callback binary identity")
    if ffi["managed_allocator_callback"] is not None:
        integer(ffi["managed_allocator_callback"])
    workspaces = raw.get("workspaces")
    if type(workspaces) is not dict or type(workspaces.get("owners")) is not list:
        raise ProbeError("Missing existing workspace owner ledger")
    stores = {}
    for record in workspaces["owners"]:
        if type(record) is not dict or type(record.get("device")) is not str:
            raise ProbeError("Malformed actual workspace placement")
        key = (record["device"], integer(record.get("storage_ptr")))
        size = integer(record.get("storage_bytes"))
        if key in stores and stores[key] != size:
            raise ProbeError("Workspace backing storage extent differs")
        stores[key] = size
    totals = {"deduplicated_storage_bytes": sum(stores.values()),
        "deduplicated_GPU_storage_bytes": sum(s for (d, _), s in stores.items() if d.startswith("cuda")),
        "deduplicated_host_storage_bytes": sum(s for (d, _), s in stores.items() if d == "cpu")}
    if any(workspaces.get(k) != v or type(workspaces.get(k)) is not int for k, v in totals.items()):
        raise ProbeError("Existing workspace baseline storage ledger differs")
    if (workspaces.get("required_size_M1_M2_M256_bytes") is not None
            or raw.get("baseline_policy") != "model_cache_existing_startup_workspaces_separate_from_incremental_verifier"):
        raise ProbeError("Unknown workspace sizes/baseline were substituted")
    layers = raw.get("layers")
    if type(layers) is not list or len(layers) != 30:
        raise ProbeError("Thirty actual cache layer owners required")
    from .speculative_native_receipt import page_bytes, disjoint_pages
    spans, owners, group_blocks, charges = [], set(), {}, []
    for layer in layers:
        if layer.get("dtype") != "torch.bfloat16" or not str(layer.get("device", "")).startswith("cuda"):
            raise ProbeError("Actual GPU BF16 cache owner required")
        if layer["owner"] in owners:
            raise ProbeError("Duplicate actual cache owner")
        owners.add(layer["owner"])
        placement = layer["placement"]
        block = integer(layer["manager_block_tokens"], 1)
        if block != layer["kernel_block_tokens"] or block not in (16, 32, 64):
            raise ProbeError("Manager/kernel cache splitting unresolved")
        pages = layer["owned_pages"]
        group = integer(layer["group"])
        if group >= len(identity["reserved_groups"]) or group >= len(identity["reserved_group_block_sizes"]):
            raise ProbeError("Actual core reservation group differs")
        if ([p[0] for p in pages] != identity["reserved_groups"][group]
                or block != identity["reserved_group_block_sizes"][group]):
            raise ProbeError("Core reservation and worker page identities differ")
        if group in group_blocks and group_blocks[group] != block:
            raise ProbeError("Actual group block geometry differs")
        group_blocks[group] = block
        if len(pages) != (2048 + 4 + block - 1) // block + 1 or len({p[0] for p in pages}) != len(pages):
            raise ProbeError("Actual reserved page capacity differs")
        for page, begin, end in pages:
            bounds = page_bytes(shape=tuple(layer["shape"]), strides=tuple(layer["strides"]), element_bytes=2,
                storage_offset_bytes=layer["storage_offset_bytes"], page=page, placement_offset=placement["offset"],
                layer_ordinal=placement["layers"].index(layer["owner"]), layer_stride=placement["layer_stride"],
                block_stride=placement["block_stride"], raw_bytes=layer["storage_bytes"], allocated_page_bytes=end-begin)
            if bounds != (begin, end):
                raise ProbeError("Actual private page interval differs")
            spans.append((layer["storage_ptr"], begin, end))
        charges.append(pages[0][2] - pages[0][1])
    disjoint_pages(spans)
    if set(group_blocks) != set(range(len(identity["reserved_groups"]))):
        raise ProbeError("Reservation group coverage unresolved")
    from .speculative_native_probe import allocation_lower_bound
    metadata = sum(4 * ((2048 + 4 + b - 1) // b + 1) + 48 for b in group_blocks.values())
    head_bytes = 2 if scalar["original_native_head_dtype"] == "torch.bfloat16" else 4
    expected_allocation = allocation_lower_bound(charges, metadata, head_element_bytes=head_bytes, reject=False)
    if raw["allocation_lower_bound"] != expected_allocation:
        raise ProbeError("Measured page padding/head/metadata lower bound differs")
    return {"schema": "megartx-native-receipt-comparison-v1", "source_head": plan["source_head"],
            "plan_sha256": plan["plan_sha256"], "receipt_sha256": scalar["receipt_sha256"],
            "diagnostic_target_forwards": 0, "existing_startup_target_forwards": scalar["existing_startup_target_forwards"],
            "actual_cache_layer_count": len(layers), "private_evidence_verified": True,
            "snapshot_coverage": coverage, "fit_admitted": False, "fit_blockers": scalar["decision"]["blockers"],
            "later_probe_authorized": False}


def compare_files(directory, plan):
    from pathlib import Path
    directory = Path(directory)
    raw = read_json(directory / "native-receipt.private.json", CAP)
    identity = read_json(directory / "native-receipt-identity.private.json", 64 << 10)
    scalar = read_json(directory / "native-receipt.scalars.json", 64 << 10)
    if identity.get("receipt_bytes") != (directory / "native-receipt.private.json").stat().st_size:
        raise ProbeError("Private receipt byte count differs")
    return verify_private_receipt(raw, identity, scalar, plan)
