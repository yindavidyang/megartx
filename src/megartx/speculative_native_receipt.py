"""Existing-owner receipt: no target call, lazy workspace getter or sizing JIT.

The full receipt contains private pointers. Only sanitized_receipt is exported
by the EngineCore utility. Missing external allocation bounds are never zero.
"""

import ctypes
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

from .speculative_native_probe import (CONFIG_HASH, GPU_CAP, GPU_FREE, HOST_FREE,
    P, REVISION, ProbeError, _class_source, _host_free, _integer, _process_start,
    allocation_lower_bound, check_torch_build, digest, inspect_sources)

FFI_SOURCES = {
    "tvm_ffi/include/dlpack/dlpack.h": "62052bef24bd69bb7b52e95428f9e99bd7951783850c36c5f2a41e71fd63c2fc",
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py": "00642488be9a3bdc9560f14fa5d5b73172e096e89b09ffc35a7f18225feae9ce",
    "tvm_ffi/cython/tvm_ffi_python_helpers.h": "a11be0560a2cc4b75845ab63fc7cb67f0dd890b8d89c32b58d7efb408172089f"}

TORCH_MEMORY_SOURCE_SHA256 = "7dc1d9d2a00b571977e6ecc6997d7ccfa4f13b620c11faf1f3f82317c548ec99"
RECEIPT_LIMITS = {"serialized_bytes": 8 << 20, "depth": 16, "values": 131072,
    "container_items": 8192, "string_chars": 4096, "integer_bits": 256,
    "cache_layers": 30, "cache_groups": 30, "cache_placements": 30,
    "owned_page_records": 3900, "attention_builders": 30,
    "workspace_records": 512, "workspace_objects": 256, "owner_fields": 256,
    "workspace_container_items": 256, "tensor_dimensions": 64,
    "private_pool_records_after_query": 64, "process_maps_bytes": 1 << 20}


def receipt_preflight(receipt):
    """Count canonical bytes with bounded scalar encoding before hashing.

    The serialized evidence cap does not claim an 8 MiB Python heap bound.
    Ordinary primitives only: no custom serializer or scalar conversion.
    """
    total, values, active = 0, 0, set()
    def charge(size):
        nonlocal total
        total += size
        if total > RECEIPT_LIMITS["serialized_bytes"]:
            raise ProbeError("Private receipt exceeds 8 MiB serialized metadata bound")
    def walk(value, depth=0):
        nonlocal values
        values += 1
        if values > RECEIPT_LIMITS["values"] or depth > RECEIPT_LIMITS["depth"]:
            raise ProbeError("Receipt value/depth acquisition limit exceeded")
        kind = type(value)
        if kind in (dict, list, tuple):
            if len(value) > RECEIPT_LIMITS["container_items"] or id(value) in active:
                raise ProbeError("Receipt container limit or cycle")
            active.add(id(value))
            charge(2)
            for index, item in enumerate(value):
                if index:
                    charge(1)
                if kind is dict:
                    if type(item) is not str:
                        raise ProbeError("Receipt dictionary key must be a plain string")
                    walk(item, depth + 1)
                    charge(1)
                    walk(value[item], depth + 1)
                else:
                    walk(item, depth + 1)
            active.remove(id(value))
            return
        if (kind not in (str, int, float, bool, type(None))
                or kind is str and len(value) > RECEIPT_LIMITS["string_chars"]
                or kind is int and value.bit_length() > RECEIPT_LIMITS["integer_bits"]
                or kind is float and not math.isfinite(value)):
            raise ProbeError("Unsupported or oversized receipt scalar")
        # Each bounded token is ASCII, so its character count is its byte count.
        charge(len(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))))
    walk(receipt)
    return total


def receipt_digest(receipt):
    expected_bytes = receipt_preflight(receipt)
    hasher, emitted = hashlib.sha256(), 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for chunk in encoder.iterencode(receipt):
        emitted += len(chunk)
        if emitted > RECEIPT_LIMITS["serialized_bytes"]:
            raise ProbeError("Private receipt exceeds 8 MiB serialized metadata bound")
        hasher.update(chunk.encode("ascii"))
    if emitted != expected_bytes:
        raise ProbeError("Receipt changed during canonical serialization")
    return hasher.hexdigest()


def _sequence(value, limit, label):
    if not isinstance(value, (list, tuple)) or len(value) > limit:
        raise ProbeError(label + " acquisition limit exceeded or unsupported container")
    return value


def _mapping(value, label):
    if type(value) is not dict or len(value) > RECEIPT_LIMITS["owner_fields"]:
        raise ProbeError(label + " field acquisition limit exceeded")
    return value


def _text(value):
    if type(value) is not str or len(value) > RECEIPT_LIMITS["string_chars"]:
        raise ProbeError("Owner label/string acquisition limit exceeded")
    return value


def _owner_key(value, torch, depth=0):
    if depth > 4:
        raise ProbeError("Workspace key depth limit exceeded")
    if type(value) is str:
        return _text(value)
    if type(value) is int and value.bit_length() <= RECEIPT_LIMITS["integer_bits"]:
        return str(value)
    if type(value) is tuple:
        _sequence(value, 8, "Workspace key")
        return _text("(" + ",".join(_owner_key(part, torch, depth + 1) for part in value) + ")")
    if isinstance(value, getattr(torch, "device", ())):
        return _text(str(value))
    raise ProbeError("Unsupported workspace key; arbitrary string conversion refused")


def _owner_limits(runner, ticket):
    """Reject excessive owner/page topology before inspecting tensor storage."""
    groups = _sequence(runner.kv_cache_config.kv_cache_groups, RECEIPT_LIMITS["cache_groups"], "Cache groups")
    pages = _sequence(ticket["groups"], RECEIPT_LIMITS["cache_groups"], "Ticket groups")
    sizes = _sequence(ticket["block_sizes"], RECEIPT_LIMITS["cache_groups"], "Ticket sizes")
    if not groups or len(groups) != len(pages) or len(groups) != len(sizes):
        raise ProbeError("Core/worker cache group identity differs")
    _sequence(runner.kv_cache_config.kv_cache_tensors, RECEIPT_LIMITS["cache_placements"], "Cache placements")
    for placement in runner.kv_cache_config.kv_cache_tensors:
        _sequence(placement.layers, RECEIPT_LIMITS["cache_layers"], "Placement layers")
    layers = owned = 0
    for group, reserved, size in zip(groups, pages, sizes):
        names = _sequence(group.layer_names, RECEIPT_LIMITS["cache_layers"], "Cache layers")
        if type(size) is not int or size not in (16, 32, 64):
            raise ProbeError("Unreviewed actual manager block size")
        _sequence(reserved, 130, "Reserved pages")
        if len(reserved) != (P + 4 + size - 1) // size + 1:
            raise ProbeError("Reserved page count differs from actual lease")
        layers += len(names)
        owned += len(names) * len(reserved)
    if layers != RECEIPT_LIMITS["cache_layers"] or owned > RECEIPT_LIMITS["owned_page_records"]:
        raise ProbeError("Cache owner/page acquisition limit exceeded")
    attention = _sequence(runner.attn_groups, RECEIPT_LIMITS["cache_groups"], "Attention groups")
    count = sum(len(_sequence(group, RECEIPT_LIMITS["attention_builders"], "Attention builders")) for group in attention)
    if count > RECEIPT_LIMITS["attention_builders"]:
        raise ProbeError("Attention builder acquisition limit exceeded")


def page_bytes(*, shape, strides, element_bytes, storage_offset_bytes, page,
               placement_offset, layer_ordinal, layer_stride, block_stride, raw_bytes,
               allocated_page_bytes=None):
    """Check actual owned page bytes against KVCacheTensor placement."""
    if (len(shape) != 4 or len(strides) != 4 or any(type(x) is not int or x < 1
            for x in (*shape, *strides, element_bytes, layer_stride, block_stride))):
        raise ProbeError("Unsupported actual cache geometry")
    for value in (storage_offset_bytes, placement_offset, layer_ordinal, raw_bytes, page):
        _integer(value)
    if page >= shape[0]:
        raise ProbeError("Owned page outside cache")
    start = storage_offset_bytes + page * strides[0] * element_bytes
    placed = placement_offset + layer_ordinal * layer_stride + page * block_stride
    extent = (sum((n - 1) * s for n, s in zip(shape[1:], strides[1:])) + 1) * element_bytes
    payload = shape[1] * shape[2] * shape[3] * element_bytes
    expected_stride = 1
    for dimension, stride in sorted(zip(shape[1:], strides[1:]), key=lambda x: x[1]):
        if dimension > 1 and stride != expected_stride:
            raise ProbeError("Non-dense or internally aliased cache page")
        expected_stride *= dimension
    allocated = payload if allocated_page_bytes is None else _integer(allocated_page_bytes, 1)
    if (start != placed or strides[0] * element_bytes != block_stride
            or extent != payload or allocated < extent or allocated > block_stride or start + allocated > raw_bytes):
        raise ProbeError("Actual page/placement byte ownership unresolved")
    # Dense only, including LBNHC. Arbitrary strided holes need a new byte-union adapter.
    return start, start + allocated


def disjoint_pages(intervals):
    ordered = sorted(intervals)
    for left, right in zip(ordered, ordered[1:]):
        if left[0] == right[0] and left[2] > right[1]:
            raise ProbeError("Owned layer/page bytes alias")


def fit_decision(allocation, *, allocated, reserved, gpu_free, host_free,
                 external_bound=None, temporary_bound=None, ffi_verified=False):
    for value in (allocated, reserved, gpu_free, host_free):
        _integer(value)
    if reserved < allocated:
        raise ProbeError("Allocator counters inconsistent")
    blockers = []
    for name, bound in (("external_CUDA_allocation", external_bound),
                        ("M1_M2_M256_original_lane_temporary", temporary_bound)):
        if bound is None:
            blockers.append(name + "_bound_unresolved")
        else:
            _integer(bound)
    if not ffi_verified:
        blockers.append("FFI_argument_exchange_allocator_coverage_unverified")
    slack = reserved - allocated
    lower = allocation["known_bytes"] + slack
    if lower > GPU_CAP:
        blockers.append("known_buffers_plus_cached_slack_exceed_8MiB")
    if gpu_free < GPU_FREE:
        blockers.append("GPU_free_below_2GiB")
    if host_free < HOST_FREE:
        blockers.append("host_free_below_8GiB")
    upper = None if external_bound is None or temporary_bound is None else lower + external_bound + temporary_bound
    if upper is not None and upper > GPU_CAP:
        blockers.append("bounded_incremental_peak_exceeds_8MiB")
    return {"admitted": not blockers, "blockers": blockers, "cached_slack_bytes": slack,
        "known_plus_slack_bytes": lower, "bounded_incremental_bytes": upper, "cap_bytes": GPU_CAP,
        "required_bytes_above_cap": max(0, (upper if upper is not None else lower) - GPU_CAP)}


def _tensor_record(tensor, owner):
    _text(owner)
    shape = _sequence(tensor.shape, RECEIPT_LIMITS["tensor_dimensions"], "Tensor shape")
    strides = _sequence(tensor.stride(), RECEIPT_LIMITS["tensor_dimensions"], "Tensor strides")
    raw = tensor.untyped_storage()
    return {"owner": owner, "device": str(tensor.device), "dtype": str(tensor.dtype),
        "storage_ptr": raw.data_ptr(), "storage_bytes": raw.nbytes(),
        "storage_offset_bytes": tensor.storage_offset() * tensor.element_size(),
        "shape": list(shape), "strides": list(strides)}


def existing_workspaces(runner, torch):
    """Read fields of existing builders/wrappers; never initialize a missing owner."""
    records, missing, visited = [], [], set()
    def add_tensor(tensor, label):
        if len(records) >= RECEIPT_LIMITS["workspace_records"]:
            raise ProbeError("Workspace record acquisition limit exceeded")
        records.append(_tensor_record(tensor, _text(label)))
    def scan(owner, label, depth=0):
        _text(label)
        if owner is None:
            if len(missing) >= RECEIPT_LIMITS["workspace_records"]:
                raise ProbeError("Missing-owner record acquisition limit exceeded")
            missing.append(label)
            return
        if id(owner) in visited or depth > 3:
            return
        if len(visited) >= RECEIPT_LIMITS["workspace_objects"]:
            raise ProbeError("Workspace object acquisition limit exceeded")
        visited.add(id(owner))
        for name, value in _mapping(vars(owner), label).items():
            _text(name)
            if isinstance(value, torch.Tensor):
                add_tensor(value, label + "." + name)
            elif (value is not None and hasattr(value, "__dict__") and
                    type(value).__module__.startswith(("flashinfer", "vllm.v1.attention.backends"))):
                scan(value, label + "." + name, depth + 1)
    groups_all = _sequence(runner.attn_groups, RECEIPT_LIMITS["cache_groups"], "Attention groups")
    if sum(len(_sequence(g, RECEIPT_LIMITS["attention_builders"], "Attention builders")) for g in groups_all) > RECEIPT_LIMITS["attention_builders"]:
        raise ProbeError("Attention builder acquisition limit exceeded")
    for gid, groups in enumerate(groups_all):
        for aid, group in enumerate(groups):
            builder = group.get_metadata_builder()
            label = f"attention.{gid}.{aid}"
            scan(builder, label)
            for name in ("_prefill_wrapper", "_decode_wrapper", "_noncausal_prefill_wrapper", "_cascade_wrapper"):
                scan(vars(builder).get(name), label + "." + name)
            for key, wrapper in _mapping(vars(builder).get("_decode_wrappers_cudagraph", {}), "Decode wrappers").items():
                scan(wrapper, label + "._decode_wrappers_cudagraph." + _owner_key(key, torch))
    for module_name, field in (("vllm.v1.attention.backends.flashinfer", "trtllm_workspace_buffer"),
                               ("flashinfer.utils", "_cache_buf")):
        module = sys.modules.get(module_name)
        value = vars(module).get(field) if module else None
        if isinstance(value, torch.Tensor):
            add_tensor(value, module_name + "." + field)
        elif isinstance(value, dict):
            for key, tensor in _mapping(value, "Existing buffer cache").items():
                if isinstance(tensor, torch.Tensor):
                    add_tensor(tensor, module_name + "." + field + "." + _owner_key(key, torch))
    module = sys.modules.get("vllm.v1.worker.workspace")
    manager = vars(module).get("_manager") if module else None
    if manager:
        fields = _mapping(vars(manager), "Workspace manager")
        for index, tensor in enumerate(_sequence(fields.get("_current_workspaces", ()), RECEIPT_LIMITS["workspace_container_items"], "Managed workspaces")):
            if tensor is not None:
                add_tensor(tensor, f"vllm.workspace.{index}")
    unique = {(r["device"], r["storage_ptr"]): r["storage_bytes"] for r in records}
    return {"owners": records, "deduplicated_storage_bytes": sum(unique.values()),
        "deduplicated_GPU_storage_bytes": sum(size for (device, _), size in unique.items() if device.startswith("cuda")),
        "deduplicated_host_storage_bytes": sum(size for (device, _), size in unique.items() if device == "cpu"),
        "missing_lazy_owners": missing, "workspace_manager_locked": vars(manager).get("_locked") if manager else None,
        "required_size_M1_M2_M256_bytes": None,
        "required_size_status": "unqueried_device_JIT_or_new_allocation_is_outside_zero_forward_phase"}


def allocator_counters(torch, device):
    """One selected-device query; no snapshot or flattening of private pools.

    Native _cuda_memoryStats constructs its result BEFORE the pool-count guard.
    Fresh eager ownership and host readings constrain context, not that native
    acquisition's peak. Its pre-allocation bound is explicitly unavailable.
    """
    host_before = _host_free()
    if host_before < HOST_FREE:
        raise ProbeError("Host free below 8 GiB before allocator counter acquisition")
    if torch.cuda.get_allocator_backend() != "native":
        raise ProbeError("Aggregate allocator counters require the native backend")
    stats = _mapping(torch.cuda.memory_stats_as_nested_dict(device), "Allocator statistics")
    host_after = _host_free()
    if host_after < HOST_FREE:
        raise ProbeError("Host free below 8 GiB after allocator counter acquisition")
    pools = stats.get("reserved_bytes_by_private_pools")
    if pools is not None and (type(pools) is not dict or len(pools) > RECEIPT_LIMITS["private_pool_records_after_query"]):
        raise ProbeError("Private-pool record limit exceeded after native counter acquisition")
    def counter(name, field):
        metric = _mapping(stats.get(name), "Aggregate allocator metric")
        all_pools = _mapping(metric.get("all"), "Aggregate allocator all pools")
        return _integer(all_pools.get(field))  # missing values are never substituted with zero
    allocated, reserved = counter("allocated_bytes", "current"), counter("reserved_bytes", "current")
    if reserved < allocated:
        raise ProbeError("Allocator counters inconsistent")
    return {"backend": "native", "allocated_bytes": allocated, "reserved_bytes": reserved,
        "cached_slack_bytes": reserved - allocated,
        "allocated_peak_bytes": counter("allocated_bytes", "peak"),
        "reserved_peak_bytes": counter("reserved_bytes", "peak"),
        "snapshot_segments": None, "snapshot_sha256": None,
        "snapshot_status": "not_collected_unbounded_global_snapshot",
        "coverage": "selected_device_aggregate_counters_only_no_segment_or_block_coverage",
        "acquisition": {"query": "memory_stats_as_nested_dict", "queries": 1,
            "device": _text(str(device)), "private_pool_records_observed": len(pools) if pools is not None else None,
            "private_pool_count_status": "observed_after_native_query" if pools is not None else "field_absent_unavailable",
            "native_query_preallocation_bound_bytes": None,
            "native_query_bound_status": "not_exposed_by_pinned_Torch_no_peak_host_bound_claim",
            "host_free_before_bytes": host_before, "host_free_after_bytes": host_after}}


def _process_maps():
    with Path("/proc/self/maps").open("rb") as stream:
        encoded = stream.read(RECEIPT_LIMITS["process_maps_bytes"] + 1)
    if len(encoded) > RECEIPT_LIMITS["process_maps_bytes"]:
        raise ProbeError("Process mapping acquisition limit exceeded")
    return encoded.decode("utf-8").splitlines()


def ffi_allocator_identity(torch, site_root):
    """Read the existing capsule table and mapped callback; never invoke it."""
    sources = {name: digest(site_root / name) for name in FFI_SOURCES}
    if sources != FFI_SOURCES:
        raise ProbeError("Installed FFI exchange ABI/helper sources differ")
    result = {"source_sha256": sources, "managed_allocator_callback": None,
              "callback_binary_sha256": None, "argument_exchange_coverage_verified": False}
    capsule = getattr(torch.Tensor, "__dlpack_c_exchange_api__", None)
    valid = ctypes.pythonapi.PyCapsule_IsValid
    valid.argtypes, valid.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_int
    if capsule is None or not valid(capsule, b"dlpack_exchange_api"):
        result["blocker"] = "loaded_Torch_exchange_capsule_missing"
        return result
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes, pointer.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_void_p
    address = pointer(capsule, b"dlpack_exchange_api")
    class Prefix(ctypes.Structure):
        _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                    ("previous", ctypes.c_void_p), ("allocator", ctypes.c_void_p)]
    maps = []
    for line in _process_maps():
        fields = line.split(maxsplit=5)
        start, end = (int(x, 16) for x in fields[0].split("-"))
        maps.append((start, end, fields[1], fields[5] if len(fields) == 6 else None))
    if not any(start <= address and address + ctypes.sizeof(Prefix) <= end and "r" in flags
               for start, end, flags, _ in maps):
        raise ProbeError("Exchange API table is outside readable loaded mapping")
    api = Prefix.from_address(address)
    if (api.major, api.minor) != (1, 3):
        result["blocker"] = "unreviewed_loaded_exchange_API_version"
        return result
    result.update(exchange_table_ptr=address, managed_allocator_callback=api.allocator, version=[api.major, api.minor])
    owners = [path for start, end, flags, path in maps
              if api.allocator and start <= api.allocator < end and "x" in flags and path]
    if len(owners) == 1 and Path(owners[0]).is_file() and Path(owners[0]).stat().st_size <= 512 << 20:
        binary = Path(owners[0])
        hasher = hashlib.sha256()
        with binary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
        result.update(callback_binary_path=str(binary), callback_binary_sha256=hasher.hexdigest())
    result["blocker"] = "callback_observed_but_transitive_FFI_argument_route_and_external_bounds_unverified"
    return result


def collect_receipt(runner, ticket, admission):
    if sys.version_info[:3] != (3, 12, 3):
        raise ProbeError("Pinned Python 3.12.3 required")
    _class_source(runner, "vllm.v1.worker.gpu_model_runner", "GPUModelRunner")
    site_root = Path(inspect.getsourcefile(type(runner))).resolve().parents[3]
    sources = inspect_sources(site_root)
    if digest(site_root / "torch/cuda/memory.py") != TORCH_MEMORY_SOURCE_SHA256:
        raise ProbeError("Installed allocator counter source differs")
    import importlib.metadata
    for package, version in (("vllm", "0.30.0"), ("flashinfer-python", "0.6.18.post1")):
        if importlib.metadata.version(package) != version:
            raise ProbeError("Installed package differs: " + package)
    import torch
    torch_build = check_torch_build(site_root, torch)
    if (not torch.is_inference_mode_enabled() or runner.input_batch.num_reqs
            or not runner.model_config.enforce_eager or runner.parallel_config.world_size != 1
            or runner.cache_config.cache_dtype != "bfloat16"):
        raise ProbeError("Actual empty eager BF16 single-worker owner required")
    forbidden = ("MEGARTX_M1_PREPARATION", "MEGARTX_CONTROLLED_DIR", "MEGARTX_M1_NORMAL_DIR",
                 "MEGARTX_LOGITS_DIR", "MEGARTX_ROUTER_SCORE_DIR", "MEGARTX_ROUTE_AUDIT_PATH")
    if (os.environ.get("MEGARTX_SCALE_MODE") != "native"
            or os.environ.get("VLLM_PLUGINS") != "megartx_scale_adapter"
            or any(os.environ.get(key) for key in forbidden)):
        raise ProbeError("Original corrected lane/M1-off receipt environment differs")
    if _host_free() < HOST_FREE:
        raise ProbeError("Host free below 8 GiB before targeted receipt acquisition")
    _owner_limits(runner, ticket)
    model = runner.get_model()
    _class_source(model, "vllm.model_executor.models.gemma4_mm", "Gemma4ForConditionalGeneration")
    if digest(Path(runner.model_config.model) / "config.json") != CONFIG_HASH:
        raise ProbeError("Loaded checkpoint config differs")
    from .speculative_native_probe import OwnedNativeProbe
    from vllm.forward_context import set_forward_context
    proof = OwnedNativeProbe.__new__(OwnedNativeProbe)
    proof.model, proof.admission, proof.startup_forwards = model, admission, None
    with set_forward_context({}, runner.vllm_config, num_tokens=0, skip_compiled=True):
        proof._scale_binding()  # reads existing startup proof; does not install writers or prepare
    closure = inspect.getclosurevars(type(model).forward).nonlocals
    startup = closure["integration_report"]
    fixtures = _sequence(startup["forced_fixtures"], 6, "Existing forced fixtures")
    integration_layers = _sequence(closure["integration_layers"], RECEIPT_LIMITS["cache_layers"], "Startup dispatch layers")
    processor, head = model.language_model.logits_processor, model.language_model.lm_head
    if (processor.head_dtype not in (None, torch.bfloat16, torch.float32)
            or processor.soft_cap != 30.0 or head.weight.dtype != torch.bfloat16):
        raise ProbeError("Actual head policy unsupported")
    head_dtype = processor.head_dtype or torch.bfloat16
    groups, layers, spans, charges = runner.kv_cache_config.kv_cache_groups, [], [], []
    if len(groups) != len(ticket["groups"]):
        raise ProbeError("Core/worker cache group identity differs")
    for gid, group in enumerate(groups):
        size = group.kv_cache_spec.block_size
        if size != ticket["block_sizes"][gid] or size != runner._kernel_block_sizes[gid]:
            raise ProbeError("Manager/kernel page splitting unsupported")
        charge = 0
        for ag in runner.attn_groups[gid]:
            builder = ag.get_metadata_builder()
            _class_source(builder, "vllm.v1.attention.backends.flashinfer", "FlashInferMetadataBuilder")
            if builder.page_size != size:
                raise ProbeError("Builder page differs")
        for name in group.layer_names:
            _text(name)
            spec = group.kv_cache_spec
            if hasattr(spec, "kv_cache_specs"):
                spec = spec.kv_cache_specs[name]
            layer = runner.compilation_config.static_forward_context[name]
            _class_source(layer, "vllm.model_executor.layers.attention.attention", "Attention")
            _class_source(layer.impl, "vllm.v1.attention.backends.flashinfer", "FlashInferImpl")
            cache = layer.kv_cache
            placements = [p for p in runner.kv_cache_config.kv_cache_tensors if name in p.layers]
            if len(placements) != 1 or placements[0].host_resident or cache.dtype != torch.bfloat16:
                raise ProbeError("Actual cache placement unresolved")
            placement, raw = placements[0], cache.untyped_storage()
            if raw.nbytes() != placement.size:
                raise ProbeError("Backing extent differs from actual allocation")
            record, owned = _tensor_record(cache, name), []
            for page in ticket["groups"][gid]:
                begin, end = page_bytes(shape=tuple(cache.shape), strides=tuple(cache.stride()),
                    element_bytes=cache.element_size(), storage_offset_bytes=record["storage_offset_bytes"],
                    page=page, placement_offset=placement.offset, layer_ordinal=placement.layers.index(name),
                    layer_stride=placement.layer_stride, block_stride=placement.block_stride, raw_bytes=raw.nbytes(),
                    allocated_page_bytes=spec.page_size_bytes)
                spans.append((raw.data_ptr(), begin, end))
                owned.append([page, begin, end])
            record.update(group=gid, manager_block_tokens=size, kernel_block_tokens=runner._kernel_block_sizes[gid],
                placement={k: getattr(placement, k) for k in ("size", "offset", "layer_stride", "block_stride", "layers")},
                owned_pages=owned)
            layers.append(record)
            charge += owned[0][2] - owned[0][1]
        charges.append(charge)
    if len(layers) != 30:
        raise ProbeError("Exactly thirty loaded cache owners required")
    disjoint_pages(spans)
    metadata = sum(4 * ((P + 4 + b - 1) // b + 1) + 48 for b in ticket["block_sizes"])
    allocation = allocation_lower_bound(charges, metadata,
        head_element_bytes=2 if head_dtype == torch.bfloat16 else 4, reject=False)
    torch.cuda.synchronize(runner.device)
    allocator = allocator_counters(torch, runner.device)
    allocated, reserved = allocator["allocated_bytes"], allocator["reserved_bytes"]
    gpu_free, gpu_total = torch.cuda.mem_get_info(runner.device)
    receipt = {"schema": "megartx-native-zero-forward-receipt-v1", "purpose": ticket["purpose"],
        "lease_nonce": ticket["nonce"], "worker_pid": os.getpid(), "worker_start": _process_start(os.getpid()),
        "runner_identity": id(runner), "model_identity": id(model), "checkpoint_revision": REVISION,
        "source_sha256": sources, "adapter_source_sha256": admission["adapter_source_sha256"],
        "diagnostic_target_forwards": 0, "existing_startup_target_forwards": proof.startup_forwards,
        "existing_forced_expert_fixture_pairs": len(fixtures),
        "existing_forced_correction_rows": sum(f["forced_correction_rows"] for f in fixtures),
        "startup_dispatch_ledger": [x._megartx["dispatch_calls"] for x in integration_layers],
        "original_native_head_dtype": str(head_dtype), "layers": layers,
        "workspaces": existing_workspaces(runner, torch), "ffi_allocator": ffi_allocator_identity(torch, site_root),
        "allocation_lower_bound": allocation, "allocator": allocator,
        "allocator_counter_source_sha256": TORCH_MEMORY_SOURCE_SHA256,
        "torch_build_identity": torch_build,
        "metadata_acquisition_limits": dict(RECEIPT_LIMITS),
        "gpu_free_bytes": gpu_free, "gpu_total_bytes": gpu_total, "host_free_bytes": _host_free(),
        "external_gpu_workspace_bound_bytes": None,
        "baseline_policy": "model_cache_existing_startup_workspaces_separate_from_incremental_verifier",
        "drained": True}
    receipt["decision"] = fit_decision(allocation, allocated=allocated, reserved=reserved,
        gpu_free=gpu_free, host_free=receipt["host_free_bytes"])
    receipt_preflight(receipt)  # reject before returning a full worker RPC payload
    return receipt


def sanitized_receipt(receipt):
    return {key: receipt[key] for key in ("schema", "diagnostic_target_forwards",
        "existing_startup_target_forwards", "existing_forced_expert_fixture_pairs", "existing_forced_correction_rows",
        "original_native_head_dtype", "allocation_lower_bound", "gpu_free_bytes", "host_free_bytes", "decision")}
