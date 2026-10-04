"""Existing-owner receipt: no target call, lazy workspace getter or sizing JIT.

The full receipt contains private pointers. Only sanitized_receipt is exported
by the EngineCore utility. Missing external allocation bounds are never zero.
"""

import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

from .speculative_native_probe import (CONFIG_HASH, GPU_CAP, GPU_FREE, HOST_FREE,
    P, REVISION, ProbeError, _class_source, _host_free, _integer, _process_start,
    allocation_lower_bound, digest, inspect_sources)

FFI_SOURCES = {
    "tvm_ffi/include/dlpack/dlpack.h": "62052bef24bd69bb7b52e95428f9e99bd7951783850c36c5f2a41e71fd63c2fc",
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py": "00642488be9a3bdc9560f14fa5d5b73172e096e89b09ffc35a7f18225feae9ce",
    "tvm_ffi/cython/tvm_ffi_python_helpers.h": "a11be0560a2cc4b75845ab63fc7cb67f0dd890b8d89c32b58d7efb408172089f"}


def receipt_digest(receipt):
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 8 << 20:
        raise ProbeError("Private receipt exceeds 8 MiB host metadata bound")
    return hashlib.sha256(encoded).hexdigest()


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
    raw = tensor.untyped_storage()
    return {"owner": owner, "device": str(tensor.device), "dtype": str(tensor.dtype),
        "storage_ptr": raw.data_ptr(), "storage_bytes": raw.nbytes(),
        "storage_offset_bytes": tensor.storage_offset() * tensor.element_size(),
        "shape": list(tensor.shape), "strides": list(tensor.stride())}


def existing_workspaces(runner, torch):
    """Read fields of existing builders/wrappers; never initialize a missing owner."""
    records, missing, visited = [], [], set()
    def scan(owner, label, depth=0):
        if owner is None:
            missing.append(label)
            return
        if id(owner) in visited or depth > 3:
            return
        visited.add(id(owner))
        for name, value in vars(owner).items():
            if isinstance(value, torch.Tensor):
                records.append(_tensor_record(value, label + "." + name))
            elif (value is not None and hasattr(value, "__dict__") and
                    type(value).__module__.startswith(("flashinfer", "vllm.v1.attention.backends"))):
                scan(value, label + "." + name, depth + 1)
    for gid, groups in enumerate(runner.attn_groups):
        for aid, group in enumerate(groups):
            builder = group.get_metadata_builder()
            label = f"attention.{gid}.{aid}"
            scan(builder, label)
            for name in ("_prefill_wrapper", "_decode_wrapper", "_noncausal_prefill_wrapper", "_cascade_wrapper"):
                scan(vars(builder).get(name), label + "." + name)
            for key, wrapper in vars(builder).get("_decode_wrappers_cudagraph", {}).items():
                scan(wrapper, label + "._decode_wrappers_cudagraph." + str(key))
    for module_name, field in (("vllm.v1.attention.backends.flashinfer", "trtllm_workspace_buffer"),
                               ("flashinfer.utils", "_cache_buf")):
        module = sys.modules.get(module_name)
        value = vars(module).get(field) if module else None
        if isinstance(value, torch.Tensor):
            records.append(_tensor_record(value, module_name + "." + field))
        elif isinstance(value, dict):
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    records.append(_tensor_record(tensor, module_name + "." + field + "." + str(key)))
    module = sys.modules.get("vllm.v1.worker.workspace")
    manager = vars(module).get("_manager") if module else None
    if manager:
        for index, tensor in enumerate(vars(manager).get("_current_workspaces", ())):
            if tensor is not None:
                records.append(_tensor_record(tensor, f"vllm.workspace.{index}"))
    unique = {(r["device"], r["storage_ptr"]): r["storage_bytes"] for r in records}
    return {"owners": records, "deduplicated_storage_bytes": sum(unique.values()),
        "deduplicated_GPU_storage_bytes": sum(size for (device, _), size in unique.items() if device.startswith("cuda")),
        "deduplicated_host_storage_bytes": sum(size for (device, _), size in unique.items() if device == "cpu"),
        "missing_lazy_owners": missing, "workspace_manager_locked": vars(manager).get("_locked") if manager else None,
        "required_size_M1_M2_M256_bytes": None,
        "required_size_status": "unqueried_device_JIT_or_new_allocation_is_outside_zero_forward_phase"}


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
    for line in Path("/proc/self/maps").read_text().splitlines():
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
    import importlib.metadata
    for package, version in (("vllm", "0.30.0"), ("flashinfer-python", "0.6.18.post1"), ("torch", "2.13.0+cu130")):
        if importlib.metadata.version(package) != version:
            raise ProbeError("Installed package differs: " + package)
    import torch
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
    fixtures = startup["forced_fixtures"]
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
    allocated, reserved = torch.cuda.memory_allocated(runner.device), torch.cuda.memory_reserved(runner.device)
    stats = torch.cuda.memory_stats(runner.device)
    snapshot = torch.cuda.memory_snapshot()
    gpu_free, gpu_total = torch.cuda.mem_get_info(runner.device)
    receipt = {"schema": "megartx-native-zero-forward-receipt-v1", "purpose": ticket["purpose"],
        "lease_nonce": ticket["nonce"], "worker_pid": os.getpid(), "worker_start": _process_start(os.getpid()),
        "runner_identity": id(runner), "model_identity": id(model), "checkpoint_revision": REVISION,
        "source_sha256": sources, "adapter_source_sha256": admission["adapter_source_sha256"],
        "diagnostic_target_forwards": 0, "existing_startup_target_forwards": proof.startup_forwards,
        "existing_forced_expert_fixture_pairs": len(fixtures),
        "existing_forced_correction_rows": sum(f["forced_correction_rows"] for f in fixtures),
        "startup_dispatch_ledger": [x._megartx["dispatch_calls"] for x in closure["integration_layers"]],
        "original_native_head_dtype": str(head_dtype), "layers": layers,
        "workspaces": existing_workspaces(runner, torch), "ffi_allocator": ffi_allocator_identity(torch, site_root),
        "allocation_lower_bound": allocation, "allocator": {"backend": torch.cuda.get_allocator_backend(),
            "allocated_bytes": allocated, "reserved_bytes": reserved, "cached_slack_bytes": reserved - allocated,
            "allocated_peak_bytes": stats.get("allocated_bytes.all.peak"),
            "reserved_peak_bytes": stats.get("reserved_bytes.all.peak"), "snapshot_segments": len(snapshot),
            "snapshot_sha256": receipt_digest(snapshot)},
        "gpu_free_bytes": gpu_free, "gpu_total_bytes": gpu_total, "host_free_bytes": _host_free(),
        "external_gpu_workspace_bound_bytes": None,
        "baseline_policy": "model_cache_existing_startup_workspaces_separate_from_incremental_verifier",
        "drained": True}
    receipt["decision"] = fit_decision(allocation, allocated=allocated, reserved=reserved,
        gpu_free=gpu_free, host_free=receipt["host_free_bytes"])
    return receipt


def sanitized_receipt(receipt):
    return {key: receipt[key] for key in ("schema", "diagnostic_target_forwards",
        "existing_startup_target_forwards", "existing_forced_expert_fixture_pairs", "existing_forced_correction_rows",
        "original_native_head_dtype", "allocation_lower_bound", "gpu_free_bytes", "host_free_bytes", "decision")}
