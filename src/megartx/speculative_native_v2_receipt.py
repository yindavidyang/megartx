"""Target-only V2 allocation receipt; never prepares or forwards a model batch."""
import inspect
import os
from pathlib import Path
import sys

from .speculative_native_v2 import V2Owner
from .speculative_native_probe import (CONFIG_HASH, GPU_CAP, GPU_FREE, HOST_FREE, P, REVISION,
    ProbeError, _class_source, _host_free, _process_start, allocation_lower_bound, check_torch_build, digest)
from .speculative_native_receipt import (TORCH_MEMORY_SOURCE_SHA256, RECEIPT_LIMITS,
    _owner_limits, _sequence, _text, _tensor_record, page_bytes, disjoint_pages,
    allocator_counters, existing_workspaces, ffi_allocator_identity, fit_decision, receipt_preflight)

def collect_receipt(owner, ticket, admission):
    """Existing V2 owners only; lifecycle constructs/authenticates V2Owner."""
    if type(owner) is not V2Owner:
        raise ProbeError("Actual V2 owner binding required")
    owner.check()
    runner = owner.runner
    if sys.version_info[:3] != (3, 12, 3):
        raise ProbeError("Pinned Python 3.12.3 required")
    site_root, sources = owner.root, owner.sources
    if digest(site_root / "torch/cuda/memory.py") != TORCH_MEMORY_SOURCE_SHA256:
        raise ProbeError("Installed allocator counter source differs")
    import importlib.metadata
    for package, version in (("vllm", "0.30.0"), ("flashinfer-python", "0.6.18.post1")):
        if importlib.metadata.version(package) != version:
            raise ProbeError("Installed package differs: " + package)
    import torch
    torch_build = check_torch_build(site_root, torch)
    if (not torch.is_inference_mode_enabled() or runner.req_states.req_id_to_index
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
        if size != ticket["block_sizes"][gid] or size != runner.kernel_block_sizes[gid]:
            raise ProbeError("Manager/kernel page splitting unsupported")
        charge = 0
        for ag in runner.attn_groups[gid]:
            builder = ag.get_metadata_builder(0)
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
            if (len(placements) != 1 or placements[0].host_resident
                    or not isinstance(cache, torch.Tensor) or cache.dtype != torch.bfloat16):
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
            record.update(group=gid, manager_block_tokens=size, kernel_block_tokens=runner.kernel_block_sizes[gid],
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
    receipt = {"schema": "megartx-native-v2-zero-forward-receipt-v1", "purpose": ticket["purpose"],
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
    receipt["v2_owner"] = owner.private_identity()
    receipt["decision"]["admitted"] = False
    receipt["decision"]["blockers"].append("V2_verifier_and_drafter_interfaces_not_admitted")
    owner.seal_storage(layers)
    owner.check()
    receipt_preflight(receipt)  # reject before returning a full worker RPC payload
    return receipt
