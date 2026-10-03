"""Default-off, source-bound owned Gemma diagnostic. No runtime imports at import.

This is deliberately separate from the CPU synthetic verifier. EngineCore owns
the real BlockPool reservation; the Worker owns model calls and their stream.
Neither an observational frame ticket nor supplied mock providers can launch it.
The public CLI only freezes/inspects sources. A reviewed lifecycle must install
the EngineCore utility and Worker extension before this adapter can be invoked.
"""

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid


BASE = "464c728a8758a558fd0bfe326f3ed7a2f10fe0c4"
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
CONFIG_HASH = "4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98"
GPU_CAP = 8 << 20
GPU_FREE = 2 << 30
HOST_FREE = 8 << 30
TRACE_CAP = 64 << 20
P = 2048
VOCAB = 262144
HEAD_PEAK = 2 * VOCAB * 4  # softcap rebind: old and new FP32 outputs coexist


class ProbeError(RuntimeError):
    pass


def digest(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ProbeError("Missing or oversized source")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    return json.loads((Path(__file__).resolve().parents[2] /
                       "docs/evidence/speculative-native-source-binding.json").read_text())


def inspect_sources(root):
    """Read files only; safe under python -S, without importing Torch/vLLM."""
    root = Path(root)
    files = source_manifest()["files"]
    result = {}
    for name, binding in files.items():
        if binding.get("matches") is not True:
            raise ProbeError("Installed/upstream source identity is unresolved: " + name)
        actual = digest(root / name)
        if actual != binding["sha256"]:
            raise ProbeError("Installed source changed: " + name)
        result[name] = actual
    return result


def _class_source(obj, module, name):
    cls = type(obj)
    # Worker extension changes bases, not the class's source identity.
    if cls.__module__ != module or cls.__name__ != name:
        raise ProbeError("Unexpected native owner class: " + name)
    path = inspect.getsourcefile(cls)
    expected = source_manifest()["files"][module.replace(".", "/") + ".py"]["sha256"]
    if path is None or digest(path) != expected:
        raise ProbeError("Native owner source changed: " + name)


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ProbeError("Invalid integer in native contract")
    return value


@dataclass(frozen=True)
class PagePlan:
    table: tuple
    slots: tuple
    copies: tuple  # (old page, private COW page, committed fragment length)
    private: tuple


def plan_pages(table, reserved, *, cached, rows, block_size):
    """Only choose from actual BlockPool-owned reservation, never free-ID guesses."""
    cached, rows, block_size = _integer(cached), _integer(rows, 1), _integer(block_size, 1)
    if rows > 2 or block_size not in (16, 32, 64):
        raise ProbeError("Unreviewed short-row/page shape")
    table, reserved = tuple(table), tuple(reserved)
    if (len(table) != (cached + block_size - 1) // block_size
            or len(set(reserved)) != len(reserved)
            or any(type(b) is not int or b < 1 for b in reserved)
            or len(set(table)) != len(table) or not set(table) <= set(reserved)):
        raise ProbeError("Invalid authoritative table/reservation")
    first = cached // block_size
    last = (cached + rows - 1) // block_size
    available = [b for b in reserved if b not in table]
    needed = last - first + 1
    if len(available) < needed:
        raise ProbeError("Actual reservation cannot stage complete query")
    private = tuple(available[:needed])
    new = list(table[:first]) + list(private)
    copies = () if cached % block_size == 0 else ((table[first], private[0], cached % block_size),)
    slots = tuple(new[p // block_size] * block_size + p % block_size
                  for p in range(cached, cached + rows))
    return PagePlan(tuple(new), slots, copies, private)


def consumed_table(plan, *, cached, consumed, block_size):
    consumed = _integer(consumed, 1)
    if consumed > len(plan.slots):
        raise ProbeError("Publication exceeds completed input rows")
    return plan.table[:(cached + consumed + block_size - 1) // block_size]


def host_frame(plan, *, cached, rows):
    """Exact host fields consumed by the installed CommonAttentionMetadata."""
    cached, rows = _integer(cached), _integer(rows, 1)
    if len(plan.slots) != rows or len(set(plan.slots)) != rows:
        raise ProbeError("Frame rows do not match actual private slots")
    return {"query_start_loc": (0, rows), "seq_lens": (cached + rows,),
            "num_reqs": 1, "num_actual_tokens": rows, "max_query_len": rows,
            "max_seq_len": cached + rows, "block_table": plan.table,
            "slot_mapping": plan.slots, "positions": tuple(range(cached, cached + rows)),
            "causal": True}


def select_greedy(anchor, candidates, argmax, remaining, eos=(1, 106)):
    """Anchor is already emitted; return NEW emissions and consumed input count."""
    _integer(anchor)
    remaining = _integer(remaining, 1)
    candidates, argmax = tuple(candidates), tuple(argmax)
    if len(candidates) > 1 or len(argmax) != len(candidates) + 1:
        raise ProbeError("Native probe supports only k0/k1")
    if any(type(t) is not int or not 0 <= t < VOCAB for t in (anchor,) + candidates + argmax):
        raise ProbeError("Invalid target token ID")
    emitted, accepted = [], 0
    for j, y in enumerate(candidates):
        token = y if y == argmax[j] else argmax[j]
        emitted.append(token)
        if y == argmax[j]:
            accepted += 1
        if y != argmax[j] or token in eos or len(emitted) == remaining:
            return tuple(emitted), accepted, len(emitted), token in eos or len(emitted) == remaining
    emitted.append(argmax[-1])
    return tuple(emitted), accepted, len(emitted), emitted[-1] in eos or len(emitted) == remaining


def allocation_lower_bound(group_bytes, metadata_bytes, hidden_rows=2):
    """Actual padded manager pages, head coexistence and retained hidden rows.

    This is a rejection bound, not admission: MoE/attention/allocator peaks are
    additionally checked against the same cap during the owned diagnostic.
    """
    group_bytes = tuple(_integer(n, 1) for n in group_bytes)
    metadata_bytes = _integer(metadata_bytes)
    hidden_rows = _integer(hidden_rows, 1)
    result = {"private_page_bytes": sum(group_bytes), "head_peak_bytes": HEAD_PEAK,
              "hidden_bytes": hidden_rows * 2816 * 2, "metadata_bytes": metadata_bytes}
    result["known_bytes"] = sum(result.values())
    result["remaining_for_native_scratch_and_allocator"] = GPU_CAP - result["known_bytes"]
    if result["known_bytes"] > GPU_CAP:
        raise ProbeError("Actual page padding/head/metadata already exceeds 8 MiB")
    return result


def _process_start(pid):
    # Pinned execution host is Linux; PID alone is not an ownership identity.
    return Path(f"/proc/{_integer(pid, 1)}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _host_free():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


def check_admission(admission):
    """Lifecycle-generated evidence; not an automatic approval mechanism."""
    protocol = Path(__file__).resolve().parents[2] / "docs/design/speculative-native-protocol.json"
    own = Path(__file__).resolve()
    required = {"protocol_sha256": digest(protocol), "implementation_sha256": digest(own),
                "compiler_memory_bytes": 2 << 30, "compiler_timeout_seconds": 300,
                "compiler_scope": "shared_host", "max_concurrent_gpu_jobs": 1,
                "max_extra_gpu_bytes": GPU_CAP, "owned_lifecycle_verified": True,
                "independent_review_clear": True, "parent_source_protocol_accepted": True}
    if not isinstance(admission, dict) or any(admission.get(k) != v for k, v in required.items()):
        raise ProbeError("Missing exact reviewed source/protocol/lifecycle admission")
    if (not isinstance(admission.get("parent_slot"), str) or not admission["parent_slot"]
            or type(admission.get("deadline_monotonic")) not in (int, float)
            or not time.monotonic() < admission["deadline_monotonic"] <= time.monotonic() + 1800):
        raise ProbeError("Missing sole-owner slot or bounded deadline")
    _integer(admission.get("external_gpu_workspace_bound_bytes"))
    if (not isinstance(admission.get("external_gpu_workspace_receipt_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", admission["external_gpu_workspace_receipt_sha256"]) is None):
        raise ProbeError("Missing source-bound external CUDA workspace receipt")
    return admission


def run_engine_core_probe(core, prompt, admission, *, enabled=False):
    """Smallest EngineCore utility exposure; MUST be called by its utility loop.

    Launcher first pauses an EMPTY fresh engine with mode=keep, clear_cache=False.
    The synchronous utility loop cannot schedule/add another request during RPC.
    On RPC/drain uncertainty retain allocator references and poison the engine;
    only owned process teardown may dispose of them. Never resume this engine.
    """
    if enabled is not True:
        raise ProbeError("Native diagnostic is default-off")
    check_admission(admission)
    # EngineCoreProc inherits EngineCore; check the defining source directly.
    if not any(c.__module__ == "vllm.v1.engine.core" and c.__name__ == "EngineCore"
               for c in type(core).__mro__):
        raise ProbeError("No actual EngineCore owner")
    inspect_sources(Path(inspect.getsourcefile(type(core))).resolve().parents[3])
    if (getattr(core, "_megartx_native_used", False) or not core.is_scheduler_paused()
            or core.scheduler.has_requests() or core.batch_queue
            or getattr(core, "pending_add_requests", ())):
        raise ProbeError("Exclusive empty paused EngineCore required")
    core._megartx_native_used = True
    core._megartx_native_poisoned = True
    manager = core.scheduler.kv_cache_manager
    _class_source(manager, "vllm.v1.core.kv_cache_manager", "KVCacheManager")
    pool = manager.block_pool
    _class_source(pool, "vllm.v1.core.block_pool", "BlockPool")
    config = core.vllm_config
    if (config.cache_config.enable_prefix_caching or config.speculative_config is not None
            or config.parallel_config.world_size != 1 or config.use_v2_model_runner
            or config.kv_transfer_config is not None or config.ec_transfer_config is not None
            or config.parallel_config.enable_dbo):
        raise ProbeError("Incompatible serving/transfer/speculative owner")
    groups = core.scheduler.kv_cache_config.kv_cache_groups
    sizes = [g.kv_cache_spec.block_size for g in groups]
    if any(b not in (16, 32, 64) for b in sizes):
        raise ProbeError("Unreviewed actual page size")
    # Full 2048 prefix retained, no scheduler sliding-ring recycling. Reserve the
    # serial continuation capacity plus ONE independently charged COW page/group.
    counts = [(P + 4 + b - 1) // b + 1 for b in sizes]
    blocks = pool.get_new_blocks(sum(counts))
    drained = False
    ticket = {"nonce": str(uuid.uuid4()), "engine_pid": os.getpid(),
              "engine_start": _process_start(os.getpid()), "block_sizes": sizes, "groups": []}
    try:
        if any(b.ref_cnt != 1 or b.is_null for b in blocks):
            raise ProbeError("BlockPool did not grant sole reservation")
        cursor = 0
        for count in counts:
            ticket["groups"].append([b.block_id for b in blocks[cursor:cursor + count]])
            cursor += count
        result = core.model_executor.collective_rpc(
            "megartx_native_probe", timeout=min(900, admission["deadline_monotonic"] - time.monotonic()),
            kwargs={"ticket": ticket, "prompt": list(prompt), "admission": admission})
        if len(result) != 1 or result[0].get("drained") is not True:
            raise ProbeError("Worker did not confirm owned device drainage")
        drained = True
        core._megartx_native_poisoned = False
        return result[0]
    finally:
        if not drained:
            try:
                core.model_executor.collective_rpc("synchronize_device", timeout=30)
                drained = True
            except BaseException:
                # Do not reuse uncertain pages or unpause a poisoned engine.
                core._megartx_native_retained_blocks = blocks
        if drained:
            pool.free_blocks(blocks)


@dataclass(frozen=True)
class NativeState:
    history: tuple
    tables: tuple
    terminal: bool = False

    @property
    def cached(self):
        return len(self.history) - 1


class NativeProbeWorkerExtension:
    """Pass via native --worker-extension-cls; installed vLLM files stay intact."""

    def megartx_native_probe(self, ticket, prompt, admission):
        check_admission(admission)
        _class_source(self, "vllm.v1.worker.gpu_worker", "Worker")
        if getattr(self, "_megartx_native_used", False):
            raise ProbeError("Worker ticket is single-use")
        self._megartx_native_used = True
        probe = OwnedNativeProbe.from_runner(self.model_runner, ticket, admission, enabled=True)
        try:
            return probe.run(prompt)
        finally:
            probe.close()  # restore instance writers only after owned device drain


class OwnedNativeProbe:
    """Concrete installed-model binding; no injected model/cache/head providers."""

    @classmethod
    def from_runner(cls, runner, ticket, admission, *, enabled=False):
        if enabled is not True:
            raise ProbeError("Native diagnostic is default-off")
        check_admission(admission)
        _class_source(runner, "vllm.v1.worker.gpu_model_runner", "GPUModelRunner")
        root = Path(inspect.getsourcefile(type(runner))).resolve().parents[3]
        inspect_sources(root)
        if sys.version_info[:3] != (3, 12, 3):
            raise ProbeError("Pinned Python 3.12.3 required")
        if (not isinstance(ticket, dict) or _process_start(ticket["engine_pid"]) != ticket["engine_start"]
                or str(uuid.UUID(ticket["nonce"])) != ticket["nonce"]):
            raise ProbeError("Stale EngineCore reservation identity")
        if (_host_free() < HOST_FREE or runner.input_batch.num_reqs != 0
                or runner.vllm_config.speculative_config is not None
                or runner.cache_config.cache_dtype != "bfloat16"
                or runner.parallel_config.world_size != 1
                or not runner.model_config.enforce_eager):
            raise ProbeError("Native runner/resource/empty-batch contract differs")
        forbidden = ("MEGARTX_M1_PREPARATION", "MEGARTX_CONTROLLED_DIR", "MEGARTX_M1_NORMAL_DIR",
                     "MEGARTX_LOGITS_DIR", "MEGARTX_ROUTER_SCORE_DIR", "MEGARTX_ROUTE_AUDIT_PATH")
        if (os.environ.get("MEGARTX_SCALE_MODE") != "native"
                or os.environ.get("VLLM_PLUGINS") != "megartx_scale_adapter"
                or any(os.environ.get(k) for k in forbidden)):
            raise ProbeError("Original scale-corrected lane/M1-off contract differs")
        import importlib.metadata
        for package, version in (("vllm", "0.30.0"), ("flashinfer-python", "0.6.18.post1"),
                                 ("torch", "2.13.0+cu130")):
            if importlib.metadata.version(package) != version:
                raise ProbeError("Installed package differs: " + package)
        import torch
        model = runner.get_model()
        _class_source(model, "vllm.model_executor.models.gemma4_mm", "Gemma4ForConditionalGeneration")
        checkpoint = Path(runner.model_config.model)
        if digest(checkpoint / "config.json") != CONFIG_HASH:
            raise ProbeError("Actual loaded model config differs")
        # Launcher must verify immutable original shards/tokenizer/template against
        # the existing checkpoint manifest; a config hash alone is insufficient.
        if (admission.get("checkpoint_revision") != REVISION
                or admission.get("checkpoint_manifest_verified") is not True
                or admission.get("tokenizer_template_verified") is not True):
            raise ProbeError("Missing loaded checkpoint/tokenizer identity receipt")
        processor = model.language_model.logits_processor
        head = model.language_model.lm_head
        if (processor.head_dtype != torch.float32 or processor.soft_cap != 30.0
                or processor.scale != 1.0 or processor.logits_as_input
                or head.weight.dtype != torch.bfloat16 or head.tp_size != 1):
            raise ProbeError("Actual head would change math or materialize expanded FP32 weights")
        obj = cls.__new__(cls)
        obj.torch, obj.runner, obj.model = torch, runner, model
        obj.admission, obj.ticket = admission, ticket
        obj.layers, obj.builders, obj.sizes, obj.reserved = {}, [], [], []
        obj.writer_originals, obj.frame, obj.closed, obj.failed = [], None, False, False
        obj.forwards, obj.peak_extra, obj.peak_reserved_extra = 0, 0, 0
        obj.cancelled = False
        groups = runner.kv_cache_config.kv_cache_groups
        if len(groups) != len(ticket["groups"]) or len(groups) != len(ticket["block_sizes"]):
            raise ProbeError("Reservation group identity differs")
        all_ids = [b for g in ticket["groups"] for b in g]
        if len(set(all_ids)) != len(all_ids) or any(type(b) is not int or b < 1 for b in all_ids):
            raise ProbeError("Aliased or null BlockPool reservation")
        spans, group_bytes = [], []
        for gid, group in enumerate(groups):
            b = group.kv_cache_spec.block_size
            if b != ticket["block_sizes"][gid] or runner._kernel_block_sizes[gid] != b:
                raise ProbeError("Manager/kernel page splitting requires separate adapter")
            owned = tuple(ticket["groups"][gid])
            if len(owned) != (P + 4 + b - 1) // b + 1:
                raise ProbeError("BlockPool reservation capacity differs")
            obj.sizes.append(b)
            obj.reserved.append(owned)
            charge = 0
            for ag in runner.attn_groups[gid]:
                builder = ag.get_metadata_builder()
                _class_source(builder, "vllm.v1.attention.backends.flashinfer", "FlashInferMetadataBuilder")
                if builder.page_size != b:
                    raise ProbeError("Builder actual page differs")
                obj.builders.append((gid, tuple(ag.layer_names), builder))
            for name in group.layer_names:
                match = re.search(r"layers\.(\d+)\..*attn$", name)
                if match is None or int(match[1]) in obj.layers:
                    raise ProbeError("Missing/duplicate actual Gemma layer")
                index = int(match[1])
                layer = runner.compilation_config.static_forward_context[name]
                _class_source(layer, "vllm.model_executor.layers.attention.attention", "Attention")
                _class_source(layer.impl, "vllm.v1.attention.backends.flashinfer", "FlashInferImpl")
                cache = layer.kv_cache
                heads, width = (2, 512) if index % 6 == 5 else (8, 256)
                if (cache.dtype != torch.bfloat16 or len(cache.shape) != 4
                        or tuple(cache.shape[1:]) != (heads, b, 2 * width)
                        or cache.stride(-1) != 1 or max(owned) >= cache.shape[0]
                        or layer.impl.kv_sharing_target_layer_name is not None
                        or layer.kv_cache_dtype != "bfloat16"):
                    raise ProbeError("Actual distinct BF16 BHNC cache contract differs")
                placements = [t for t in runner.kv_cache_config.kv_cache_tensors if name in t.layers]
                if len(placements) != 1 or placements[0].host_resident:
                    raise ProbeError("Unresolved actual allocator placement")
                placement = placements[0]
                page_span = cache.stride(0) * 2
                # Conservative padding charge. Interleaved layouts may fail this
                # bound; never reinterpret logical payload as physical allocation.
                charge += page_span
                raw = cache.untyped_storage()
                for page in owned:
                    start = cache.storage_offset() * 2 + page * page_span
                    end = start + ((heads - 1) * cache.stride(1)
                                   + (b - 1) * cache.stride(2) + 2 * width) * 2
                    if start < 0 or end > raw.nbytes() or end > placement.size:
                        raise ProbeError("Cache page exceeds actual backing byte bounds")
                    spans.append((raw.data_ptr(), start, end))
                obj.layers[index] = (gid, name, layer, cache, width)
            group_bytes.append(charge)
        if set(obj.layers) != set(range(30)):
            raise ProbeError("Exactly thirty actual layer/cache owners required")
        for left, right in zip(sorted(spans), sorted(spans)[1:]):
            if left[0] == right[0] and left[2] > right[1]:
                raise ProbeError("Reserved layer/page storage aliases")
        metadata = sum(4 * ((P + 4 + b - 1) // b + 1) + 48 for b in obj.sizes)
        obj.allocation = allocation_lower_bound(group_bytes, metadata)
        external = admission["external_gpu_workspace_bound_bytes"]
        if obj.allocation["known_bytes"] + external > GPU_CAP:
            raise ProbeError("Actual allocation plus external workspace cannot fit cap")
        obj.allocation["external_gpu_workspace_bound_bytes"] = external
        try:
            obj._install_writers()
        except BaseException:
            obj.close()
            raise
        return obj

    def _check(self):
        if (self.failed or self.closed or time.monotonic() >= self.admission["deadline_monotonic"]
                or self.forwards >= 96 or _host_free() < HOST_FREE):
            raise ProbeError("Poisoned/expired/bounded native lease")
        if self.torch.cuda.mem_get_info(self.runner.device)[0] < GPU_FREE:
            raise ProbeError("Less than 2 GiB free GPU reserve")

    def _cpu(self, tensor):
        # Device->host copies of strided individual views, no full-cache gather
        # or GPU contiguous() temporary. Returned bytes remain private.
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def cancel(self):
        """Signal cancellation; submitted work must drain before disposal."""
        self.cancelled = True

    def _install_writers(self):
        for index, (_, _, layer, cache, width) in self.layers.items():
            impl = layer.impl
            if any(owner is impl for owner, _, _ in self.writer_originals):
                raise ProbeError("Shared writer instance needs separate binding")
            had = "do_kv_cache_update" in impl.__dict__
            saved = impl.__dict__.get("do_kv_cache_update")
            original = impl.do_kv_cache_update

            def writer(actual_layer, key, value, actual_cache, slots,
                       index=index, cache=cache, width=width, original=original):
                frame = self.frame
                if frame is None or index in frame["kv"] or actual_cache is not cache:
                    raise ProbeError("Stale/duplicate/unowned cache writer")
                gid = self.layers[index][0]
                plan = frame["plans"][gid]
                if tuple(self._cpu(slots).tolist()) != plan.slots:
                    raise ProbeError("Actual writer mapping differs from private page table")
                # One layer at a time: <=16 MiB before +16 MiB after at P2048,
                # below 64 MiB host trace ceiling. No all-layer prefix snapshots.
                old = [(page, self._cpu(cache[page])) for page in frame["old_tables"][gid]]
                if sum(t.numel() * t.element_size() for _, t in old) * 2 > TRACE_CAP - (8 << 20):
                    raise ProbeError("Old-byte comparison exceeds bounded host trace")
                n = len(plan.slots)
                k, v = self._cpu(key[:n]).reshape(n, cache.shape[1], width), self._cpu(value[:n]).reshape(n, cache.shape[1], width)
                if not self.torch.isfinite(k).all() or not self.torch.isfinite(v).all():
                    raise ProbeError("Processed native K/V is nonfinite")
                original(actual_layer, key, value, actual_cache, slots)
                self.torch.cuda.current_stream(self.runner.device).synchronize()
                for page, before in old:
                    if not self.torch.equal(before.view(self.torch.uint8), self._cpu(cache[page]).view(self.torch.uint8)):
                        raise ProbeError("Writer changed old committed page bytes")
                actual = self.torch.stack([self._cpu(cache[s // cache.shape[2], :, s % cache.shape[2], :]) for s in plan.slots])
                if (not self.torch.equal(k.view(self.torch.uint8), actual[:, :, :width].contiguous().view(self.torch.uint8))
                        or not self.torch.equal(v.view(self.torch.uint8), actual[:, :, width:].contiguous().view(self.torch.uint8))):
                    raise ProbeError("Actual staged cache differs from distinct processed K/V")
                frame["kv"][index] = actual[-frame["keep_rows"]:].clone()
            impl.do_kv_cache_update = writer
            self.writer_originals.append((impl, had, saved))

    def close(self):
        if self.closed:
            return
        # A failure here leaves writers installed and the worker poisoned until
        # owned teardown, avoiding use-after-free of callbacks or reservations.
        self.torch.cuda.synchronize(self.runner.device)
        self.frame = None
        for impl, had, saved in reversed(self.writer_originals):
            if had:
                impl.do_kv_cache_update = saved
            else:
                del impl.do_kv_cache_update
        self.closed = True

    def _forward(self, inputs, cached, tables, *, diagnostic=True, cancel_at=None,
                 prefill=False, head=True):
        self._check()
        t = self.torch
        n = len(inputs)
        if prefill:
            if n != 256 or cached % 256 or not 0 <= cached < P:
                raise ProbeError("Prefill requires exact eight aligned 256-row chunks")
            plans = []
            for table, owned, b in zip(tables, self.reserved, self.sizes):
                new = owned[:(cached + n) // b]
                if tuple(table) != new[:cached // b]:
                    raise ProbeError("Prefill prefix differs from actual reservation")
                slots = tuple(new[p // b] * b + p % b for p in range(cached, cached + n))
                plans.append(PagePlan(new, slots, (), new[cached // b:]))
        else:
            plans = [plan_pages(table, owned, cached=cached, rows=n, block_size=b)
                     for table, owned, b in zip(tables, self.reserved, self.sizes)]
        if self.frame is not None:
            raise ProbeError("Reentrant native transaction")
        baseline_a, baseline_r = t.cuda.memory_allocated(), t.cuda.memory_reserved()
        fraction = None
        external = self.allocation["external_gpu_workspace_bound_bytes"]
        if diagnostic:
            # Bound the native PyTorch allocator BEFORE any query allocation,
            # including cached-block/rounding growth. External CUDA libraries
            # require a separate frozen source-bound workspace upper bound.
            if t.cuda.get_allocator_backend() != "native":
                raise ProbeError("Unreviewed allocator cannot enforce incremental cap")
            t.cuda.synchronize(self.runner.device)
            t.cuda.empty_cache()
            baseline_a, baseline_r = t.cuda.memory_allocated(), t.cuda.memory_reserved()
            if baseline_a != baseline_r:
                raise ProbeError("Live allocator padding makes incremental baseline ambiguous")
            fraction = t.cuda.get_per_process_memory_fraction(self.runner.device)
            total = t.cuda.get_device_properties(self.runner.device).total_memory
            limit = baseline_r + GPU_CAP - self.allocation["private_page_bytes"] - external
            t.cuda.set_per_process_memory_fraction(min(fraction, limit / total), self.runner.device)
        t.cuda.reset_peak_memory_stats()
        metadata, mappings, commons = {}, {}, {}
        self.frame = {"plans": plans, "old_tables": tables, "kv": {}, "keep_rows": 1 if prefill else n}
        try:
            from vllm.v1.attention.backend import CommonAttentionMetadata
            from vllm.forward_context import set_forward_context
            ids = t.tensor(inputs, dtype=t.int64, device=self.runner.device)
            positions = t.arange(cached, cached + n, dtype=t.int64, device=self.runner.device)
            for gid, plan in enumerate(plans):
                frame = host_frame(plan, cached=cached, rows=n)
                for _, (group, _, _, cache, _) in self.layers.items():
                    if group == gid:
                        for old, private, fragment in plan.copies:
                            cache[private].zero_()
                            cache[private, :, :fragment, :].copy_(cache[old, :, :fragment, :])
                cpu_q = t.tensor(frame["query_start_loc"], dtype=t.int32)
                cpu_seq = t.tensor(frame["seq_lens"], dtype=t.int32)
                commons[gid] = CommonAttentionMetadata(
                    query_start_loc=cpu_q.to(self.runner.device), query_start_loc_cpu=cpu_q,
                    seq_lens=cpu_seq.to(self.runner.device), seq_lens_cpu_upper_bound=cpu_seq,
                    num_reqs=1, num_actual_tokens=n, max_query_len=n, max_seq_len=cached + n,
                    block_table_tensor=t.tensor([plan.table], dtype=t.int32, device=self.runner.device),
                    slot_mapping=t.tensor(plan.slots, dtype=t.int64, device=self.runner.device),
                    causal=True, positions=positions)
            for gid, names, builder in self.builders:
                # Reuse the actual initialized owner/workspace; a new wrapper may
                # still need scratch, which belongs to the diagnostic peak cap.
                if diagnostic and builder._workspace_buffer is None:
                    raise ProbeError("Uninitialized attention workspace is not incremental 8 MiB work")
                meta = builder.build(0, commons[gid])
                if meta.num_actual_tokens != n or not meta.causal or meta.use_cascade:
                    raise ProbeError("Actual metadata is not one causal unpadded query")
                for name in names:
                    metadata[name], mappings[name] = meta, commons[gid].slot_mapping
            with set_forward_context(metadata, self.runner.vllm_config, num_tokens=n,
                                     slot_mapping=mappings, skip_compiled=True):
                hidden = self.model(input_ids=ids, positions=positions)
                if tuple(hidden.shape) != (n, 2816) or hidden.dtype != t.bfloat16:
                    raise ProbeError("Missing actual hidden rows for every target input")
                self.forwards += 1
                # Retain both actual hidden rows until their individual heads
                # complete; never use runner's last-row ordinary selection.
                host_hidden = self._cpu(hidden[-1:] if prefill else hidden)
                laws, argmax = [], []
                head_rows = ((n - 1,) if prefill else range(n)) if head else ()
                for row in head_rows:
                    logits = self.model.compute_logits(hidden[row:row + 1])
                    if tuple(logits.shape) != (1, VOCAB) or logits.dtype != t.float32:
                        raise ProbeError("Actual softcapped/suppressed target head differs")
                    host = self._cpu(logits[0])
                    if t.isnan(host).any() or t.isposinf(host).any() or not t.isfinite(host).any():
                        raise ProbeError("Native target logits are invalid")
                    argmax.append(int(host.argmax()))  # first index resolves ties
                    laws.append(host)
                    del logits
                t.cuda.synchronize(self.runner.device)
                if set(self.frame["kv"]) != set(range(30)):
                    raise ProbeError("Missing completed actual layer writer")
                kv = self.frame["kv"]
                del hidden
            if diagnostic:
                extra_a = self.allocation["private_page_bytes"] + external + max(0, t.cuda.max_memory_allocated() - baseline_a)
                extra_r = self.allocation["private_page_bytes"] + external + max(0, t.cuda.max_memory_reserved() - baseline_r)
                self.peak_extra, self.peak_reserved_extra = max(self.peak_extra, extra_a), max(self.peak_reserved_extra, extra_r)
                if max(extra_a, extra_r) > GPU_CAP:
                    raise ProbeError("Measured staging/scratch/logits/allocator peak exceeds 8 MiB")
            self._check()
            if cancel_at == "after_forward":
                self.cancelled = True
            return plans, tuple(argmax), tuple(laws), host_hidden, kv
        except BaseException:
            self.failed = True
            raise
        finally:
            try:
                t.cuda.synchronize(self.runner.device)
            finally:
                self.frame = None
                if fraction is not None:
                    t.cuda.set_per_process_memory_fraction(fraction, self.runner.device)

    def _discard_suffix(self, plans, cached, consumed):
        # Zero ONLY private rows after the published prefix; old pages never
        # receive verifier writes. Device completion precedes state assignment.
        for _, (gid, _, _, cache, _) in self.layers.items():
            b = self.sizes[gid]
            for offset, page in enumerate(plans[gid].private):
                page_start = (cached // b + offset) * b
                begin = max(0, cached + consumed - page_start)
                if begin < b:
                    cache[page, :, begin:, :].zero_()
        self.torch.cuda.synchronize(self.runner.device)
        for _, (gid, _, _, cache, _) in self.layers.items():
            b = self.sizes[gid]
            for offset, page in enumerate(plans[gid].private):
                begin = max(0, cached + consumed - (cached // b + offset) * b)
                if begin < b and self.torch.count_nonzero(self._cpu(cache[page, :, begin:, :])):
                    raise ProbeError("Rejected suffix bytes were not discarded")

    def cycle(self, state, candidates, *, remaining=4, cancel_at=None):
        if state.terminal:
            raise ProbeError("Terminal pending token must not be forwarded")
        if len(candidates) > 1 or state.cached not in (P, P + 1, P + 2):
            raise ProbeError("Unreviewed native k/context")
        if self.cancelled or cancel_at == "before_forward":
            self.cancelled = False
            return state, None
        before = state
        plans, ids, laws, hidden, kv = self._forward((state.history[-1],) + tuple(candidates), state.cached, state.tables,
                                                   cancel_at=cancel_at)
        emitted, accepted, consumed, terminal = select_greedy(state.history[-1], candidates, ids, remaining)
        new_tables = tuple(consumed_table(p, cached=state.cached, consumed=consumed, block_size=b)
                           for p, b in zip(plans, self.sizes))
        if self.cancelled or cancel_at == "before_publication":
            self._discard_suffix(plans, state.cached, 0)
            self.cancelled = False
            return before, None
        self._discard_suffix(plans, state.cached, consumed)
        self._check()
        if self.cancelled:
            self._discard_suffix(plans, state.cached, 0)
            self.cancelled = False
            return before, None
        # All 30 writers, head, drainage, cap and suffix disposal completed.
        # One immutable assignment is the diagnostic publication point; there
        # is no scheduler/output acknowledgement or stochastic RNG publication.
        after = NativeState(state.history + emitted, new_tables, terminal)
        return after, {"accepted": accepted, "emitted": emitted, "consumed": consumed,
                       "argmax": ids, "logits": laws, "hidden": hidden, "kv": kv}

    def prefill(self, prompt):
        if len(prompt) != P or any(type(x) is not int or not 0 <= x < VOCAB for x in prompt):
            raise ProbeError("Exactly P2048 target prompt IDs required")
        # Reset only the allocator-owned pages after prior work drained; old
        # sessions are no longer live. Prefix tables retain all 2048 inputs and
        # the installed local attention window applies its own causal bounds.
        self.torch.cuda.synchronize(self.runner.device)
        for _, (gid, _, _, cache, _) in self.layers.items():
            for page in self.reserved[gid]:
                cache[page].zero_()
        tables = tuple(() for _ in self.sizes)
        for start in range(0, P, 256):
            plans, ids, _, _, _ = self._forward(prompt[start:start + 256], start, tables,
                diagnostic=False, prefill=True, head=start == P - 256)
            tables = tuple(p.table for p in plans)
        # Prefill logits at the LAST prompt input predict position P. The first
        # output is now emitted once; only the pending anchor enters cycle().
        anchor = ids[0]
        return NativeState(tuple(prompt) + (anchor,), tables, anchor in (1, 106))

    def _equal_rows(self, actual, references):
        t = self.torch
        if len(actual["argmax"]) != len(references):
            raise ProbeError("Serial comparison has missing target rows")
        for row, reference in enumerate(references):
            if (actual["argmax"][row] != reference["argmax"][0]
                    or not t.equal(actual["logits"][row].view(t.uint8), reference["logits"][0].view(t.uint8))
                    or not t.equal(actual["hidden"][row].view(t.uint8), reference["hidden"][0].view(t.uint8))):
                raise ProbeError("Actual target row differs from separate serial replay")
            for layer in range(30):
                if not t.equal(actual["kv"][layer][row].view(t.uint8), reference["kv"][layer][0].view(t.uint8)):
                    raise ProbeError("Actual processed KV row differs from separate serial replay")

    def run(self, prompt):
        receipts = []
        # Sequentially replay the same prompt using one loaded model and one
        # physical arena; no second 2K GPU prefix/model is resident.
        for case in ("k0", "forced_rejection", "full_accept"):
            serial = self.prefill(prompt)
            if serial.terminal:
                raise ProbeError("Bootstrap EOS: frozen continuation case is unavailable")
            first = serial.history[-1]
            refs = []
            for _ in range(3):
                serial, row = self.cycle(serial, (), remaining=4)
                if serial.terminal:
                    raise ProbeError("Serial EOS: frozen continuation case is unavailable")
                refs.append(row)
            greedy = refs[0]["argmax"][0]
            wrong = (greedy + 1) % VOCAB
            while wrong in (1, 106):
                wrong = (wrong + 1) % VOCAB
            candidate = () if case == "k0" else (wrong,) if case == "forced_rejection" else (greedy,)
            target_refs = refs[:len(candidate) + 1]
            if case == "forced_rejection":
                # Row1 is conditioned on the SUPPLIED wrong input, not the
                # greedy fallback. Replay that hypothetical branch separately.
                branch = self.prefill(prompt)
                branch, anchor_row = self.cycle(branch, ())
                branch = NativeState(branch.history[:-1] + candidate, branch.tables)
                _, wrong_row = self.cycle(branch, ())
                target_refs = [anchor_row, wrong_row]
            verified = self.prefill(prompt)
            if verified.history[-1] != first:
                raise ProbeError("Separately replayed prefill bootstrap differs")
            verified, actual = self.cycle(verified, candidate, remaining=4)
            self._equal_rows(actual, target_refs)
            consumed = actual["consumed"]
            expected_history = tuple(prompt) + (first,) + tuple(r["emitted"][0] for r in refs[:consumed])
            if verified.history != expected_history or verified.cached != P + consumed:
                raise ProbeError("Skipped/duplicated token or wrong consumed frontier")
            verified, next_row = self.cycle(verified, (), remaining=4)
            self._equal_rows(next_row, [refs[consumed]])
            receipts.append({"case": case, "candidate_count": len(candidate),
                             "accepted": actual["accepted"], "consumed": consumed,
                             "all_hidden_logit_KV_rows_bit_equal": True,
                             "next_single_input_bit_equal": True,
                             "old_committed_writer_bytes_unchanged": True,
                             "rejected_suffix_zeroed": True})
        # Cancellation is local to the synchronous diagnostic transaction. It
        # is observed AFTER submitted work drains; it never claims to preempt a
        # CUDA kernel. No publication occurs at either tested cancellation site.
        initial = self.prefill(prompt)
        candidate = (greedy,)
        for point in ("before_forward", "after_forward", "before_publication"):
            unchanged, result = self.cycle(initial, candidate, cancel_at=point)
            if unchanged is not initial or result is not None:
                raise ProbeError("Cancellation published native state")
        retry, row = self.cycle(initial, candidate)
        self._equal_rows(row, refs[:2])
        retry, next_row = self.cycle(retry, ())
        self._equal_rows(next_row, [refs[2]])
        self.torch.cuda.synchronize(self.runner.device)
        return {"schema": "megartx-speculative-native-scalars-v1", "drained": True,
                "native_executed": True, "cases": receipts, "cancellation_boundaries": 3,
                "cancel_retry_next_forward_bit_equal": True, "forwards": self.forwards,
                "allocation_lower_bound": self.allocation, "peak_extra_allocated_bytes": self.peak_extra,
                "peak_extra_reserved_bytes": self.peak_reserved_extra,
                "scope": "greedy_correctness_diagnostic_no_quality_or_speedup_claim"}
