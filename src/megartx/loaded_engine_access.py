"""Source-bound observation access. This interface grants no mutable lease."""
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys

from .controlled_kv_capture import CONFIG_SHA256, SOURCE_HASHES
from .prefill_diagnostic_plan import INSTALLED, REVISION, file_sha, checkpoint_identity, digest


@dataclass(frozen=True)
class LoadedModelIdentity:
    owner_pid: int
    owner_start_ticks: int
    checkpoint_revision: str
    config_sha256: str
    source_hashes: tuple
    checkpoint_files_sha256: str


@dataclass(frozen=True)
class OwnedFrame:
    """Retained actual references; single-use enforced by the provider."""
    owner: object
    sequence: int
    request_id: str
    context: object
    input_ids: object
    positions: object
    slots: dict
    identities: dict
    block_tables: dict


def owned_page_ranges(identity, blocks):
    """Check dense BHNC/HND inner pages, allowing disjoint group overlays."""
    _, pointer, nbytes, offset, shape, strides, device = identity
    dense = (len(shape) == 4 and strides[3] == 1 and strides[0] >= shape[1]*shape[2]*shape[3]
             and ((strides[2] == shape[3] and strides[1] == shape[2]*shape[3])
                  or (strides[1] == shape[3] and strides[2] == shape[1]*shape[3])))
    if not dense:
        raise ValueError("Unsupported within-page striding")
    page_bytes = shape[1]*shape[2]*shape[3]*2
    result = []
    for block in sorted(set(blocks)):
        if type(block) is not int or not 0 <= block < shape[0]:
            raise ValueError("Owned page outside actual cache")
        low = (offset + block*strides[0])*2
        if low < 0 or low + page_bytes > nbytes:
            raise ValueError("Owned page exceeds backing allocation")
        result.append((device, pointer + low, pointer + low + page_bytes))
    return result


def reject_page_alias(regions):
    flattened = sorted((device, low, high, layer) for layer, pages in regions.items()
                       for device, low, high in pages)
    previous = None
    for current in flattened:
        if previous and current[0] == previous[0] and current[1] < previous[2]:
            raise ValueError("Overlapping actually owned cache pages")
        previous = current


class LoadedEngineAccess:
    """Checked loaded runner identity/metadata, with no allocation or lease API."""
    def __init__(self, runner, model, torch_module):
        self._runner, self.model, self.torch = runner, model, torch_module
        cls = type(runner)
        if cls.__module__ != "vllm.v1.worker.gpu_model_runner" or cls.__name__ != "GPUModelRunner":
            raise RuntimeError("Actual GPUModelRunner required")
        for module, expected in {**SOURCE_HASHES, **INSTALLED}.items():
            source = sys.modules.get(module)
            if source is None or file_sha(inspect.getsourcefile(source)) != expected:
                raise RuntimeError("Installed observation source drift: " + module)
        checkpoint = Path(os.environ["MEGARTX_CHECKPOINT_PATH"])
        if file_sha(checkpoint / "config.json") != CONFIG_SHA256:
            raise RuntimeError("Loaded checkpoint config differs")
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        self.identity = LoadedModelIdentity(os.getpid(), ticks, REVISION, CONFIG_SHA256,
                                           tuple(sorted({**SOURCE_HASHES, **INSTALLED}.items())),
                                           digest(checkpoint_identity(checkpoint)))
        config = runner.vllm_config
        if (runner.get_model() is not model or runner.use_async_scheduling
                or config.scheduler_config.async_scheduling is not False
                or config.scheduler_config.enable_chunked_prefill is not True
                or config.scheduler_config.disable_hybrid_kv_cache_manager is not True
                or config.scheduler_config.max_num_seqs != 1
                or config.scheduler_config.max_num_batched_tokens != 256
                or config.model_config.max_model_len != 2304
                or config.model_config.enforce_eager is not True
                or config.cache_config.enable_prefix_caching
                or runner.speculative_config is not None):
            raise RuntimeError("Native observation requires exact synchronous eager configuration")
        self.registry = config.compilation_config.static_forward_context
        self.groups, self.builders, self.common = {}, {}, {}
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if group.host_resident or group.is_eagle_group:
                raise RuntimeError("Unexpected cache group ownership")
            for name in group.layer_names:
                if name in self.groups:
                    raise RuntimeError("Duplicate cache-group layer")
                spec = group.kv_cache_spec
                spec = spec.kv_cache_specs[name] if hasattr(spec, "kv_cache_specs") else spec
                if type(spec).__name__ != "FullAttentionSpec":
                    raise RuntimeError("Full-context allocation promotion not observed")
                self.groups[name] = (gid, spec)
            for attention_group in runner.attn_groups[gid]:
                builder = attention_group.get_metadata_builder(0)
                if (type(builder).__module__ != "vllm.v1.attention.backends.flashinfer"
                        or type(builder).__name__ != "FlashInferMetadataBuilder"
                        or builder.supports_update_block_table):
                    raise RuntimeError("Unknown actual metadata builder")
                self.builders[id(builder)] = (builder, gid, tuple(attention_group.layer_names))

    @classmethod
    def from_runner(cls, runner, torch_module):
        return cls(runner, runner.get_model(), torch_module)

    def record_metadata(self, builder, common, metadata):
        entry = self.builders.get(id(builder))
        if entry is None or entry[0] is not builder:
            raise RuntimeError("Metadata built by an unowned builder")
        self.common[id(metadata)] = (metadata, common, entry[1], entry[2])

    def bind_frame(self, owner, sequence, context, input_ids, positions, slots, identities):
        runner, torch = self._runner, self.torch
        if runner.get_model() is not self.model or context.no_compile_layers is not self.registry:
            raise RuntimeError("Loaded model/ForwardContext registry changed")
        if runner.input_batch.num_reqs != 1 or len(runner.input_batch.req_ids) != 1:
            raise RuntimeError("Exactly one actual request required")
        rows, end = len(positions), int(positions[-1].item()) + 1
        blocks, regions = {}, {}
        for layer, identity in identities.items():
            name = owner.layers[layer][0]
            meta = context.attn_metadata[name]
            receipt = self.common.get(id(meta))
            if receipt is None or receipt[0] is not meta or name not in receipt[3]:
                raise RuntimeError("Actual metadata/common-frame binding absent")
            common, gid = receipt[1], receipt[2]
            table = runner.input_batch.block_table[gid]
            actual = table.get_device_tensor(1)
            if (common.num_reqs != 1 or common.num_actual_tokens != rows
                    or common.query_start_loc.cpu().tolist() != [0, rows]
                    or common.query_start_loc_cpu.tolist() != [0, rows]
                    or common.seq_lens.cpu().tolist() != [end] or common.causal is not True
                    or common.block_table_tensor.data_ptr() != actual.data_ptr()
                    or not torch.equal(common.slot_mapping, meta.slot_mapping)
                    or table.block_size != identity[4][2]
                    or table.block_size != runner._kernel_block_sizes[gid]):
                raise RuntimeError("Actual query/sequence/block-table/slot geometry changed")
            count = (end + table.block_size - 1) // table.block_size
            if int(table.num_blocks_per_row[0]) < count:
                raise RuntimeError("Actual request lacks required allocated blocks")
            ids = [int(x) for x in actual[0, :count].cpu().tolist()]
            if len(set(ids)) != len(ids) or any(x <= 0 for x in ids):
                raise RuntimeError("Null/repeated owned blocks")
            expected = [ids[int(p)//table.block_size]*table.block_size + int(p)%table.block_size
                        for p in positions.tolist()]
            if slots[layer] != expected:
                raise RuntimeError("Actual block-table to writer-slot correspondence changed")
            placements = [t for t in runner.kv_cache_config.kv_cache_tensors if name in t.layers]
            if len(placements) != 1:
                raise RuntimeError("Ambiguous cache allocation placement")
            placement, spec = placements[0], self.groups[name][1]
            if (placement.host_resident or identity[2] != placement.size
                    or identity[3]*2 != placement.offset + placement.layers.index(name)*placement.layer_stride
                    or identity[5][0]*2 != placement.block_stride or spec.block_size != table.block_size
                    or spec.page_size_bytes < identity[4][1]*identity[4][2]*identity[4][3]*2):
                raise RuntimeError("Actual cache view differs from allocation placement")
            blocks[layer], regions[layer] = ids, owned_page_ranges(identity, ids)
        reject_page_alias(regions)
        return OwnedFrame(owner, sequence, runner.input_batch.req_ids[0], context,
                          input_ids, positions, slots, identities, blocks)
