"""Source-bound observation access. This interface grants no mutable lease."""
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
from types import MethodType

from .controlled_kv_capture import CONFIG_SHA256, SOURCE_HASHES
from .prefill_diagnostic_plan import INSTALLED, REVISION, file_sha, checkpoint_identity, digest
from .prefill_runner_binding import require_default_selection


# Additional head-path modules are checked as actual loaded files below. This
# additive vector does not rewrite the historical runner-selection catalog.
HEAD_SOURCES = {
    "vllm.config.model": "f4168b67cc93ce72ff2a6d4c66974223e32fd0f1fab4c05e44bd68a275e18d69",
    "vllm.model_executor.models.gemma4_mm": "ee77a4f9884524e35219e8230a65f73eef675f2a23540cb267a25bf7029ea9b7",
    "vllm.model_executor.layers.logits_processor": "6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f",
    "vllm.model_executor.layers.vocab_parallel_embedding": "d42574ac79bd7d7847548784d2272c7c40cbd93fe6cdadb15e8c2db79d2074d7",
    "vllm.model_executor.layers.linear": "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d"}


class NativeHeadBinding:
    """Freeze the actual native projection dtype and exact suppression mask.

    The processed model head precedes sampler-specific transforms. It is not a
    probability distribution, and its configured negative infinities are valid.
    No casting, mutation, normalized-law comparison or extra model call occurs.
    """
    def __init__(self, runner, model, torch):
        self.runner, self.model, self.torch = runner, model, torch
        self.language = model.language_model
        self.processor, self.head = self.language.logits_processor, self.language.lm_head
        self.quant_method, self.weight = self.head.quant_method, self.head.weight
        self.callbacks = tuple((owner, name, getattr(owner, name).__func__) for owner, name in (
            (self.language, 'compute_logits'), (self.processor, 'forward'),
            (self.processor, '_get_logits'), (self.processor, '_apply_head'),
            (self.quant_method, 'apply')))
        self.signature = self._read()
        self.dtype = torch.bfloat16 if self.processor.head_dtype is None else self.processor.head_dtype
        self.suppressed = frozenset(self.signature[-1])

    def _read(self):
        t, p, h = self.torch, self.processor, self.head
        known = ((self.model, 'vllm.model_executor.models.gemma4_mm', 'Gemma4ForConditionalGeneration'),
                 (self.language, 'vllm.model_executor.models.gemma4', 'Gemma4ForCausalLM'),
                 (p, 'vllm.model_executor.layers.logits_processor', 'LogitsProcessor'),
                 (h, 'vllm.model_executor.layers.vocab_parallel_embedding', 'ParallelLMHead'))
        if any(type(obj).__module__ != module or type(obj).__name__ != name for obj, module, name in known):
            raise RuntimeError('Unknown actual native head owner')
        if ((type(self.quant_method).__module__, type(self.quant_method).__name__) not in {
                ('vllm.model_executor.layers.vocab_parallel_embedding', 'UnquantizedEmbeddingMethod'),
                ('vllm.model_executor.layers.linear', 'UnquantizedLinearMethod')}
                or self.runner.vllm_config.model_config.dtype != t.bfloat16
                or p.head_dtype not in (None, t.bfloat16, t.float32)
                or self.runner.vllm_config.model_config.head_dtype != p.head_dtype
                or p.soft_cap != 30.0 or p.scale != 1.0 or p.logits_as_input is not False
                or p.vocab_size != 262144 or p.org_vocab_size != 262144
                or h.weight.dtype != t.bfloat16 or h.weight.device != self.runner.device
                or tuple(h.weight.shape) != (262144, 2816) or h.tp_size != 1):
            raise RuntimeError('Actual native head dtype/configuration changed')
        suppressed = self.model._suppress_token_ids
        if suppressed is None:
            suppressed = ()
        if (type(suppressed) not in (list, tuple) or len(suppressed) >= 262144
                or any(type(i) is not int or not 0 <= i < 262144 for i in suppressed)
                or len(set(suppressed)) != len(suppressed)):
            raise RuntimeError('Actual native head suppression mask is invalid')
        return (p.head_dtype, p.soft_cap, p.scale, p.logits_as_input, p.vocab_size,
                p.org_vocab_size, h.tp_size, tuple(suppressed))

    def require_current(self):
        if (self.runner.get_model() is not self.model or self.model.language_model is not self.language
                or self.language.logits_processor is not self.processor or self.language.lm_head is not self.head
                or self.head.quant_method is not self.quant_method or self.head.weight is not self.weight):
            raise RuntimeError('Actual native head ownership changed')
        for owner, name, function in self.callbacks:
            callback = getattr(owner, name, None)
            if (type(callback) is not MethodType or callback.__self__ is not owner
                    or callback.__func__ is not function):
                raise RuntimeError('Actual native head callback changed')
        if self._read() != self.signature:
            raise RuntimeError('Actual native head configuration/suppression changed')

    def receipt(self):
        return {'native_logits_dtype': str(self.dtype),
                'head_dtype_setting': None if self.processor.head_dtype is None else str(self.processor.head_dtype),
                'suppressed_token_count': len(self.suppressed),
                'suppressed_token_ids_sha256': digest(sorted(self.suppressed)),
                'tensor_semantics': 'processed_native_head_before_sampler_transforms',
                'normalized_probabilities': False}


@dataclass(frozen=True)
class LoadedModelIdentity:
    owner_pid: int
    owner_start_ticks: int
    checkpoint_revision: str
    config_sha256: str
    source_hashes: tuple
    checkpoint_files_sha256: str
    runner_module: str
    runner_class: str
    source_origins: tuple


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


class MetadataBuilderOwnership:
    """Retain the exact eager dispatch structure, not just a builder cache."""
    def __init__(self, runner):
        from vllm.v1.worker.utils import AttentionGroup
        self.group_type, self.getter = AttentionGroup, AttentionGroup.get_metadata_builder
        self.root = runner.attn_groups
        if type(self.root) is not list or not self.root:
            raise RuntimeError("Actual attention-group ownership is absent")
        groups, seen_groups, seen_builders, seen_layers = [], set(), set(), set()
        for gid, members in enumerate(self.root):
            if type(members) is not list:
                raise RuntimeError("Actual attention-group members must be a list")
            retained = []
            for group in members:
                if type(group) is not AttentionGroup or id(group) in seen_groups:
                    raise RuntimeError("Unknown or repeated actual attention group")
                callback = getattr(group, 'get_metadata_builder', None)
                builders, layers = group.metadata_builders, group.layer_names
                if (type(callback) is not MethodType or callback.__self__ is not group
                        or callback.__func__ is not self.getter
                        or type(group.kv_cache_group_id) is not int or group.kv_cache_group_id != gid
                        or type(builders) is not list or len(builders) != 1
                        or type(layers) is not list or not layers
                        or any(type(name) is not str or not name for name in layers)
                        or len(set(layers)) != len(layers) or seen_layers.intersection(layers)):
                    raise RuntimeError("Actual metadata-builder selection is unsupported")
                builder = builders[0]
                if callback(0) is not builder or id(builder) in seen_builders:
                    raise RuntimeError("Repeated or changed selected metadata builder")
                retained.append((group, builders, builder, layers, tuple(layers)))
                seen_groups.add(id(group)); seen_builders.add(id(builder)); seen_layers.update(layers)
            groups.append((members, tuple(retained)))
        if not seen_builders:
            raise RuntimeError("Selected metadata-builder ownership is empty")
        self.groups = tuple(groups)
        self.builder_ids = frozenset(seen_builders)

    def matches(self, runner, builders):
        """Reject owner/mapping/getter drift before executing metadata work."""
        if (runner.attn_groups is not self.root or type(self.root) is not list
                or len(self.root) != len(self.groups)
                or self.group_type.get_metadata_builder is not self.getter
                or type(builders) is not dict or set(builders) != self.builder_ids):
            return False
        for gid, (members, retained) in enumerate(self.groups):
            if self.root[gid] is not members or type(members) is not list or len(members) != len(retained):
                return False
            for index, (group, selected, builder, layers, names) in enumerate(retained):
                callback = getattr(group, 'get_metadata_builder', None)
                if (members[index] is not group or type(group) is not self.group_type
                        or type(group.kv_cache_group_id) is not int or group.kv_cache_group_id != gid
                        or group.metadata_builders is not selected or type(selected) is not list
                        or len(selected) != 1 or selected[0] is not builder
                        or group.layer_names is not layers or type(layers) is not list
                        or tuple(layers) != names
                        or type(callback) is not MethodType or callback.__self__ is not group
                        or callback.__func__ is not self.getter):
                    return False
                # Invoke only the unchanged source-bound getter, never an
                # instance replacement, and reconcile its actual selection.
                if callback(0) is not builder:
                    return False
                record = builders[id(builder)]
                if (type(record) is not tuple or len(record) != 3 or record[0] is not builder
                        or type(record[1]) is not int or record[1] != gid or record[2] != names):
                    return False
        return True


class LoadedEngineAccess:
    """Checked loaded runner identity/metadata, with no allocation or lease API."""
    def __init__(self, runner, model, torch_module):
        self._runner, self.model, self.torch = runner, model, torch_module
        cls = type(runner)
        if cls.__module__ != "vllm.v1.worker.gpu.model_runner" or cls.__name__ != "GPUModelRunner":
            raise RuntimeError("Actual default V2 GPUModelRunner required")
        origins = []
        for module, expected in {**SOURCE_HASHES, **INSTALLED, **HEAD_SOURCES}.items():
            source = sys.modules.get(module)
            if source is None or file_sha(inspect.getsourcefile(source)) != expected:
                raise RuntimeError("Installed observation source drift: " + module)
            root = Path(inspect.getsourcefile(sys.modules['vllm'])).resolve().parent
            path = root/Path(*module.split('.')[1:]).with_suffix('.py')
            if not path.is_file():
                path = root/Path(*module.split('.')[1:])/'__init__.py'
            if path.is_symlink() or Path(inspect.getsourcefile(source)).resolve() != path:
                raise RuntimeError('Loaded observation module origin drift: ' + module)
            origins.append((module, str(path)))
        checkpoint = Path(os.environ["MEGARTX_CHECKPOINT_PATH"])
        if file_sha(checkpoint / "config.json") != CONFIG_SHA256:
            raise RuntimeError("Loaded checkpoint config differs")
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        self.identity = LoadedModelIdentity(os.getpid(), ticks, REVISION, CONFIG_SHA256,
                                           tuple(sorted({**SOURCE_HASHES, **INSTALLED, **HEAD_SOURCES}.items())),
                                           digest(checkpoint_identity(checkpoint)),
                                           cls.__module__, cls.__name__, tuple(sorted(origins)))
        config = runner.vllm_config
        self.runner_policy = require_default_selection(config)
        if (runner.get_model() is not model
                or config.scheduler_config.async_scheduling is not False
                or config.scheduler_config.enable_chunked_prefill is not True
                or config.scheduler_config.disable_hybrid_kv_cache_manager is not True
                or config.scheduler_config.max_num_seqs != 1
                or config.scheduler_config.max_num_batched_tokens != 256
                or config.model_config.max_model_len != 2304
                or config.model_config.enforce_eager is not True
                or config.cache_config.enable_prefix_caching
                or runner.speculative_config is not None
                or any(getattr(config.parallel_config, key) != 1 for key in
                       ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
                        'decode_context_parallel_size', 'prefill_context_parallel_size'))
                or config.parallel_config.enable_dbo
                or runner.pcp_manager is not None or runner.ubatch_runner is not None
                or runner.batch_sharder is not None or runner.fast_prefill is not None
                or type(runner.model_state).__module__ != 'vllm.v1.worker.gpu.model_states.default'
                or type(runner.model_state).__name__ != 'DefaultModelState'
                or runner.model_state.rope_state is not None):
            raise RuntimeError("Native observation requires exact synchronous eager configuration")
        self.head_binding = NativeHeadBinding(runner, model, torch_module)
        self.registry = config.compilation_config.static_forward_context
        self.groups, self.builders, self.common = {}, {}, {}
        self.builder_ownership = MetadataBuilderOwnership(runner)
        self.input_batch, self.block_tables, self.slot_mappings = None, None, None
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

    def prepare_inputs(self, batch):
        cls = type(batch)
        if cls.__module__ != 'vllm.v1.worker.gpu.input_batch' or cls.__name__ != 'InputBatch':
            raise RuntimeError('Actual V2 InputBatch required')
        if (batch.num_reqs != 1 or len(batch.req_ids) != 1 or batch.num_reqs_after_padding != 1
                or batch.num_tokens != batch.num_tokens_after_padding
                or batch.num_draft_tokens != 0 or batch.fast_prefill is not None
                or batch.has_structured_output_reqs or batch.cu_num_logits_np.tolist() != [0, 1]
                or batch.idx_mapping_np.tolist() != [self._runner.req_states.req_id_to_index.get(batch.req_ids[0])]):
            raise RuntimeError('V2 batch requires one ordinary unpadded owned request')
        self.common.clear()
        self.input_batch, self.block_tables, self.slot_mappings = batch, None, None

    def prepare_attn(self, batch, result):
        if batch is not self.input_batch or self.block_tables is not None:
            raise RuntimeError('V2 attention preparation escaped its actual InputBatch')
        self.block_tables, self.slot_mappings = result
        tables = self._runner.block_tables
        if (len(self.block_tables) != tables.num_kv_cache_groups
                or self.slot_mappings.data_ptr() != tables.slot_mappings.data_ptr()
                or any(actual.data_ptr() != persistent.data_ptr() for actual, persistent in
                       zip(self.block_tables, tables.input_block_tables))):
            raise RuntimeError('V2 gathered tables/slot maps are not incumbent buffers')

    def bind_frame(self, owner, sequence, context, input_ids, positions, slots, identities,
                   retained_slots=None):
        runner, torch = self._runner, self.torch
        if runner.get_model() is not self.model or context.no_compile_layers is not self.registry:
            raise RuntimeError("Loaded model/ForwardContext registry changed")
        batch = self.input_batch
        if batch is None or batch.num_reqs != 1 or len(batch.req_ids) != 1 or self.block_tables is None:
            raise RuntimeError("Exactly one actual request required")
        if (input_ids is not batch.input_ids or positions is not batch.positions
                or batch.num_computed_tokens_np.tolist() != [int(positions[0].item())]
                or batch.prefill_len_np.tolist() != [2048]
                or batch.query_start_loc_np.tolist() != [0, len(positions)]
                or batch.idx_mapping.cpu().tolist() != batch.idx_mapping_np.tolist()
                or runner.req_states.req_id_to_index.get(batch.req_ids[0]) != int(batch.idx_mapping_np[0])):
            raise RuntimeError('V2 actual forward tensors/request-state row identity changed')
        rows, start, end = len(positions), int(positions[0].item()), int(positions[-1].item()) + 1
        if retained_slots is None:
            if sequence != 0 or start != 0:
                raise RuntimeError("Retained native query mapping receipt required")
            retained_slots = {layer: {} for layer in identities}
        blocks, regions = {}, {}
        for layer, identity in identities.items():
            name = owner.layers[layer][0]
            meta = context.attn_metadata[name]
            receipt = self.common.get(id(meta))
            if receipt is None or receipt[0] is not meta or name not in receipt[3]:
                raise RuntimeError("Actual metadata/common-frame binding absent")
            common, gid = receipt[1], receipt[2]
            table = runner.block_tables
            actual = self.block_tables[gid]
            block_size = runner.kernel_block_sizes[gid]
            ratio = table.blocks_per_kv_block[gid]
            state_index = int(batch.idx_mapping_np[0])
            if (common.num_reqs != 1 or common.num_actual_tokens != rows
                    or common.query_start_loc.cpu().tolist() != [0, rows]
                    or common.query_start_loc_cpu.tolist() != [0, rows]
                    or common.seq_lens.cpu().tolist() != [end] or common.causal is not True
                    or common.block_table_tensor is not actual
                    or not torch.equal(common.slot_mapping, meta.slot_mapping)
                    or common.slot_mapping.data_ptr() != self.slot_mappings[gid].data_ptr()
                    or context.slot_mapping[name].data_ptr() != self.slot_mappings[gid].data_ptr()
                    or not torch.equal(batch.seq_lens, common.seq_lens)
                    or block_size != identity[4][2] or block_size != table.kernel_block_sizes[gid]
                    or type(ratio) is not int or ratio < 1
                    or table.block_sizes[gid] != block_size*ratio):
                raise RuntimeError("Actual query/sequence/block-table/slot geometry changed")
            count = (end + block_size - 1) // block_size
            if int(table.num_blocks.np[gid, state_index]) < count:
                raise RuntimeError("Actual request lacks required allocated blocks")
            ids = [int(x) for x in actual[0, :count].cpu().tolist()]
            if len(set(ids)) != len(ids) or any(x <= 0 for x in ids):
                raise RuntimeError("Null/repeated owned blocks")
            if ids != [int(x) for x in table.block_tables[gid].gpu[state_index, :count].cpu().tolist()]:
                raise RuntimeError('V2 gathered block table differs from actual owned state row')
            expected = [ids[int(p)//block_size]*block_size + int(p)%block_size
                        for p in positions.tolist()]
            if slots[layer] != expected:
                raise RuntimeError("Actual block-table to writer-slot correspondence changed")
            # The writer suffix alone cannot bind the prefix queried by this
            # frame. Check every still-required absolute historical row against
            # the actual current table before any model work, and again after.
            low = 0 if layer % 6 == 5 else max(0, start-1023)
            retained = retained_slots.get(layer, {})
            for position in range(low, start):
                current_slot = ids[position//block_size]*block_size + position%block_size
                if retained.get(position) != current_slot:
                    raise RuntimeError("Retained native query block mapping changed")
            placements = [t for t in runner.kv_cache_config.kv_cache_tensors if name in t.layers]
            if len(placements) != 1:
                raise RuntimeError("Ambiguous cache allocation placement")
            placement, spec = placements[0], self.groups[name][1]
            if (placement.host_resident or identity[2] != placement.size
                    or identity[3]*2 != placement.offset + placement.layers.index(name)*placement.layer_stride
                    or identity[5][0]*2*ratio != placement.block_stride
                    or spec.block_size != table.block_sizes[gid]
                    or spec.page_size_bytes < identity[4][1]*identity[4][2]*identity[4][3]*2*ratio):
                raise RuntimeError("Actual cache view differs from allocation placement")
            blocks[layer], regions[layer] = ids, owned_page_ranges(identity, ids)
        reject_page_alias(regions)
        return OwnedFrame(owner, sequence, batch.req_ids[0], context,
                          input_ids, positions, slots, identities, blocks)
