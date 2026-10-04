"""Default-off actual-input to stored-BF16 native control; no arithmetic claim.

One owned P2048/chunk256 context only. All storage checks stop after input2048;
remaining254 inputs retain metadata and the actual head/sampler/token frontier.
"""
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import struct
import sys

from .prefill_native import NativeProvider, memory_sample, retained_host_bytes
from .prefill_diagnostic_plan import digest, remaining, INSTALLED
from .controlled_kv_capture import SOURCE_HASHES
from .prefill_storage import (ManagerInputs, InstanceHook, SpecialCallHook, TransferBudget,
                              native_bytes, stored_row, require_source, canonical_runtime, table_owner_signature, require_method_source, add_failure_note)
from .prefill_storage_plan import CompactEvidence, load_plan, verify_adapter_sources, raw_manifest, file_sha


class StorageProvider(NativeProvider):
    evidence_factory = CompactEvidence
    source_verifier = staticmethod(verify_adapter_sources)
    observe_cache_hashes = False

    def __init__(self, runner, plan, directory, torch, hook_checks):
        self.torch = torch
        self.storage_ready, self.storage_hooks = False, []
        self.failure_context, self.failure_emitted = None, False
        self.transfer = TransferBudget()
        self.counts = {'pre': 0, 'processed': 0, 'post': 0}
        self.roots = {key: hashlib.sha256() for key in self.counts}
        self.head_events, self.emissions, self.decode_inputs = [], [], []
        self.sampler_logits, self.sampler_result, self.sampler_identity = None, None, None
        self.sample_count, self.raw_heads, self.storage_frames = 0, 0, 0
        self.writer_seen = set()
        self.manager_request_id = None
        self.table_owners = table_owner_signature(runner.block_tables)
        self.manager_owner, self.sampler_owner = runner.block_tables, runner.sampler
        root = Path(os.environ['MEGARTX_PREFILL_NATIVE_SOURCE_ROOT']).resolve()
        path = root / 'numerical_reference/prefill_native_control.py'
        expected = plan['source_hashes']['numerical_reference/prefill_native_control.py']
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Independent CPU checker source differs')
        name = 'megartx_prefill_storage_independent_contract'
        if name in sys.modules:
            raise RuntimeError('Storage checker already loaded; one fresh context required')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.contract, self.checker_path, self.checker_sha = module, path, expected
        self.control = module.ProcessedKVControl()
        self.manager = ManagerInputs(runner.block_tables.block_sizes,
                                     runner.block_tables.kernel_block_sizes,
                                     runner.block_tables.blocks_per_kv_block)
        if (runner.vllm_config.kv_transfer_config is not None
                or runner.kv_connector is not sys.modules[type(runner).__module__].NO_OP_KV_CONNECTOR):
            raise RuntimeError('Storage control forbids KV connector transfers')
        require_source(runner.block_tables.num_blocks, 'vllm.v1.worker.gpu.buffer_utils', 'UvaBackedTensor',
                       '6597f701a30261bf250e9ad5a29737264624a1caa20527aeea0dad630325b093')
        self.connector = runner.kv_connector
        require_source(self.connector, 'vllm.v1.worker.gpu.kv_connector', 'KVConnector',
                       'b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81')
        try:
            super().__init__(runner, plan, directory, torch, hook_checks)
            self.install_storage_hooks()
            self.storage_ready = True
            self.require_hooks()
            self.evidence.write('storage-binding.json', {
                'schema': 'megartx-prefill-storage-binding-v1', 'plan_sha256': plan['plan_sha256'],
                'source_head': plan['source_head'], 'owner_pid': self.access.identity.owner_pid,
                'owner_start_ticks': self.access.identity.owner_start_ticks,
                'purpose': 'first-nine-frame-storage-frontier-control',
                'writer_hook_bound': True, 'checker_sha256': expected})
        except BaseException as error:
            self.poison_storage(error)
            raise

    def row_diagnostic(self, phase, layer, position, slot, key=None, value=None):
        tag = struct.pack('<II', layer, position)
        hashed = (hashlib.sha256(b'processed-k\0'+tag+key).hexdigest(),
                  hashlib.sha256(b'processed-v\0'+tag+value).hexdigest()) if key is not None else (None, None)
        retained = self.control.expected.get(layer, {}).get(position)
        expected = tuple(x.hex() for x in retained) if retained is not None else (None, None)
        if phase == 'processed':
            expected, hashed = hashed, (None, None)
        self.failure_context = {'phase': phase, 'layer': layer, 'absolute_position': position,
            'slot': slot, 'expected_k_sha256': expected[0], 'expected_v_sha256': expected[1],
            'observed_k_sha256': hashed[0], 'observed_v_sha256': hashed[1]}

    def record_failure(self, primary):
        """One bounded provenance record; never invent discarded expected words."""
        if primary is None or getattr(self, 'failure_emitted', False):
            return
        self.failure_emitted = True
        record = {'schema': 'megartx-prefill-storage-failure-v1', 'status': 'failed',
            'plan_sha256': getattr(self, 'plan', {}).get('plan_sha256'),
            'phase': 'admission', 'layer': None, 'absolute_position': None, 'slot': None,
            'expected_k_sha256': None, 'expected_v_sha256': None,
            'observed_k_sha256': None, 'observed_v_sha256': None,
            **(getattr(self, 'failure_context', None) or {}),
            'exception_type': type(primary).__name__, 'word_comparison_performed': False,
            'first_differing_word_index': None, 'expected_word': None, 'observed_word': None,
            'diagnostic_semantics': 'role_separated_row_digests_no_second_tensor_capture'}
        add_failure_note(primary, 'Storage failure provenance: ' + json.dumps(record, sort_keys=True, separators=(',', ':')))
        try:
            self.evidence.write('storage-failure.json', record)
        except BaseException as error:
            add_failure_note(primary, 'Secondary storage failure-record write failed: '+type(error).__name__)

    def poison_storage(self, primary):
        try:
            self.record_failure(primary)
        except BaseException as error:
            add_failure_note(primary, 'Secondary storage provenance construction failed: '+type(error).__name__)
        self.failed = True
        self.control.poisoned = True
        self.manager.poisoned = True
        if hasattr(self, 'ledger'):
            self.ledger.abort()
        self.restore_storage(primary)

    def restore_storage(self, primary=None):
        errors = []
        if any(hook.active for hook in self.storage_hooks):
            try:
                self.torch.cuda.synchronize()
            except BaseException as error:
                self.failed = True
                if primary is None:
                    raise RuntimeError('Storage callback drain failed; owned teardown required') from error
                add_failure_note(primary, 'Storage drain failed; hooks retained for owned teardown: ' + repr(error))
                return
        for hook in reversed(self.storage_hooks):
            try:
                hook.restore()
            except BaseException as error:
                errors.append(error)
        if errors:
            self.failed = True
            if primary is None:
                raise RuntimeError('Storage callback restoration failed') from errors[0]
            add_failure_note(primary, 'Storage callback restoration failed: ' + repr(errors[0]))

    def require_hooks(self):
        super().require_hooks()
        if not self.storage_ready:
            return
        try:
            runner = self.access._runner
            if (runner.block_tables is not self.manager_owner or runner.sampler is not self.sampler_owner
                    or table_owner_signature(runner.block_tables) != self.table_owners
                    or runner.kv_connector is not self.connector or runner.vllm_config.kv_transfer_config is not None
                    or tuple(runner.block_tables.block_sizes) != self.manager.sizes
                    or tuple(runner.block_tables.kernel_block_sizes) != self.manager.kernels
                    or tuple(runner.block_tables.blocks_per_kv_block) != self.manager.ratios
                    or runner.block_tables.cp_size != 1 or runner.block_tables.cp_rank != 0
                    or not all(runner.block_tables._slot_mapping_enabled)
                    or hashlib.sha256(self.checker_path.read_bytes()).hexdigest() != self.checker_sha):
                raise RuntimeError('Storage manager/sampler/connector/checker ownership changed')
            for hook in self.storage_hooks:
                hook.require_current()
        except BaseException as error:
            self.poison_storage(error)
            raise

    def install_storage_hooks(self):
        manager, sampler = self.manager_owner, self.sampler_owner
        self.writer_policies = {}
        require_source(manager, 'vllm.v1.worker.gpu.block_table', 'BlockTables',
                       INSTALLED['vllm.v1.worker.gpu.block_table'])
        require_source(sampler, 'vllm.v1.worker.gpu.sample.sampler', 'Sampler',
                       INSTALLED['vllm.v1.worker.gpu.sample.sampler'])
        require_method_source(manager, 'append_block_ids')
        require_method_source(sampler, 'add_request')
        require_method_source(sampler, '__call__')
        def manager_before(arguments):
            if not self.active():
                return None
            index = arguments['req_index']
            identities = self.access._runner.req_states.req_id_to_index
            if len(identities) != 1 or list(identities.values()) != [index]:
                raise RuntimeError('Manager inputs do not name the sole actual request')
            request_id = next(iter(identities))
            from .prefill_diagnostic_plan import native_request_identity
            native_request_identity(self.plan, request_id)
            if self.manager_request_id is not None and self.manager_request_id != request_id:
                raise RuntimeError('Manager request row was reused')
            candidate = copy.deepcopy(self.manager)
            candidate.append(index, arguments['new_block_ids'], arguments['overwrite'])
            return candidate, request_id
        def manager_after(ticket, result):
            if ticket is not None:
                self.manager, self.manager_request_id = ticket
        self.storage_hooks.append(InstanceHook(manager, 'append_block_ids', manager_before,
                                               manager_after, self.poison_storage))
        def added_before(arguments):
            if not self.active():
                return None
            if self.sampler_identity is not None:
                raise RuntimeError('Repeated actual sampler request setup')
            params = arguments['sampling_params']
            require_source(params, 'vllm.sampling_params', 'SamplingParams',
                           'f21efc98bf8ce8aab88eb67adb4207285e509a167d976e14d3e2e8282e024bea')
            if (arguments['prompt_len'] != 2048 or params.temperature != 0 or params.seed != 1234
                    or params.ignore_eos is not True or params.n != 1 or params.max_tokens != 256):
                raise RuntimeError('Actual SamplingParams differ from control request')
            return arguments['req_idx'], digest(canonical_runtime(params))
        def added_after(ticket, result):
            if ticket is not None:
                self.sampler_identity = ticket
        self.storage_hooks.append(InstanceHook(sampler, 'add_request', added_before,
                                               added_after, self.poison_storage))
        def sampling_before(arguments):
            if not self.started or self.completed:
                return None
            self.require_hooks()
            if (self.failed or not self.logit_seen or arguments['logits'] is not self.sampler_logits
                    or arguments['input_batch'] is not self.access.input_batch or self.sampler_result is not None):
                raise RuntimeError('Actual sampler did not consume the observed native head')
            return self.sampler_logits
        def sampling_after(ticket, result):
            if ticket is not None:
                self.sampler_result = result
                self.sampler_logits = None
        self.storage_hooks.append(SpecialCallHook(sampler, sampling_before, sampling_after, self.poison_storage))
        for ordinal in range(30):
            impl = self.layers[ordinal][3]
            require_source(impl, 'vllm.v1.attention.backends.flashinfer', 'FlashInferImpl',
                           SOURCE_HASHES['vllm.v1.attention.backends.flashinfer'])
            require_method_source(impl, 'do_kv_cache_update')
            attn = self.layers[ordinal][2]
            self.writer_policies[ordinal] = (attn.attn_backend, attn.kv_cache_dtype, impl.cache_dtype)
            self.storage_hooks.append(InstanceHook(impl, 'do_kv_cache_update',
                lambda arguments, layer=ordinal: self.before_writer(layer, arguments),
                self.after_writer, self.poison_storage))

    def reconstruct_tables(self, ticket, start, end):
        """Observe both native tables, independently expected from captured manager IDs."""
        self.require_hooks()
        table, batch = self.manager_owner, self.access.input_batch
        index = int(batch.idx_mapping_np[0])
        if (self.manager.req_index != index or ticket.request_id != self.manager_request_id
                or self.sampler_identity is None or self.sampler_identity[0] != index):
            raise RuntimeError('Missing manager/sampler capture or request identity mismatch')
        group_tables = {}
        for group in range(table.num_kv_cache_groups):
            count = int(table.num_blocks.np[group, index])
            if count < 1 or count > (2304 + self.manager.kernels[group]-1)//self.manager.kernels[group]:
                raise RuntimeError('Actual manager kernel count exceeds controlled capacity')
            need = (end + self.manager.kernels[group]-1)//self.manager.kernels[group]
            # These actual kernel observations are checked against manager inputs;
            # they are never passed into slots() as expected-address authority.
            self.transfer.reserve((count+need)*4, 'manager_table_metadata')
            persistent = table.block_tables[group].gpu[index, :count].cpu().tolist()
            gathered = self.access.block_tables[group][0, :need].cpu().tolist()
            self.manager.reconcile(group, persistent, gathered, count, end)
            group_tables[group] = self.manager.expanded(group)
        expected = {}
        for layer in range(30):
            name = self.layers[layer][0]
            group = self.access.groups[name][0]
            expected[layer] = self.manager.slots(group, self.contract.required_positions(layer, start, end))
            if ticket.slots[layer] != [expected[layer][p] for p in range(start, end)]:
                raise RuntimeError('Actual writer slots differ from original manager inputs')
            if tuple(ticket.block_tables[layer]) != group_tables[group][:len(ticket.block_tables[layer])]:
                raise RuntimeError('Existing owner receipt differs from independent manager input mapping')
        return expected

    def record_row(self, phase, layer, position, key, value):
        self.counts[phase] += 1
        self.roots[phase].update(struct.pack('<II', layer, position))
        self.roots[phase].update(hashlib.sha256(key).digest())
        self.roots[phase].update(hashlib.sha256(value).digest())

    def retained_union(self, phase, start, end, tables):
        for layer in range(30):
            descriptor, cache = self.descriptors[layer], self.layers[layer][2].kv_cache
            positions = self.contract.required_positions(layer, start, end)
            stop = start if phase == 'pre' else end
            for position in range(positions.start, stop):
                remaining(self.deadline)
                slot = tables[layer][position]
                self.row_diagnostic(phase, layer, position, slot)
                key, value = stored_row(cache, slot, descriptor['kv_heads'], descriptor['head_dim'],
                                        self.torch, self.transfer, phase)
                self.row_diagnostic(phase, layer, position, slot, key, value)
                self.control.retained(layer, position, slot, key, value, phase=phase)
                self.record_row(phase, layer, position, key, value)
                self.failure_context = None

    def begin(self, model, tokens, positions):
        # Source-audited conservative D2H quota covers existing loaded-access,
        # token/position/slot/table validation and scalar sampler calls. Charge
        # it before entering those inherited methods; it is not measured traffic.
        if (self.directory / 'request.json').exists():
            self.transfer.reserve(1 << 20 if self.ledger.end < 2048 else 128 << 10,
                                  'inherited_metadata_upper_bound')
        ticket = super().begin(model, tokens, positions)
        if ticket is None:
            return None
        start, end, slots = self.ledger.pending
        tables = self.reconstruct_tables(ticket, start, end)
        self.writer_seen = set()
        if start >= 2048:
            self.decode_inputs.append((start, int(tokens[0].item())))
        if end <= 2049:
            self.control.begin(self.contract.Frame(start, end), tables,
                               {i: dict(zip(range(start, end), slots[i])) for i in range(30)})
            self.retained_union('pre', start, end, tables)
        if start == 0:
            self.evidence.write('storage-geometry.json', {'schema': 'megartx-storage-geometry-v1',
                'plan_sha256': self.plan['plan_sha256'], **self.manager.receipt(),
                'page_sizes': [int(self.layers[i][2].kv_cache.shape[2]) for i in range(30)],
                'sample_positions': list(self.contract.sample_positions((16,)*30)),
                'layer_groups': [self.access.groups[self.layers[i][0]][0] for i in range(30)]})
        return ticket

    def before_writer(self, layer, arguments):
        if not self.started or self.completed:
            return None
        self.require_hooks()
        frame = self.frame
        if self.failed or frame is None or layer in self.writer_seen:
            raise RuntimeError('Missing frame or repeated actual layer writer')
        name, parent, attn, impl = self.layers[layer]
        self.failure_context = {'phase': 'writer_dispatch', 'layer': layer}
        backend, cache_mode, impl_mode = self.writer_policies[layer]
        descriptor = self.descriptors[layer]
        if (parent.attn is not attn or attn.impl is not impl or attn.attn_backend is not backend
                or parent.is_kv_shared_layer is not False
                or attn.kv_sharing_target_layer_name is not None
                or impl.kv_sharing_target_layer_name is not None
                or backend.forward_includes_kv_cache_update is not False
                or impl.is_kvcache_nvfp4 is not False
                or attn.num_kv_heads != descriptor['kv_heads'] or impl.num_kv_heads != descriptor['kv_heads']
                or attn.head_size != descriptor['head_dim'] or attn.head_size_v != descriptor['head_dim']
                or impl.head_size != descriptor['head_dim']
                or attn.sliding_window != descriptor['window_size']
                or impl.window_left != (1023 if descriptor['window_size'] else -1)
                or attn.kv_cache_dtype != cache_mode or impl.cache_dtype != impl_mode
                or cache_mode not in ('auto', 'bfloat16') or impl_mode != cache_mode):
            raise RuntimeError('Actual selected writer sharing/dispatch policy changed')
        key, value = arguments['key'], arguments['value']
        if (arguments['layer'] is not attn or arguments['kv_cache'] is not attn.kv_cache
                or arguments['slot_mapping'] is not frame.context.slot_mapping[name]):
            raise RuntimeError('Selected writer actual layer/cache/slot owner changed')
        start, end, slots = self.ledger.pending
        mapping = arguments['slot_mapping']
        if (not isinstance(mapping, self.torch.Tensor) or mapping.dtype != self.torch.int64
                or tuple(mapping.shape) != (end-start,) or mapping.device != attn.kv_cache.device):
            raise RuntimeError('Actual selected writer slot tensor changed')
        self.transfer.reserve((end-start)*8, 'writer_slot_metadata')
        if mapping.cpu().tolist() != slots[layer]:
            raise RuntimeError('Actual selected writer slot values changed before cache write')
        cache = attn.kv_cache
        storage = cache.untyped_storage()
        identity = (id(cache), storage.data_ptr(), storage.nbytes(), cache.storage_offset(),
                    tuple(cache.shape), tuple(cache.stride()), str(cache.device))
        if identity != frame.identities[layer]:
            raise RuntimeError('Actual selected writer cache allocation changed')
        descriptor = self.descriptors[layer]
        shape = (end-start, descriptor['kv_heads'], descriptor['head_dim'])
        for tensor in (key, value):
            if (not isinstance(tensor, self.torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != self.torch.bfloat16 or tensor.device != attn.kv_cache.device
                    or tensor.untyped_storage().data_ptr() == attn.kv_cache.untyped_storage().data_ptr()):
                raise RuntimeError('Writer inputs lost processed BF16 row provenance')
        # Equal numerical K/V values are legal; aliasing their logical source
        # address is not. Both may be disjoint views into one QKV allocation.
        if key.data_ptr() == value.data_ptr():
            raise RuntimeError('Processed K/V roles alias the same input address')
        if end <= 2049:
            with self.scratch.scope('processed_writer'):
                for row, position in enumerate(range(start, end)):
                    remaining(self.deadline)
                    k = native_bytes(key[row], self.torch, self.transfer, 'processed')
                    v = native_bytes(value[row], self.torch, self.transfer, 'processed')
                    self.row_diagnostic('processed', layer, position, slots[layer][row], k, v)
                    self.control.processed(layer, position, slots[layer][row], k, v)
                    self.record_row('processed', layer, position, k, v)
                    if position in (15, 16, 1023, 1024, 2047, 2048):
                        self.evidence.raw(f'kv-layer-{layer:02d}-position-{position:04d}.bf16', k+v)
                        self.sample_count += 1
        self.writer_seen.add(layer)
        self.failure_context = {'phase': 'native_writer', 'layer': layer}
        return frame

    def after_writer(self, ticket, result):
        if ticket is not None:
            self.failure_context = None

    def finish(self, ticket, result):
        if ticket is not self.frame or ticket.owner is not self or ticket.sequence != self.ledger.frames:
            raise RuntimeError('Stale storage-control frame')
        self.require_hooks()
        self.torch.cuda.synchronize()
        if self.writer_seen != set(range(30)):
            raise RuntimeError('Not all selected native writers ran exactly once')
        slots, _, identities, _ = self.read_frame(ticket.positions, ticket.input_ids)
        from vllm.forward_context import get_forward_context
        if get_forward_context() is not ticket.context:
            raise RuntimeError('Query completion escaped actual ForwardContext')
        current = self.access.bind_frame(self, ticket.sequence, ticket.context, ticket.input_ids,
                                         ticket.positions, slots, identities, self.ledger.positions)
        if identities != ticket.identities or slots != ticket.slots or current.block_tables != ticket.block_tables:
            raise RuntimeError('Actual cache ownership changed during query')
        start, end, _ = self.ledger.pending
        tables = self.reconstruct_tables(current, start, end)
        checked = end <= 2049
        if checked:
            self.retained_union('post', start, end, tables)
            self.control.finish(queries_complete=True)
            self.storage_frames += 1
        if (not isinstance(result, self.torch.Tensor) or tuple(result.shape) != (end-start, 2816)
                or result.dtype != self.torch.bfloat16):
            raise RuntimeError('Actual final hidden row geometry changed')
        self.hidden = result
        self.ledger.complete()
        self.evidence.write('control-frames.jsonl', {'plan_sha256': self.plan['plan_sha256'],
            'sequence': ticket.sequence, 'start': start, 'end': end,
            'input_ids_sha256': digest(ticket.input_ids.tolist()), 'storage_checked': checked,
            'queries_complete': True, 'manager_tables_sha256': self.manager.receipt()['manager_table_sha256'],
            'storage_counts': dict(self.counts) if checked else None,
            'storage_roots': {k: v.hexdigest() for k, v in self.roots.items()} if checked else None,
            'memory': memory_sample(self.torch, 'storage' if checked else 'metadata', self.scratch, self.ledger)}, append=True)
        self.frame = None

    def head(self, model, hidden, logits):
        if not self.started or self.completed:
            return
        self.require_hooks()
        torch, binding = self.torch, self.access.head_binding
        binding.require_current()
        if (self.failed or model is not self.access.model or self.frame is not None or self.hidden is None
                or self.logit_seen or binding.dtype != torch.bfloat16 or binding.suppressed
                or not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, 2816)
                or hidden.dtype != torch.bfloat16 or hidden.device != self.access._runner.device
                or not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 262144)
                or logits.dtype != torch.bfloat16 or logits.device != self.access._runner.device):
            raise RuntimeError('Actual BF16 native head/mask/ownership changed')
        indices = self.logits_indices
        if (indices is not self.access.input_batch.logits_indices or tuple(indices.shape) != (1,)
                or indices.dtype != torch.int64 or indices.device != self.access._runner.device):
            raise RuntimeError('Actual native index tensor changed')
        self.transfer.reserve(8, 'head_index')
        selected = indices.cpu().tolist()
        if selected != [len(self.hidden)-1]:
            raise RuntimeError('Actual native head selection differs')
        bits = native_bytes(hidden, torch, self.transfer, 'selected_hidden')
        expected = native_bytes(self.hidden[selected[0]:selected[0]+1], torch, self.transfer, 'selected_hidden')
        if bits != expected:
            raise RuntimeError('Native processed head selected a different hidden row')
        position = self.ledger.end-1
        raw_hash = None
        if position in (2047, 2048):
            raw = native_bytes(logits, torch, self.transfer, 'raw_heads')
            self.contract.finite_bf16(raw, 262144)
            self.evidence.raw(f'head-position-{position}.bf16', raw)
            self.raw_heads += 1
            raw_hash = hashlib.sha256(raw).hexdigest()
        else:
            # Scalar finite qualification without retaining/downloading every
            # native logit row. The bounded boolean GPU temporary is charged.
            with self.scratch.allocation(262145):
                finite = torch.isfinite(logits)
                all_finite = finite.all()
                self.transfer.reserve(1, 'head_finite_scalar')
                okay = bool(all_finite.item())
                del all_finite, finite
            if not okay:
                raise RuntimeError('Unexpected nonfinite native head with empty loaded mask')
        phase = 'intermediate_prompt_chunk' if position < 2047 else 'final_prompt' if position == 2047 else 'decode'
        event = (position, position+1, position < 2047)
        self.head_events.append(event)
        self.evidence.write('heads.jsonl', {'plan_sha256': self.plan['plan_sha256'],
            'sequence': self.ledger.frames-1, 'phase': phase, 'logit_position': position,
            'predicts_position': position+1, 'expected_sampler_discard': event[2],
            'selected_row_index': selected[0], 'hidden_bits_sha256': hashlib.sha256(bits).hexdigest(),
            'logit_bits_sha256': raw_hash, 'native_logits_dtype': 'torch.bfloat16',
            'suppressed_token_count': 0, 'soft_cap': 30.0, 'before_sampler_transforms': True}, append=True)
        self.head_counts[phase] += 1
        self.logit_seen, self.hidden, self.sampler_logits = True, None, logits

    def sampled(self, result):
        if not self.started or self.completed:
            return
        self.require_hooks()
        if self.sampler_result is not result or self.sampler_logits is not None:
            raise RuntimeError('Runner sample result differs from actual observed sampler output')
        before = len(self.ledger.outputs)
        super().sampled(result)
        self.sampler_result = None
        if len(self.ledger.outputs) != before:
            self.emissions.append((self.ledger.frames-1, self.ledger.outputs[-1]))
        if self.completed:
            frontier = self.contract.verify_frontier(tuple(self.plan['tokens']), self.head_events,
                        self.emissions, self.decode_inputs, self.ledger.end)
            expected = self.contract.capture_work_budget()
            transferred = sum(self.transfer.kinds.get(k, 0) for k in ('pre', 'processed', 'post'))
            if (self.control.end != 2049 or self.control.poisoned or self.storage_frames != 9
                    or self.sample_count != 180 or self.raw_heads != 2
                    or self.counts != {'pre': 212355, 'processed': 61470, 'post': 273825}
                    or transferred != expected['transferred_bytes_per_context']):
                raise RuntimeError('Incomplete bounded all-layer storage control')
            self.restore_storage()
            self.evidence.write('control.json', {'schema': 'megartx-prefill-storage-control-v1',
                'plan_sha256': self.plan['plan_sha256'], 'status': 'storage_frontier_observed',
                'capture_frames': 9, 'capture_end': 2049, 'checked_layer_frames': 270,
                'pre_rows': self.counts['pre'], 'processed_rows': self.counts['processed'],
                'post_rows': self.counts['post'], 'transferred_bytes': transferred,
                'metadata_only_decode_inputs': 254, 'storage_exact': True, 'frontier_verified': True,
                'storage_callbacks_restored': True,
                'sample_combined_kv_rows': self.sample_count, 'raw_head_rows': self.raw_heads,
                'frontier': frontier, 'storage_roots': {k: v.hexdigest() for k, v in self.roots.items()},
                'transfer': self.transfer.receipt(), 'manager': self.manager.receipt(),
                'actual_sampling_params_sha256': self.sampler_identity[1],
                'processed_digest_host_bytes': retained_host_bytes(self.control.expected),
                'numerical_qualified': False, 'performance_qualified': False,
                'independent_arithmetic_qualified': False, 'quality_qualified': False,
                'sampled_repeatability_qualified': False, 'natural_positive_correction_coverage': None,
                'raw_samples_sha256': digest(raw_manifest(self.directory)),
                'frame_records_sha256': file_sha(self.directory/'control-frames.jsonl'),
                'head_records_sha256': file_sha(self.directory/'heads.jsonl'),
                'sample_records_sha256': file_sha(self.directory/'samples.jsonl'),
                'sample_hashes_sha256': digest([digest([token]) for token in self.ledger.outputs])})

    def abort(self):
        primary = sys.exc_info()[1]
        try:
            self.record_failure(primary)
        except BaseException as error:
            add_failure_note(primary, 'Secondary storage provenance construction failed: '+type(error).__name__)
        self.control.poisoned = True
        self.manager.poisoned = True
        try:
            super().abort()
        finally:
            self.restore_storage(primary)


def install_storage_observer(torch, model_cls):
    if os.environ.get('MEGARTX_PREFILL_STORAGE_CONTROL') != '1':
        raise RuntimeError('Storage control requires its explicit exclusive purpose')
    from .prefill_native import install_native_observer
    return install_native_observer(torch, model_cls, provider_factory=StorageProvider,
                                   plan_loader=load_plan, source_verifier=verify_adapter_sources)
