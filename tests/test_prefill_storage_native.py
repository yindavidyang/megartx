"""CPU fake-runtime exercises of the actual default-off storage provider.

NumPy stores BF16 *words*, never approximated BF16 arithmetic. No real Torch or
vLLM imports/devices are used. Passing these tests is not native execution proof.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'numerical_reference'))
import prefill_native_control as contract
from megartx.prefill_native import GpuScratch, NativeProvider
from megartx.prefill_storage import (InstanceHook, ManagerInputs, SpecialCallHook,
                                    TransferBudget, native_bytes, stored_row,
                                    table_owner_signature)
from megartx.prefill_storage_native import StorageProvider

BF16, I64, I32, U8 = 'bfloat16', 'int64', 'int32', 'uint8'


class Device:
    def __init__(self, kind): self.type = kind
    def __str__(self): return self.type + ':0'


CUDA, CPU = Device('cuda'), Device('cpu')


class Tensor:
    """A host array with explicit fake-device and backing-allocation identity."""
    def __init__(self, array, dtype=BF16, device=CUDA, *, backing=None, events=None, label='tensor'):
        self.array, self.dtype, self.device = np.asarray(array), dtype, device
        self.backing = self.array if backing is None else backing
        self.events, self.label = events if events is not None else [], label
    @property
    def shape(self): return self.array.shape
    def __len__(self): return len(self.array)
    def __getitem__(self, index):
        return Tensor(self.array[index], self.dtype, self.device, backing=self.backing,
                      events=self.events, label=self.label)
    def stride(self): return tuple(x // self.array.itemsize for x in self.array.strides)
    def numel(self): return int(self.array.size)
    def data_ptr(self): return int(self.array.ctypes.data)
    def untyped_storage(self):
        return NS(data_ptr=lambda: int(self.backing.ctypes.data), nbytes=lambda: self.backing.nbytes)
    def storage_offset(self): return (self.data_ptr() - int(self.backing.ctypes.data)) // self.array.itemsize
    def detach(self): return self
    def cpu(self):
        self.events.append(('cpu', self.label, self.array.nbytes))
        return Tensor(self.array.copy(), self.dtype, CPU, events=self.events, label=self.label)
    def contiguous(self):
        if self.device is CUDA:
            raise AssertionError('Observer made a fake-GPU contiguous copy')
        return Tensor(np.ascontiguousarray(self.array), self.dtype, CPU,
                      events=self.events, label=self.label)
    def view(self, dtype):
        if dtype != U8: raise AssertionError('Unexpected bit cast')
        return Tensor(self.array.view(np.uint8), dtype, self.device, backing=self.backing,
                      events=self.events, label=self.label)
    def numpy(self):
        if self.device is CUDA: raise AssertionError('Host access before .cpu')
        return self.array
    def tolist(self): return self.array.tolist()
    def item(self): return self.array.item()
    def all(self): return Tensor(np.array(self.array.all()), 'bool', self.device, events=self.events)


class FakeCuda:
    def __init__(self, events): self.events, self.calls, self.failures = events, 0, {}
    def synchronize(self, *args):
        self.calls += 1
        self.events.append(('synchronize', self.calls))
        if self.calls in self.failures: raise self.failures[self.calls]
    def memory_allocated(self): return 0
    def memory_reserved(self): return 0
    def max_memory_allocated(self): return 0
    def max_memory_reserved(self): return 0
    def reset_peak_memory_stats(self): pass


def fake_torch(events):
    def as_strided(tensor, shape, strides, *, storage_offset):
        base = tensor.backing.reshape(-1)[storage_offset:]
        array = np.lib.stride_tricks.as_strided(base, shape=shape,
                                               strides=tuple(s * 2 for s in strides))
        return Tensor(array, tensor.dtype, tensor.device, backing=tensor.backing,
                      events=events, label=tensor.label)
    return NS(Tensor=Tensor, bfloat16=BF16, int64=I64, int32=I32, uint8=U8,
              cuda=FakeCuda(events), as_strided=as_strided,
              isfinite=lambda t: Tensor((t.array & 0x7f80) != 0x7f80, 'bool', events=events))


class Evidence:
    def __init__(self): self.records, self.raws = [], {}
    def write(self, name, value, **kwargs): self.records.append((name, value))
    def raw(self, name, value):
        if name in self.raws: raise AssertionError('Duplicate raw sample')
        self.raws[name] = value


class RecordingControl:
    def __init__(self): self.processed_rows, self.retained_rows, self.poisoned, self.expected = [], [], False, {}
    def processed(self, *row): self.processed_rows.append(row)
    def retained(self, *row, phase): self.retained_rows.append((phase, *row))


class Ledger:
    def __init__(self, start, end, slots):
        self.pending, self.end = (start, end, slots), start
        self.frames = start // 256 if start < 2048 else 8 + start - 2048
        self.positions, self.outputs, self.failed = {i: {} for i in range(30)}, [], False
    def complete(self):
        start, end, slots = self.pending
        for layer in range(30): self.positions[layer].update(zip(range(start, end), slots[layer]))
        self.end, self.pending, self.frames = end, None, self.frames + 1
    def abort(self): self.failed = True


def manager_table(events, ids):
    class Manager:
        def append_block_ids(self, req_index, new_block_ids, overwrite): pass
    table = Manager()
    table.block_sizes, table.kernel_block_sizes, table.blocks_per_kv_block = [16], [16], [1]
    table.cp_size, table.cp_rank, table._slot_mapping_enabled = 1, 0, [True]
    table.num_kv_cache_groups = 1
    table.block_tables = [NS(gpu=Tensor(np.array([ids], dtype=np.int32), I32, events=events))]
    table.input_block_tables = [Tensor(np.array([ids], dtype=np.int32), I32, events=events)]
    cpu = Tensor(np.array([[len(ids)]], dtype=np.int32), I32, CPU, events=events)
    buffers = []
    for _ in range(2):
        pinned = Tensor(cpu.array.copy(), I32, CPU, events=events)
        uva = Tensor(pinned.array, I32, CUDA, backing=pinned.array, events=events)
        buffers.append(NS(cpu=pinned, np=pinned.array, _uva=uva))
    pool = NS(_uva_bufs=buffers, _curr=0, max_concurrency=2, size=(1, 1), dtype=I32)
    table.num_blocks = NS(cpu=cpu, np=cpu.array, pool=pool, gpu=buffers[0]._uva[:1])
    return table


class Harness:
    """Skip startup only; execute real provider methods and hook classes."""
    def __init__(self, *, all_layers=False, start=0, end=256, real_control=False):
        p = self.p = StorageProvider.__new__(StorageProvider)
        self.events = []
        p.torch = fake_torch(self.events)
        p.storage_hooks, p.storage_ready = [], False
        p.failed, p.started, p.completed = False, True, False
        p.transfer, p.evidence = TransferBudget(), Evidence()
        p.counts = dict.fromkeys(('pre', 'processed', 'post'), 0)
        p.roots = {k: hashlib.sha256() for k in p.counts}
        p.writer_seen, p.sample_count, p.raw_heads, p.storage_frames = set(), 0, 0, 0
        p.head_events, p.emissions, p.decode_inputs = [], [], []
        p.head_counts = dict.fromkeys(('intermediate_prompt_chunk', 'final_prompt', 'decode'), 0)
        p.sampler_logits, p.sampler_result, p.sampler_identity = None, None, (0, 'b' * 64)
        p.manager_request_id, p.plan, p.deadline = 'request', {'plan_sha256': 'a' * 64}, float('inf')
        p.hidden, p.logit_seen = None, False
        p.contract, p.control = contract, contract.ProcessedKVControl() if real_control else RecordingControl()
        p.checker_path = ROOT / 'numerical_reference/prefill_native_control.py'
        p.checker_sha = hashlib.sha256(p.checker_path.read_bytes()).hexdigest()
        ids = list(range(1, (end + 15) // 16 + 1))
        p.manager_owner = manager_table(self.events, ids)
        p.table_owners = table_owner_signature(p.manager_owner)
        p.manager = ManagerInputs([16], [16], [1])
        p.manager.append(0, (ids,), True)
        p.connector = object()
        events = self.events
        class Sampler:
            def add_request(self, req_idx, prompt_len, sampling_params): return None
            def __call__(self, logits, input_batch):
                events.append('actual_sampler')
                logits.array[...] = 1  # Model actual in-place sampler transforms.
                return self.result
        p.sampler_owner = Sampler()
        p.sampler_owner.result = object()
        runner = NS(block_tables=p.manager_owner, sampler=p.sampler_owner, kv_connector=p.connector,
                    vllm_config=NS(kv_transfer_config=None), device=CUDA,
                    req_states=NS(req_id_to_index={'request': 0}))
        p.access = NS(_runner=runner, model=object(), groups={}, input_batch=NS(idx_mapping_np=[0]),
                      block_tables=p.manager_owner.input_block_tables,
                      head_binding=NS(dtype=BF16, suppressed=set(), require_current=lambda: None))
        p.layers, p.descriptors, mappings, slots, identities = {}, [], {}, {}, {}
        class Writer:
            def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
                events.append(('actual_writer', self.ordinal))
                self.calls += 1
                for row, slot in enumerate(slot_mapping.tolist()):
                    block, token = divmod(slot, 16)
                    kv_cache.array[block, :, token, :self.dim] = key.array[row]
                    kv_cache.array[block, :, token, self.dim:] = value.array[row]
                if self.mutate_inputs:
                    key.array[...] = 123
                    value.array[...] = 456
                if self.error is not None: raise self.error
        for layer in range(30):
            heads, dim = (2, 512) if layer % 6 == 5 else (8, 256)
            pages = len(ids) + 1 if all_layers or layer == 0 else 1
            # Construct physical BNHC then expose logical BHNC to exercise LBNHC.
            backing = np.zeros((pages, 16, heads, 2 * dim), dtype=np.uint16)
            cache = Tensor(backing.transpose(0, 2, 1, 3), backing=backing, events=events,
                           label='cache-' + str(layer))
            attn, impl = NS(kv_cache=cache, kv_cache_dtype="auto", kv_sharing_target_layer_name=None,
                            attn_backend=NS(forward_includes_kv_cache_update=False)), Writer()
            impl.ordinal, impl.dim, impl.calls, impl.mutate_inputs, impl.error = layer, dim, 0, False, None
            impl.kv_sharing_target_layer_name = None
            impl.is_kvcache_nvfp4 = False
            impl.cache_dtype = "auto"
            impl.num_kv_heads, impl.head_size = heads, dim
            impl.window_left = -1 if layer % 6 == 5 else 1023
            attn.num_kv_heads, attn.head_size, attn.head_size_v = heads, dim, dim
            attn.sliding_window = None if layer % 6 == 5 else 1024
            attn.impl = impl
            name = 'layer-' + str(layer)
            p.layers[layer] = (name, NS(attn=attn, is_kv_shared_layer=False), attn, impl)
            p.descriptors.append({'kv_heads': heads, 'head_dim': dim, 'window_size': attn.sliding_window})
            p.access.groups[name] = (0, None)
            slots[layer] = [position + 16 for position in range(start, end)]
            mappings[name] = Tensor(np.array(slots[layer], dtype=np.int64), I64, events=events,
                                    label='slots-' + str(layer))
            identities[layer] = self.identity(cache)
        p.ledger = Ledger(start, end, slots)
        p.frame = NS(owner=p, sequence=p.ledger.frames, request_id='request',
                     context=NS(slot_mapping=mappings), slots=slots, identities=identities,
                     block_tables={i: list(ids) for i in range(30)},
                     positions=Tensor(np.arange(start, end, dtype=np.int64), I64, events=events),
                     input_ids=Tensor(np.arange(start, end, dtype=np.int32), I32, events=events))
        p.read_frame = lambda *args: (slots, {}, identities, [])
        p.access.bind_frame = lambda *args: p.frame
        p.active = lambda: True
        if real_control:
            tables = {i: dict(zip(range(start, end), slots[i])) for i in range(30)}
            p.control.begin(contract.Frame(start, end), tables, tables)
        with patch('megartx.prefill_storage_native.require_source'), \
                patch('megartx.prefill_storage_native.require_method_source'):  # Host fakes are not the installed classes

            p.install_storage_hooks()
        p.storage_ready = True
        p.scratch = GpuScratch(p.torch, 8 << 20)

    @staticmethod
    def identity(cache):
        storage = cache.untyped_storage()
        return (id(cache), storage.data_ptr(), storage.nbytes(), cache.storage_offset(),
                tuple(cache.shape), tuple(cache.stride()), str(cache.device))

    def values(self, layer=0, k_word=0x0000, v_word=0x8000):
        p = self.p
        start, end, _ = p.ledger.pending
        descriptor = p.descriptors[layer]
        shape = (end - start, descriptor['kv_heads'], descriptor['head_dim'])
        key = Tensor(np.full(shape, k_word, dtype=np.uint16), events=self.events, label='key')
        value = Tensor(np.full(shape, v_word, dtype=np.uint16), events=self.events, label='value')
        return key, value

    def write(self, layer=0, key=None, value=None):
        p = self.p
        if key is None: key, value = self.values(layer)
        name, _, attn, impl = p.layers[layer]
        return impl.do_kv_cache_update(layer=attn, key=key, value=value, kv_cache=attn.kv_cache,
                                      slot_mapping=p.frame.context.slot_mapping[name])

    def finish(self, result=None):
        p, ticket = self.p, self.p.frame
        if result is None:
            result = Tensor(np.zeros((len(ticket.positions), 2816), dtype=np.uint16), events=self.events)
        context_module = ModuleType('vllm.forward_context')
        context_module.get_forward_context = lambda: ticket.context
        # Production's outer model wrapper poisons exceptions from finish.
        with patch.dict(sys.modules, {'vllm.forward_context': context_module}), \
                patch('megartx.prefill_storage_native.memory_sample', return_value={}):
            try:
                p.finish(ticket, result)
            except BaseException as error:
                p.poison_storage(error)
                raise
        return result

    def setup_head(self, position=2047):
        p = self.p
        p.frame = None
        rows = 256 if position < 2048 else 1
        p.ledger.end = position + 1
        p.ledger.frames = position // 256 + 1 if position < 2048 else 9 + position - 2048
        words = np.broadcast_to(np.arange(rows, dtype=np.uint16)[:, None], (rows, 2816)).copy()
        p.hidden = Tensor(words, events=self.events, label='full-hidden')
        p.logits_indices = Tensor(np.array([rows - 1], dtype=np.int64), I64,
                                  events=self.events, label='head-index')
        p.access.input_batch.logits_indices = p.logits_indices
        selected = Tensor(words[-1:].copy(), events=self.events, label='selected-hidden')
        logits = Tensor(np.full((1, 262144), 0x8000, dtype=np.uint16), events=self.events, label='logits')
        return selected, logits


class ProviderTests(unittest.TestCase):
    def setUp(self):
        # Existing legacy admissions have their own source-extracted suite. Keep
        # actual StorageProvider.require_hooks and every storage method intact.
        self.legacy = patch.object(NativeProvider, 'require_hooks', return_value=None)
        self.legacy.start()
        self.addCleanup(self.legacy.stop)

    def harness(self, **kwargs):
        h = Harness(**kwargs)
        self.addCleanup(self.cleanup, h)
        return h

    @staticmethod
    def cleanup(h):
        h.p.torch.cuda.failures.clear()
        # Restore only hooks that remain ours; drift tests intentionally retain
        # foreign replacements and their fake classes are isolated per fixture.
        for hook in reversed(h.p.storage_hooks):
            try: hook.restore()
            except RuntimeError: pass

    def test_all_rows_are_copied_before_original_writer_and_signed_zero_is_exact(self):
        h = self.harness()
        h.p.layers[0][3].mutate_inputs = True
        key, value = h.values()
        h.write(key=key, value=value)
        rows = h.p.control.processed_rows
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0][3], b'\x00\x00' * 2048)
        self.assertEqual(rows[0][4], b'\x00\x80' * 2048)
        self.assertTrue((key.array == 123).all())
        writer_index = h.events.index(('actual_writer', 0))
        copies = [i for i, e in enumerate(h.events) if isinstance(e, tuple) and e[:1] == ('cpu',)
                  and e[1] in ('key', 'value')]
        self.assertEqual(len(copies), 512)
        self.assertTrue(all(index < writer_index for index in copies))
        self.assertEqual(h.p.layers[0][3].calls, 1)
        self.assertEqual(set(h.p.evidence.raws), {'kv-layer-00-position-0015.bf16',
                                               'kv-layer-00-position-0016.bf16'})

    def test_selected_rows_at_all_declared_edges(self):
        for start, end, wanted in ((0, 256, (15, 16)), (768, 1024, (1023,)),
                                    (1024, 1280, (1024,)), (1792, 2048, (2047,)),
                                    (2048, 2049, (2048,))):
            with self.subTest(start=start):
                h = self.harness(start=start, end=end)
                h.write()
                self.assertEqual(set(h.p.evidence.raws), {
                    f'kv-layer-00-position-{p:04d}.bf16' for p in wanted})

    def test_missing_duplicate_wrong_owner_wrong_slot_and_replaced_cache_are_rejected(self):
        for change in ('missing', 'duplicate', 'layer', 'slot_owner', 'slot_contents',
                       'slot_dtype', 'slot_shape', 'cache_view'):
            with self.subTest(change=change):
                h = self.harness()
                p = h.p
                if change == 'missing': p.frame = None
                elif change == 'duplicate': p.writer_seen.add(0)
                elif change == 'layer': p.layers[0][2].impl = p.layers[1][3]
                elif change == 'slot_contents': p.frame.context.slot_mapping['layer-0'].array[0] += 1
                elif change == 'slot_dtype': p.frame.context.slot_mapping['layer-0'].dtype = I32
                elif change == 'slot_shape':
                    p.frame.context.slot_mapping['layer-0'].array = p.frame.context.slot_mapping['layer-0'].array.reshape(128, 2)
                elif change == 'cache_view':
                    p.layers[0][2].kv_cache.array = p.layers[0][2].kv_cache.array[:, :, ::-1, :]
                key, value = h.values()
                name, _, attn, impl = p.layers[0]
                slots = (Tensor(p.frame.context.slot_mapping[name].array.copy(), I64, events=h.events)
                         if change == 'slot_owner' else p.frame.context.slot_mapping[name] if p.frame else object())
                passed_layer = object() if change == 'layer' else attn
                with self.assertRaises((RuntimeError, ValueError)):
                    impl.do_kv_cache_update(passed_layer, key, value, attn.kv_cache, slots)
                self.assertTrue(p.failed)
                self.assertTrue(p.manager.poisoned)
                self.assertEqual(impl.calls, 0)

    def test_source_role_alias_and_source_cache_alias_are_rejected(self):
        for change in ('roles', 'cache'):
            with self.subTest(change=change):
                h = self.harness()
                key, value = h.values()
                if change == 'roles': value = key
                else: key.backing = h.p.layers[0][2].kv_cache.backing
                with self.assertRaisesRegex(RuntimeError, 'alias|provenance'):
                    h.write(key=key, value=value)
                self.assertTrue(h.p.failed)

    def test_equal_kv_values_with_distinct_source_ownership_are_legal(self):
        h = self.harness()
        key, value = h.values(k_word=0x3f80, v_word=0x3f80)
        self.assertNotEqual(key.data_ptr(), value.data_ptr())
        h.write(key=key, value=value)
        self.assertEqual(h.p.layers[0][3].calls, 1)
        self.assertFalse(h.p.failed)

    def test_nonfinite_processed_words_fail_before_original_writer(self):
        for word in (0x7f80, 0x7fc1):
            with self.subTest(word=word):
                h = self.harness(real_control=True)
                key, value = h.values(k_word=word)
                with self.assertRaisesRegex(ValueError, 'Nonfinite BF16'):
                    h.write(key=key, value=value)
                self.assertEqual(h.p.layers[0][3].calls, 0)
                self.assertTrue(h.p.failed)

    def test_metadata_only_decode_does_not_copy_or_qualify_processed_storage(self):
        h = self.harness(start=2049, end=2050)
        h.write()
        self.assertEqual(h.p.writer_seen, {0})
        self.assertEqual(h.p.counts, {'pre': 0, 'processed': 0, 'post': 0})
        self.assertEqual(h.p.evidence.raws, {})
        self.assertNotIn('processed', h.p.transfer.kinds)
        self.assertEqual(h.p.transfer.kinds['writer_slot_metadata'], 8)

    def test_full_first_frame_all_thirty_layers_stored_exact_and_finished(self):
        h = self.harness(all_layers=True, real_control=True)
        for layer in range(30): h.write(layer)
        self.assertEqual(h.p.writer_seen, set(range(30)))
        self.assertEqual(h.p.counts['processed'], 30 * 256)
        hidden = h.finish()
        self.assertEqual(h.p.control.end, 256)
        self.assertEqual(h.p.counts, {'pre': 0, 'processed': 7680, 'post': 7680})
        self.assertEqual(h.p.storage_frames, 1)
        self.assertEqual(h.p.sample_count, 60)
        self.assertIs(h.p.hidden, hidden)
        self.assertIsNone(h.p.frame)
        self.assertEqual(h.p.ledger.end, 256)

    def test_missing_one_writer_blocks_finish_before_cache_reads(self):
        h = self.harness()
        h.p.writer_seen = set(range(29))
        h.events.clear()
        with self.assertRaisesRegex(RuntimeError, 'Not all selected native writers'):
            h.finish()
        self.assertTrue(h.p.failed)
        self.assertFalse(any(isinstance(e, tuple) and e[0] == 'cpu' for e in h.events))

    def test_swapped_roles_or_changed_signed_zero_fail_real_storage_control(self):
        for mutation in ('swap', 'zero_sign'):
            with self.subTest(mutation=mutation):
                h = self.harness(all_layers=True, real_control=True)
                for layer in range(30): h.write(layer)
                cache = h.p.layers[0][2].kv_cache.array
                if mutation == 'swap':
                    cache[1, :, 0, :] = np.concatenate((cache[1, :, 0, 256:].copy(),
                                                       cache[1, :, 0, :256].copy()), axis=-1)
                else: cache[1, 0, 0, 0] = 0x8000
                with self.assertRaisesRegex(ValueError, 'Stored processed K/V'):
                    h.finish()
                self.assertTrue(h.p.failed)
                self.assertTrue(h.p.control.poisoned)

    def test_complete_required_union_is_checked_for_every_layer(self):
        h = self.harness()
        start, end = 1280, 1536
        tables = {layer: {p: p + 100 for p in contract.required_positions(layer, start, end)}
                  for layer in range(30)}
        with patch('megartx.prefill_storage_native.stored_row', return_value=(b'k', b'v')):
            h.p.retained_union('pre', start, end, tables)
            h.p.retained_union('post', start, end, tables)
        for layer in range(30):
            low = 0 if layer % 6 == 5 else 257
            for phase, stop in (('pre', start), ('post', end)):
                observed = [r[2] for r in h.p.control.retained_rows if r[0] == phase and r[1] == layer]
                self.assertEqual(observed, list(range(low, stop)))
        self.assertEqual(h.p.counts['pre'], 25 * 1023 + 5 * 1280)
        self.assertEqual(h.p.counts['post'], 25 * 1279 + 5 * 1536)

    def test_finish_synchronize_failure_or_interruption_poisons_and_drains_before_restore(self):
        for kind in (RuntimeError, KeyboardInterrupt):
            with self.subTest(kind=kind.__name__):
                h = self.harness()
                error = kind('query completion failed')
                h.p.torch.cuda.failures[h.p.torch.cuda.calls + 1] = error
                with self.assertRaises(kind) as caught: h.finish()
                self.assertIs(caught.exception, error)
                self.assertTrue(h.p.failed)
                self.assertTrue(h.p.control.poisoned)
                self.assertTrue(h.p.ledger.failed)
                self.assertGreaterEqual(h.p.torch.cuda.calls, 2)
                self.assertTrue(all(not hook.active for hook in h.p.storage_hooks))

    def test_failed_drain_retains_hooks_and_preserves_primary_exception(self):
        h = self.harness()
        primary = ValueError('primary control failure')
        h.p.torch.cuda.failures[h.p.torch.cuda.calls + 1] = RuntimeError('drain failed')
        h.p.poison_storage(primary)
        self.assertTrue(h.p.failed)
        self.assertTrue(all(hook.active for hook in h.p.storage_hooks))
        self.assertTrue(any('drain' in note.lower() for note in getattr(primary, '__notes__', ())))

    def test_successful_drain_precedes_every_callback_restore(self):
        h = self.harness()
        order = []
        native_synchronize = h.p.torch.cuda.synchronize
        def synchronize(*args):
            order.append('drain')
            native_synchronize(*args)
        h.p.torch.cuda.synchronize = synchronize
        for hook in h.p.storage_hooks:
            original = hook.restore
            def restore(original=original):
                order.append('restore')
                return original()
            hook.restore = restore
        h.p.restore_storage()
        self.assertEqual(order, ['drain'] + ['restore'] * len(h.p.storage_hooks))

    def test_foreign_callback_replacement_survives_poison_restoration(self):
        h = self.harness()
        foreign = lambda *args, **kwargs: None
        h.p.layers[0][3].do_kv_cache_update = foreign
        primary = ValueError('primary owner failure')
        h.p.poison_storage(primary)
        self.assertIs(h.p.layers[0][3].do_kv_cache_update, foreign)
        self.assertTrue(h.p.failed)
        self.assertTrue(any('restoration' in note for note in getattr(primary, '__notes__', ())))

    def test_same_content_manager_list_tensor_and_counter_replacement_is_rejected(self):
        for component in ('block_sizes', 'block_tables', 'persistent', 'input_block_tables', 'gathered',
                          'counts_np', 'pool_buffers', 'pool_buffer_np'):
            with self.subTest(component=component):
                h = self.harness()
                t = h.p.manager_owner
                if component == 'block_sizes': t.block_sizes = list(t.block_sizes)
                elif component == 'block_tables': t.block_tables = list(t.block_tables)
                elif component == 'persistent': t.block_tables[0].gpu = Tensor(t.block_tables[0].gpu.array.copy(), I32)
                elif component == 'input_block_tables': t.input_block_tables = list(t.input_block_tables)
                elif component == 'gathered': t.input_block_tables[0] = Tensor(t.input_block_tables[0].array.copy(), I32)
                elif component == 'counts_np': t.num_blocks.np = t.num_blocks.np.copy()
                elif component == 'pool_buffers': t.num_blocks.pool._uva_bufs = list(t.num_blocks.pool._uva_bufs)
                else: t.num_blocks.pool._uva_bufs[0].np = t.num_blocks.pool._uva_bufs[0].np.copy()
                with self.assertRaisesRegex(RuntimeError, 'ownership changed|no longer aliases'):
                    h.p.require_hooks()
                self.assertTrue(h.p.failed)

    def test_source_defined_count_uva_pool_rotation_is_allowed(self):
        h = self.harness()
        counts = h.p.manager_owner.num_blocks
        counts.pool._curr = 1
        current = counts.pool._uva_bufs[1]
        current.np[:] = counts.np
        counts.gpu = current._uva[:len(counts.np)]
        # Actual UvaBackedTensor.copy_to_uva changes this view every frame.
        h.p.require_hooks()
        self.assertFalse(h.p.failed)

    def test_arbitrary_equal_content_count_gpu_view_is_not_source_owned_rotation(self):
        h = self.harness()
        counts = h.p.manager_owner.num_blocks
        counts.gpu = Tensor(counts.gpu.array.copy(), I32)
        with self.assertRaises(RuntimeError): h.p.require_hooks()
        self.assertTrue(h.p.failed)

    def test_raw_head_is_copied_before_sampler_mutation_and_result_identity_is_bound(self):
        h = self.harness()
        hidden, logits = h.setup_head()
        h.p.head(h.p.access.model, hidden, logits)
        raw = h.p.evidence.raws['head-position-2047.bf16']
        self.assertEqual(raw, b'\x00\x80' * 262144)
        self.assertIs(h.p.sampler_logits, logits)
        result = h.p.sampler_owner(logits=logits, input_batch=h.p.access.input_batch)
        self.assertIs(result, h.p.sampler_owner.result)
        self.assertIs(h.p.sampler_result, result)
        self.assertIsNone(h.p.sampler_logits)
        self.assertTrue((logits.array == 1).all())
        self.assertEqual(h.p.evidence.raws['head-position-2047.bf16'], raw)
        self.assertEqual(h.p.head_events, [(2047, 2048, False)])
        with self.assertRaisesRegex(RuntimeError, 'Runner sample result differs'):
            h.p.sampled(object())

    def test_first_decode_head_captures_index_zero_at_the_consumed_anchor(self):
        h = self.harness()
        hidden, logits = h.setup_head(2048)
        h.p.head(h.p.access.model, hidden, logits)
        self.assertEqual(h.p.head_events, [(2048, 2049, False)])
        self.assertEqual(set(h.p.evidence.raws), {'head-position-2048.bf16'})
        head = next(value for name, value in h.p.evidence.records if name == 'heads.jsonl')
        self.assertEqual(head['selected_row_index'], 0)
        self.assertEqual(head['phase'], 'decode')

    def test_repeated_sampler_call_cannot_reuse_the_head(self):
        h = self.harness()
        hidden, logits = h.setup_head()
        h.p.head(h.p.access.model, hidden, logits)
        h.p.sampler_owner(logits, h.p.access.input_batch)
        with self.assertRaisesRegex(RuntimeError, 'did not consume'):
            h.p.sampler_owner(logits, h.p.access.input_batch)
        self.assertEqual(h.events.count('actual_sampler'), 1)
        self.assertTrue(h.p.failed)

    def test_actual_writer_and_head_budgets_reject_before_metadata_cpu_copy(self):
        for phase in ('writer', 'head'):
            with self.subTest(phase=phase):
                h = self.harness()
                h.p.transfer = TransferBudget(7)
                h.events.clear()
                with self.assertRaisesRegex(RuntimeError, 'before transfer'):
                    if phase == 'writer':
                        h.write()
                    else:
                        hidden, logits = h.setup_head()
                        h.p.head(h.p.access.model, hidden, logits)
                self.assertFalse(any(isinstance(e, tuple) and e[0] == 'cpu' for e in h.events))

    def test_sampler_rejects_equal_content_replacement_logits_or_input_batch(self):
        for component in ('logits', 'batch'):
            with self.subTest(component=component):
                h = self.harness()
                hidden, logits = h.setup_head()
                h.p.head(h.p.access.model, hidden, logits)
                actual_logits = Tensor(logits.array.copy()) if component == 'logits' else logits
                batch = NS(**vars(h.p.access.input_batch)) if component == 'batch' else h.p.access.input_batch
                with self.assertRaisesRegex(RuntimeError, 'did not consume'):
                    h.p.sampler_owner(actual_logits, batch)
                self.assertTrue(h.p.failed)
                self.assertNotIn('actual_sampler', h.events)

    def test_sampler_class_or_instance_call_replacement_is_detected(self):
        for location in ('class', 'instance'):
            with self.subTest(location=location):
                h = self.harness()
                replacement = lambda *args, **kwargs: None
                target = type(h.p.sampler_owner) if location == 'class' else h.p.sampler_owner
                target.__call__ = replacement
                with self.assertRaisesRegex(RuntimeError, 'sampler call ownership'):
                    h.p.require_hooks()
                self.assertTrue(h.p.failed)
                self.assertIs(target.__call__, replacement)

    def test_head_rejects_index_owner_value_hidden_bits_mask_and_nonfinite(self):
        for change in ('index_owner', 'index_value', 'hidden_bits', 'mask', 'nonfinite'):
            with self.subTest(change=change):
                h = self.harness()
                hidden, logits = h.setup_head()
                if change == 'index_owner': h.p.logits_indices = Tensor(np.array([255], dtype=np.int64), I64)
                elif change == 'index_value': h.p.logits_indices.array[0] = 0
                elif change == 'hidden_bits': hidden.array[0, 0] ^= 0x8000
                elif change == 'mask': h.p.access.head_binding.suppressed = {1}
                else: logits.array[0, 0] = 0x7f80
                with self.assertRaises((RuntimeError, ValueError)):
                    h.p.head(h.p.access.model, hidden, logits)
                self.assertEqual(h.p.head_events, [])
                self.assertEqual(h.p.evidence.raws, {})

    def test_intermediate_head_has_scalar_finite_check_and_no_raw_logit_capture(self):
        h = self.harness()
        hidden, logits = h.setup_head(255)
        # install_native_observer.observed owns the surrounding head scope.
        with h.p.scratch.scope('head'):
            h.p.head(h.p.access.model, hidden, logits)
        self.assertEqual(h.p.evidence.raws, {})
        self.assertEqual(h.p.head_events, [(255, 256, True)])
        self.assertEqual(h.p.transfer.kinds['head_finite_scalar'], 1)
        self.assertFalse(any(isinstance(e, tuple) and e[:2] == ('cpu', 'logits') for e in h.events))


class CopyAndHookTests(unittest.TestCase):
    def test_transfer_rejected_before_any_cpu_copy(self):
        events = []
        t = Tensor(np.zeros(4, dtype=np.uint16), events=events)
        with self.assertRaisesRegex(RuntimeError, 'before transfer'):
            native_bytes(t, fake_torch(events), TransferBudget(7), 'processed')
        self.assertEqual(events, [])

    def test_independent_storage_reader_respects_offset_and_lbn_hc_strides(self):
        events = []
        backing = np.arange(32 + 3 * 16 * 2 * 8, dtype=np.uint16)
        physical = backing[32:].reshape(3, 16, 2, 8)
        cache = Tensor(physical.transpose(0, 2, 1, 3), backing=backing, events=events)
        key, value = stored_row(cache, 17, 2, 4, fake_torch(events), TransferBudget(), 'post')
        self.assertEqual(key, physical[1, 1, :, :4].copy().tobytes())
        self.assertEqual(value, physical[1, 1, :, 4:].copy().tobytes())
        self.assertEqual(cache.storage_offset(), 32)

    def test_special_call_hook_changes_selected_instance_only_and_restores(self):
        calls = []
        class Sampler:
            def __call__(self, logits, input_batch): calls.append((self, logits, input_batch)); return logits
        selected, other = Sampler(), Sampler()
        before, after, failures = [], [], []
        original = Sampler.__call__
        hook = SpecialCallHook(selected, lambda a: before.append(a) or a['logits'],
                               lambda ticket, result: after.append((ticket, result)), failures.append)
        logits, batch = object(), object()
        self.assertIs(selected(logits=logits, input_batch=batch), logits)
        self.assertIs(other(logits, batch), logits)
        self.assertEqual(len(before), 1)
        self.assertIs(before[0]['self'], selected)
        self.assertEqual(after, [(logits, logits)])
        self.assertEqual(failures, [])
        hook.restore()
        self.assertIs(Sampler.__call__, original)

    def test_fresh_runtime_module_import_does_not_import_torch_or_vllm(self):
        code = "import sys; import megartx.prefill_storage_native; assert not any(n == 'torch' or n.startswith('torch.') or n == 'vllm' or n.startswith('vllm.') for n in sys.modules)"
        result = subprocess.run([sys.executable, '-S', '-c', code],
                                env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)




class ReviewRegressionTests(unittest.TestCase):
    setUp = ProviderTests.setUp
    tearDown = ProviderTests.tearDown
    harness = ProviderTests.harness
    cleanup = staticmethod(ProviderTests.cleanup)
    def test_selected_writer_dispatch_mutations_stop_before_any_native_call(self):
        for kind in ('parent_sharing','attention_sharing','impl_sharing','included_writer','nvfp4','cache_mode','head_size','head_count','window','parent_owner'):
            with self.subTest(kind=kind):
                h=self.harness();p=h.p;name,parent,attn,impl=p.layers[0]
                if kind=='parent_sharing': parent.is_kv_shared_layer=True
                elif kind=='attention_sharing': attn.kv_sharing_target_layer_name='other'
                elif kind=='impl_sharing': impl.kv_sharing_target_layer_name='other'
                elif kind=='included_writer': attn.attn_backend.forward_includes_kv_cache_update=True
                elif kind=='nvfp4': impl.is_kvcache_nvfp4=True
                elif kind=='cache_mode': impl.cache_dtype='fp8'
                elif kind=='head_size': impl.head_size=257
                elif kind=='head_count': impl.num_kv_heads=7
                elif kind=='window': impl.window_left=1022
                else: parent.attn=object()
                with self.assertRaisesRegex(RuntimeError,'sharing/dispatch'):
                    h.write(0)
                self.assertEqual(impl.calls,0)
                self.assertTrue(p.failed)

    def test_actual_extracted_writer_cannot_silently_skip_after_sharing_drift(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('storage_extracted_regression',ROOT/'tests/test_prefill_storage_extracted.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        h=self.harness();p=h.p;p.restore_storage();p.storage_hooks=[];p.storage_ready=False
        klass,namespace=fixture.extracted('writer');impl=klass()
        impl.kv_sharing_target_layer_name=None;impl.is_kvcache_nvfp4=False;impl.cache_dtype='auto'
        impl.num_kv_heads=8;impl.head_size=256;impl.window_left=1023
        calls=[];namespace['torch']=NS(ops=NS(_C_cache_ops=NS(reshape_and_cache_flash=lambda *args:calls.append(args))))
        name,parent,attn,_=p.layers[13];attn.impl=impl;p.layers[13]=(name,parent,attn,impl)
        with patch('megartx.prefill_storage_native.require_source'),patch('megartx.prefill_storage_native.require_method_source'):
            p.install_storage_hooks()
        p.storage_ready=True;impl.kv_sharing_target_layer_name='foreign-sharing-target'
        with self.assertRaisesRegex(RuntimeError,'sharing/dispatch'):
            h.write(13,*h.values(13,k_word=0,v_word=0))
        self.assertEqual(calls,[]);self.assertTrue(p.failed);self.assertNotIn(13,p.writer_seen)

    def test_mismatch_retains_one_bounded_role_provenance_record(self):
        h=self.harness(all_layers=True,real_control=True)
        for layer in range(30): h.write(layer)
        h.p.layers[13][2].kv_cache.array[2,3,7,61]^=0x8000
        with self.assertRaisesRegex(ValueError,'Stored processed') as caught: h.finish()
        records=[value for name,value in h.p.evidence.records if name=='storage-failure.json']
        self.assertEqual(len(records),1);record=records[0]
        self.assertEqual((record['phase'],record['layer'],record['absolute_position'],record['slot']),('post',13,23,39))
        self.assertEqual(record['expected_k_sha256'],h.p.control.expected[13][23][0].hex())
        self.assertNotEqual(record['expected_k_sha256'],record['observed_k_sha256'])
        self.assertEqual(record['expected_v_sha256'],record['observed_v_sha256'])
        self.assertIsNone(record['first_differing_word_index']);self.assertIsNone(record['expected_word'])
        self.assertTrue(any('Storage failure provenance' in note for note in caught.exception.__notes__))
        import json
        self.assertLessEqual(len(json.dumps(record).encode()),4096)

    def test_failed_mismatch_diagnostic_io_preserves_primary_error_and_poison(self):
        h=self.harness(all_layers=True,real_control=True)
        for layer in range(30): h.write(layer)
        h.p.layers[13][2].kv_cache.array[2,3,7,61]^=0x8000
        original=h.p.evidence.write
        def write(name,*args,**kwargs):
            if name=='storage-failure.json': raise OSError('secondary disk failure')
            return original(name,*args,**kwargs)
        h.p.evidence.write=write
        with self.assertRaisesRegex(ValueError,'Stored processed') as caught: h.finish()
        self.assertTrue(h.p.failed)
        self.assertTrue(any('failure-record write failed' in note for note in caught.exception.__notes__))


class SourceBoundMethodFixture:
    def operation(self, value):
        return value


class CallbackOriginTests(unittest.TestCase):
    def test_preinstallation_function_substitution_is_rejected(self):
        from megartx.prefill_storage import require_method_source
        owner = SourceBoundMethodFixture()
        require_method_source(owner, 'operation')
        original = SourceBoundMethodFixture.operation
        try:
            SourceBoundMethodFixture.operation = lambda self, value: value
            with self.assertRaisesRegex(RuntimeError, 'source-defined'):
                require_method_source(owner, 'operation')
        finally:
            SourceBoundMethodFixture.operation = original
        owner.operation = lambda value: value
        with self.assertRaisesRegex(RuntimeError, 'source-defined'):
            require_method_source(owner, 'operation')


if __name__ == '__main__':
    unittest.main()
