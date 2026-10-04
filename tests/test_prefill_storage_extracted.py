"""Execute frozen native method bodies with host fakes and the real adapter core.

No Torch/vLLM import, device work, native arithmetic, or fit claim is involved.
The exact excerpts retain upstream method parameters and statements; future
annotations defer their optional runtime type names without rewriting bodies.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from megartx.prefill_storage import InstanceHook, ManagerInputs

FIXTURES = ROOT / 'tests' / 'fixtures' / 'prefill-storage-source'
PINS = {
    'writer': ('8ee541fde43ed92a417b3f01ea1e6fc64af8ab2eb54cbfc28308a49aaca4ed7f',
               'bfee76131f53a9c1710c939e54d804a8147088c1e2e7da41a08b28e71da197da'),
    'manager_append': ('61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a',
                       '6f2c2d7ade31117b8a41f946c3cb7b2b836ace9b464654444e343d0f0c053fc3'),
}


def extracted(kind):
    manifest = json.loads((FIXTURES / 'provenance.json').read_text())
    item = manifest['sources'][kind]
    data = (FIXTURES / item['file']).read_bytes()
    whole_hash, excerpt_hash = PINS[kind]
    if (item['whole_source_sha256'] != whole_hash
            or item['excerpt_sha256'] != excerpt_hash
            or hashlib.sha256(data).hexdigest() != excerpt_hash):
        raise AssertionError('Frozen native source provenance changed')
    # Preserve the source's class indentation and every original statement.
    source = 'from __future__ import annotations\nclass ' + item['class'] + ':\n' + data.decode()
    namespace = {}
    exec(compile(source, str(FIXTURES / item['file']), 'exec'), namespace)
    return namespace[item['class']], namespace


class Counts:
    def __init__(self):
        self.values = {}

    def __getitem__(self, index):
        return self.values.get(index, 0)

    def __setitem__(self, index, value):
        self.values[index] = value


class StagedTable:
    def __init__(self, events, capacity=160):
        self.gpu = NS(shape=(1, capacity))
        self.events, self.rows = events, {}
        self.failure = None

    def stage_write(self, request, start, values):
        self.events.append(('stage', request, start, tuple(values)))
        if self.failure is not None:
            raise self.failure
        row = self.rows.setdefault(request, [None] * self.gpu.shape[1])
        row[start:start + len(values)] = list(values)


class ManagerHarness:
    """Real InstanceHook/ManagerInputs surrounding the extracted native append.

    Models provider poisoning, not a runnable GPU adapter. Native staged table
    buffers are deliberately separate from immutable expected manager inputs.
    """
    def __init__(self, ratios=(1, 2, 4)):
        klass, _ = extracted('manager_append')
        self.native = klass()
        self.events, self.failures = [], []
        self.native.num_kv_cache_groups = len(ratios)
        self.native.blocks_per_kv_block = list(ratios)
        self.native.num_blocks = NS(np=Counts())
        self.native.block_tables = [StagedTable(self.events) for _ in ratios]
        self.inputs = ManagerInputs([16 * r for r in ratios], [16] * len(ratios), ratios)
        self.hook = InstanceHook(self.native, 'append_block_ids', self.before, self.after, self.fail)

    def before(self, arguments):
        self.events.append('before')
        self.inputs.append(**arguments)
        return self.inputs.tables

    def after(self, snapshot, result):
        if self.inputs.tables != snapshot or result is not None:
            raise AssertionError('Unexpected native append completion')
        self.events.append('after')

    def fail(self, error):
        self.inputs.poisoned = True
        self.failures.append(error)
        self.events.append('poison')

    def persistent(self, group):
        count = self.native.num_blocks.np[group, 0]
        return tuple(self.native.block_tables[group].rows[0][:count])


class WriterHarness:
    def __init__(self):
        klass, namespace = extracted('writer')
        self.native, self.events, self.failures = klass(), [], []
        self.native.kv_sharing_target_layer_name = None
        self.native.is_kvcache_nvfp4 = False
        self.native.head_size, self.native.cache_dtype = 256, 'auto'
        self.key, self.value, self.slots = object(), object(), object()
        self.k_cache, self.v_cache = object(), object()
        self.layer = NS(_k_scale=object(), _v_scale=object())
        self.calls, self.before_error, self.native_error, self.after_error = [], None, None, None
        harness = self

        class Cache:
            def transpose(self, left, right):
                if (left, right) != (1, 2):
                    raise AssertionError('Changed native cache transpose')
                harness.events.append('transpose')
                return self

            def split(self, width, *, dim):
                if (width, dim) != (256, -1):
                    raise AssertionError('Changed native BF16 K/V split')
                harness.events.append('split')
                return harness.k_cache, harness.v_cache

        self.cache = Cache()
        namespace['torch'] = NS(ops=NS(_C_cache_ops=NS(reshape_and_cache_flash=self.write)))
        self.original = self.native.do_kv_cache_update
        self.hook = InstanceHook(self.native, 'do_kv_cache_update', self.before, self.after, self.fail)

    def before(self, arguments):
        self.events.append('copy_processed_before')
        expected = dict(layer=self.layer, key=self.key, value=self.value,
                        kv_cache=self.cache, slot_mapping=self.slots)
        if set(arguments) != set(expected) or any(arguments[k] is not v for k, v in expected.items()):
            raise AssertionError('Original writer input identity changed')
        if self.before_error is not None:
            raise self.before_error
        return object()

    def write(self, *arguments):
        expected = (self.key, self.value, self.k_cache, self.v_cache, self.slots,
                    self.native.cache_dtype, self.layer._k_scale, self.layer._v_scale)
        if any(actual is not wanted for actual, wanted in zip(arguments, expected)):
            raise AssertionError('Original native write arguments changed')
        if len(arguments) != len(expected):
            raise AssertionError('Original native write argument count changed')
        self.calls.append(arguments)
        self.events.append('native_write')
        if self.native_error is not None:
            raise self.native_error

    def after(self, ticket, result):
        if ticket is None or result is not None:
            raise AssertionError('Original native return was changed')
        self.events.append('after')
        if self.after_error is not None:
            raise self.after_error

    def fail(self, error):
        self.failures.append(error)
        self.events.append('poison')

    def invoke(self, style='positional'):
        values = (self.layer, self.key, self.value, self.cache, self.slots)
        if style == 'positional':
            return self.native.do_kv_cache_update(*values)
        if style == 'keyword':
            return self.native.do_kv_cache_update(**dict(zip(
                ('layer', 'key', 'value', 'kv_cache', 'slot_mapping'), values)))
        return self.native.do_kv_cache_update(self.layer, self.key, value=self.value,
                                            kv_cache=self.cache, slot_mapping=self.slots)


class SourceExtractionTests(unittest.TestCase):
    def test_manifest_and_bodies_are_pinned(self):
        manifest = json.loads((FIXTURES / 'provenance.json').read_text())
        self.assertEqual(manifest['upstream_revision'], 'ced6857afa0ea7b2e3f0846a62e1394e90f15607')
        self.assertIs(manifest['runtime_imports'], False)
        for kind, first, last, parameters in (
                ('writer', 2577, 2614, ('self', 'layer', 'key', 'value', 'kv_cache', 'slot_mapping')),
                ('manager_append', 113, 133, ('self', 'req_index', 'new_block_ids', 'overwrite'))):
            with self.subTest(kind=kind):
                klass, _ = extracted(kind)
                entry = manifest['sources'][kind]
                self.assertEqual((entry['first_line'], entry['last_line']), (first, last))
                self.assertEqual(tuple(inspect.signature(getattr(klass, entry['method'])).parameters), parameters)

    def test_fresh_import_and_source_execution_do_not_import_runtime(self):
        script = '''
import ast, pathlib, runpy, sys
from megartx.prefill_storage import InstanceHook, ManagerInputs
scope = runpy.run_path(sys.argv[1])
scope['extracted']('writer')
scope['extracted']('manager_append')
assert not any(x == 'torch' or x.startswith('torch.') or x == 'vllm' or x.startswith('vllm.') for x in sys.modules)
'''
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        result = subprocess.run([sys.executable, '-S', '-c', script, __file__], env=environment,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class ExtractedWriterTests(unittest.TestCase):
    def test_positional_keyword_and_mixed_calls_preserve_identity_order_and_count(self):
        for style in ('positional', 'keyword', 'mixed'):
            with self.subTest(style=style):
                h = WriterHarness()
                self.assertIsNone(h.invoke(style))
                self.assertEqual(len(h.calls), 1)
                self.assertEqual(h.events, ['copy_processed_before', 'transpose', 'split', 'native_write', 'after'])
                self.assertEqual(h.failures, [])
                h.hook.restore()
                self.assertNotIn('do_kv_cache_update', vars(h.native))
                self.assertIs(h.native.do_kv_cache_update.__self__, h.native)
                self.assertIs(h.native.do_kv_cache_update.__func__, h.original.__func__)

    def test_legacy_keyword_aliases_fail_before_native_write(self):
        h = WriterHarness()
        with self.assertRaises(TypeError):
            h.native.do_kv_cache_update(h.layer, h.key, h.value, cache=h.cache, slots=h.slots)
        self.assertEqual(h.calls, [])
        self.assertEqual(h.events, ['poison'])

    def test_duplicate_argument_fails_before_native_write(self):
        h = WriterHarness()
        with self.assertRaises(TypeError):
            h.native.do_kv_cache_update(h.layer, h.key, h.value, h.cache, h.slots, key=h.key)
        self.assertEqual(h.calls, [])
        self.assertEqual(h.events, ['poison'])

    def test_before_native_and_after_interruptions_propagate_and_poison(self):
        for phase in ('before', 'native', 'after'):
            for error_class in (RuntimeError, KeyboardInterrupt):
                with self.subTest(phase=phase, error=error_class.__name__):
                    h = WriterHarness()
                    error = error_class('injected ' + phase)
                    setattr(h, phase + '_error', error)
                    with self.assertRaises(error_class) as caught:
                        h.invoke()
                    self.assertIs(caught.exception, error)
                    self.assertEqual(h.failures, [error])
                    self.assertEqual(len(h.calls), 0 if phase == 'before' else 1)
                    self.assertEqual(h.events[-1], 'poison')
                    h.hook.restore()
                    self.assertIs(h.native.do_kv_cache_update.__func__, h.original.__func__)

    def test_other_instances_are_unchanged(self):
        h = WriterHarness()
        other = type(h.native)()
        self.assertIs(other.do_kv_cache_update.__func__, h.original.__func__)
        self.assertNotIn('do_kv_cache_update', vars(other))

    def test_class_replacement_blocks_saved_wrapper_and_restoration(self):
        h = WriterHarness()
        saved = h.native.do_kv_cache_update
        replacement = lambda self, *args, **kwargs: None
        type(h.native).do_kv_cache_update = replacement
        with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
            saved(h.layer, h.key, h.value, h.cache, h.slots)
        self.assertEqual(h.calls, [])
        with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
            h.hook.restore()
        self.assertIs(type(h.native).do_kv_cache_update, replacement)

    def test_instance_replacement_blocks_saved_wrapper_and_is_never_overwritten(self):
        for kind in ('function', 'bound', 'wrong_owner', 'same_function_wrong_owner'):
            with self.subTest(kind=kind):
                h = WriterHarness()
                saved = h.native.do_kv_cache_update
                other = type(h.native)()
                replacement = lambda *args, **kwargs: None
                if kind == 'bound':
                    replacement = MethodType(replacement, h.native)
                elif kind == 'wrong_owner':
                    replacement = MethodType(replacement, other)
                elif kind == 'same_function_wrong_owner':
                    replacement = MethodType(h.hook.wrapper, other)
                h.native.do_kv_cache_update = replacement
                with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
                    saved(h.layer, h.key, h.value, h.cache, h.slots)
                with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
                    h.hook.restore()
                self.assertIs(h.native.do_kv_cache_update, replacement)
                self.assertEqual(h.calls, [])

    def test_direct_wrong_receiver_is_rejected(self):
        h = WriterHarness()
        with self.assertRaisesRegex(RuntimeError, 'owner changed'):
            h.hook.wrapper(object(), h.layer, h.key, h.value, h.cache, h.slots)
        self.assertEqual(h.calls, [])

    def test_preexisting_override_is_rejected(self):
        h = WriterHarness()
        with self.assertRaisesRegex(RuntimeError, 'preexisting instance'):
            InstanceHook(h.native, 'do_kv_cache_update', h.before, h.after, h.fail)

    def test_restore_is_idempotent_but_saved_wrapper_cannot_run(self):
        h = WriterHarness()
        saved = h.native.do_kv_cache_update
        h.hook.restore()
        h.hook.restore()
        with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
            saved(h.layer, h.key, h.value, h.cache, h.slots)
        self.assertEqual(h.calls, [])


class ExtractedManagerTests(unittest.TestCase):
    def test_ratios_one_two_four_match_actual_native_expansion_and_slots(self):
        h = ManagerHarness()
        original_inputs = ([7, 9], [13, 17], [19, 23])
        h.native.append_block_ids(0, original_inputs, overwrite=True)
        for group, ratio in enumerate((1, 2, 4)):
            with self.subTest(ratio=ratio):
                native = h.persistent(group)
                self.assertEqual(native, h.inputs.expanded(group))
                self.assertEqual(h.native.num_blocks.np[group, 0], 2 * ratio)
                end = 2 * 16 * ratio
                slots = h.inputs.slots(group, range(end))
                # Independent manager-domain equation, no writer slots as input.
                expected = {p: original_inputs[group][p // (16 * ratio)] * (16 * ratio)
                            + p % (16 * ratio) for p in range(end)}
                self.assertEqual(slots, expected)
                self.assertEqual(list(slots.values()), [native[p // 16] * 16 + p % 16 for p in range(end)])
                h.inputs.reconcile(group, native, native, len(native), end)
        self.assertEqual(original_inputs, ([7, 9], [13, 17], [19, 23]))
        self.assertEqual(h.events[0], 'before')
        self.assertEqual(h.events[-1], 'after')

    def test_append_positional_keyword_mixed_and_unchanged_empty_group(self):
        h = ManagerHarness()
        h.native.append_block_ids(0, ([7], [13], [19]), True)
        h.native.append_block_ids(req_index=0, new_block_ids=([9], [], [23]), overwrite=False)
        h.native.append_block_ids(0, new_block_ids=([], [17], []), overwrite=False)
        self.assertEqual(h.inputs.tables, ((7, 9), (13, 17), (19, 23)))
        self.assertEqual(h.inputs.calls, 3)
        for group in range(3):
            self.assertEqual(h.persistent(group), h.inputs.expanded(group))

    def test_caller_mutation_after_capture_cannot_change_expected_tables(self):
        h = ManagerHarness((2,))
        inputs = ([7, 9],)
        h.native.append_block_ids(0, inputs, True)
        inputs[0][0] = 99
        inputs[0].append(111)
        self.assertEqual(h.inputs.tables, ((7, 9),))
        self.assertEqual(h.inputs.slots(0, [0, 31, 32]), {0: 224, 31: 255, 32: 288})
        self.assertEqual(h.persistent(0), (14, 15, 18, 19))

    def test_prefix_preserving_overwrite_accepts_same_table_and_extension(self):
        h = ManagerHarness((2,))
        h.native.append_block_ids(0, ([7, 9],), True)
        h.native.append_block_ids(0, ([7, 9],), True)
        h.native.append_block_ids(0, ([7, 9, 11],), True)
        self.assertEqual(h.inputs.tables, ((7, 9, 11),))
        self.assertEqual(h.persistent(0), (14, 15, 18, 19, 22, 23))
        self.assertTrue(all(event[2] == 0 for event in h.events if isinstance(event, tuple)))

    def test_overwrite_cannot_remap_reorder_shorten_or_duplicate_committed_prefix(self):
        for replacement in ([8, 9], [9, 7], [7], [7, 9, 7]):
            with self.subTest(replacement=replacement):
                h = ManagerHarness((1,))
                h.native.append_block_ids(0, ([7, 9],), True)
                before = h.persistent(0)
                with self.assertRaisesRegex(ValueError, 'Recycled/reordered'):
                    h.native.append_block_ids(0, (replacement,), True)
                self.assertEqual(h.persistent(0), before)
                self.assertTrue(h.inputs.poisoned)
                with self.assertRaisesRegex(ValueError, 'unpoisoned'):
                    h.inputs.slots(0, [0])

    def test_append_cannot_reuse_previously_owned_manager_block(self):
        h = ManagerHarness((4,))
        h.native.append_block_ids(0, ([7, 9],), True)
        before = h.persistent(0)
        with self.assertRaisesRegex(ValueError, 'Recycled/reordered'):
            h.native.append_block_ids(0, ([7],), False)
        self.assertEqual(h.persistent(0), before)
        self.assertTrue(h.inputs.poisoned)

    def test_missing_initial_capture_cannot_be_recovered_from_actual_table(self):
        h = ManagerHarness((2,))
        # Simulate installation too late: native original ran with no hook.
        h.hook.original(0, ([7, 9],), True)
        self.assertEqual(h.persistent(0), (14, 15, 18, 19))
        with self.assertRaisesRegex(ValueError, 'unpoisoned original manager'):
            h.inputs.reconcile(0, h.persistent(0), h.persistent(0), 4, 64)
        with self.assertRaisesRegex(ValueError, 'Missing first actual manager overwrite'):
            h.native.append_block_ids(0, ([11],), False)
        self.assertTrue(h.inputs.poisoned)

    def test_original_manager_inputs_are_independent_of_corrupted_actual_tables(self):
        for corrupted in ('persistent', 'gathered', 'count'):
            with self.subTest(corrupted=corrupted):
                h = ManagerHarness((2,))
                h.native.append_block_ids(0, ([7, 9],), True)
                persistent, gathered, count = list(h.persistent(0)), list(h.persistent(0)), 4
                if corrupted == 'persistent':
                    persistent[0] = 100
                elif corrupted == 'gathered':
                    gathered[0] = 100
                else:
                    count += 1
                self.assertEqual(h.inputs.slots(0, [0]), {0: 224})
                with self.assertRaisesRegex(ValueError, 'differs from original manager'):
                    h.inputs.reconcile(0, persistent, gathered, count, 64)
                self.assertTrue(h.inputs.poisoned)

    def test_native_append_interruption_poison_prevents_partial_commit_use_and_restores(self):
        for error_class in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error_class.__name__):
                h = ManagerHarness((1, 2))
                error = error_class('native group 1 interrupted')
                h.native.block_tables[1].failure = error
                with self.assertRaises(error_class) as caught:
                    h.native.append_block_ids(0, ([7], [13]), True)
                self.assertIs(caught.exception, error)
                self.assertEqual(h.persistent(0), (7,))  # Actual source partially wrote group 0.
                self.assertEqual(h.native.num_blocks.np[1, 0], 0)
                self.assertEqual(h.failures, [error])
                self.assertTrue(h.inputs.poisoned)
                self.assertNotIn('after', h.events)
                with self.assertRaisesRegex(ValueError, 'unpoisoned'):
                    h.inputs.expanded(0)
                before = list(h.events)
                with self.assertRaisesRegex(ValueError, 'Poisoned'):
                    h.native.append_block_ids(0, ([9], [17]), False)
                self.assertEqual([x for x in h.events[len(before):] if isinstance(x, tuple)], [])
                original = h.hook.original.__func__
                h.hook.restore()
                self.assertNotIn('append_block_ids', vars(h.native))
                self.assertIs(h.native.append_block_ids.__func__, original)

    def test_native_capacity_error_also_poisons_manager_receipt(self):
        h = ManagerHarness((2,))
        h.native.block_tables[0].gpu.shape = (1, 1)
        with self.assertRaisesRegex(RuntimeError, 'row capacity'):
            h.native.append_block_ids(0, ([7],), True)
        self.assertTrue(h.inputs.poisoned)
        self.assertEqual(h.native.num_blocks.np[0, 0], 0)

    def test_manager_class_and_instance_replacements_cannot_run_saved_wrapper(self):
        for location in ('class', 'instance'):
            with self.subTest(location=location):
                h = ManagerHarness((2,))
                saved = h.native.append_block_ids
                replacement = lambda *args, **kwargs: None
                if location == 'class':
                    type(h.native).append_block_ids = replacement
                else:
                    h.native.append_block_ids = replacement
                with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
                    saved(0, ([7],), True)
                self.assertTrue(h.inputs.poisoned)
                self.assertEqual(h.events, ['poison'])
                self.assertEqual(h.native.num_blocks.np[0, 0], 0)
                with self.assertRaisesRegex(RuntimeError, 'class/instance replacement'):
                    h.hook.restore()
                target = type(h.native) if location == 'class' else h.native
                self.assertIs(getattr(target, 'append_block_ids'), replacement)

    def test_second_request_index_is_rejected_before_native_append(self):
        h = ManagerHarness((1,))
        h.native.append_block_ids(0, ([7],), True)
        with self.assertRaisesRegex(ValueError, 'More than one manager request'):
            h.native.append_block_ids(1, ([9],), True)
        self.assertNotIn(1, h.native.block_tables[0].rows)
        self.assertTrue(h.inputs.poisoned)

    def test_table_owner_replacement_is_detected_when_reconciling_contents(self):
        h = ManagerHarness((2,))
        h.native.append_block_ids(0, ([7],), True)
        h.native.block_tables[0] = StagedTable(h.events)
        h.native.block_tables[0].rows[0] = [90, 91]
        with self.assertRaisesRegex(ValueError, 'differs from original manager'):
            h.inputs.reconcile(0, h.persistent(0), (14, 15), 2, 32)
        self.assertTrue(h.inputs.poisoned)
        # Equal-content owner replacement requires the runtime owner's identity
        # checks; ManagerInputs deliberately qualifies values, not live owners.

    def test_extraction_fixture_contains_no_runtime_import_statements(self):
        for kind in PINS:
            _, namespace = extracted(kind)
            entry = json.loads((FIXTURES / 'provenance.json').read_text())['sources'][kind]
            source = 'class Native:\n' + (FIXTURES / entry['file']).read_text()
            self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.parse(source))))
            self.assertNotIn('torch', namespace)
            self.assertNotIn('vllm', namespace)


if __name__ == '__main__':
    unittest.main()
