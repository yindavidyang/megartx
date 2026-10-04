"""Current selected-builder ownership on exact upstream CPU dispatch excerpts."""
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest

from megartx.prefill_diagnostic_plan import INSTALLED
import test_prefill_native_v2 as native_tests

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT/'tests/fixtures/prefill-native-builder-selection.json').read_text())
namespace = {'torch': NS(Tensor=type('Tensor', (), {})),
             'CommonAttentionMetadata': lambda **kwargs: NS(**kwargs)}
exec('from __future__ import annotations\n'+FIXTURE['getter']+'\n'+FIXTURE['dispatch'], namespace)
GETTER, DISPATCH = namespace['get_metadata_builder'], namespace['build_attn_metadata']


def dispatch(runner):
    n = len(runner.attn_groups)
    return DISPATCH(runner.attn_groups, 1, 256, [0, 256], [0, 256], 256, [256], 256,
                    [[1]]*n, [list(range(256))]*n, NS(kv_cache_groups=[object()]*n))


class SelectedBuilderOwnershipTests(unittest.TestCase):
    fixture = native_tests.BoundHookIdentityTests.fixture

    def test_exact_pinned_getter_and_dispatch_accept_original_owners(self):
        self.assertEqual(FIXTURE['files'], {name: INSTALLED[name] for name in FIXTURE['files']})
        with self.fixture(metadata_dispatch=dispatch, metadata_getter=GETTER) as (runner, provider, effects):
            provider.require_hooks()
            runner.execute_model(NS(total_num_scheduled_tokens=256))
            self.assertEqual(effects, ['metadata-work', 'model-work'])
            self.assertEqual(provider.log, ['record-metadata', 'begin', 'finish'])
            self.assertFalse(provider.failed)

    def test_selected_builder_and_owner_drift_reject_before_any_dispatch(self):
        mutations = ('selected-builder', 'builder-list', 'builder-list-empty', 'builder-list-added',
                     'outer-list', 'outer-remove', 'outer-add', 'inner-list', 'group-remove', 'group-add',
                     'group-object', 'layer-list', 'layer-name', 'gid', 'gid-bool',
                     'getter-function', 'getter-wrong-function', 'getter-wrong-owner', 'getter-class',
                     'retained-remove', 'retained-add', 'retained-gid', 'retained-layers')
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.fixture(metadata_dispatch=dispatch, metadata_getter=GETTER) as (runner, provider, effects):
                group = runner.attn_groups[0][0]
                original = group.metadata_builders[0]
                replacement = type(original)()
                replacement.build = lambda **kwargs: effects.append('replacement-work') or object()
                if mutation == 'selected-builder': group.metadata_builders[0] = replacement
                elif mutation == 'builder-list': group.metadata_builders = list(group.metadata_builders)
                elif mutation == 'builder-list-empty': group.metadata_builders.clear()
                elif mutation == 'builder-list-added': group.metadata_builders.append(original)
                elif mutation == 'outer-list': runner.attn_groups = list(runner.attn_groups)
                elif mutation == 'outer-remove': runner.attn_groups.clear()
                elif mutation == 'outer-add': runner.attn_groups.append([])
                elif mutation == 'inner-list': runner.attn_groups[0] = list(runner.attn_groups[0])
                elif mutation == 'group-remove': runner.attn_groups[0].clear()
                elif mutation == 'group-add': runner.attn_groups[0].append(group)
                elif mutation == 'group-object': runner.attn_groups[0][0] = type(group)(original)
                elif mutation == 'layer-list': group.layer_names = list(group.layer_names)
                elif mutation == 'layer-name': group.layer_names[0] = 'changed-layer'
                elif mutation == 'gid': group.kv_cache_group_id = 1
                elif mutation == 'gid-bool': group.kv_cache_group_id = False
                elif mutation == 'getter-function': group.get_metadata_builder = lambda *args: effects.append('getter-work') or original
                elif mutation == 'getter-wrong-function': group.get_metadata_builder = MethodType(lambda *args: effects.append('getter-work') or original, group)
                elif mutation == 'getter-wrong-owner': group.get_metadata_builder = MethodType(GETTER, type(group)(original))
                elif mutation == 'getter-class': type(group).get_metadata_builder = lambda *args: effects.append('getter-work') or original
                elif mutation == 'retained-remove': provider.access.builders.clear()
                elif mutation == 'retained-add': provider.access.builders[id(replacement)] = (replacement, 0, ('other',))
                elif mutation == 'retained-gid': provider.access.builders[id(original)] = (original, 1, tuple(group.layer_names))
                elif mutation == 'retained-layers': provider.access.builders[id(original)] = (original, 0, ('other',))
                with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                    runner.execute_model(NS(total_num_scheduled_tokens=256))
                self.assertEqual(effects, [])
                self.assertTrue(provider.failed)
                self.assertIn('abort', provider.log)
                self.assertNotIn('begin', provider.log)
                with self.assertRaisesRegex(RuntimeError, 'Poisoned'):
                    runner.execute_model(NS(total_num_scheduled_tokens=256))

    def test_order_and_partial_retained_set_drift_reject(self):
        def multiple(runner):
            first = runner.attn_groups[0][0]
            second = type(first)(type(runner.builder)())
            second.layer_names = ['layer-1']
            runner.attn_groups[0].append(second)
            third = type(first)(type(runner.builder)())
            third.layer_names, third.kv_cache_group_id = ['layer-2'], 1
            runner.attn_groups.append([third])
        for mutation in ('outer-order', 'inner-order', 'partial-retained', 'partial-selected', 'repeated-group', 'repeated-builder'):
            with self.subTest(mutation=mutation), self.fixture(before_cache=multiple, metadata_dispatch=dispatch, metadata_getter=GETTER) as (runner, provider, effects):
                if mutation == 'outer-order': runner.attn_groups.reverse()
                elif mutation == 'inner-order': runner.attn_groups[0].reverse()
                elif mutation == 'partial-retained': provider.access.builders.pop(id(runner.builder))
                elif mutation == 'partial-selected': runner.attn_groups[0].pop()
                elif mutation == 'repeated-group': runner.attn_groups[0][1] = runner.attn_groups[0][0]
                elif mutation == 'repeated-builder': runner.attn_groups[0][1].metadata_builders[0] = runner.builder
                with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                    runner.execute_model(NS(total_num_scheduled_tokens=256))
                self.assertEqual(effects, [])
                self.assertTrue(provider.failed)

    def test_invalid_initial_selection_rejects_without_invoking_shadowed_getter(self):
        for mutation in ('getter', 'duplicate-group', 'duplicate-builder', 'empty', 'wrong-gid'):
            effects = []
            def change(runner):
                group = runner.attn_groups[0][0]
                if mutation == 'getter':
                    group.get_metadata_builder = MethodType(lambda *args: effects.append('getter-work') or runner.builder, group)
                elif mutation == 'duplicate-group': runner.attn_groups[0].append(group)
                elif mutation == 'duplicate-builder':
                    second = type(group)(runner.builder)
                    second.layer_names = ['layer-1']
                    runner.attn_groups[0].append(second)
                elif mutation == 'empty': runner.attn_groups[0].clear()
                elif mutation == 'wrong-gid': group.kv_cache_group_id = 1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                with self.fixture(before_cache=change, metadata_dispatch=dispatch, metadata_getter=GETTER):
                    self.fail('Invalid ownership cannot publish admission')
            self.assertEqual(effects, [])

    def test_selection_getter_interrupt_preserves_primary_when_abort_fails(self):
        with self.fixture(metadata_dispatch=dispatch, metadata_getter=GETTER) as (runner, provider, effects):
            primary = KeyboardInterrupt('ownership admission interruption')
            def interrupted(*args): raise primary
            def failed_abort(): raise OSError('secondary abort failure')
            provider.access.builder_ownership.matches = interrupted
            provider.abort = failed_abort
            with self.assertRaises(KeyboardInterrupt) as caught:
                runner.execute_model(NS(total_num_scheduled_tokens=256))
            self.assertIs(caught.exception, primary)
            self.assertTrue(provider.failed)
            self.assertEqual(effects, [])
            if hasattr(primary, 'add_note'):
                self.assertTrue(any('secondary abort failure' in note for note in primary.__notes__))


if __name__ == '__main__': unittest.main()
