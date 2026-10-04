"""Execute pinned selection excerpts and actual guard code on CPU boundaries.

Substituted external objects exercise rejection and routing only. No fixture
admits a native runner, tensor observation, fit, or numerical comparison.
"""
import ast
import copy
from contextlib import contextmanager
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from megartx.prefill_diagnostic_plan import INSTALLED, RUNNER_POLICY
from megartx.prefill_runner_binding import HOOKS, require_default_selection, validate_binding
from megartx.prefill_native import NativeProvider, install_native_observer
from test_prefill_native_safety import Tensor, TORCH, Provider, frame_fixture, hook_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
FIXTURE = json.loads((ROOT/'tests/fixtures/prefill-native-runner-selection.json').read_text())


class Selection(unittest.TestCase):
    def resolve(self, explicit=None, triton=True, unsupported=(), hisparse=None, watermark=None):
        platform = ModuleType('vllm.platforms')
        platform.current_platform = NS(is_rocm=lambda: False)
        globals_ = {'envs': NS(VLLM_USE_V2_MODEL_RUNNER=explicit), 'HAS_TRITON': triton,
                    'logger': NS(warning_once=lambda *a: None, info_once=lambda *a: None),
                    'ROCM_DEFAULT_MRV1_ARCHITECTURES': ()}
        exec('class Config:\n'+'\n'.join('    '+line for line in FIXTURE['property'].splitlines()), globals_)
        config = globals_['Config']()
        config.attention_config = NS(hisparse_config=hisparse)
        config.watermark_config, config.model_config = watermark, NS()
        config._get_v2_model_runner_unsupported_features = lambda: unsupported
        with patch.dict(sys.modules, {'vllm': ModuleType('vllm'), 'vllm.platforms': platform}):
            return config.use_v2_model_runner

    def test_actual_default_selector_and_fallbacks(self):
        self.assertEqual(FIXTURE['files'], {k: INSTALLED[k] for k in FIXTURE['files']})
        self.assertIs(self.resolve(), True)
        self.assertIs(self.resolve(explicit=False), False)
        self.assertIs(self.resolve(triton=False), False)
        self.assertIs(self.resolve(unsupported=('cpu boundary',)), False)
        self.assertIs(self.resolve(explicit=False, watermark=object()), True)
        with self.assertRaises(ValueError): self.resolve(explicit=False, hisparse=object())

    def test_actual_worker_import_selects_distinct_class(self):
        modules, v2, _ = hook_fixture()
        v1 = modules['vllm.v1.worker.gpu_model_runner'].GPUModelRunner
        for selected, expected in ((True, v2), (False, v1)):
            owner = NS(use_v2_model_runner=selected, vllm_config=NS(is_mm_encoder_only=False), device=None)
            # Constructor boundaries stand in for CUDA-dependent classes.
            v2.__init__ = lambda self, *a: None
            v1.__init__ = lambda self, *a: None
            with patch.dict(sys.modules, modules):
                exec(FIXTURE['worker_branch'], {'self': owner, 'GPUModelRunner': v1})
            self.assertIs(type(owner.model_runner), expected)

    def test_actual_environment_selector_and_no_override_policy(self):
        globals_ = {'os': os, 'maybe_convert_bool': lambda s: None if s is None else bool(int(s))}
        selector = eval(FIXTURE['environment_selector'], globals_)
        for value, expected in ((None, None), ('0', False), ('1', True)):
            with patch.dict(os.environ, {}, clear=True):
                if value is not None: os.environ['VLLM_USE_V2_MODEL_RUNNER'] = value
                self.assertIs(selector(), expected)
                config = NS(use_v2_model_runner=self.resolve(explicit=expected), is_mm_encoder_only=False)
                if value is None: self.assertEqual(require_default_selection(config), RUNNER_POLICY)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'no override'): require_default_selection(config)
        with patch.dict(os.environ, {}, clear=True):
            for resolved in (False, None, 1):
                with self.assertRaises(RuntimeError):
                    require_default_selection(NS(use_v2_model_runner=resolved, is_mm_encoder_only=False))


class DispatchGate(unittest.TestCase):
    def test_source_drift_rejected_without_runtime_import(self):
        from megartx.prefill_runner_binding import verify_installed_files
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'__init__.py').write_text('# CPU substituted source root\n')
            path=root/'envs.py';path.write_text('# CPU source boundary\n')
            package=ModuleType('vllm');package.__file__=str(root/'__init__.py')
            pins={'vllm.envs':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.dict(sys.modules, {'vllm':package}),patch('megartx.prefill_runner_binding.INSTALLED',pins):
                verify_installed_files()
                path.write_text('# changed CPU boundary\n')
                with self.assertRaisesRegex(RuntimeError,'source drift'):verify_installed_files()

    def receipt(self):
        return {'schema': 'megartx-prefill-runner-binding-v1', 'plan_sha256': 'a'*64,
                'source_head': 'b'*40, 'owner_pid': 101, 'owner_start_ticks': 17,
                'runner_policy': RUNNER_POLICY.copy(), 'installed_sources': INSTALLED.copy(),
                'hook_bindings': dict.fromkeys(HOOKS, True), 'mutable_lease_granted': False}

    def test_missing_binding_prevents_http_import_marker_and_post(self):
        from prefill_diagnostic_client import run
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'absent before request dispatch'):
                run({}, d, 100)
            self.assertFalse((Path(d)/'request.json').exists())

    def test_binding_rejects_stale_unowned_source_unknown_hooks_and_lease(self):
        good = self.receipt()
        plan = {k: good[k] for k in ('plan_sha256', 'source_head', 'runner_policy')}
        with tempfile.TemporaryDirectory() as d, patch('megartx.prefill_runner_binding.start_ticks', return_value=17):
            path = Path(d)/'runner-binding.json'
            path.write_text(json.dumps(good))
            validate_binding(plan, d, {(101, 17)})
            for key, value in (('owner_start_ticks', 18), ('installed_sources', {}),
                               ('hook_bindings', {**good['hook_bindings'], 'model_forward': False}),
                               ('mutable_lease_granted', True), ('plan_sha256', 'c'*64),
                               ('runner_policy', {**RUNNER_POLICY, 'resolved_v2': 1})):
                path.write_text(json.dumps({**good, key: value}))
                with self.subTest(key=key), self.assertRaises(RuntimeError): validate_binding(plan, d, {(101, 17)})
            path.write_text(json.dumps(good))
            with self.assertRaisesRegex(RuntimeError, 'owned process'): validate_binding(plan, d, {(102, 17)})
            with patch('megartx.prefill_runner_binding.start_ticks', side_effect=ProcessLookupError):
                with self.assertRaises(ProcessLookupError): validate_binding(plan, d)

    def test_first_active_forward_refuses_unbound_model_before_call(self):
        modules, runner, model = hook_fixture()
        with tempfile.TemporaryDirectory() as d:
            env = {'MEGARTX_PREFILL_NATIVE_PLAN': 'cpu-boundary', 'MEGARTX_PREFILL_NATIVE_DIR': d,
                   'MEGARTX_SCALE_MODE': 'native'}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True), \
                 patch('megartx.prefill_runner_binding.verify_installed_files'), \
                 patch('megartx.prefill_native.load_plan', return_value={}), \
                 patch('megartx.prefill_native.verify_adapter_sources', return_value={}):
                install_native_observer(TORCH, model)
                (Path(d)/'request.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'no bound V2 model observer'): model().forward([], [])
                with self.assertRaisesRegex(RuntimeError, 'unbound V1'):
                    modules['vllm.v1.worker.gpu_model_runner'].GPUModelRunner()
                with self.assertRaisesRegex(RuntimeError, 'resolved default V2'):
                    runner(NS(use_v2_model_runner=False, is_mm_encoder_only=False))
                selected = runner(NS(use_v2_model_runner=True, is_mm_encoder_only=False))
                with self.assertRaisesRegex(RuntimeError, 'before execute_model'):
                    selected.execute_model(NS(total_num_scheduled_tokens=256))
                selected.execute_model(NS(total_num_scheduled_tokens=0))

    def test_actual_live_hook_guard_rejects_lost_binding(self):
        provider = NativeProvider.__new__(NativeProvider)
        provider.access = NS(_runner=NS(vllm_config=NS(use_v2_model_runner=True, is_mm_encoder_only=False)))
        provider.failed = False
        provider.abort = lambda: None
        provider.hook_checks = {key: lambda p: True for key in HOOKS}
        with patch.dict(os.environ, {}, clear=True):
            provider.require_hooks()
            provider.hook_checks['model_forward'] = lambda p: False
            with self.assertRaisesRegex(RuntimeError, 'hook binding changed'): provider.require_hooks()


class BoundHookIdentityTests(unittest.TestCase):
    @contextmanager
    def fixture(self, before_cache=None, records=None, metadata_dispatch=None, metadata_getter=None):
        # Actual installer/guards; substituted model and cache boundaries only.
        modules, runner_cls, model_cls = hook_fixture()
        effects = []
        model_cls.forward = lambda self, *args: effects.append('model-work')
        if metadata_getter is not None:
            modules['vllm.v1.worker.utils'].AttentionGroup.get_metadata_builder = metadata_getter
        if metadata_dispatch is not None:
            modules['vllm.v1.attention.backends.flashinfer'].FlashInferMetadataBuilder.build = \
                lambda self, *args, **kwargs: effects.append('metadata-work') or object()
        def execute(runner, output):
            if metadata_dispatch is not None:
                metadata_dispatch(runner)
            return runner.model.forward([], [])
        runner_cls.execute_model = execute
        records = [] if records is None else records
        class BoundaryProvider(Provider):
            require_hooks = NativeProvider.require_hooks
            active = NativeProvider.active
            def __init__(self, runner, plan, directory, torch, checks):
                super().__init__(runner)
                self.access.record_metadata = lambda *args: self.log.append('record-metadata')
                self.plan, self.directory, self.hook_checks = plan, Path(directory), checks
                self.adapter_sources = {}
                self.deadline = time.time() + 30
                self.started, self.completed, self.startup_admitted = False, False, False
                records.append(self)
                self.require_hooks()
                self.startup_admitted = True
        with tempfile.TemporaryDirectory() as directory:
            plan = {'plan_sha256': 'a'*64}
            env = {'MEGARTX_PREFILL_NATIVE_PLAN': 'synthetic-cpu-plan',
                   'MEGARTX_PREFILL_NATIVE_DIR': directory, 'MEGARTX_SCALE_MODE': 'native'}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True), \
                 patch('megartx.prefill_native.load_plan', return_value=plan), \
                 patch('megartx.prefill_native.NativeProvider', BoundaryProvider), \
                 patch('megartx.prefill_native.verify_adapter_sources', return_value={}), \
                 patch('megartx.prefill_runner_binding.verify_installed_files'):
                install_native_observer(TORCH, model_cls)
                runner = runner_cls(NS(use_v2_model_runner=True, is_mm_encoder_only=False))
                if before_cache is not None:
                    before_cache(runner)
                runner.initialize_kv_cache()
                Path(directory, 'request.json').write_text(json.dumps({
                    'schema': 'megartx-prefill-native-request-v1', 'plan_sha256': plan['plan_sha256']}))
                yield runner, records[0], effects

    def test_ordinary_bound_callbacks_admit_actual_begin_and_finish(self):
        with self.fixture() as (runner, provider, effects):
            provider.require_hooks()
            runner.execute_model(NS(total_num_scheduled_tokens=256))
            self.assertEqual(effects, ['model-work'])
            self.assertEqual(provider.log, ['begin', 'finish'])
            self.assertFalse(provider.failed)

    def test_each_instance_callback_shadow_poisoned_before_work(self):
        names = ('__init__', 'initialize_kv_cache', 'execute_model', 'prepare_inputs', 'prepare_attn',
                 'sample', 'forward', 'compute_logits', 'build')
        for name in names:
            with self.subTest(name=name), self.fixture() as (runner, provider, effects):
                owner = runner.model if name in ('forward', 'compute_logits') else runner.builder if name == 'build' else runner
                setattr(owner, name, lambda *args: effects.append('bypass'))
                with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                    provider.require_hooks()
                self.assertTrue(provider.failed)
                self.assertIn('abort', provider.log)
                self.assertEqual(effects, [])
                with self.assertRaisesRegex(RuntimeError, 'Poisoned'):
                    provider.require_hooks()

    def test_active_forward_and_builder_shadows_rejected_by_execute_guard(self):
        for name in ('forward', 'build'):
            with self.subTest(name=name), self.fixture() as (runner, provider, effects):
                owner = runner.model if name == 'forward' else runner.builder
                setattr(owner, name, lambda *args: effects.append('unobserved-work'))
                with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                    runner.execute_model(NS(total_num_scheduled_tokens=256))
                self.assertTrue(provider.failed)
                self.assertNotIn('begin', provider.log)
                self.assertEqual(effects, [])
                with self.assertRaisesRegex(RuntimeError, 'Poisoned'):
                    runner.execute_model(NS(total_num_scheduled_tokens=256))

    def test_callback_shadow_before_provider_creation_rejects_startup(self):
        records = []
        with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
            with self.fixture(lambda runner: setattr(runner.model, 'forward', lambda *args: None), records):
                self.fail('A shadowed callback cannot publish startup admission')
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].failed)
        self.assertFalse(records[0].startup_admitted)

    def test_bound_method_of_wrong_model_owner_is_rejected(self):
        with self.fixture() as (runner, provider, effects):
            runner.model.forward = type(runner.model)().forward
            with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                provider.require_hooks()
            self.assertTrue(provider.failed)
            self.assertEqual(effects, [])

    def test_replaced_loaded_model_or_absent_owned_builders_is_rejected(self):
        for mutation in ('model', 'builders'):
            with self.subTest(mutation=mutation), self.fixture() as (runner, provider, effects):
                if mutation == 'model':
                    runner.model = type(runner.model)()
                else:
                    provider.access.builders.clear()
                with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                    provider.require_hooks()
                self.assertTrue(provider.failed)
                self.assertEqual(effects, [])

    def test_class_callback_replacement_remains_rejected(self):
        with self.fixture() as (runner, provider, effects):
            type(runner.model).forward = lambda *args: effects.append('class-bypass')
            with self.assertRaisesRegex(RuntimeError, 'hook binding changed'):
                provider.require_hooks()
            self.assertTrue(provider.failed)
            self.assertEqual(effects, [])

    def test_guard_interrupt_preserves_primary_when_abort_fails(self):
        with self.fixture() as (runner, provider, effects):
            primary = KeyboardInterrupt('hook guard interrupt')
            def interrupted(provider): raise primary
            def failed_abort(): raise OSError('abort failure')
            provider.hook_checks['model_forward'] = interrupted
            provider.abort = failed_abort
            with self.assertRaises(KeyboardInterrupt) as caught:
                provider.require_hooks()
            self.assertIs(caught.exception, primary)
            self.assertTrue(provider.failed)
            if hasattr(primary, 'add_note'):
                self.assertTrue(any('abort failure' in note for note in getattr(primary, '__notes__', [])))
            self.assertEqual(effects, [])


class V2Objects(unittest.TestCase):
    def test_v2_i32_override_retains_every_other_historical_read_check(self):
        from megartx.prefill_kv import PrefillKVObserver
        old = ast.parse(textwrap.dedent(inspect.getsource(PrefillKVObserver.read_frame))).body[0]
        new = ast.parse(textwrap.dedent(inspect.getsource(NativeProvider.read_frame))).body[0]
        old.decorator_list = []
        new.body[0] = copy.deepcopy(old.body[0])
        old_if = next(n for n in old.body if isinstance(n, ast.If) and 'tokens.dtype' in ast.unparse(n.test))
        new_if = next(n for n in new.body if isinstance(n, ast.If) and 'tokens.dtype' in ast.unparse(n.test))
        expected = ast.unparse(old_if.test).replace('tokens.dtype != torch.int64', 'tokens.dtype != torch.int32')
        self.assertEqual(ast.dump(new_if.test), ast.dump(ast.parse(expected, mode='eval').body))
        new.body[new.body.index(new_if)] = copy.deepcopy(old_if)
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_actual_read_i32_only_and_all_thirty_cache_checks(self):
        from test_prefill_collect import NativeDTypeTests
        from megartx.controlled_kv_capture import SOURCE_HASHES
        for dtype in ('int32', 'int64', 'bool', 'string'):
            old, positions, tokens, module, _ = NativeDTypeTests().frame()
            provider = NativeProvider.__new__(NativeProvider)
            provider.torch, provider.layers, provider.descriptors = old.torch, old.layers, old.descriptors
            tokens.dtype = 'torch.int32' if dtype == 'string' else getattr(old.torch, dtype)
            with patch.dict(sys.modules, {'vllm.forward_context': module}), \
                 patch('megartx.prefill_native._digest', return_value=SOURCE_HASHES['vllm.forward_context']):
                if dtype == 'int32':
                    slots, capacities, identities, bindings = provider.read_frame(positions, tokens)
                    self.assertEqual(set(slots), set(range(30)))
                    self.assertEqual(len(bindings), 30)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'V2 I32 token'): provider.read_frame(positions, tokens)

    def test_actual_frame_rejects_token_object_and_state_row_substitution(self):
        for mutation in ('tokens', 'state', 'gather', 'slot'):
            a, o, c, p, s, identities = frame_fixture()
            tokens = a.input_batch.input_ids
            if mutation == 'tokens': tokens = Tensor(tokens.values)
            if mutation == 'state': a._runner.req_states.req_id_to_index['request'] = 1
            if mutation == 'gather': a._runner.block_tables.block_tables[5].gpu = Tensor([[999]*72])
            if mutation == 'slot': c.slot_mapping['layer-5'] = Tensor(s[5], pointer=999)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                a.bind_frame(o, 0, c, tokens, p, s, identities)

    def test_kernel_subdivision_placement_and_owned_page_alias(self):
        a, o, c, p, s, identities = frame_fixture()
        table = a._runner.block_tables
        for layer in range(30):
            table.block_sizes[layer], table.blocks_per_kv_block[layer] = 64, 2
            a.groups[f'layer-{layer}'][1].block_size = 64
            a.groups[f'layer-{layer}'][1].page_size_bytes *= 2
            a._runner.kv_cache_config.kv_cache_tensors[layer].block_stride *= 2
        a.bind_frame(o, 0, c, a.input_batch.input_ids, p, s, identities)
        a._runner.kv_cache_config.kv_cache_tensors[5].block_stride -= 2
        with self.assertRaisesRegex(RuntimeError, 'allocation placement'):
            a.bind_frame(o, 0, c, a.input_batch.input_ids, p, s, identities)

    def test_actual_sample_count_tensor_controls_intermediate_discard(self):
        for end, count, expected in ((256, 0, True), (2048, 1, False), (2049, 1, False)):
            provider = NativeProvider.__new__(NativeProvider)
            provider.started, provider.completed, provider.failed, provider.logit_seen = True, False, False, True
            provider.torch = NS(Tensor=Tensor, int32='int32', int64='int64')
            provider.access = NS(_runner=NS(device='cpu-substituted-boundary'))
            log = []
            provider.ledger = NS(sampled=lambda *args: log.append(args), outputs=[17], frames=1, end=end, complete_request=False)
            provider.evidence = NS(write=lambda *a, **kw: None)
            Output = type('SamplerOutput', (), {'__module__': 'vllm.v1.worker.gpu.sample.output'})
            def tensor(value, shape, dtype):
                result=Tensor(value);result.shape,result.dtype,result.device=shape,dtype,'cpu-substituted-boundary'
                return result
            output = Output(); output.sampled_token_ids = tensor(17,(1,1),'int64')
            output.num_sampled = tensor(count,(1,),'int32')
            output.num_rejected = tensor(0,(1,),'int32')
            provider.sampled(output)
            self.assertEqual(log, [(17, expected)])
            output.num_sampled = tensor(2,(1,),'int32')
            with self.assertRaisesRegex(RuntimeError, 'counts changed'): provider.sampled(output)
            output.num_sampled = tensor(1,(1,),'bool')
            with self.assertRaisesRegex(RuntimeError, 'device/dtype'): provider.sampled(output)


if __name__ == '__main__': unittest.main()
