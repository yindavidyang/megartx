"""Pinned-parent composition and actual launcher lifecycle, with CPU host fakes.

The two ownership implementations are intentionally separate: warmed decode
needs exact observation intervals, whereas the validated prefill packet retains
its original compact evidence contract. Neither policy is rewritten here.
"""
import ast
import copy
from contextlib import ExitStack
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_scale_validation.py'
FIXTURES = ROOT / 'tests/fixtures/shared-launcher-composition'
PARENTS = {
    'main': ('036ec1c63fd41a740a8a9b67e76ee294b1c74990',
             '9a07af53ac5cbbf73592c36adfda3abc7a2cb1c55c98ca7996331e1df9346203'),
    'prefill': ('6493affbcf0827f8702e736636bb4f2549d07305',
                '27f47525141ba511e5c4282f9717fb5fba44e4131fec3ae832143e1958b83384'),
}
HELPERS = {
    'main': ('m1_owned_processes', 'b193ac3b991723d82f3d92b4908d5dfd5e04fce32c0b9d62435cc66f95797253'),
    'prefill': ('prefill_owned_processes', 'df01a29a8089db09530eef3d36efc04d2871033b4fa6b357eb73a0aca99dae4b'),
}
sys.path.insert(0, str(ROOT / 'scripts'))
import m1_owned_processes as decode_owner
import prefill_owned_processes as prefill_owner
from megartx.prefill_storage_plan import CompactEvidence


def parent_source(purpose):
    data = (FIXTURES / (purpose + '.source')).read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENTS[purpose][1]:
        raise AssertionError('Exact launcher parent fixture changed: ' + purpose)
    return data.decode()


def tree(source=None):
    return ast.parse(LAUNCHER.read_text() if source is None else source)


def run_node(source=None):
    return next(n for n in tree(source).body if isinstance(n, ast.Try) and n.finalbody)


def function(name, source=None):
    return copy.deepcopy(next(n for n in tree(source).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))


def code(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                   str(LAUNCHER), 'exec')


def dump(node):
    return ast.dump(node, include_attributes=False)


class SelectPurpose(ast.NodeTransformer):
    """Evaluate only explicit purpose selectors, not arbitrary statements.

    Exact parent proofs cover the entire try/except/finally lifecycle. Two
    mechanical adaptations are allowed: the verified helper module rename and
    the otherwise unused native server-ready return binding added for decode.
    The separate attention selector remains off for every historical purpose.
    """
    def __init__(self, native, storage=False):
        self.native, self.storage = native, storage

    def visit_Name(self, node):
        values = {'prefill_native': self.native, 'prefill_storage': self.storage,
                  'prefill_attention': False}
        if self.native:
            values['eager_benchmark'] = False
        if node.id in values and isinstance(node.ctx, ast.Load):
            return ast.Constant(values[node.id])
        return node

    def visit_Attribute(self, node):
        if self.native and dump(node) == dump(ast.parse('args.m1_warmed_timing', mode='eval').body):
            return ast.Constant(False)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        decisive = False if is_and else True
        if any(isinstance(x, ast.Constant) and x.value is decisive for x in node.values):
            return ast.Constant(decisive)
        values = [x for x in node.values if not isinstance(x, ast.Constant)
                  or x.value is not (not decisive)]
        return (ast.Constant(not decisive) if not values else values[0]
                if len(values) == 1 else ast.BoolOp(op=node.op, values=values))

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return node.body if node.test.value else node.orelse
        return node

    def visit_ImportFrom(self, node):
        if self.native and node.module == 'prefill_owned_processes':
            node.module = 'm1_owned_processes'
        return node

    def visit_Assign(self, node):
        expected = ast.parse('server_ready_ns = phase("server_ready")').body[0]
        if self.native and dump(node) == dump(expected):
            return ast.Expr(value=node.value)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        # These two values extend only the declared diagnostic client set.
        # They cannot be selected on the decode/default parent branch.
        if (not self.native and dump(node.left) == dump(ast.parse('args.client', mode='eval').body)
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.In)
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Set)
                and {x.value for x in node.comparators[0].elts if isinstance(x, ast.Constant)}
                    == {'controlled', 'normal', 'prefill-native', 'prefill-storage'}):
            node.comparators[0].elts = [x for x in node.comparators[0].elts
                                       if x.value in {'controlled', 'normal'}]
        return node


class ParentCompositionTests(unittest.TestCase):
    def test_helpers_are_exact_pinned_parent_bytes(self):
        for name, digest in HELPERS.values():
            self.assertEqual(hashlib.sha256((ROOT / 'scripts' / (name + '.py')).read_bytes()).hexdigest(), digest)

    def test_entire_decode_lifecycle_equals_exact_current_main(self):
        selected = SelectPurpose(False).visit(run_node())
        self.assertEqual(dump(selected), dump(run_node(parent_source('main'))))

    def test_entire_prefill_lifecycle_equals_exact_validated_parent(self):
        for storage in (False, True):
            with self.subTest(storage=storage):
                selected = SelectPurpose(True, storage).visit(run_node())
                previous = SelectPurpose(True, storage).visit(run_node(parent_source('prefill')))
                self.assertEqual(dump(selected), dump(previous))

    def test_sampling_and_phase_functions_preserve_both_exact_parents(self):
        for name in ('phase', 'compiler_guard', 'sampler'):
            with self.subTest(function=name):
                current = function(name)
                dispatch = ast.parse('if prefill_native:\n    return prefill_' + name +
                                     ('(name, **fields)' if name == 'phase' else '()')).body[0]
                self.assertEqual(dump(current.body.pop(0)), dump(dispatch))
                self.assertEqual(dump(current), dump(function(name, parent_source('main'))))
                native = function('prefill_' + name)
                native.name = name
                native = SelectPurpose(True).visit(native)
                previous = SelectPurpose(True).visit(function(name, parent_source('prefill')))
                self.assertEqual(dump(native), dump(previous))

    def test_native_warmed_flag_rejected_before_plan_or_runtime_work(self):
        for client in ('prefill-native', 'prefill-storage'):
            result = subprocess.run([sys.executable, '-S', str(LAUNCHER), '--mode', 'native',
                                     '--label', 'not-created', '--client', client, '--trials', '1',
                                     '--m1-warmed-timing', '--prefill-native-plan', 'missing-plan',
                                     '--prefill-native-clearance', 'missing-clearance'],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('requires its exact default-off', result.stderr)
            self.assertNotIn('Traceback', result.stderr)


class OwnershipBindingTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {'pathlib': __import__('pathlib'), 'hashlib': hashlib, 'sys': sys}
        exec(code([function('verify_owned_process_module')]), self.namespace)
        self.verify = self.namespace['verify_owned_process_module']

    def hashes(self, purpose):
        name, digest = HELPERS['main' if purpose == 'decode' else 'prefill']
        return {'scripts/' + name + '.py': digest}

    def test_actual_loaded_helpers_have_exact_distinct_origins_and_source_vectors(self):
        for purpose, module in (('decode', decode_owner), ('prefill', prefill_owner)):
            self.verify(module, ROOT, self.hashes(purpose), purpose)
            with self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                self.verify(module, ROOT, self.hashes('prefill' if purpose == 'decode' else 'decode'), purpose)

    def test_missing_wrong_and_mixed_module_identity_origin_and_hash_fail_closed(self):
        for purpose, module, other in (('decode', decode_owner, prefill_owner),
                                       ('prefill', prefill_owner, decode_owner)):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                    self.verify(other, ROOT, self.hashes(purpose), purpose)
                for field, value in (('__file__', None), ('__file__', other.__file__),
                                     ('__spec__', None), ('__spec__', NS(origin=None)),
                                     ('__spec__', NS(origin=other.__file__))):
                    with patch.object(module, field, value), self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                        self.verify(module, ROOT, self.hashes(purpose), purpose)
                with patch.dict(sys.modules, {module.__name__: other}), self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                    self.verify(module, ROOT, self.hashes(purpose), purpose)
                for hashes in ({}, {key: '0' * 64 for key in self.hashes(purpose)}):
                    with self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                        self.verify(module, ROOT, hashes, purpose)

    def test_loaded_helper_file_mutation_or_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            path = root / 'scripts/prefill_owned_processes.py'
            path.write_bytes((ROOT / 'scripts/prefill_owned_processes.py').read_bytes())
            spec = importlib.util.spec_from_file_location('prefill_owned_processes', path)
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {'prefill_owned_processes': module}):
                spec.loader.exec_module(module)
                self.verify(module, root, self.hashes('prefill'), 'prefill')
                path.write_bytes(path.read_bytes() + b'\n# changed after import\n')
                with self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                    self.verify(module, root, self.hashes('prefill'), 'prefill')
                path.unlink()
                path.symlink_to(ROOT / 'scripts/prefill_owned_processes.py')
                with self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                    self.verify(module, root, self.hashes('prefill'), 'prefill')

    def test_actual_admission_gates_reject_wrong_helper_before_resource_operations(self):
        top_level = tree().body
        runtime_import = next(i for i, n in enumerate(top_level) if isinstance(n, ast.Import)
                              and any(a.name == 'requests' for a in n.names))
        gates = []
        for index, node in enumerate(top_level):
            if not isinstance(node, ast.If):
                continue
            for offset, statement in enumerate(node.body):
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and statement.value.func.id == 'verify_owned_process_module'):
                    self.assertLess(index, runtime_import)
                    gates.append(node.body[offset-1:offset+1])
        self.assertEqual(len(gates), 2)
        env = {**self.namespace, 'project_root': ROOT,
               'prefill_plan': {'source_hashes': self.hashes('prefill')},
               'benchmark_plan': {'driver_source_hashes': self.hashes('decode')}}
        for gate in gates:
            exec(code(gate), env)
            module_name = gate[0].names[0].name
            other = decode_owner if module_name == 'prefill_owned_processes' else prefill_owner
            with patch.dict(sys.modules, {module_name: other}), self.assertRaisesRegex(RuntimeError, 'origin/source differs'):
                exec(code(gate), env)


class FastEvent(threading.Event):
    """Keep real synchronization while shortening idle polling in CPU tests."""
    def wait(self, timeout=None):
        return super().wait(min(timeout, .002) if timeout is not None else None)


class LauncherHarness:
    """Run the real top-level lifecycle with only external boundaries faked."""
    def __init__(self, case, purpose, *, client_action=None, final_sample_failure=False):
        self.case, self.purpose = case, purpose
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.native = purpose != 'main'
        self.storage = purpose == 'storage'
        self.module = prefill_owner if self.native else decode_owner
        self.other_module = decode_owner if self.native else prefill_owner
        self.root = self.module.Process(20, 20, 10)
        self.reaper = self.module.Process(10, 10, 1)
        self.server = NS(pid=20, returncode=None)
        self.server.poll = lambda: self.server.returncode
        self.snapshots = []
        self.signals = []
        self.imports = []
        self.snapshot_lock = threading.Lock()
        self.active_snapshots = self.max_active_snapshots = 0
        self.telemetry_seen = threading.Event()
        self.telemetry_calls = []
        self.client_calls = []
        self.client_action = client_action
        self.final_sample_failure = final_sample_failure
        self.publish = Mock()
        self.summary = Mock()
        self.evidence = None
        if self.storage:
            self.evidence = CompactEvidence(self.output / 'prefill-storage')
        elif self.native:
            from megartx.prefill_diagnostic_plan import Evidence
            self.evidence = Evidence(self.output / 'prefill-native')
        phases = open(os.devnull, 'w') if self.storage else (self.output / 'server-phases.jsonl').open('a')
        self.args = NS(client='prefill-storage' if self.storage else
                       'prefill-native' if self.native else 'm1-eager-benchmark',
                       m1_warmed_timing=False, m1_decode_profile=False, m1_external_observer=False,
                       m1_eager_benchmark_plan=Path('cpu-plan'), prefill_native_plan=Path('cpu-plan'),
                       profile=False, activation_only=False, routing_diagnostic=False,
                       router_score_only=False)
        self.env = {
            'os': os, 'pathlib': __import__('pathlib'), 'sys': sys, 'signal': signal,
            'time': time, 'datetime': datetime, 'threading': threading, 'json': json,
            'math': math, 'output': self.output, 'base': Path('cpu-runtime'), 'project': ROOT,
            'env': {}, 'command': ['cpu-fake-server'], 'args': self.args,
            'prefill_native': self.native, 'prefill_storage': self.storage,
            'prefill_attention': False,
            'prefill_plan': {'plan_sha256': 'prefill-cpu'}, 'prefill_evidence': self.evidence,
            'prefill_evidence_failed': False, 'prefill_deadline': time.monotonic() + 1800,
            'eager_benchmark': not self.native, 'benchmark_plan': {'plan_sha256': 'decode-cpu'},
            'metadata_timing': None, 'server': None, 'ownership': None, 'guard_failure': None,
            'guard_thread': None, 'stop_guard': FastEvent(), 'stop_sample': FastEvent(),
            'idle': NS(returncode=0, stdout='GPU, 12.0, 100, 4096, 0, driver, 500'),
            'client': NS(get=lambda *a, **k: NS(status_code=200)),
            'requests': NS(RequestException=RuntimeError), 'activation': Mock(),
            'phases': phases, 'print': lambda *a, **k: None,
            'subprocess': NS(run=self.run_command, Popen=lambda *a, **k: self.server,
                             STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
        }
        definitions = [n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name != 'gpu_jobs']
        exec(code(definitions), self.env)
        self.env['gpu_jobs'] = lambda: []
        self.env['sample_thread'] = threading.Thread(target=self.env['sampler'], daemon=True)

    def snapshot(self):
        with self.snapshot_lock:
            self.active_snapshots += 1
            self.max_active_snapshots = max(self.max_active_snapshots, self.active_snapshots)
        try:
            # A tiny bounded wait makes snapshot overlap observable, while the
            # actual main ownership lock must serialize every observation.
            threading.Event().wait(.001)
            self.snapshots.append(threading.current_thread().name)
            return {} if self.server.returncode is not None else {20: self.root}
        finally:
            with self.snapshot_lock:
                self.active_snapshots -= 1

    def signal_identity(self, process, signum):
        self.signals.append((process.identity, signum))
        self.server.returncode = 0

    def run_command(self, command, **kwargs):
        if command[0] == 'nvidia-smi':
            final = self.env['stop_sample'].is_set()
            self.telemetry_calls.append(final)
            self.telemetry_seen.set()
            if final and self.final_sample_failure:
                return NS(returncode=1, stdout='', stderr='last sample unavailable')
            return NS(returncode=0, stdout='100, 4096, 0, 50, 40, 100, 100, P8', stderr='')
        return self.dispatch(command, self.env['env'])

    def dispatch(self, command, environment, *args):
        self.client_calls.append((command, environment, args))
        if not self.telemetry_seen.wait(2):
            raise AssertionError('Real launcher sampler did not run')
        if self.client_action:
            self.client_action(self)
        return NS(returncode=0)

    def execute(self):
        read_text = Path.read_text
        import builtins
        original_import = builtins.__import__

        def read(path, *args, **kwargs):
            return 'MemAvailable: 16777216 kB\n' if str(path) == '/proc/meminfo' else read_text(path, *args, **kwargs)

        def record_import(name, *args, **kwargs):
            self.imports.append(name)
            return original_import(name, *args, **kwargs)

        with ExitStack() as stack:
            for module in (self.module, self.other_module):
                chosen = module is self.module
                for name, replacement in (
                    ('snapshot', self.snapshot), ('read_process', lambda pid: self.root),
                    ('enable_subreaper', lambda: self.reaper), ('signal_identity', self.signal_identity),
                ):
                    stack.enter_context(patch.object(module, name, side_effect=replacement if chosen else
                                                     AssertionError('Wrong-purpose ownership helper used')))
            stack.enter_context(patch.object(Path, 'read_text', read))
            stack.enter_context(patch.object(signal, 'signal'))
            stack.enter_context(patch('m1_eager_benchmark_client.summarize_run', self.summary))
            for module in ('megartx.prefill_storage_plan', 'megartx.prefill_diagnostic_plan'):
                stack.enter_context(patch(module + '.load_plan', return_value=self.env['prefill_plan']))
            stack.enter_context(patch('megartx.prefill_storage_plan.validate_binding', return_value={'runner_policy': 'cpu'}))
            stack.enter_context(patch('megartx.prefill_runner_binding.validate_binding', return_value={'runner_policy': 'cpu'}))
            stack.enter_context(patch('megartx.prefill_storage_plan.publish_storage', self.publish))
            stack.enter_context(patch('megartx.prefill_diagnostic_plan.publish_fit', self.publish))
            stack.enter_context(patch('megartx.prefill_storage_process.run_bounded_client', side_effect=self.dispatch))
            stack.enter_context(patch.object(builtins, '__import__', record_import))
            try:
                exec(code([run_node()]), self.env)
            finally:
                if 'logfile' in self.env:
                    self.env['logfile'].close()
                self.env['phases'].close()

    def cleanup(self):
        path = (self.evidence.directory / 'cleanup.json' if self.storage else self.output /
                ('prefill-native-cleanup.json' if self.native else 'eager-benchmark-cleanup.json'))
        return json.loads(path.read_text())


class ExtractedLauncherTests(unittest.TestCase):
    def test_complete_lifecycle_selects_only_correct_helper_and_evidence_domain(self):
        for purpose in ('main', 'native', 'storage'):
            with self.subTest(purpose=purpose):
                harness = LauncherHarness(self, purpose)
                harness.execute()
                self.assertFalse(harness.env['prefill_attention'])
                selected = 'prefill_owned_processes' if harness.native else 'm1_owned_processes'
                other = 'm1_owned_processes' if harness.native else 'prefill_owned_processes'
                self.assertIn(selected, harness.imports)
                self.assertNotIn(other, harness.imports)
                self.assertEqual(type(harness.env['ownership']).__module__, selected)
                self.assertTrue(harness.cleanup()['cleanup_complete'])
                self.assertEqual((harness.output / 'run.exit').read_text(), '0\n')
                self.assertTrue(harness.signals)
                if harness.native:
                    self.assertEqual(Path(harness.client_calls[0][0][1]).name,
                                     'prefill_storage_client.py' if harness.storage
                                     else 'prefill_diagnostic_client.py')
                    harness.publish.assert_called_once()
                    harness.summary.assert_not_called()
                    self.assertNotIn('compiler_observation_intervals_ns', harness.cleanup())
                else:
                    harness.summary.assert_called_once_with(harness.output)
                    harness.publish.assert_not_called()
                    self.assertTrue(harness.cleanup()['compiler_observation_clock_exact'])
                    self.assertTrue(harness.telemetry_calls[-1])
                if harness.storage:
                    for name in ('client.log', 'benchmark.exit', 'gpu-telemetry.jsonl',
                                 'server-phases.jsonl', 'status.json', 'owned-processes.json'):
                        self.assertFalse((harness.output / name).exists(), name)
                    rows = (harness.evidence.directory / 'telemetry.jsonl').read_text().splitlines()
                    self.assertEqual(json.loads(rows[0])['schema'], 'megartx-prefill-storage-telemetry-v1')
                    self.assertTrue(all(isinstance(json.loads(row), list) for row in rows[1:]))
                    self.assertLess(harness.evidence.sizes()['metadata_bytes'], 2 << 20)

    def test_decode_guard_and_dispatch_snapshot_order_is_locked(self):
        harness = LauncherHarness(self, 'main')
        harness.execute()
        self.assertEqual(harness.max_active_snapshots, 1)
        self.assertGreater(len(set(harness.snapshots)), 1)
        intervals = harness.cleanup()['compiler_observation_intervals_ns']
        self.assertGreater(len(intervals), 2)
        self.assertTrue(all(a <= b for a, b in intervals))
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])))

    def test_decode_last_sample_failure_rejects_final_admission(self):
        harness = LauncherHarness(self, 'main', final_sample_failure=True)
        with self.assertRaisesRegex(RuntimeError, 'resource telemetry failed'):
            harness.execute()
        self.assertTrue(harness.telemetry_calls[-1])
        self.assertTrue(harness.cleanup()['cleanup_complete'])
        self.assertIn('resource telemetry failed', harness.cleanup()['failure'])
        self.assertEqual((harness.output / 'run.exit').read_text(), '1\n')
        harness.summary.assert_not_called()

    def test_actual_warmed_dispatch_records_monotonic_boundaries_and_exact_flag(self):
        harness = LauncherHarness(self, 'main')
        harness.args.m1_warmed_timing = True
        harness.env.update({'bench_command': ['cpu-client'], 'WARMED_RULE': 'cpu-warmed-rule'})
        dispatch = next(node for node in run_node().body if isinstance(node, ast.If)
                        and ast.unparse(node.test) == 'prefill_native'
                        and any(isinstance(child, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == 'client_launch_ns'
                                        for target in child.targets) for child in node.orelse))
        monotonic_values = iter((100, 200, 300))
        harness.env['time'] = NS(monotonic_ns=lambda: next(monotonic_values),
                                 time_ns=lambda: 999, perf_counter_ns=lambda: self.fail('Decode used prefill clock'))
        harness.env['subprocess'].run = Mock(return_value=NS(returncode=0))
        try:
            harness.env['server_ready_ns'] = harness.env['phase']('server_ready')
            exec(code([dispatch]), harness.env)
        finally:
            harness.env['phases'].close()
        boundary = json.loads((harness.output / 'warmed-launch-boundaries.json').read_text())
        self.assertEqual([boundary[key] for key in ('server_ready_ns', 'client_launch_ns', 'client_returned_ns')],
                         [100, 200, 300])
        self.assertEqual(harness.env['bench_command'], ['cpu-client', '--m1-warmed-timing'])
        self.assertEqual(harness.env['subprocess'].run.call_args.kwargs['timeout'], 3600)
        self.assertEqual((harness.output / 'benchmark.exit').read_text(), '0\n')

    def test_interruption_preserves_primary_failure_and_completes_owned_cleanup(self):
        for purpose in ('main', 'native', 'storage'):
            with self.subTest(purpose=purpose):
                harness = LauncherHarness(self, purpose, client_action=lambda h: h.env['interrupted'](signal.SIGTERM, None))
                with self.assertRaisesRegex(InterruptedError, 'interrupted by signal'):
                    harness.execute()
                self.assertTrue(harness.cleanup()['cleanup_complete'])
                self.assertEqual((harness.output / 'run.exit').read_text(), '1\n')
                self.assertFalse(harness.env['sample_thread'].is_alive())
                self.assertFalse(harness.env['guard_thread'].is_alive())
                harness.summary.assert_not_called()
                harness.publish.assert_not_called()

    def test_storage_evidence_cap_failure_retains_cleanup_and_forbids_publication(self):
        def fill_metadata(harness):
            evidence = harness.evidence
            harness.env['stop_sample'].set()
            harness.env['sample_thread'].join(timeout=2)
            self.assertFalse(harness.env['sample_thread'].is_alive())
            # Use the real cross-process writer and its unchanged 4KiB reserve.
            # Exhaust the remaining metadata allowance before post-client phases.
            remaining = evidence.metadata_limit - evidence.sizes()['metadata_bytes'] - 4096
            evidence.write('fill.json', {'padding': 'x' * (remaining - 32)})

        harness = LauncherHarness(self, 'storage', client_action=fill_metadata)
        with self.assertRaisesRegex((ValueError, RuntimeError), 'overflow|evidence writer failed|resource telemetry failed'):
            harness.execute()
        self.assertEqual(harness.server.returncode, 0)
        self.assertTrue(harness.env['ownership_report']['cleanup_complete'])
        self.assertEqual((harness.output / 'run.exit').read_text(), '1\n')
        self.assertTrue(harness.env['prefill_evidence_failed'])
        self.assertLessEqual(harness.evidence.sizes()['metadata_bytes'], 2 << 20)
        self.assertLessEqual(harness.evidence.sizes()['total_bytes'], 8 << 20)
        self.assertFalse((harness.evidence.directory / 'storage.json').exists())
        harness.publish.assert_not_called()

    def test_storage_phase_and_sampler_concurrency_keeps_compact_complete_rows(self):
        phase_started, telemetry_during_phase = threading.Event(), threading.Event()

        def concurrent_phases(harness):
            for index in range(30):
                harness.env['phase']('client_parallel_test', index=index)

        harness = LauncherHarness(self, 'storage', client_action=concurrent_phases)
        real_write = harness.evidence.write

        def overlapping_write(name, value, **kwargs):
            if name == 'phases.jsonl' and value['phase'] == 'client_parallel_test':
                phase_started.set()
                self.assertTrue(telemetry_during_phase.wait(2))
            elif name == 'telemetry.jsonl' and phase_started.is_set():
                telemetry_during_phase.set()
            return real_write(name, value, **kwargs)

        harness.evidence.write = overlapping_write
        harness.execute()
        phases = [json.loads(row) for row in (harness.evidence.directory / 'phases.jsonl').read_text().splitlines()]
        self.assertEqual([row['index'] for row in phases if row['phase'] == 'client_parallel_test'], list(range(30)))
        telemetry = [json.loads(row) for row in (harness.evidence.directory / 'telemetry.jsonl').read_text().splitlines()]
        self.assertGreater(len(telemetry), 2)
        self.assertTrue(all(isinstance(row, list) and len(row) == 4 for row in telemetry[1:]))
        self.assertTrue(harness.cleanup()['cleanup_complete'])
        self.assertFalse((harness.output / 'gpu-telemetry.jsonl').exists())

    def test_storage_last_inflight_sample_failure_rejects_final_admission(self):
        entered, release = threading.Event(), threading.Event()

        def finish_during_sample(harness):
            self.assertTrue(entered.wait(2))
            harness.env['stop_sample'].set()
            release.set()
            harness.env['sample_thread'].join(timeout=2)
            self.assertFalse(harness.env['sample_thread'].is_alive())

        harness = LauncherHarness(self, 'storage', client_action=finish_during_sample)

        def last_sample(command, **kwargs):
            if command[0] == 'nvidia-smi' and harness.telemetry_calls:
                entered.set()
                self.assertTrue(release.wait(2))
                harness.telemetry_calls.append(harness.env['stop_sample'].is_set())
                return NS(returncode=1, stdout='', stderr='last inflight sample failed')
            return harness.run_command(command, **kwargs)

        harness.env['subprocess'].run = last_sample
        with self.assertRaisesRegex(RuntimeError, 'resource telemetry failed'):
            harness.execute()
        self.assertTrue(harness.telemetry_calls[-1])
        self.assertTrue(harness.cleanup()['cleanup_complete'])
        self.assertIn('resource telemetry failed', harness.cleanup()['failure'])
        self.assertEqual((harness.output / 'run.exit').read_text(), '1\n')
        harness.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
