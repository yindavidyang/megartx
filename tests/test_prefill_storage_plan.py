"""Source-only storage admission and accounting; synthetic receipts are not proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from megartx import prefill_storage_plan as storage
from megartx import prefill_diagnostic_plan as legacy
from megartx.controlled_kv_capture import CONFIG_SHA256
from megartx.prefill_runner_binding import HOOKS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prefill_storage_client import (StreamLedger, payload, client_receipt, stream_observation,
                                   storage_client_receipt, validate_observation, run)


def source_fixture(root):
    """A private CPU fixture tree, never a repository freeze or GPU clearance."""
    for name in storage.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / name
        path.write_bytes(original.read_bytes() if original.is_file() else b'# synthetic CPU source\n')
    composition = storage.map_borrow.load_catalog(ROOT)
    for record in composition['superseded_sources']:
        target = root/record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/record['path']).read_bytes())
    for name in [*composition['parent_ledgers'], *composition['preserved_borrow_sources']]:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    catalog = json.loads((ROOT/storage.CATALOG_SOURCE).read_text())
    equivalence = catalog['default_fit_equivalence']
    for name in [*catalog['unchanged_historical_catalogs'], *catalog['parent_ledgers'],
                 equivalence['test_path'], *equivalence['additional_tests'], *equivalence['source_fixtures'],
                 *(record['path'] for record in catalog['superseded_sources'])]:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    # Copy the immutable catalogs exactly; the terminal overlay admits current
    # proof and validator bytes without refreshing any historical hash.
    return storage.freeze_plan(list(range(2048)), '3' * 40, root, {
        'config_sha256': CONFIG_SHA256, 'index_sha256': 'a' * 64,
        'shard_stats': {'synthetic.safetensors': {'size': 1, 'mtime_ns': 1}},
    })


def stream_fixture(plan):
    stream = StreamLedger(plan)
    stream.consume('data: ' + json.dumps({'id': stream.response_id, 'choices': [
        {'index': 0, 'token_ids': [17] * 256, 'finish_reason': 'length'}]}))
    stream.consume('data: ' + json.dumps({'id': stream.response_id, 'choices': [],
        'usage': {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}}))
    stream.consume('data: [DONE]')
    engine_id = stream.response_id + '-0-1234abcd'
    observer = {'schema': 'megartx-prefill-native-observation-v1', 'status': 'request_observed',
        'plan_sha256': plan['plan_sha256'], 'engine_request_id': engine_id,
        'request_identity': legacy.native_request_identity(plan, engine_id),
        'output_ids_sha256': storage.digest([17] * 256), 'prompt_frames': 8,
        'decode_input_rows': 255, 'emitted_outputs': 256, 'committed_length': 2303,
        'numerical_qualified': False, 'performance_qualified': False}
    return stream, observer


def binding_fixture(plan):
    binding = {'schema': 'megartx-prefill-runner-binding-v1',
        'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'owner_pid': 1, 'owner_start_ticks': 1, 'runner_policy': legacy.RUNNER_POLICY,
        'installed_sources': {**legacy.INSTALLED, **legacy.TRANSPORT_FILES},
        'hook_bindings': dict.fromkeys(HOOKS, True), 'mutable_lease_granted': False}
    mode = {'schema': 'megartx-prefill-storage-binding-v1',
        'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'owner_pid': 1, 'owner_start_ticks': 1, 'purpose': storage.PURPOSE,
        'writer_hook_bound': True, 'checker_sha256': plan['source_hashes'][storage.CHECKER_SOURCE]}
    return binding, mode


def receipt_fixture(directory, plan):
    evidence = storage.CompactEvidence(directory)
    stream, observer = stream_fixture(plan)
    evidence.write('telemetry.jsonl', {'monotonic_ns': 1, 'unix_ns': 2,
        'fields': storage.TELEMETRY_FIELDS,
        'values': ['32000','10855','100','142.57','70','1215','10501','P0'], 'exit': 0}, append=True)
    observer['observer_gpu_scratch'] = {
        'domain': 'incremental_gpu_allocator_bytes', 'cap_bytes': 8 << 20,
        'measured_phases': 789, 'managed_tensor_simultaneous_peak_bytes': 0,
        'measured_phase_allocator_increment_peak_bytes': 0, 'measured_phase_reserved_increment_peak_bytes': 0,
        'counter_policy': 'process_global_resets_with_explicit_runwide_peak_preservation',
        'runwide_allocator_allocated_peak_bytes': 0, 'runwide_allocator_reserved_peak_bytes': 0,
        'host_heap_excluded': True}
    binding, mode_binding = binding_fixture(plan)
    for name, size in storage.RAW_FILES.items():
        evidence.raw(name, bytes(size))
    counts = {'pre': 0, 'processed': 0, 'post': 0}
    roots = dict.fromkeys(counts, 'b' * 64)
    for sequence in range(263):
        start = sequence * 256 if sequence < 8 else 2048 + sequence - 8
        end = start + (256 if sequence < 8 else 1)
        if sequence < 9:
            for layer in range(30):
                low = 0 if layer % 6 == 5 else max(0, start - 1023)
                counts['pre'] += start - low
                counts['processed'] += end - start
                counts['post'] += end - low
        evidence.write('control-frames.jsonl', {'sequence': sequence, 'start': start, 'end': end,
            'plan_sha256': plan['plan_sha256'], 'storage_checked': sequence < 9,
            'storage_counts': dict(counts) if sequence < 9 else None,
            'storage_roots': roots if sequence < 9 else None,
            'queries_complete': True, 'input_ids_sha256': storage.digest(
                plan['tokens'][start:end] if sequence < 8 else [17])}, append=True)
        evidence.write('heads.jsonl', {'sequence': sequence, 'logit_position': end - 1,
            'plan_sha256': plan['plan_sha256'], 'before_sampler_transforms': True,
            'selected_row_index': 255 if sequence < 8 else 0,
            'native_logits_dtype': 'torch.bfloat16', 'suppressed_token_count': 0,
            'predicts_position': end, 'expected_sampler_discard': sequence < 7,
            'hidden_bits_sha256': 'c'*64,
            'logit_bits_sha256': hashlib.sha256(bytes(524288)).hexdigest() if sequence in (7,8) else None}, append=True)
    for index in range(256):
        evidence.write('samples.jsonl', {'output_index': index, 'head_sequence': index + 7,
            'head_phase': 'final_prompt' if index == 0 else 'decode',
            'sample_sha256': storage.digest([17]), 'cached_length': 2048 + index,
            'pending_anchor_position': 2048 + index, 'anchor_kv_written': False}, append=True)
    frontier = {'checked_heads': 263, 'emitted_tokens': 256, 'decode_inputs': 255,
        'committed_length': 2303, 'uncached_output_position': 2303,
        'token_ids_encoding': 'canonical-compact-json-integer-array-utf8',
        'token_ids_sha256': observer['output_ids_sha256']}
    domains = {'pre': 1550868480, 'processed': 461598720, 'post': 2012467200,
        'raw_heads': 1048576, 'head_index': 2104, 'selected_hidden': 2962432,
        'head_finite_scalar': 261, 'manager_table_metadata': 1132032,
        'inherited_metadata_upper_bound': 41811968, 'writer_slot_metadata': 552720}
    control = {'schema': 'megartx-prefill-storage-control-v1', 'plan_sha256': plan['plan_sha256'],
        'status': 'storage_frontier_observed', 'storage_exact': True, 'frontier_verified': True,
        'storage_callbacks_restored': True,
        **storage.CONTROL_COUNTS, **storage.QUALIFICATIONS, 'frontier': frontier,
        'storage_roots': roots, 'actual_sampling_params_sha256': 'd'*64,
        'processed_digest_host_bytes': 100, 'natural_positive_correction_coverage': None,
        'transfer': {'limit_bytes': 4 << 30, 'transferred_bytes': sum(domains.values()),
                     'copy_calls': 1095300, 'domains': domains},
        'manager': {'manager_block_tokens': [16,16], 'kernel_block_tokens': [16,16],
            'manager_to_kernel_ratios':[1,1], 'append_calls': 9, 'manager_table_sha256':'e'*64,
            'expected_address_source':'actual_append_block_ids_before_native_subdivision'},
        'raw_samples_sha256': storage.digest(storage.raw_manifest(directory)),
        'frame_records_sha256': storage.file_sha(directory / 'control-frames.jsonl'),
        'head_records_sha256': storage.file_sha(directory / 'heads.jsonl'),
        'sample_records_sha256': storage.file_sha(directory / 'samples.jsonl'),
        'sample_hashes_sha256': storage.digest([storage.digest([t]) for t in stream.tokens])}
    client = client_receipt(plan, stream, observer)
    for name, value in (
        ('runner-binding.json', binding), ('storage-binding.json', mode_binding),
        ('observer.json', observer), ('client.json', client),
        ('loaded.json', {'plan_sha256': plan['plan_sha256'], 'mutable_lease_granted': False,
                         'identity': {'owner_pid': 1, 'owner_start_ticks': 1}}),
        ('geometry.json', {'physical_policy': 'full_context', 'capacity_tokens': 2304,
                          'actual_owned_page_ranges_disjoint': True}),
        ('client-stream.json', stream_observation(plan, stream)), ('control.json', control),
        ('storage-client.json', storage_client_receipt(plan, control, client, stream)),
    ):
        evidence.write(name, value)
    ownership = {'cleanup_complete': True, 'failure': None, 'owned_identities_remaining': [],
                 'owned_gpu_pids_remaining': [], 'cleanup_errors': []}
    return evidence, control, ownership


class StoragePlanTests(unittest.TestCase):
    def test_imports_are_cpu_only(self):
        result = subprocess.run([sys.executable, '-S', '-c',
            "import megartx.prefill_storage_plan; import sys; sys.path.insert(0,'scripts'); "
            "import prefill_storage_client; assert not any(n.split('.')[0] in "
            "{'torch','vllm','requests','numpy'} for n in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_separate_schema_source_vector_and_immutable_one_context_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root)
            self.assertEqual(storage.validate_plan(plan, root), plan)
            self.assertEqual(set(plan['source_hashes']), set(storage.SOURCES))
            self.assertTrue(set(legacy.SOURCES) < set(storage.SOURCES))
            self.assertEqual(plan['bounds'], legacy.BOUNDS)
            self.assertEqual(plan['control_spec']['max_contexts'], 1)
            for key, value in [('schema', 'megartx-prefill-native-plan-v3'), ('purpose', 'fit'),
                ('tokens', plan['tokens'][:-1]), ('source_head', '0' * 39),
                ('control_spec', {**storage.CONTROL_SPEC, 'max_contexts': 2}),
                ('control_spec', {**storage.CONTROL_SPEC, 'capture_frames': 9.0}),
                ('control_spec', {**storage.CONTROL_SPEC, 'numerical_qualified': True}),
                ('source_hashes', {k: v for k, v in plan['source_hashes'].items() if k != storage.CHECKER_SOURCE})]:
                altered = copy.deepcopy(plan); altered[key] = value
                altered['plan_sha256'] = storage.digest({k: v for k, v in altered.items() if k != 'plan_sha256'})
                with self.subTest(field=key, value=value):
                    with self.assertRaises(ValueError): storage.validate_plan(altered, root)
            (root / storage.CHECKER_SOURCE).write_text('# changed independent checker\n')
            with self.assertRaisesRegex(ValueError, 'source drift'): storage.validate_plan(plan, root)

    def test_plan_bounded_file_and_duplicate_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root); path = root / 'plan.json'
            path.write_text(json.dumps(plan)); self.assertEqual(storage.load_plan(path, root), plan)
            link = root / 'link.json'; link.symlink_to(path)
            with self.assertRaises(ValueError): storage.load_plan(link, root)
            path.write_text('{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'): storage.load_plan(path, root)
            path.write_bytes(b' ' * ((1 << 20) + 1))
            with self.assertRaisesRegex(ValueError, 'bounded'): storage.load_plan(path, root)

    def test_storage_clearance_rejects_fit_clearance_and_truthy_flags(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root); path = root / 'cpu-fixture.json'
            value = {'schema': 'megartx-prefill-storage-clearance-v1', 'plan_sha256': plan['plan_sha256'],
                'source_head': plan['source_head'], 'cpu_review_passed': True, 'parent_gpu_slot_clearance': True}
            path.write_text(json.dumps(value)); storage.require_clearance(path, plan)
            for field, bad in [('schema', 'megartx-prefill-native-clearance-v1'),
                ('cpu_review_passed', 1), ('parent_gpu_slot_clearance', 1),
                ('source_head', 'f' * 40), ('plan_sha256', '0' * 64)]:
                path.write_text(json.dumps({**value, field: bad}))
                with self.subTest(field=field):
                    with self.assertRaises(ValueError): storage.require_clearance(path, plan)

    def test_counts_match_independent_cpu_contract(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('storage_test_contract', ROOT / storage.CHECKER_SOURCE)
        contract = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = contract
        try:
            spec.loader.exec_module(contract)
            work = contract.capture_work_budget()
            for field in ('capture_frames', 'capture_end', 'single_row_equivalent_reads',
                          'transferred_bytes_per_context', 'transfer_limit_bytes_per_context'):
                self.assertEqual(storage.CONTROL_SPEC[field], work[field])
            budget = contract.evidence_budget((16,) * 30, contexts=1)
            self.assertEqual(storage.CONTROL_SPEC['sample_kv_bytes'], budget['kv_bytes'])
            self.assertEqual(storage.CONTROL_SPEC['raw_head_bytes'], budget['head_bytes'])
            self.assertEqual(sum(storage.RAW_FILES.values()), budget['kv_bytes'] + budget['head_bytes'])
            self.assertEqual(sum(storage.CONTROL_SPEC[k] for k in ('pre_rows','processed_rows','post_rows')),
                             work['single_row_equivalent_reads'])
        finally:
            del sys.modules[spec.name]

    def test_historical_binding_cannot_admit_storage_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root); binding, mode = binding_fixture(plan)
            (root / 'runner-binding.json').write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, 'bounded regular'):
                storage.validate_binding(plan, root, require_live=False)
            path = root / 'storage-binding.json'; path.write_text(json.dumps(mode))
            self.assertEqual(storage.validate_binding(plan, root, require_live=False), binding)
            for key, bad in [('writer_hook_bound', 1), ('owner_pid', 2), ('checker_sha256', '0'*64),
                              ('purpose', 'native_fit')]:
                path.write_text(json.dumps({**mode, key: bad}))
                with self.subTest(key=key):
                    with self.assertRaises(ValueError): storage.validate_binding(plan, root, require_live=False)

    def test_source_catalog_rejects_mixed_previous_plugin_before_plan_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root)
            catalog = root/storage.CATALOG_SOURCE
            value = json.loads(catalog.read_text())
            self.assertEqual(storage.validate_source_catalog(root, plan['source_hashes']), value)
            old = ROOT/'tests/fixtures/prefill-storage-base/src__megartx__vllm_scale_plugin.py.source'
            (root/'src/megartx/vllm_scale_plugin.py').write_bytes(old.read_bytes())
            with self.assertRaisesRegex(ValueError, 'mixed'):
                storage.freeze_plan(plan['tokens'], plan['source_head'], root, plan['checkpoint_identity'])

    def test_storage_catalog_wrong_purpose_and_equivalence_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root)
            path = root/storage.CATALOG_SOURCE; original = json.loads(path.read_text())
            for key, bad in [('purpose','native_fit'), ('default_fit_equivalence',{}), ('gpu_authorized',True)]:
                path.write_text(json.dumps({**original,key:bad}))
                hashes = {**plan['source_hashes'], storage.CATALOG_SOURCE:storage.file_sha(path)}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    storage.validate_source_catalog(root, hashes)


class CompactEvidenceTests(unittest.TestCase):
    def test_metadata_append_limit_includes_unknown_temporary_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evidence = storage.CompactEvidence(root, 5096, 4196)
            evidence.write('events.jsonl', {'v': 'a' * 20}, append=True)
            old = (root / 'events.jsonl').read_bytes()
            (root / '.publication-temp').write_bytes(b'x' * 50)
            with self.assertRaisesRegex(ValueError, 'before write'):
                evidence.write('events.jsonl', {'v': 'b' * 20}, append=True)
            self.assertEqual((root / 'events.jsonl').read_bytes(), old)
            with self.assertRaisesRegex(ValueError, 'before write'):
                evidence.write('storage.json', {'v': 'b' * 20})
            self.assertFalse((root / 'storage.json').exists())

    def test_raw_whitelist_extent_and_total_are_shared_with_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evidence = storage.CompactEvidence(root, 12346, 4196)
            name = 'kv-layer-00-position-0015.bf16'
            evidence.raw(name, bytes(8192))
            evidence.write('summary.json', {'ok': True})
            with self.assertRaisesRegex(ValueError, 'before write'):
                evidence.raw('kv-layer-00-position-0016.bf16', bytes(8192))
            with self.assertRaises((FileExistsError, ValueError)): evidence.raw(name, bytes(8192))
            for name, data in [('not-a-sample.bf16', bytes(2)),
                ('kv-layer-00-position-0016.bf16', bytes(2)), ('../outside.bf16', bytes(2))]:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError): evidence.raw(name, data)
            self.assertLessEqual(evidence.sizes()['total_bytes'], 8250)

    def test_directory_symlink_hardlink_and_lock_symlink_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); target = root / 'target'; target.mkdir()
            link = root / 'link'; link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError): storage.CompactEvidence(link)
            evidence = storage.CompactEvidence(target)
            source = root / 'private'; source.write_bytes(b'unchanged')
            (target / '.budget-lock').symlink_to(source)
            with self.assertRaises(OSError): evidence.write('file.json', {})
            self.assertEqual(source.read_bytes(), b'unchanged')
            (target / '.budget-lock').unlink(); os.link(source, target / 'linked')
            with self.assertRaises(ValueError): evidence.write('file.json', {})

    def test_legacy_full_transcript_and_historical_fit_are_never_written(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d)
            for name in ('frames.jsonl', 'head-invocations.jsonl', 'fit.json'):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, 'Legacy'): evidence.write(name, {})
                    self.assertFalse((Path(d) / name).exists())

    def test_telemetry_projection_preserves_every_record_and_fixed_field_identity(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d)
            value = {'monotonic_ns': 123456789000000, 'unix_ns': 1791144600000000000,
                'fields': storage.TELEMETRY_FIELDS,
                'values': ['32000','10855','100','142.57','70','1215','10501','P0'], 'exit': 0}
            for index in range(10):
                evidence.write('telemetry.jsonl', {**value, 'monotonic_ns': value['monotonic_ns']+index}, append=True)
            lines = (Path(d)/'telemetry.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(lines[0]), storage.TELEMETRY_HEADER)
            self.assertEqual(len(lines), 11)
            for index, line in enumerate(lines[1:]):
                self.assertLessEqual(len(line.encode())+1, 130)
                self.assertEqual(json.loads(line), [value['monotonic_ns']+index, value['unix_ns'], value['values'], 0])
            self.assertEqual(storage.validate_telemetry(d)['records'], 10)
            self.assertEqual(storage.validate_telemetry(d)['minimum_gpu_free_mib'], 10855)
            for key, bad in [('fields', list(reversed(value['fields']))), ('exit', False),
                             ('monotonic_ns', 1.0), ('values', value['values'][:-1]), ('extra', True)]:
                with self.subTest(key=key):
                    with self.assertRaises(ValueError): evidence.write('telemetry.jsonl', {**value, key:bad}, append=True)
            self.assertEqual(len((Path(d)/'telemetry.jsonl').read_text().splitlines()), 11)

    def test_failed_telemetry_is_retained_but_never_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d)
            value = {'monotonic_ns': 1, 'unix_ns': 2, 'fields': storage.TELEMETRY_FIELDS,
                     'values': [''], 'exit': 1}
            evidence.write('telemetry.jsonl', value, append=True)
            self.assertEqual(json.loads((Path(d)/'telemetry.jsonl').read_text().splitlines()[1]), [1,2,[''],1])
            with self.assertRaisesRegex(ValueError, 'Telemetry loss'): storage.validate_telemetry(d)

    def test_failure_reservation_survives_normal_overflow_and_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'prefill-storage'
            evidence = storage.CompactEvidence(root, 4500, 4500)
            evidence.write('normal.json', {'data':'x'*350})
            initial = evidence.sizes()
            self.assertEqual(initial['metadata_bytes'], (root/'normal.json').stat().st_size+2)
            with self.assertRaisesRegex(ValueError, 'before write'):
                evidence.write('overflow.json', {'data':'x'*100})
            self.assertFalse((root/'overflow.json').exists())
            evidence.write(storage.FAILURE_FILE, {'schema':'cpu-failure', 'phase':'post',
                'layer':13, 'position':23, 'slot':23, 'expected_k_sha256':'a'*64,
                'observed_k_sha256':'b'*64, 'expected_v_sha256':'c'*64, 'observed_v_sha256':'c'*64})
            original = (root/storage.FAILURE_FILE).read_bytes()
            self.assertLessEqual(len(original), 4096)
            with self.assertRaises(FileExistsError): evidence.write(storage.FAILURE_FILE, {})
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                evidence.write(storage.FAILURE_FILE, {}, append=True)
            with self.assertRaisesRegex(ValueError, 'forbids'):
                evidence.write('storage.json', {'status':'success'})
            self.assertEqual((root/storage.FAILURE_FILE).read_bytes(), original)
            self.assertLessEqual(evidence.sizes()['metadata_bytes'], 4500)

    def test_failure_larger_than_reserve_rejected_without_consuming_it(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d, 4500, 4500)
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                evidence.write(storage.FAILURE_FILE, {'data':'x'*4096})
            self.assertFalse((Path(d)/storage.FAILURE_FILE).exists())
            evidence.write(storage.FAILURE_FILE, {'status':'failed'})
            self.assertTrue((Path(d)/storage.FAILURE_FILE).is_file())

    def test_competing_failure_writers_can_consume_reservation_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d, 4500, 4500)
            evidence.write('normal.json', {'data':'x'*350})
            code = """from megartx.prefill_storage_plan import CompactEvidence
import sys
try:
    CompactEvidence(sys.argv[1],4500,4500).write('storage-failure.json', {'writer':sys.argv[2]})
except FileExistsError:
    sys.exit(3)
"""
            processes = [subprocess.Popen([sys.executable, '-c', code, d, str(index)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE) for index in range(4)]
            statuses = []
            for process in processes:
                _, error = process.communicate(timeout=20)
                self.assertIn(process.returncode, (0,3), error.decode())
                statuses.append(process.returncode)
            self.assertEqual(statuses.count(0), 1)
            self.assertLessEqual(evidence.sizes()['total_bytes'], 4500)

    def test_cross_process_budget_is_serialized(self):
        with tempfile.TemporaryDirectory() as d:
            code = """from megartx.prefill_storage_plan import CompactEvidence
import sys
try:
    CompactEvidence(sys.argv[1], 5096, 4446).write(sys.argv[2]+'.json', {'payload':'x'*180})
except ValueError:
    sys.exit(3)
"""
            processes = [subprocess.Popen([sys.executable, '-c', code, d, str(i)],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(4)]
            statuses = []
            for process in processes:
                _, err = process.communicate(timeout=20)
                self.assertIn(process.returncode, (0, 3), err.decode())
                statuses.append(process.returncode)
            self.assertEqual(statuses.count(0), 1)
            self.assertLessEqual(storage.CompactEvidence(d, 5096, 4446).sizes()['metadata_bytes'], 350)


class StoragePublicationTests(unittest.TestCase):
    def test_synthetic_proof_reaches_only_separate_intercepted_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root / 'source')
            evidence, _, ownership = receipt_fixture(root / 'evidence', plan)
            with patch.object(evidence, 'write') as write:
                storage.publish_storage(evidence, plan, ownership)
            write.assert_called_once()
            name, receipt = write.call_args.args
            self.assertEqual(name, 'storage.json')
            self.assertEqual(receipt['storage_capture_end'], 2049)
            self.assertEqual(receipt['metadata_only_decode_inputs'], 254)
            for field in storage.QUALIFICATIONS: self.assertIs(receipt[field], False)
            self.assertFalse((evidence.directory / 'fit.json').exists())
            self.assertFalse((evidence.directory / 'storage.json').exists())

    def test_control_counts_roots_and_frontier_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root / 'source')
            _, control, _ = receipt_fixture(root / 'evidence', plan)
            for key in control:
                value = {k: v for k, v in control.items() if k != key}
                with self.subTest(missing=key):
                    with self.assertRaises(ValueError): storage.validate_control(plan, value)
            for key in storage.CONTROL_COUNTS:
                for bad in (control[key] - 1, float(control[key]), True):
                    with self.subTest(field=key, value=bad):
                        with self.assertRaises(ValueError): storage.validate_control(plan, {**control, key: bad})
            for key in storage.QUALIFICATIONS:
                with self.subTest(field=key):
                    with self.assertRaises(ValueError): storage.validate_control(plan, {**control, key: True})
            wrong = copy.deepcopy(control); wrong['frontier']['token_ids_sha256'] = '0'*64
            with self.assertRaises(ValueError): storage.validate_control(plan, wrong, control['frontier']['token_ids_sha256'])

    def test_missing_tampered_raw_and_incomplete_cleanup_never_publish(self):
        for mutation in ('raw-missing', 'raw-bit', 'raw-nonfinite', 'frame-missing', 'head-shift',
                         'stream-stale', 'mode-stale', 'cleanup', 'scratch'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d); plan = source_fixture(root / 'source')
                evidence, control, ownership = receipt_fixture(root / 'evidence', plan)
                raw = evidence.directory / 'kv-layer-00-position-0015.bf16'
                if mutation == 'raw-missing': raw.unlink()
                if mutation == 'raw-bit': raw.write_bytes(b'\x01\x00' + bytes(8190))
                if mutation == 'raw-nonfinite': raw.write_bytes(b'\x80\x7f' + bytes(8190))
                if mutation == 'frame-missing': (evidence.directory / 'control-frames.jsonl').unlink()
                if mutation == 'head-shift':
                    path = evidence.directory / 'heads.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[7]['logit_position'] = 2048
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                if mutation in ('stream-stale', 'mode-stale', 'scratch'):
                    name = {'stream-stale':'client-stream.json', 'mode-stale':'storage-client.json',
                            'scratch':'observer.json'}[mutation]
                    path = evidence.directory / name; value = json.loads(path.read_text())
                    if mutation == 'scratch': value['observer_gpu_scratch']['cap_bytes'] = 16 << 20
                    else: value['plan_sha256'] = '0'*64
                    path.write_text(json.dumps(value))
                if mutation == 'cleanup': ownership['cleanup_complete'] = False
                with patch.object(evidence, 'write') as write:
                    with self.assertRaises((ValueError, OSError)): storage.publish_storage(evidence, plan, ownership)
                    write.assert_not_called()
                self.assertFalse((evidence.directory / 'fit.json').exists())

    def test_required_samples_and_consumed_input_chain_fail_closed_even_with_rebound_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root/'source')
            evidence, control, ownership = receipt_fixture(root/'evidence', plan)
            directory = evidence.directory
            names = ('samples.jsonl','control-frames.jsonl','control.json','storage-client.json')
            originals = {name:(directory/name).read_bytes() for name in names}
            mutations = [('missing',None),('invalid',None),('truncated',None),('duplicate',None),
                         ('sample-hash',None),('decode-input',None),('prompt-input',None)]
            sample = json.loads(originals['samples.jsonl'].splitlines()[0])
            mutations += [('field:'+key,None) for key in sample]
            mutations += [('field-type:'+key,None) for key in ('output_index','head_sequence',
                'cached_length','pending_anchor_position','anchor_kv_written')]
            for mutation, _ in mutations:
                with self.subTest(mutation=mutation):
                    for name, data in originals.items(): (directory/name).write_bytes(data)
                    records = [json.loads(line) for line in originals['samples.jsonl'].splitlines()]
                    frames = [json.loads(line) for line in originals['control-frames.jsonl'].splitlines()]
                    if mutation == 'missing': (directory/'samples.jsonl').unlink()
                    elif mutation == 'invalid': (directory/'samples.jsonl').write_text('{"invalid":true}\n')
                    else:
                        if mutation == 'truncated': records.pop()
                        elif mutation == 'duplicate': records[-1] = records[-2]
                        elif mutation == 'sample-hash': records[100]['sample_sha256'] = 'f'*64
                        elif mutation.startswith('field:'): del records[17][mutation.split(':')[1]]
                        elif mutation.startswith('field-type:'):
                            key = mutation.split(':')[1]
                            records[17][key] = 0 if key == 'anchor_kv_written' else float(records[17][key])
                        elif mutation == 'decode-input': frames[25]['input_ids_sha256'] = 'f'*64
                        elif mutation == 'prompt-input': frames[3]['input_ids_sha256'] = 'f'*64
                        (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
                    (directory/'control-frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames))
                    altered = copy.deepcopy(control)
                    if (directory/'samples.jsonl').exists():
                        altered['sample_records_sha256'] = storage.file_sha(directory/'samples.jsonl')
                    altered['frame_records_sha256'] = storage.file_sha(directory/'control-frames.jsonl')
                    (directory/'control.json').write_text(json.dumps(altered))
                    mode = json.loads(originals['storage-client.json'])
                    mode['control_sha256'] = storage.digest(altered)
                    (directory/'storage-client.json').write_text(json.dumps(mode))
                    with patch.object(evidence, 'write') as write:
                        with self.assertRaises(ValueError): storage.publish_storage(evidence, plan, ownership)
                        write.assert_not_called()

    def test_actual_client_independently_binds_final_uncached_sample_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root/'source')
            _, control, _ = receipt_fixture(root/'evidence', plan)
            stream, observer = stream_fixture(plan)
            client = client_receipt(plan, stream, observer)
            mode = storage_client_receipt(plan, control, client, stream)
            self.assertEqual(mode['sample_hashes_sha256'], storage.digest([storage.digest([17])]*256))
            forged_hashes = [storage.digest([17])]*255+[storage.digest([18])]
            forged = {**control, 'sample_hashes_sha256':storage.digest(forged_hashes)}
            with self.assertRaisesRegex(ValueError, 'scalar sample hash root'):
                storage_client_receipt(plan, forged, client, stream)

    def test_exact_inherited_and_per_group_manager_transfer_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root/'source')
            _, control, _ = receipt_fixture(root/'evidence', plan)
            for domain, bad in [('inherited_metadata_upper_bound',1),
                ('inherited_metadata_upper_bound',41811967), ('inherited_metadata_upper_bound',41811969),
                ('manager_table_metadata',1), ('manager_table_metadata',566016*2-1),
                ('manager_table_metadata',585984*2+1)]:
                altered = copy.deepcopy(control)
                altered['transfer']['domains'][domain] = bad
                altered['transfer']['transferred_bytes'] = sum(altered['transfer']['domains'].values())
                with self.subTest(domain=domain,bad=bad), self.assertRaises(ValueError):
                    storage.validate_control(plan, altered)
            for good in (566016*2, 585984*2):
                altered = copy.deepcopy(control)
                altered['transfer']['domains']['manager_table_metadata'] = good
                altered['transfer']['transferred_bytes'] = sum(altered['transfer']['domains'].values())
                storage.validate_control(plan, altered)

    def test_failure_file_blocks_publication_before_any_success_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = source_fixture(root/'source')
            evidence, _, ownership = receipt_fixture(root/'evidence', plan)
            evidence.write(storage.FAILURE_FILE, {'phase':'post','layer':13,'position':23})
            with patch.object(evidence,'write') as write:
                with self.assertRaisesRegex(ValueError,'forbids'): storage.publish_storage(evidence,plan,ownership)
                write.assert_not_called()

    def test_known_ledger_mismatch_survives_secondary_evidence_failure(self):
        from prefill_diagnostic_client import LedgerMismatch
        plan = {'plan_sha256':'a'*64,'tokens':list(range(2048))}
        stream, observer = stream_fixture(plan)
        def failed_write(*args, **kwargs): raise OSError('secondary disk failure')
        evidence = SimpleNamespace(write=failed_write)
        with self.assertRaises(LedgerMismatch) as caught:
            validate_observation(plan,stream,{**observer,'output_ids_sha256':'f'*64},evidence)
        self.assertIn('output_ids_sha256',caught.exception.diagnostic['failed_fields'])
        self.assertIsInstance(caught.exception.__cause__,OSError)
        if hasattr(caught.exception,'__notes__'):
            self.assertIn('OSError',' '.join(caught.exception.__notes__))
        with self.assertRaisesRegex(OSError,'secondary disk failure'):
            validate_observation(plan,stream,observer,evidence)

    def test_client_primary_http_failure_survives_close_failure_and_cleans_marker(self):
        plan = {'plan_sha256':'a'*64, 'tokens':list(range(2048))}
        def fail_post(*args, **kwargs):
            raise ValueError('primary HTTP failure')
        def fail_close():
            raise RuntimeError('secondary close failure')
        session = SimpleNamespace(post=fail_post, close=fail_close)
        with tempfile.TemporaryDirectory() as d, patch('prefill_storage_client.validate_binding'), \
                patch.dict(sys.modules, {'requests': SimpleNamespace(Session=lambda: session)}):
            with self.assertRaisesRegex(ValueError, 'primary HTTP failure'):
                run(plan, d, float('inf'))
            self.assertFalse((Path(d)/'request.json').exists())
            self.assertTrue((Path(d)/'client-stream.json').is_file())
            self.assertFalse(session.trust_env)

    def test_client_reuses_strict_transport_without_runtime_admission(self):
        plan = {'plan_sha256':'a'*64, 'tokens':list(range(2048))}
        stream, observer = stream_fixture(plan)
        with tempfile.TemporaryDirectory() as d:
            evidence = storage.CompactEvidence(d)
            client = validate_observation(plan, stream, observer, evidence)
            compact = json.loads((Path(d)/'client-ledger-check.json').read_text())
            self.assertEqual(compact['checks'], [])
            self.assertGreater(compact['checked_fields'], 30)
            self.assertEqual(client['emitted_outputs'], 256)
            self.assertEqual(payload(plan)['prompt'], plan['tokens'])
        with self.assertRaises(ValueError): validate_observation(plan, stream, {**observer, 'output_ids_sha256':'0'*64})
        with self.assertRaises(ValueError): StreamLedger(plan).consume('data: [DONE]')
        with patch('prefill_storage_client.validate_binding', side_effect=ValueError('not bound')):
            with self.assertRaisesRegex(ValueError, 'not bound'): run(plan, '/unused', 1)


if __name__ == '__main__': unittest.main()
